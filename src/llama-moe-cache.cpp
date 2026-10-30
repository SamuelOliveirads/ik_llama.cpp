#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-moe-cache-share.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

struct moe_cache_lru {
    struct fill { int32_t expert; int32_t slot; };
    enum class slot_state : uint8_t { FREE, COPYING, READY };

    int32_t n_expert = 0;
    int32_t n_slots = 0;
    std::vector<int32_t *> slot_map;
    std::vector<int32_t> key_of;
    std::vector<uint64_t> slot_generation;
    std::vector<uint32_t> pin_count;
    std::vector<slot_state> state;
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    std::vector<uint32_t> seen;
    std::vector<int32_t> uniq;
    uint32_t seen_gen = 0;
    int32_t head = -1;
    int32_t tail = -1;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        slot_generation.assign(n_slots, 0);
        pin_count.assign(n_slots, 0);
        state.assign(n_slots, slot_state::FREE);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    void touch(int32_t s) {
        if (s == tail) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        prev[next[s]] = prev[s];
        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    void unlink(int32_t s) {
        if (head != s && prev[s] < 0 && next[s] < 0) return;
        if (prev[s] >= 0) next[prev[s]] = next[s]; else head = next[s];
        if (next[s] >= 0) prev[next[s]] = prev[s]; else tail = prev[s];
        prev[s] = next[s] = -1;
    }

    void link_tail(int32_t s) {
        prev[s] = tail;
        next[s] = -1;
        if (tail >= 0) next[tail] = s; else head = s;
        tail = s;
    }

    uint64_t advance_generation(int32_t s) {
        // Advance on assignment and on each pin epoch so a replayed release
        // cannot decrement a later read of the same resident owner.
        if (++slot_generation[s] == 0) ++slot_generation[s];
        return slot_generation[s];
    }

    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit, size_t & n_eviction) {
        fills.clear();
        n_hit = 0;
        n_eviction = 0;
        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) return false;
        int32_t * slots = slot_map[il];
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                ++n_hit;
            }
        }
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) continue;
            const int32_t s = head;
            if (key_of[s] >= 0) {
                const int32_t old = key_of[s];
                slot_map[old / n_expert][old % n_expert] = -1;
                ++n_eviction;
            }
            key_of[s] = il*n_expert + e;
            advance_generation(s);
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

static std::vector<ggml_tensor *> layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_up_gate_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t) res.push_back(t);
    }
    return res;
}

static bool is_host_weight(const ggml_tensor * t) {
    return t && t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

static bool same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) return false;
    }
    return true;
}

} // namespace

struct llama_moe_cache::impl {
    struct device {
        ggml_backend * backend = nullptr;
        ggml_backend_buffer_type * buft = nullptr;
        int32_t model_device = -1;
        double split = 0.0;
        size_t host_bytes = 0;
        size_t buffer_size = 0;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
    };
    struct group {
        int32_t device = -1;
        std::vector<ggml_tensor *> ref;
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks;
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        uint64_t generation = 0;
        bool frozen = false;
        moe_cache_lru lru;
    };
    struct layer {
        int32_t group = -1;
        ggml_tensor * slot_map = nullptr;
        ggml_tensor * host_up = nullptr;
        ggml_tensor * host_gate = nullptr;
        ggml_tensor * host_down = nullptr;
        std::vector<ggml_tensor *> experts;
    };
    struct binding {
        int32_t layer;
        int32_t part;
        ggml_tensor * cached;
    };
    struct stats { size_t hits = 0; size_t misses = 0; size_t evictions = 0; size_t bytes = 0; };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings;
    std::unordered_map<const ggml_tensor *, int32_t> layer_of;
    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;
    stats stats_small, stats_large;
    size_t requested_size;
    mutable size_t large_batch_skips = 0;
    mutable size_t incompatible_layer_skips = 0;
    mutable std::mutex state_mutex;

    impl(const llama_model & model, const std::vector<ggml_backend *> & backends,
            const std::vector<ggml_backend_buffer_type *> & bufts, size_t size)
            : n_expert_used(model.hparams.n_expert_used), layers(model.layers.size()), requested_size(size) {
        if (model.hparams.n_expert == 0 || n_expert_used <= 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (!model.rpc_servers.empty()) throw std::runtime_error("MoE cache does not support RPC devices");
        if (backends.size() != bufts.size()) throw std::runtime_error("MoE cache backend metadata is inconsistent");

        for (size_t i = 0; i < backends.size(); ++i) {
            if (!ggml_backend_is_cpu(backends[i])) {
                device d;
                d.backend = backends[i];
                d.buft = bufts[i];
                devices.push_back(std::move(d));
            }
        }
        if (devices.empty()) throw std::runtime_error("MoE cache requires a GPU backend");

        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), is_host_weight)) continue;
            if (!std::all_of(experts.begin(), experts.end(), [&](const ggml_tensor * t) {
                    return t->ne[2] == experts[0]->ne[2];
                })) continue;
            if (il >= model.default_layer_device.size()) continue;
            const int32_t model_device = model.default_layer_device[il];
            if (model_device < 0 || (size_t) model_device >= model.devices.size()) continue;
            const auto * layer_buft = model.default_buffer_type_offload(model_device);
            auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return d.buft == layer_buft; });
            if (it_dev == devices.end()) continue;
            const int32_t id = (int32_t) (it_dev - devices.begin());
            it_dev->model_device = model_device;
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.device == id && same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->device = id;
                it->ref = experts;
            }
            it->layers.push_back((int32_t) il);
            for (ggml_tensor * t : experts) {
                it->host_bytes += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no complete host-resident expert layer is eligible. MoE cache is inactive\n", __func__);
            return;
        }

        std::vector<bool> eligible(model.splits.size(), false);
        for (const device & d : devices) {
            if (d.host_bytes && d.model_device >= 0 && (size_t) d.model_device < eligible.size()) {
                eligible[d.model_device] = true;
            }
        }
        const std::vector<double> shares = llama_moe_cache_detail::device_shares(model.splits, eligible);
        for (device & d : devices) {
            d.split = d.model_device >= 0 && (size_t) d.model_device < shares.size() ? shares[d.model_device] : 0.0;
        }

        double split_sum = 0.0;
        for (const device & d : devices) if (d.host_bytes) split_sum += d.split;
        if (split_sum <= 0.0) {
            for (device & d : devices) if (d.host_bytes) d.split = 1.0;
            split_sum = std::count_if(devices.begin(), devices.end(), [](const device & d) { return d.host_bytes != 0; });
        }
        auto alloc_size = [&](const group & g, int32_t slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.device].buft);
            size_t total = 0;
            for (const ggml_tensor * t : g.ref) {
                const size_t bytes = t->nb[2] * (size_t) (slots + 1);
                if (bytes > SIZE_MAX - alignment) return SIZE_MAX;
                const size_t padded = GGML_PAD(bytes, alignment);
                if (padded > SIZE_MAX - total) return SIZE_MAX;
                total += padded;
            }
            return total;
        };

        size_t host_layer_count = 0;
        std::vector<size_t> device_tensor_counts(devices.size(), 0);
        for (group & g : groups) {
            const device & d = devices[g.device];
            const size_t budget = (size_t) ((double) size * d.split / split_sum * g.host_bytes / d.host_bytes);
            const int32_t n_expert = (int32_t) g.ref[0]->ne[2];
            const int32_t max_slots = (int32_t) std::min<size_t>(INT32_MAX, g.layers.size() * (size_t) n_expert);
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) ++g.n_slots;
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: cache budget leaves %zu expert layers uncached because they cannot hold one token\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            g.lru.init((int32_t) layers.size(), n_expert, g.n_slots);
            device_tensor_counts[g.device] += g.ref.size() * (1 + g.layers.size());
            host_layer_count += g.layers.size();
        }
        if (host_layer_count == 0) throw std::runtime_error("MoE cache budget is too small to hold one token");

        auto make_ctx = [](size_t n_tensors) {
            ggml_init_params params = { n_tensors * ggml_tensor_overhead(), nullptr, true };
            ggml_context_ptr ctx(ggml_init(params));
            if (!ctx) throw std::runtime_error("failed to create MoE cache tensor metadata");
            return ctx;
        };
        for (size_t id = 0; id < devices.size(); ++id) if (device_tensor_counts[id]) devices[id].ctx = make_ctx(device_tensor_counts[id]);
        ctx_host = make_ctx(host_layer_count);
        const auto buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (group & g : groups) {
            if (g.n_slots == 0) continue;
            auto & d = devices[g.device];
            for (ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(d.ctx.get(), t->type, t->ne[0], t->ne[1], g.n_slots + 1);
                if (bank->nb[2] != t->nb[2]) throw std::runtime_error("MoE cache expert layout is not contiguous by expert");
                ggml_format_name(bank, "moe_cache.%s", t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.group = (int32_t) (&g - groups.data());
                l.experts = layer_experts(model.layers[il]);
                l.host_up = model.layers[il].ffn_up_exps;
                l.host_gate = model.layers[il].ffn_gate_exps;
                l.host_down = model.layers[il].ffn_down_exps;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(d.ctx.get(), bank, bank->ne[0], bank->ne[1], g.n_slots,
                            bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
            }
            d.buffer_size += alloc_size(g, g.n_slots);
        }

        for (device & d : devices) {
            if (!d.ctx) continue;
            d.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
            if (!d.buffer) throw std::runtime_error("failed to allocate MoE cache buffers");
            ggml_backend_buffer_clear(d.buffer.get(), 0);
            d.buffer_size = ggml_backend_buffer_get_size(d.buffer.get());
        }
        buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
        if (!buf_host) throw std::runtime_error("failed to allocate MoE cache slot maps");
        ggml_backend_buffer_clear(buf_host.get(), 0xff);
        buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
        for (group & g : groups) for (int32_t il : g.layers) g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
        for (device & d : devices) if (d.buffer) ggml_backend_buffer_set_usage(d.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if (buf_host) ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        for (size_t i = 0; i < devices.size(); ++i) {
            const device & d = devices[i];
            if (!d.host_bytes || !d.buffer_size) continue;
            LLAMA_LOG_INFO("%s: %s MoE cache uses %.2f MiB for %.2f MiB of host experts\n", __func__,
                    ggml_backend_buft_name(d.buft), d.buffer_size / 1048576.0, d.host_bytes / 1048576.0);
            for (const group & g : groups) if (g.device == (int32_t) i && g.n_slots) {
                LLAMA_LOG_INFO("%s: %zu layer(s), %d expert slots (%.1f%% coverage)\n", __func__,
                        g.layers.size(), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]));
            }
        }
    }

    ~impl() {
        GGML_ASSERT(std::none_of(groups.begin(), groups.end(), [](const group & g) { return g.frozen; }) &&
                "MoE cache destroyed while a frozen group lease is active");
        log_stats();
    }

    ggml_backend * backend(int32_t il) const {
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return nullptr;
        return devices[groups[layers[il].group].device].backend;
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        const layer & l = layers[il];
        const group & g = groups[l.group];
        if (g.frozen) return nullptr;
        if (n_tokens <= 0 || n_tokens > max_batch) {
            ++large_batch_skips;
            return nullptr;
        }
        if (std::min<int64_t>(n_tokens*n_expert_used, l.slot_map->ne[1]) > g.n_slots) {
            ++incompatible_layer_skips;
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * t) const {
        auto it = bindings.find(t);
        return it == bindings.end() ? nullptr : it->second.cached;
    }

    bool copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
            ggml_backend_sched_copy_phase phase) {
        auto it = layer_of.find(src);
        if (it == layer_of.end()) return false;
        std::lock_guard<std::mutex> lock(state_mutex);
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.group];
        if (g.frozen) return false;
        if (phase == GGML_BACKEND_SCHED_COPY_PHASE_QUERY) return true;
        const device & d = devices[g.device];
        GGML_ASSERT(backend == d.backend);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < ggml_graph_n_nodes(graph) && !lookup; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) lookup = node;
        }
        GGML_ASSERT(lookup && "MoE cache slot map must feed GET_ROWS");
        const ggml_tensor * selected = lookup->src[1];
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == selected || node == selected->view_src) {
                GGML_ABORT("MoE experts for layer %d are selected in the same scheduler split as the cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(selected));
        ids.resize(ggml_nelements(selected));
        ggml_backend_tensor_get_async(backend, selected, ids.data(), 0, ggml_nbytes(selected));
        ggml_backend_synchronize(backend);
        size_t n_hit = 0;
        size_t n_eviction = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit, n_eviction)) GGML_ABORT("MoE cache slot count is smaller than the routed expert set");
        if (++g.generation == 0) ++g.generation;
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i+n < fills.size() && fills[i+n].expert == fills[i].expert + (int32_t)n &&
                        fills[i+n].slot == fills[i].slot + (int32_t)n) ++n;
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size,
                        fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits += n_hit;
        st.misses += fills.size();
        st.evictions += n_eviction;
        st.bytes += bytes;
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));
        return true;
    }

    bool freeze_group_for_layer(int32_t il, ggml_backend * expected_backend,
            const std::vector<int32_t> & seed_experts, llama_moe_cache_frozen_layer & out) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (il < 0 || (size_t) il >= layers.size() || layers[il].group < 0) return false;
        layer & l = layers[il];
        group & g = groups[l.group];
        device & d = devices[g.device];
        if (g.frozen || d.backend != expected_backend || !expected_backend ||
                l.host_up == nullptr || l.host_gate == nullptr || l.host_down == nullptr ||
                l.experts.size() != g.banks.size()) return false;

        const int32_t n_experts = g.lru.n_expert;
        std::vector<uint8_t> seen(n_experts, 0);
        size_t n_unique = 0;
        for (const int32_t expert : seed_experts) {
            if (expert < 0 || expert >= n_experts) return false;
            if (!seen[expert]) { seen[expert] = 1; ++n_unique; }
        }
        if (n_unique > (size_t) g.n_slots) return false;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            if (!l.experts[ip] || !l.experts[ip]->data || !g.banks[ip]) return false;
            auto it = bindings.find(l.experts[ip]);
            if (it == bindings.end() || !it->second.cached) return false;
        }

        // This is invocation-local setup state. The ordinary cache uses the
        // shared fills scratch, but a prior seed plan must never leak into an
        // empty freeze of another layer in this group.
        std::vector<moe_cache_lru::fill> seed_fills;
        if (!seed_experts.empty()) {
            size_t n_hit = 0;
            size_t n_eviction = 0;
            if (!g.lru.plan(il, seed_experts.data(), seed_experts.size(), seed_fills, n_hit, n_eviction)) return false;
        }
        size_t seed_bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * weights = l.experts[ip];
            ggml_tensor * bank = g.banks[ip];
            const size_t expert_size = weights->nb[2];
            for (size_t i = 0; i < seed_fills.size();) {
                size_t n = 1;
                while (i + n < seed_fills.size() && seed_fills[i+n].expert == seed_fills[i].expert + (int32_t)n &&
                        seed_fills[i+n].slot == seed_fills[i].slot + (int32_t)n) ++n;
                ggml_backend_tensor_set_async(d.backend, bank,
                        (const uint8_t *) weights->data + seed_fills[i].expert * expert_size,
                        seed_fills[i].slot * expert_size, n * expert_size);
                seed_bytes += n * expert_size;
                i += n;
            }
        }
        if (!seed_fills.empty()) ggml_backend_synchronize(d.backend);

        for (int32_t s = 0; s < g.lru.n_slots; ++s) {
            g.lru.state[s] = g.lru.key_of[s] >= 0 ? moe_cache_lru::slot_state::READY : moe_cache_lru::slot_state::FREE;
        }

        if (++g.generation == 0) ++g.generation;
        g.frozen = true;
        out = {};
        out.layer = il;
        out.group = l.group;
        out.device_group = g.device;
        out.owner_backend = d.backend;
        out.n_experts = n_experts;
        out.n_slots = g.n_slots;
        out.generation = g.generation;
        out.seed_bytes = seed_bytes;
        out.host_up = l.host_up;
        out.host_gate = l.host_gate;
        out.host_down = l.host_down;
        auto bank_for = [&](const ggml_tensor * host) -> ggml_tensor * {
            auto it = bindings.find(host);
            return it == bindings.end() ? nullptr : g.banks[it->second.part];
        };
        auto cached_for = [&](const ggml_tensor * host) -> ggml_tensor * {
            auto it = bindings.find(host);
            return it == bindings.end() ? nullptr : it->second.cached;
        };
        out.bank_up = bank_for(out.host_up);
        out.bank_gate = bank_for(out.host_gate);
        out.bank_down = bank_for(out.host_down);
        out.cached_up = cached_for(out.host_up);
        out.cached_gate = cached_for(out.host_gate);
        out.cached_down = cached_for(out.host_down);
        out.group_layers = g.layers;
        GGML_ASSERT(out.bank_up && out.bank_gate && out.bank_down && out.cached_up && out.cached_gate && out.cached_down);
        return true;
    }

    bool frozen_snapshot_valid(const llama_moe_cache_frozen_layer & snapshot) const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (snapshot.group < 0 || (size_t) snapshot.group >= groups.size()) return false;
        const group & g = groups[snapshot.group];
        if (!g.frozen || snapshot.generation != g.generation || snapshot.layer < 0 ||
                (size_t) snapshot.layer >= layers.size() || layers[snapshot.layer].group != snapshot.group) return false;
        const layer & l = layers[snapshot.layer];
        auto same_view = [&](const ggml_tensor * host, ggml_tensor * cached, const ggml_tensor * bank) {
            auto it = bindings.find(host);
            return it != bindings.end() && it->second.layer == snapshot.layer &&
                    it->second.cached == cached && g.banks[it->second.part] == bank;
        };
        return snapshot.host_up == l.host_up && snapshot.host_gate == l.host_gate && snapshot.host_down == l.host_down &&
                same_view(l.host_up, snapshot.cached_up, snapshot.bank_up) &&
                same_view(l.host_gate, snapshot.cached_gate, snapshot.bank_gate) &&
                same_view(l.host_down, snapshot.cached_down, snapshot.bank_down);
    }

    bool frozen_layer_snapshot(const llama_moe_cache_frozen_layer & group_snapshot, int32_t il,
            llama_moe_cache_frozen_layer & snapshot) const {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (group_snapshot.group < 0 || (size_t) group_snapshot.group >= groups.size() ||
                il < 0 || (size_t) il >= layers.size()) return false;
        const group & g = groups[group_snapshot.group];
        if (!g.frozen || g.generation != group_snapshot.generation ||
                group_snapshot.layer < 0 || (size_t) group_snapshot.layer >= layers.size() ||
                layers[group_snapshot.layer].group != group_snapshot.group ||
                layers[il].group != group_snapshot.group ||
                std::find(g.layers.begin(), g.layers.end(), il) == g.layers.end()) return false;
        const layer & anchor = layers[group_snapshot.layer];
        const layer & l = layers[il];
        auto matches = [&](const ggml_tensor * host, ggml_tensor * cached, const ggml_tensor * bank,
                const ggml_tensor * expected_host) {
            const auto it = bindings.find(expected_host);
            return it != bindings.end() && it->second.layer == group_snapshot.layer &&
                    it->second.cached == cached && g.banks[it->second.part] == bank && host == expected_host;
        };
        if (!matches(group_snapshot.host_up, group_snapshot.cached_up, group_snapshot.bank_up, anchor.host_up) ||
                !matches(group_snapshot.host_gate, group_snapshot.cached_gate, group_snapshot.bank_gate, anchor.host_gate) ||
                !matches(group_snapshot.host_down, group_snapshot.cached_down, group_snapshot.bank_down, anchor.host_down) ||
                !l.host_up || !l.host_gate || !l.host_down || l.experts.size() != g.banks.size()) return false;
        auto bank_for = [&](const ggml_tensor * host) -> const ggml_tensor * {
            const auto it = bindings.find(host);
            return it == bindings.end() ? nullptr : g.banks[it->second.part];
        };
        auto cached_for = [&](const ggml_tensor * host) -> ggml_tensor * {
            const auto it = bindings.find(host);
            return it == bindings.end() ? nullptr : it->second.cached;
        };
        snapshot = {};
        snapshot.layer = il;
        snapshot.group = group_snapshot.group;
        snapshot.device_group = g.device;
        snapshot.owner_backend = devices[g.device].backend;
        snapshot.n_experts = g.lru.n_expert;
        snapshot.n_slots = g.n_slots;
        snapshot.generation = g.generation;
        snapshot.host_up = l.host_up;
        snapshot.host_gate = l.host_gate;
        snapshot.host_down = l.host_down;
        snapshot.bank_up = bank_for(l.host_up);
        snapshot.bank_gate = bank_for(l.host_gate);
        snapshot.bank_down = bank_for(l.host_down);
        snapshot.cached_up = cached_for(l.host_up);
        snapshot.cached_gate = cached_for(l.host_gate);
        snapshot.cached_down = cached_for(l.host_down);
        snapshot.group_layers = g.layers;
        return snapshot.bank_up && snapshot.bank_gate && snapshot.bank_down &&
                snapshot.cached_up && snapshot.cached_gate && snapshot.cached_down;
    }

    bool release_frozen_group(const llama_moe_cache_frozen_layer & snapshot) {
        std::lock_guard<std::mutex> lock(state_mutex);
        if (snapshot.group < 0 || (size_t) snapshot.group >= groups.size()) return false;
        group & g = groups[snapshot.group];
        if (!g.frozen || snapshot.generation != g.generation) return false;
        for (int32_t s = 0; s < g.lru.n_slots; ++s) {
            if (g.lru.pin_count[s] != 0 || g.lru.state[s] == moe_cache_lru::slot_state::COPYING) return false;
        }
        g.frozen = false;
        if (++g.generation == 0) ++g.generation;
        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & s) {
            const size_t n = s.hits + s.misses;
            if (n) LLAMA_LOG_INFO("llama_moe_cache: %s hits=%zu misses=%zu evictions=%zu hit-rate=%.2f%% uploaded=%.2f MiB\n",
                    name, s.hits, s.misses, s.evictions, 100.0*s.hits/n, s.bytes/1048576.0);
        };
        size_t actual_size = buf_host_size;
        for (const device & d : devices) actual_size += d.buffer_size;
        LLAMA_LOG_INFO("llama_moe_cache: requested %.2f MiB, allocated %.2f MiB\n",
                requested_size / 1048576.0, actual_size / 1048576.0);
        log("ubatch <= 8", stats_small);
        log("ubatch 9-32", stats_large);
        if (large_batch_skips || incompatible_layer_skips) {
            LLAMA_LOG_INFO("llama_moe_cache: skipped %zu large-batch and %zu ineligible layer accesses\n",
                    large_batch_skips, incompatible_layer_skips);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model,
        const std::vector<ggml_backend *> & backends,
        const std::vector<ggml_backend_buffer_type *> & bufts, size_t size)
        : pimpl(new impl(model, backends, bufts, size)) {}

llama_moe_cache::~llama_moe_cache() = default;

llama_moe_cache::frozen_lease::frozen_lease(frozen_lease && other) noexcept
        : owner(other.owner), descriptor(std::move(other.descriptor)) {
    other.owner = nullptr;
    other.descriptor = {};
}

llama_moe_cache::frozen_lease & llama_moe_cache::frozen_lease::operator=(frozen_lease && other) noexcept {
    if (this != &other) {
        if (owner && !release()) GGML_ABORT("failed to release frozen MoE cache group during lease move assignment");
        owner = other.owner;
        descriptor = std::move(other.descriptor);
        other.owner = nullptr;
        other.descriptor = {};
    }
    return *this;
}

llama_moe_cache::frozen_lease::~frozen_lease() {
    if (owner && !release()) GGML_ABORT("failed to release frozen MoE cache group during lease destruction");
}

bool llama_moe_cache::frozen_lease::release() {
    if (!owner) return false;
    if (!owner->release_frozen_group(descriptor)) return false;
    owner = nullptr;
    descriptor = {};
    return true;
}

ggml_backend * llama_moe_cache::backend(int32_t il) const { return pimpl->backend(il); }
ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const { return pimpl->get_slot_map(il, n_tokens, n_expert_used); }
ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * t) const { return pimpl->get_experts(t); }
bool llama_moe_cache::copy(ggml_backend * backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph,
        ggml_backend_sched_copy_phase phase) { return pimpl->copy(backend, src, dst, graph, phase); }

bool llama_moe_cache::resolve_and_pin(int32_t device, int32_t group_id, int32_t il, uint64_t owner_generation,
        const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result) {
    if (!ids || !result || n_ids <= 0 || n_ids > GGML_CUDA_MOE_FFN_MAX_SELECTED) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    if (group_id < 0 || (size_t) group_id >= pimpl->groups.size() ||
            il < 0 || (size_t) il >= pimpl->layers.size()) return false;
    auto & g = pimpl->groups[group_id];
    auto & l = pimpl->layers[il];
    if (!g.frozen || g.generation != owner_generation || g.device != device ||
            l.group != group_id || !g.lru.slot_map[il]) return false;

    ggml_cuda_moe_ffn_resolution resolved{};
    resolved.n_draws = n_ids;
    for (int32_t i = 0; i < n_ids; ++i) {
        const int32_t expert = ids[i];
        if (expert < 0 || expert >= g.lru.n_expert) return false;
        const int32_t slot = g.lru.slot_map[il][expert];
        if (slot < -1 || slot >= g.lru.n_slots) return false;
        resolved.draw_slots[i] = slot;
        if (slot < 0) continue;
        if (g.lru.state[slot] != moe_cache_lru::slot_state::READY ||
                g.lru.key_of[slot] != il*g.lru.n_expert + expert ||
                g.lru.slot_generation[slot] == 0 || g.lru.pin_count[slot] != 0) return false;
        bool already_added = false;
        for (int32_t p = 0; p < resolved.n_pins; ++p) {
            already_added |= resolved.pins[p].slot == slot;
        }
        if (!already_added) {
            auto & token = resolved.pins[resolved.n_pins++];
            token.owner = this;
            token.device = device;
            token.group = group_id;
            token.layer = il;
            token.slot = slot;
            token.key = il*g.lru.n_expert + expert;
            token.owner_generation = owner_generation;
        }
    }

    for (int32_t p = 0; p < resolved.n_pins; ++p) {
        auto & token = resolved.pins[p];
        token.generation = g.lru.advance_generation(token.slot);
        ++g.lru.pin_count[token.slot];
        g.lru.touch(token.slot);
    }
    *result = resolved;
    return true;
}

bool llama_moe_cache::release_pins(const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins) {
    if (!pins || n_pins <= 0 || n_pins > GGML_CUDA_MOE_FFN_MAX_SELECTED) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    for (int32_t i = 0; i < n_pins; ++i) {
        const auto & token = pins[i];
        if (token.owner != this || token.group < 0 || (size_t) token.group >= pimpl->groups.size() ||
                token.layer < 0 || (size_t) token.layer >= pimpl->layers.size()) return false;
        const auto & g = pimpl->groups[token.group];
        const auto & l = pimpl->layers[token.layer];
        if (!g.frozen || g.device != token.device || l.group != token.group ||
                g.generation != token.owner_generation ||
                token.key < 0 || token.key / g.lru.n_expert != token.layer ||
                token.slot < 0 || token.slot >= g.lru.n_slots || !g.lru.slot_map[token.layer] ||
                g.lru.state[token.slot] != moe_cache_lru::slot_state::READY ||
                g.lru.key_of[token.slot] != token.key ||
                g.lru.slot_generation[token.slot] != token.generation ||
                g.lru.pin_count[token.slot] != 1 ||
                g.lru.slot_map[token.layer][token.key % g.lru.n_expert] != token.slot) return false;
        for (int32_t j = 0; j < i; ++j) {
            if (pins[j].owner == token.owner && pins[j].group == token.group &&
                    pins[j].slot == token.slot && pins[j].generation == token.generation) return false;
        }
    }
    for (int32_t i = 0; i < n_pins; ++i) --pimpl->groups[pins[i].group].lru.pin_count[pins[i].slot];
    return true;
}

bool llama_moe_cache::reserve_fill(int32_t device, int32_t group_id, int32_t il,
        uint64_t owner_generation, int32_t expert, ggml_cuda_moe_ffn_fill_job * job) {
    if (!job) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    if (group_id < 0 || (size_t) group_id >= pimpl->groups.size() ||
            il < 0 || (size_t) il >= pimpl->layers.size()) return false;
    auto & g = pimpl->groups[group_id];
    auto & l = pimpl->layers[il];
    if (!g.frozen || g.generation != owner_generation || g.device != device ||
            l.group != group_id || expert < 0 || expert >= g.lru.n_expert ||
            !g.lru.slot_map[il] || l.experts.size() != GGML_CUDA_MOE_FFN_MAX_FILL_PARTS ||
            g.banks.size() != l.experts.size() || g.lru.slot_map[il][expert] >= 0) return false;

    const int64_t key64 = int64_t(il) * g.lru.n_expert + expert;
    if (key64 < 0 || key64 > std::numeric_limits<int32_t>::max()) return false;
    const int32_t key = (int32_t) key64;
    for (int32_t s = 0; s < g.lru.n_slots; ++s) {
        if (g.lru.state[s] == moe_cache_lru::slot_state::COPYING && g.lru.key_of[s] == key) return false;
    }

    int32_t slot = -1;
    int32_t old_key = -1;
    for (int32_t s = 0; s < g.lru.n_slots; ++s) {
        if (g.lru.state[s] == moe_cache_lru::slot_state::FREE &&
                g.lru.key_of[s] < 0 && g.lru.pin_count[s] == 0) { slot = s; break; }
    }
    if (slot < 0) {
        for (int32_t s = g.lru.head; s >= 0; s = g.lru.next[s]) {
            if (g.lru.state[s] != moe_cache_lru::slot_state::READY ||
                    g.lru.key_of[s] < 0 || g.lru.pin_count[s] != 0) continue;
            const int32_t resident_key = g.lru.key_of[s];
            const int32_t resident_layer = resident_key / g.lru.n_expert;
            const int32_t resident_expert = resident_key % g.lru.n_expert;
            if (resident_layer < 0 || (size_t) resident_layer >= pimpl->layers.size() ||
                    pimpl->layers[resident_layer].group != group_id ||
                    !g.lru.slot_map[resident_layer] ||
                    g.lru.slot_map[resident_layer][resident_expert] != s) return false;
            slot = s;
            old_key = resident_key;
            break;
        }
    }
    if (slot < 0) return false;

    ggml_cuda_moe_ffn_fill_job pending{};
    pending.owner = this;
    pending.device = device;
    pending.group = group_id;
    pending.layer = il;
    pending.slot = slot;
    pending.key = key;
    pending.owner_generation = owner_generation;
    pending.n_parts = GGML_CUDA_MOE_FFN_MAX_FILL_PARTS;
    const ggml_tensor * canonical_parts[GGML_CUDA_MOE_FFN_MAX_FILL_PARTS] = {
        l.host_up, l.host_gate, l.host_down,
    };
    for (int32_t ip = 0; ip < pending.n_parts; ++ip) {
        const ggml_tensor * weights = canonical_parts[ip];
        const auto part = std::find(l.experts.begin(), l.experts.end(), weights);
        if (!weights || part == l.experts.end()) return false;
        const size_t bank_index = std::distance(l.experts.begin(), part);
        ggml_tensor * bank = g.banks[bank_index];
        if (!weights || !weights->data || !bank || !bank->data || weights->nb[2] == 0 ||
                (size_t) expert > std::numeric_limits<size_t>::max() / weights->nb[2] ||
                (size_t) slot > std::numeric_limits<size_t>::max() / weights->nb[2]) return false;
        const size_t bytes = weights->nb[2];
        const size_t source_offset = (size_t) expert * bytes;
        const size_t destination_offset = (size_t) slot * bytes;
        const size_t source_bytes = ggml_nbytes(weights);
        const size_t bank_bytes = ggml_nbytes(bank);
        if (source_offset > source_bytes || bytes > source_bytes - source_offset ||
                destination_offset > bank_bytes || bytes > bank_bytes - destination_offset ||
                pending.total_bytes > std::numeric_limits<size_t>::max() - bytes) return false;
        pending.parts[ip] = { (const uint8_t *) weights->data + source_offset,
                (uint8_t *) bank->data + destination_offset, bytes };
        pending.total_bytes += bytes;
    }
    if (pending.total_bytes == 0) return false;

    if (old_key >= 0) {
        const int32_t old_layer = old_key / g.lru.n_expert;
        const int32_t old_expert = old_key % g.lru.n_expert;
        g.lru.slot_map[old_layer][old_expert] = -1;
    }
    pending.generation = g.lru.advance_generation(slot);
    g.lru.key_of[slot] = key;
    g.lru.state[slot] = moe_cache_lru::slot_state::COPYING;
    g.lru.unlink(slot);
    *job = pending;
    return true;
}

bool llama_moe_cache::finish_fill(const ggml_cuda_moe_ffn_fill_job * job, bool copied) {
    if (!job || job->owner != this || job->n_parts != GGML_CUDA_MOE_FFN_MAX_FILL_PARTS) return false;
    std::lock_guard<std::mutex> lock(pimpl->state_mutex);
    if (job->group < 0 || (size_t) job->group >= pimpl->groups.size() ||
            job->layer < 0 || (size_t) job->layer >= pimpl->layers.size()) return false;
    auto & g = pimpl->groups[job->group];
    auto & l = pimpl->layers[job->layer];
    if (!g.frozen || g.generation != job->owner_generation || g.device != job->device ||
            l.group != job->group || job->slot < 0 || job->slot >= g.lru.n_slots ||
            job->key < 0 || job->key / g.lru.n_expert != job->layer ||
            !g.lru.slot_map[job->layer] || g.lru.state[job->slot] != moe_cache_lru::slot_state::COPYING ||
            g.lru.key_of[job->slot] != job->key || g.lru.slot_generation[job->slot] != job->generation ||
            g.lru.pin_count[job->slot] != 0 || g.lru.slot_map[job->layer][job->key % g.lru.n_expert] >= 0) return false;

    if (copied) {
        g.lru.slot_map[job->layer][job->key % g.lru.n_expert] = job->slot;
        g.lru.state[job->slot] = moe_cache_lru::slot_state::READY;
        g.lru.link_tail(job->slot);
    } else {
        g.lru.key_of[job->slot] = -1;
        g.lru.state[job->slot] = moe_cache_lru::slot_state::FREE;
        g.lru.link_tail(job->slot);
    }
    return true;
}

namespace {
bool llama_moe_cache_resolve_callback(void * context, int32_t device, int32_t group, int32_t layer,
        uint64_t owner_generation, const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result) {
    return context && static_cast<llama_moe_cache *>(context)->resolve_and_pin(
            device, group, layer, owner_generation, ids, n_ids, result);
}

bool llama_moe_cache_release_callback(void * context, const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins) {
    return context && static_cast<llama_moe_cache *>(context)->release_pins(pins, n_pins);
}

bool llama_moe_cache_reserve_fill_callback(void * context, int32_t device, int32_t group,
        int32_t layer, uint64_t owner_generation, int32_t expert, ggml_cuda_moe_ffn_fill_job * job) {
    return context && static_cast<llama_moe_cache *>(context)->reserve_fill(
            device, group, layer, owner_generation, expert, job);
}

bool llama_moe_cache_finish_fill_callback(void * context, const ggml_cuda_moe_ffn_fill_job * job, bool copied) {
    return context && static_cast<llama_moe_cache *>(context)->finish_fill(job, copied);
}
} // namespace

const ggml_cuda_moe_ffn_cache_callbacks * llama_moe_cache_get_cuda_callbacks() {
    static const ggml_cuda_moe_ffn_cache_callbacks callbacks = {
        llama_moe_cache_resolve_callback,
        llama_moe_cache_release_callback,
        llama_moe_cache_reserve_fill_callback,
        llama_moe_cache_finish_fill_callback,
    };
    return &callbacks;
}

bool llama_moe_cache::freeze_group_for_layer(int32_t il, ggml_backend * expected_backend,
        const std::vector<int32_t> & seed_experts, frozen_lease & lease) {
    if (lease) return false;
    llama_moe_cache_frozen_layer descriptor;
    if (!pimpl->freeze_group_for_layer(il, expected_backend, seed_experts, descriptor)) return false;
    lease.owner = this;
    lease.descriptor = std::move(descriptor);
    return true;
}

bool llama_moe_cache::frozen_snapshot_valid(const llama_moe_cache_frozen_layer & descriptor) const {
    return pimpl->frozen_snapshot_valid(descriptor);
}

bool llama_moe_cache::frozen_layer_snapshot(const llama_moe_cache_frozen_layer & group_snapshot,
        int32_t il, llama_moe_cache_frozen_layer & snapshot) const {
    return pimpl->frozen_layer_snapshot(group_snapshot, il, snapshot);
}

bool llama_moe_cache::release_frozen_group(const llama_moe_cache_frozen_layer & descriptor) {
    return pimpl->release_frozen_group(descriptor);
}
