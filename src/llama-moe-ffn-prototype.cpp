#include "llama-moe-ffn-prototype.h"

#include "llama-model.h"

#include "ggml-moe-ffn.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace {

constexpr size_t CPU_GRAPH_CONTEXT_BYTES = 1024 * 1024;
constexpr int32_t N_EXPERTS_USED = 10;
constexpr int32_t N_EMBD = 2560;
constexpr int32_t N_FF = 640;

bool model_layer_is_eligible(const llama_model & model, int32_t il) {
    if (il < 0 || (size_t) il >= model.layers.size()) return false;
    const auto & w = model.layers[il];
    return w.ffn_up_exps && w.ffn_gate_exps && w.ffn_down_exps &&
            w.ffn_up_exps->type == GGML_TYPE_IQ3_S && w.ffn_gate_exps->type == GGML_TYPE_IQ3_S &&
            w.ffn_down_exps->type == GGML_TYPE_IQ4_NL &&
            w.ffn_up_exps->ne[0] == N_EMBD && w.ffn_up_exps->ne[1] == N_FF && w.ffn_up_exps->ne[2] == 512 &&
            w.ffn_gate_exps->ne[0] == N_EMBD && w.ffn_gate_exps->ne[1] == N_FF && w.ffn_gate_exps->ne[2] == 512 &&
            w.ffn_down_exps->ne[0] == N_FF && w.ffn_down_exps->ne[1] == N_EMBD && w.ffn_down_exps->ne[2] == 512 &&
            !w.ffn_up_exps_b && !w.ffn_gate_exps_b && !w.ffn_down_exps_b &&
            !w.ffn_up_gate_exps && !w.ffn_down_exps_s &&
            !model.hadamard_rotation(w.ffn_up_exps) && !model.hadamard_rotation(w.ffn_gate_exps) &&
            !model.hadamard_rotation(w.ffn_down_exps);
}

bool layer_is_eligible(const llama_model & model, int32_t il,
        const llama_moe_cache_frozen_layer & s, int32_t device) {
    if (!model_layer_is_eligible(model, il)) return false;
    const auto & w = model.layers[il];
    return s.layer == il && s.owner_backend && s.device_group == device &&
            s.host_up == w.ffn_up_exps && s.host_gate == w.ffn_gate_exps && s.host_down == w.ffn_down_exps &&
            s.cached_up && s.cached_gate && s.cached_down;
}

template<typename T>
void add_stats(T & dst, const T & src) {
    static_assert(sizeof(T) % sizeof(uint64_t) == 0);
    auto * d = reinterpret_cast<uint64_t *>(&dst);
    const auto * s = reinterpret_cast<const uint64_t *>(&src);
    for (size_t i = 0; i < sizeof(T) / sizeof(uint64_t); ++i) d[i] += s[i];
}

} // namespace

bool llama_moe_ffn_prototype_state::initialize(llama_moe_cache & cache, const llama_model & model,
        const llama_moe_ffn_prototype_config & config) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(cache);
    GGML_UNUSED(model);
    GGML_UNUSED(config);
    return false;
#else
    const auto fail = [](const char * stage) {
        LLAMA_LOG_ERROR("%s: internal MoE FFN prototype initialization failed at %s\n", __func__, stage);
        return false;
    };
    if ((!config.all_devices && (config.layer < 0 || config.device < 0)) || config.cpu_threads <= 0 ||
            model.arch != LLM_ARCH_QWEN4EXP || model.hparams.n_layer != 48 ||
            model.hparams.n_expert != 512 || model.hparams.n_expert_used != N_EXPERTS_USED) {
        return fail("model/config eligibility");
    }

    this->cache = &cache;
    lifecycle = config.lifecycle;
    std::vector<ggml_tensor *> prior_maps;
    if (lifecycle) {
        prior_maps.resize(model.layers.size(), nullptr);
        for (size_t il = 0; il < model.layers.size(); ++il)
            prior_maps[il] = cache.get_slot_map((int32_t) il, 1, N_EXPERTS_USED);
    }

    const int32_t first_layer = config.all_devices ? 0 : config.layer;
    const int32_t last_layer = config.all_devices ? (int32_t) model.layers.size() : config.layer + 1;
    for (int32_t il = first_layer; il < last_layer; ++il) {
        if (!model_layer_is_eligible(model, il)) continue;
        ggml_backend * owner = cache.backend(il);
        if (!owner) continue;
        bool group_seen = false;
        for (const auto & d : devices) for (const auto & g : d.groups) {
            if (std::find(g.snapshot.group_layers.begin(), g.snapshot.group_layers.end(), il) != g.snapshot.group_layers.end()) {
                group_seen = true;
                break;
            }
        }
        if (group_seen) continue;

        device_state * dstate = nullptr;
        for (auto & d : devices) if (d.cuda_backend == owner) { dstate = &d; break; }
        if (!dstate) {
            devices.emplace_back();
            dstate = &devices.back();
            dstate->cuda_backend = owner;
        }
        group_state group;
        if (!cache.freeze_group_for_layer(il, owner, config.seed_experts, group.lease))
            return fail("cache group freeze");
        group.snapshot = group.lease.snapshot();
        if (lifecycle) ++lifecycle->live_leases;
        if (group.snapshot.owner_backend != owner || group.snapshot.group_layers.empty())
            return fail("frozen group identity");
        if (lifecycle) for (const int32_t member : group.snapshot.group_layers)
            group.stale_slot_maps.push_back(prior_maps[member]);
        if (dstate->device < 0) dstate->device = group.snapshot.device_group;
        if (dstate->device != group.snapshot.device_group) return fail("backend spans incompatible cache devices");
        dstate->groups.push_back(std::move(group));
        if (!config.all_devices) break;
    }
    if (devices.empty()) return fail("no eligible cache groups discovered");

    size_t eligible_count = 0;
    int32_t ineligible_count = 0;
    for (const auto & d : devices) for (const auto & g : d.groups) {
        for (const int32_t il : g.snapshot.group_layers) {
            llama_moe_cache_frozen_layer layer_snapshot;
            if (cache.frozen_layer_snapshot(g.snapshot, il, layer_snapshot) &&
                    layer_is_eligible(model, il, layer_snapshot, d.device)) ++eligible_count;
            else ++ineligible_count;
        }
    }
    if (eligible_count == 0) return fail("no compatible layers in frozen groups");

    for (auto & d : devices) {
        d.cpu_backend = ggml_backend_cpu_init();
        if (!d.cpu_backend) return fail("CPU backend allocation");
        ggml_backend_cpu_set_n_threads(d.cpu_backend, config.cpu_threads);
        ggml_cuda_moe_ffn_cpu_resource resource = { d.cpu_backend, N_EMBD, N_FF, N_EXPERTS_USED, config.cpu_threads };
        d.cpu_resource_id = ggml_backend_cuda_moe_ffn_cpu_resource_register(d.cuda_backend, &resource);
        if (d.cpu_resource_id <= 0) return fail("CUDA-owned CPU resource registration");
        d.cpu_backend_transferred = true;
        if (lifecycle) ++lifecycle->live_cpu_resources;
    }

    for (auto & d : devices) for (const auto & g : d.groups) {
        for (const int32_t il : g.snapshot.group_layers) {
            llama_moe_cache_frozen_layer snapshot;
            if (!cache.frozen_layer_snapshot(g.snapshot, il, snapshot) ||
                    !layer_is_eligible(model, il, snapshot, d.device)) continue;
            layer_binding layer;
            layer.snapshot = std::move(snapshot);
            layer.cuda_backend = d.cuda_backend;
            layer.cpu_resource_id = d.cpu_resource_id;
            layer.n_embd = N_EMBD;
            layer.n_ff = N_FF;
            layer.cpu_ctx = ggml_init({ CPU_GRAPH_CONTEXT_BYTES, nullptr, true });
            if (!layer.cpu_ctx) return fail("per-layer CPU graph context allocation");
            layer.cpu_activations = ggml_new_tensor_2d(layer.cpu_ctx, GGML_TYPE_F32, N_EMBD, 1);
            layer.cpu_ids = ggml_new_tensor_2d(layer.cpu_ctx, GGML_TYPE_I32, 1, 1);
            layer.cpu_routes = ggml_new_tensor_3d(layer.cpu_ctx, GGML_TYPE_F32, 1, 1, 1);
            ggml_tensor * up_gate = ggml_moe_up_gate(layer.cpu_ctx,
                    const_cast<ggml_tensor *>(layer.snapshot.host_up),
                    const_cast<ggml_tensor *>(layer.snapshot.host_gate),
                    layer.cpu_activations, layer.cpu_ids, GGML_UNARY_OP_SILU);
            layer.cpu_output = ggml_mul_multi_add(layer.cpu_ctx,
                    ggml_mul_mat_id(layer.cpu_ctx, const_cast<ggml_tensor *>(layer.snapshot.host_down),
                            up_gate, layer.cpu_ids), layer.cpu_routes);
            if (!layer.cpu_activations || !layer.cpu_ids || !layer.cpu_routes || !up_gate || !layer.cpu_output ||
                    layer.cpu_output->ne[0] != N_EMBD || layer.cpu_output->ne[1] != 1 || layer.cpu_output->ne[2] != 1) {
                ggml_free(layer.cpu_ctx);
                return fail("per-layer CPU graph construction");
            }
            layer.cpu_graph = ggml_new_graph_custom(layer.cpu_ctx, 64, false);
            ggml_build_forward_expand(layer.cpu_graph, layer.cpu_output);
            layer.cpu_buffer = ggml_backend_alloc_ctx_tensors(layer.cpu_ctx, d.cpu_backend);
            if (!layer.cpu_buffer) {
                ggml_free(layer.cpu_ctx);
                return fail("per-layer CPU graph workspace allocation");
            }
            if (config.test_poison_cpu_priming_inputs) {
                std::memset(layer.cpu_activations->data, 0xa5, ggml_nbytes(layer.cpu_activations));
                std::memset(layer.cpu_ids->data, 0xa5, ggml_nbytes(layer.cpu_ids));
                std::memset(layer.cpu_routes->data, 0xa5, ggml_nbytes(layer.cpu_routes));
            }
            const std::vector<float> priming_activations(N_EMBD, 0.0f);
            const int32_t priming_id = 0;
            const float priming_route = 1.0f;
            ggml_backend_tensor_set(layer.cpu_activations,
                    priming_activations.data(), 0, ggml_nbytes(layer.cpu_activations));
            ggml_backend_tensor_set(layer.cpu_ids,
                    &priming_id, 0, ggml_nbytes(layer.cpu_ids));
            ggml_backend_tensor_set(layer.cpu_routes,
                    &priming_route, 0, ggml_nbytes(layer.cpu_routes));
            d.layer_bindings.push_back(std::move(layer));
            auto & registered = d.layer_bindings.back();
            if (lifecycle) {
                ++lifecycle->live_cpu_graphs;
                lifecycle->cpu_graph_context_bytes += CPU_GRAPH_CONTEXT_BYTES;
            }

            ggml_cuda_moe_ffn_binding binding = {};
            binding.generation = registered.snapshot.generation;
            binding.device = d.device;
            binding.n_experts = registered.snapshot.n_experts;
            binding.n_slots = registered.snapshot.n_slots;
            binding.n_experts_used = N_EXPERTS_USED;
            binding.n_embd = N_EMBD;
            binding.n_ff = N_FF;
            binding.up_type = registered.snapshot.host_up->type;
            binding.gate_type = registered.snapshot.host_gate->type;
            binding.down_type = registered.snapshot.host_down->type;
            binding.unary_op = GGML_UNARY_OP_SILU;
            binding.limit = 0.0f;
            binding.host_up = registered.snapshot.host_up;
            binding.host_gate = registered.snapshot.host_gate;
            binding.host_down = registered.snapshot.host_down;
            binding.cuda_up = registered.snapshot.cached_up;
            binding.cuda_gate = registered.snapshot.cached_gate;
            binding.cuda_down = registered.snapshot.cached_down;
            binding.cache_callbacks = llama_moe_cache_get_cuda_callbacks();
            binding.cache_context = &cache;
            binding.cache_group = registered.snapshot.group;
            binding.cache_layer = registered.snapshot.layer;
            binding.cpu_resource_id = d.cpu_resource_id;
            binding.cpu_graph = registered.cpu_graph;
            binding.cpu_activations = registered.cpu_activations;
            binding.cpu_ids = registered.cpu_ids;
            binding.cpu_route_weights = registered.cpu_routes;
            binding.cpu_output = registered.cpu_output;
            registered.binding_id = ggml_backend_cuda_moe_ffn_register(d.cuda_backend, &binding);
            if (registered.binding_id <= 0) return fail("per-layer semantic binding registration");
            if (lifecycle) ++lifecycle->live_bindings;
        }
    }
    LLAMA_LOG_INFO("%s: devices=%zu groups=%zu compatible=%zu ineligible=%d resources=%zu threads/device=%d graph-context=%zu MiB\n",
            __func__, devices.size(), std::accumulate(devices.begin(), devices.end(), size_t(0),
                [](size_t n, const device_state & d) { return n + d.groups.size(); }),
            layer_count(), ineligible_count, devices.size(), config.cpu_threads,
            layer_count() * CPU_GRAPH_CONTEXT_BYTES / (1024 * 1024));
    return true;
#endif
}

void llama_moe_ffn_prototype_state::shutdown() {
#ifdef GGML_USE_CUDA
    for (auto & d : devices) if (d.cpu_resource_id > 0 &&
            !ggml_backend_cuda_moe_ffn_cpu_resource_stop(d.cuda_backend, d.cpu_resource_id)) std::abort();
    for (auto & d : devices) for (auto & layer : d.layer_bindings) if (layer.binding_id > 0) {
        if (!ggml_backend_cuda_moe_ffn_unregister(d.cuda_backend, layer.binding_id)) std::abort();
        layer.binding_id = 0;
        if (lifecycle) --lifecycle->live_bindings;
    }
    for (auto & d : devices) {
        for (auto & layer : d.layer_bindings) {
            if (layer.cpu_buffer) { ggml_backend_buffer_free(layer.cpu_buffer); layer.cpu_buffer = nullptr; }
            if (layer.cpu_ctx) { ggml_free(layer.cpu_ctx); layer.cpu_ctx = nullptr; }
            if (lifecycle) {
                --lifecycle->live_cpu_graphs;
                lifecycle->cpu_graph_context_bytes -= CPU_GRAPH_CONTEXT_BYTES;
            }
        }
        d.layer_bindings.clear();
        if (d.cpu_resource_id > 0) {
            if (!ggml_backend_cuda_moe_ffn_cpu_resource_unregister(d.cuda_backend, d.cpu_resource_id)) std::abort();
            d.cpu_resource_id = 0;
            d.cpu_backend = nullptr;
            d.cpu_backend_transferred = false;
            if (lifecycle) --lifecycle->live_cpu_resources;
        } else if (d.cpu_backend && !d.cpu_backend_transferred) {
            ggml_backend_free(d.cpu_backend);
            d.cpu_backend = nullptr;
        }
        for (auto & g : d.groups) if (g.lease) {
            if (!g.lease.release()) std::abort();
            if (lifecycle) --lifecycle->live_leases;
        }
        d.groups.clear();
    }
#endif
    devices.clear();
    cache = nullptr;
}

const llama_moe_ffn_prototype_state::layer_binding * llama_moe_ffn_prototype_state::find_layer(int32_t il) const {
    for (const auto & d : devices) for (const auto & layer : d.layer_bindings)
        if (layer.snapshot.layer == il) return &layer;
    return nullptr;
}

const llama_moe_ffn_prototype_state::group_state * llama_moe_ffn_prototype_state::find_group(int32_t group) const {
    for (const auto & d : devices) for (const auto & g : d.groups)
        if (g.snapshot.group == group) return &g;
    return nullptr;
}

size_t llama_moe_ffn_prototype_state::layer_count() const {
    size_t n = 0;
    for (const auto & d : devices) n += d.layer_bindings.size();
    return n;
}

bool llama_moe_ffn_prototype_state::valid() const {
#ifdef GGML_USE_CUDA
    if (!cache || devices.empty()) return false;
    for (const auto & d : devices) {
        if (d.cpu_resource_id <= 0 || d.groups.empty() || d.layer_bindings.empty()) return false;
        for (const auto & g : d.groups) if (!g.lease || !cache->frozen_snapshot_valid(g.snapshot)) return false;
        for (const auto & layer : d.layer_bindings)
            if (layer.binding_id <= 0 || !cache->frozen_snapshot_valid(layer.snapshot)) return false;
    }
    return true;
#else
    return false;
#endif
}

bool llama_moe_ffn_prototype_state::collect_stats(ggml_cuda_moe_ffn_stats & stats) const {
    std::memset(&stats, 0, sizeof(stats));
    for (const auto & d : devices) {
        ggml_cuda_moe_ffn_stats current{};
        if (!collect_device_stats(d.cuda_backend, current)) return false;
        add_stats(stats, current);
    }
    return true;
}

bool llama_moe_ffn_prototype_state::collect_device_stats(ggml_backend * backend,
        ggml_cuda_moe_ffn_stats & stats) const {
#ifdef GGML_USE_CUDA
    std::memset(&stats, 0, sizeof(stats));
    bool found = false;
    for (const auto & d : devices) if (d.cuda_backend == backend) {
        found = true;
        for (const auto & layer : d.layer_bindings) {
            ggml_cuda_moe_ffn_stats current{};
            if (layer.binding_id <= 0 || !ggml_backend_cuda_moe_ffn_get_stats(d.cuda_backend, layer.binding_id, &current)) return false;
            add_stats(stats, current);
        }
    }
    return found;
#else
    GGML_UNUSED(backend); GGML_UNUSED(stats); return false;
#endif
}

bool llama_moe_ffn_prototype_state::collect_resource_stats(ggml_cuda_moe_ffn_cpu_resource_stats & stats) const {
    std::memset(&stats, 0, sizeof(stats));
    for (const auto & d : devices) {
        ggml_cuda_moe_ffn_cpu_resource_stats current{};
        if (!collect_device_resource_stats(d.cuda_backend, current)) return false;
        add_stats(stats, current);
    }
    return !devices.empty();
}

bool llama_moe_ffn_prototype_state::collect_device_resource_stats(ggml_backend * backend,
        ggml_cuda_moe_ffn_cpu_resource_stats & stats) const {
#ifdef GGML_USE_CUDA
    std::memset(&stats, 0, sizeof(stats));
    for (const auto & d : devices) if (d.cuda_backend == backend)
        return d.cpu_resource_id > 0 && ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(backend, d.cpu_resource_id, &stats);
    return false;
#else
    GGML_UNUSED(backend); GGML_UNUSED(stats); return false;
#endif
}

ggml_tensor * llama_moe_ffn_prototype_state::build_node(ggml_context * ctx, ggml_backend_sched_t sched,
        int32_t il, ggml_tensor * activation, ggml_tensor * selected_ids, ggml_tensor * route_weights) {
#ifdef GGML_USE_CUDA
    const layer_binding * layer = find_layer(il);
    if (!layer || !sched || !valid() || !activation || !selected_ids || !route_weights ||
            activation->type != GGML_TYPE_F32 || activation->ne[0] != layer->n_embd || activation->ne[1] != 1 ||
            selected_ids->type != GGML_TYPE_I32 || selected_ids->ne[0] != N_EXPERTS_USED || selected_ids->ne[1] != 1 ||
            route_weights->type != GGML_TYPE_F32 || route_weights->ne[0] != 1 ||
            route_weights->ne[1] != N_EXPERTS_USED || route_weights->ne[2] != 1 ||
            !ggml_is_contiguous(activation) || !ggml_is_contiguous(selected_ids) || !ggml_is_contiguous(route_weights)) return nullptr;
    ggml_tensor * op = ggml_moe_ffn(ctx, activation, selected_ids, route_weights,
            layer->snapshot.device_group, layer->binding_id, layer->snapshot.generation,
            layer->n_embd, layer->n_ff, GGML_UNARY_OP_SILU, 0.0f);
    if (!ggml_backend_supports_op(layer->cuda_backend, op)) return nullptr;
    ggml_backend_sched_set_tensor_backend(sched, op, layer->cuda_backend);
    return op;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(sched); GGML_UNUSED(il); GGML_UNUSED(activation);
    GGML_UNUSED(selected_ids); GGML_UNUSED(route_weights); return nullptr;
#endif
}
