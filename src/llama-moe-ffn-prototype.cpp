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

bool model_layer_layout(const llama_model & model, int32_t il, ggml_moe_ffn_layout & layout) {
    if (il < 0 || (size_t) il >= model.layers.size()) return false;
    const auto & w = model.layers[il];
    if (!w.ffn_up_exps || !w.ffn_gate_exps || !w.ffn_down_exps) return false;
    layout = {};
    layout.n_experts = model.hparams.n_expert;
    layout.n_experts_used = model.hparams.n_expert_used;
    layout.n_embd = w.ffn_up_exps->ne[0];
    layout.n_ff = w.ffn_up_exps->ne[1];
    layout.n_parts = GGML_MOE_FFN_PARTS;
    layout.cache_layer = il;
    layout.up = w.ffn_up_exps;
    layout.gate = w.ffn_gate_exps;
    layout.down = w.ffn_down_exps;
    layout.unary_op = GGML_UNARY_OP_SILU;
    layout.limit = model.swiglu_limit(il, false);
    if (layout.up->buffer && !ggml_backend_buffer_is_host(layout.up->buffer)) return false;
    if (layout.gate->buffer && !ggml_backend_buffer_is_host(layout.gate->buffer)) return false;
    if (layout.down->buffer && !ggml_backend_buffer_is_host(layout.down->buffer)) return false;
    layout.flags = GGML_MOE_FFN_LAYOUT_HOST_RESIDENT;
    if (!w.ffn_up_gate_exps) layout.flags |= GGML_MOE_FFN_LAYOUT_SEPARATE_PARTS;
    if (ggml_moe_ffn_expert_execution_inputs_supported(w.ffn_up_exps_b,
            w.ffn_gate_exps_b, w.ffn_down_exps_b, w.ffn_up_gate_exps,
            w.ffn_up_gate_exps_b, w.ffn_down_exps_s))
        layout.flags |= GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS;
    if (!model.hadamard_rotation(w.ffn_up_exps) && !model.hadamard_rotation(w.ffn_gate_exps) &&
            !model.hadamard_rotation(w.ffn_down_exps)) layout.flags |= GGML_MOE_FFN_LAYOUT_NO_HADAMARD;
    return ggml_moe_ffn_layout_supported(&layout);
}

bool layer_is_eligible(const llama_model & model, int32_t il, const llama_moe_cache & cache,
        const llama_moe_cache_frozen_layer & s, int32_t device, ggml_backend * backend,
        ggml_moe_ffn_layout & layout) {
    if (!model_layer_layout(model, il, layout)) return false;
    layout.n_slots = s.n_slots;
    layout.device = s.device_group;
    layout.cache_group = s.group;
    layout.cache_layer = s.layer;
    layout.owner_backend = s.owner_backend;
    layout.cache_owner = &cache;
    layout.bank_up = s.cached_up;
    layout.bank_gate = s.cached_gate;
    layout.bank_down = s.cached_down;
    return s.layer == il && s.owner_backend == backend && s.device_group == device &&
            s.n_experts == layout.n_experts && s.host_up == layout.up &&
            s.host_gate == layout.gate && s.host_down == layout.down &&
            ggml_moe_ffn_layout_supported(&layout);
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
    if ((!config.all_devices && (config.layer < 0 || config.device < 0)) || config.cpu_threads <= 0) {
        return fail("model/config eligibility");
    }

    this->cache = &cache;
    lifecycle = config.lifecycle;
    std::vector<ggml_tensor *> prior_maps;
    if (lifecycle) {
        prior_maps.resize(model.layers.size(), nullptr);
        for (size_t il = 0; il < model.layers.size(); ++il)
            prior_maps[il] = cache.get_slot_map((int32_t) il, 1, GGML_MOE_FFN_MAX_SELECTED);
    }

    const int32_t first_layer = config.all_devices ? 0 : config.layer;
    const int32_t last_layer = config.all_devices ? (int32_t) model.layers.size() : config.layer + 1;
    for (int32_t il = first_layer; il < last_layer; ++il) {
        ggml_moe_ffn_layout candidate{};
        if (!model_layer_layout(model, il, candidate)) continue;
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
        if (dstate->n_embd && (candidate.n_embd != dstate->n_embd ||
                candidate.n_ff != dstate->n_ff || candidate.n_experts_used != dstate->n_experts_used)) continue;
        group_state group;
        if (!cache.freeze_group_for_layer(il, owner, config.seed_experts, group.lease))
            return fail("cache group freeze");
        group.snapshot = group.lease.snapshot();
        if (group.snapshot.owner_backend != owner || group.snapshot.group_layers.empty())
            return fail("frozen group identity");
        llama_moe_cache_frozen_layer candidate_snapshot;
        ggml_moe_ffn_layout descriptor{};
        if (!cache.frozen_layer_snapshot(group.snapshot, il, candidate_snapshot) ||
                !layer_is_eligible(model, il, cache, candidate_snapshot, group.snapshot.device_group,
                    owner, descriptor)) continue;
        if (lifecycle) ++lifecycle->live_leases;
        if (lifecycle) for (const int32_t member : group.snapshot.group_layers)
            group.stale_slot_maps.push_back(prior_maps[member]);
        if (dstate->device < 0) dstate->device = group.snapshot.device_group;
        if (dstate->device != group.snapshot.device_group) return fail("backend spans incompatible cache devices");
        if (!dstate->n_embd) {
            dstate->n_embd = descriptor.n_embd;
            dstate->n_ff = descriptor.n_ff;
            dstate->n_experts_used = descriptor.n_experts_used;
        }
        dstate->groups.push_back(std::move(group));
        if (!config.all_devices) break;
    }
    if (devices.empty()) return fail("no eligible cache groups discovered");

    size_t eligible_count = 0;
    int32_t ineligible_count = 0;
    for (const auto & d : devices) for (const auto & g : d.groups) {
        for (const int32_t il : g.snapshot.group_layers) {
            llama_moe_cache_frozen_layer layer_snapshot;
            ggml_moe_ffn_layout descriptor{};
            if (cache.frozen_layer_snapshot(g.snapshot, il, layer_snapshot) &&
                    layer_is_eligible(model, il, cache, layer_snapshot, d.device,
                        d.cuda_backend, descriptor) && descriptor.n_embd == d.n_embd &&
                    descriptor.n_ff == d.n_ff && descriptor.n_experts_used == d.n_experts_used) ++eligible_count;
            else ++ineligible_count;
        }
    }
    if (eligible_count == 0) return fail("no compatible layers in frozen groups");

    for (auto & d : devices) {
        d.cpu_backend = ggml_backend_cpu_init();
        if (!d.cpu_backend) return fail("CPU backend allocation");
        ggml_backend_cpu_set_n_threads(d.cpu_backend, config.cpu_threads);
        ggml_cuda_moe_ffn_cpu_resource resource = { d.cpu_backend, d.n_embd, d.n_ff,
            d.n_experts_used, config.cpu_threads };
        d.cpu_resource_id = ggml_backend_cuda_moe_ffn_cpu_resource_register(d.cuda_backend, &resource);
        if (d.cpu_resource_id <= 0) return fail("CUDA-owned CPU resource registration");
        d.cpu_backend_transferred = true;
        if (lifecycle) ++lifecycle->live_cpu_resources;
    }

    for (auto & d : devices) for (const auto & g : d.groups) {
        for (const int32_t il : g.snapshot.group_layers) {
            llama_moe_cache_frozen_layer snapshot;
            ggml_moe_ffn_layout layout{};
            if (!cache.frozen_layer_snapshot(g.snapshot, il, snapshot) ||
                    !layer_is_eligible(model, il, cache, snapshot, d.device, d.cuda_backend, layout) ||
                    layout.n_embd != d.n_embd || layout.n_ff != d.n_ff ||
                    layout.n_experts_used != d.n_experts_used) continue;
            layer_binding layer;
            layer.snapshot = std::move(snapshot);
            layer.cuda_backend = d.cuda_backend;
            layer.cpu_resource_id = d.cpu_resource_id;
            layer.layout = layout;
            layer.cpu_ctx = ggml_init({ CPU_GRAPH_CONTEXT_BYTES, nullptr, true });
            if (!layer.cpu_ctx) return fail("per-layer CPU graph context allocation");
            layer.cpu_activations = ggml_new_tensor_2d(layer.cpu_ctx, GGML_TYPE_F32, layout.n_embd, 1);
            layer.cpu_ids = ggml_new_tensor_2d(layer.cpu_ctx, GGML_TYPE_I32, 1, 1);
            layer.cpu_routes = ggml_new_tensor_3d(layer.cpu_ctx, GGML_TYPE_F32, 1, 1, 1);
            ggml_tensor * up_gate = ggml_moe_up_gate(layer.cpu_ctx,
                    const_cast<ggml_tensor *>(layer.snapshot.host_up),
                    const_cast<ggml_tensor *>(layer.snapshot.host_gate),
                    layer.cpu_activations, layer.cpu_ids, GGML_UNARY_OP_SILU);
            *(float *)(up_gate->op_params + 1) = layout.limit;
            layer.cpu_output = ggml_mul_multi_add(layer.cpu_ctx,
                    ggml_mul_mat_id(layer.cpu_ctx, const_cast<ggml_tensor *>(layer.snapshot.host_down),
                            up_gate, layer.cpu_ids), layer.cpu_routes);
            if (!layer.cpu_activations || !layer.cpu_ids || !layer.cpu_routes || !up_gate || !layer.cpu_output ||
                    layer.cpu_output->ne[0] != layout.n_embd || layer.cpu_output->ne[1] != 1 || layer.cpu_output->ne[2] != 1) {
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
            const std::vector<float> priming_activations(layout.n_embd, 0.0f);
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
            binding.layout = registered.layout;
            binding.cache_callbacks = llama_moe_cache_get_cuda_callbacks();
            binding.cache_context = &cache;
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
        const ggml_moe_ffn_layout & layout) {
#ifdef GGML_USE_CUDA
    const layer_binding * layer = find_layer(layout.cache_layer);
    if (!layer || !sched || !valid() || !ggml_moe_ffn_layout_supported(&layout) ||
            !ggml_moe_ffn_graph_supported(&layout) || layout.owner_backend != layer->cuda_backend ||
            layout.cache_owner != cache || layout.device != layer->layout.device ||
            layout.cache_group != layer->layout.cache_group || layout.cache_layer != layer->layout.cache_layer ||
            layout.n_experts != layer->layout.n_experts || layout.n_experts_used != layer->layout.n_experts_used ||
            layout.n_embd != layer->layout.n_embd || layout.n_ff != layer->layout.n_ff ||
            layout.n_slots != layer->layout.n_slots || layout.n_parts != layer->layout.n_parts ||
            layout.up != layer->layout.up || layout.gate != layer->layout.gate || layout.down != layer->layout.down ||
            layout.bank_up != layer->layout.bank_up || layout.bank_gate != layer->layout.bank_gate ||
            layout.bank_down != layer->layout.bank_down || layout.unary_op != layer->layout.unary_op ||
            layout.limit != layer->layout.limit || layout.flags != layer->layout.flags) return nullptr;
    ggml_tensor * op = ggml_moe_ffn(ctx, const_cast<ggml_tensor *>(layout.activation),
            const_cast<ggml_tensor *>(layout.selected_ids), const_cast<ggml_tensor *>(layout.route_weights),
            layout.device, layer->binding_id, layer->snapshot.generation,
            layout.n_embd, layout.n_ff, layout.unary_op, layout.limit);
    if (!ggml_backend_supports_op(layer->cuda_backend, op)) return nullptr;
    ggml_backend_sched_set_tensor_backend(sched, op, layer->cuda_backend);
    return op;
#else
    GGML_UNUSED(ctx); GGML_UNUSED(sched); GGML_UNUSED(layout); return nullptr;
#endif
}
