#include "llama-moe-cache-ffn.h"

#include "llama-model.h"

#include "ggml-moe-ffn.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#endif

#include <algorithm>
#include <cstdlib>
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
    layout.storage_topology = w.ffn_up_gate_exps ? GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN :
        GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN;
    layout.cache_layer = il;
    layout.up = w.ffn_up_exps;
    layout.gate = w.ffn_gate_exps;
    layout.down = w.ffn_down_exps;
    layout.up_gate_storage = w.ffn_up_gate_exps;
    layout.unary_op = GGML_UNARY_OP_SILU;
    layout.limit = model.swiglu_limit(il, false);
    if (layout.up->buffer && !ggml_backend_buffer_is_host(layout.up->buffer)) return false;
    if (layout.gate->buffer && !ggml_backend_buffer_is_host(layout.gate->buffer)) return false;
    if (layout.down->buffer && !ggml_backend_buffer_is_host(layout.down->buffer)) return false;
    layout.flags = GGML_MOE_FFN_LAYOUT_HOST_RESIDENT;
    if (ggml_moe_ffn_expert_execution_inputs_supported(w.ffn_up_exps_b,
            w.ffn_gate_exps_b, w.ffn_down_exps_b, w.ffn_up_gate_exps_b, w.ffn_down_exps_s))
        layout.flags |= GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS;
    layout.flags |= GGML_MOE_FFN_LAYOUT_PHYSICAL_STORAGE_VALID;
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
    layout.up_gate_storage = s.host_up_gate_storage;
    layout.bank_up_gate_storage = s.cached_up_gate_storage;
    const bool identity = s.layer == il && s.owner_backend == backend && s.device_group == device &&
            s.n_experts == layout.n_experts && s.host_up == layout.up &&
            s.host_gate == layout.gate && s.host_down == layout.down;
    const bool cuda = ggml_backend_cuda_moe_ffn_layout_supported(backend, &layout);
    return identity && cuda;
}

template<typename T>
void add_stats(T & dst, const T & src) {
    static_assert(sizeof(T) % sizeof(uint64_t) == 0);
    auto * d = reinterpret_cast<uint64_t *>(&dst);
    const auto * s = reinterpret_cast<const uint64_t *>(&src);
    for (size_t i = 0; i < sizeof(T) / sizeof(uint64_t); ++i) d[i] += s[i];
}

} // namespace

bool llama_moe_cache_ffn_state::initialize(llama_moe_cache & cache, const llama_model & model,
        int32_t cpu_threads) {
#ifndef GGML_USE_CUDA
    GGML_UNUSED(cache);
    GGML_UNUSED(model);
    GGML_UNUSED(cpu_threads);
    return false;
#else
    const auto fail = [](const char * stage) {
        LLAMA_LOG_ERROR("%s: MoE cache FFN initialization failed at %s\n", __func__, stage);
        return false;
    };
    if (cpu_threads <= 0) return fail("invalid CPU thread count");

    this->cache = &cache;

    const int32_t first_layer = 0;
    const int32_t last_layer = (int32_t) model.layers.size();
    for (int32_t il = first_layer; il < last_layer; ++il) {
        ggml_moe_ffn_layout candidate{};
        if (!model_layer_layout(model, il, candidate)) continue;
        ggml_backend * owner = cache.backend(il);
        if (!owner) continue;
        candidate.owner_backend = owner;
        candidate.device = -1; // The CUDA-owned query resolves and checks this backend's device.
        if (!ggml_backend_cuda_moe_ffn_layout_supported(owner, &candidate)) continue;
        bool group_seen = false;
        for (const auto & d : devices) for (const auto & g : d.groups) {
            const auto & frozen_group = g.lease.snapshot();
            if (std::find(frozen_group.group_layers.begin(), frozen_group.group_layers.end(), il) !=
                    frozen_group.group_layers.end()) {
                group_seen = true;
                break;
            }
        }
        if (group_seen) continue;

        device_state * dstate = nullptr;
        for (auto & d : devices) if (d.cuda_backend == owner) { dstate = &d; break; }
        group_state group;
        if (!cache.freeze_group_for_layer(il, owner, {}, group.lease))
            return fail("cache group freeze");
        const auto & frozen_group = group.lease.snapshot();
        if (frozen_group.owner_backend != owner || frozen_group.group_layers.empty())
            return fail("frozen group identity");
        llama_moe_cache_frozen_layer candidate_snapshot;
        ggml_moe_ffn_layout descriptor{};
        const bool snapshot_ok = cache.frozen_layer_snapshot(frozen_group, il, candidate_snapshot);
        const bool eligible = snapshot_ok && layer_is_eligible(model, il, cache, candidate_snapshot,
                frozen_group.device_group, owner, descriptor);
        if (!eligible) continue;
        if (!dstate) {
            devices.emplace_back();
            dstate = &devices.back();
            dstate->cuda_backend = owner;
        }
        if (dstate->device < 0) dstate->device = frozen_group.device_group;
        if (dstate->device != frozen_group.device_group) return fail("backend spans incompatible cache devices");
        dstate->groups.push_back(std::move(group));
    }
    if (devices.empty()) return fail("no eligible cache groups discovered");

    size_t eligible_count = 0;
    int32_t ineligible_count = 0;
    for (auto & d : devices) for (const auto & g : d.groups) {
        const auto & frozen_group = g.lease.snapshot();
        for (const int32_t il : frozen_group.group_layers) {
            llama_moe_cache_frozen_layer layer_snapshot;
            ggml_moe_ffn_layout descriptor{};
            if (cache.frozen_layer_snapshot(frozen_group, il, layer_snapshot) &&
                    layer_is_eligible(model, il, cache, layer_snapshot, d.device,
                        d.cuda_backend, descriptor)) {
                const ggml_cuda_moe_ffn_resource_signature signature{
                    descriptor.n_embd, descriptor.n_ff, descriptor.n_experts_used };
                if (std::find_if(d.resource_signatures.begin(), d.resource_signatures.end(), [&](const auto & current) {
                            return current.n_embd == signature.n_embd && current.n_ff == signature.n_ff &&
                                current.n_experts_used == signature.n_experts_used;
                        }) == d.resource_signatures.end()) d.resource_signatures.push_back(signature);
                d.n_embd = std::max(d.n_embd, descriptor.n_embd);
                d.n_ff = std::max(d.n_ff, descriptor.n_ff);
                d.n_experts_used = std::max(d.n_experts_used, descriptor.n_experts_used);
                ++eligible_count;
            } else ++ineligible_count;
        }
    }
    if (eligible_count == 0) return fail("no compatible layers in frozen groups");

    for (auto & d : devices) {
        d.cpu_backend = ggml_backend_cpu_init();
        if (!d.cpu_backend) return fail("CPU backend allocation");
        ggml_backend_cpu_set_n_threads(d.cpu_backend, cpu_threads);
        ggml_cuda_moe_ffn_cpu_resource resource = { d.cpu_backend, d.n_embd, d.n_ff,
            d.n_experts_used, cpu_threads, d.resource_signatures.data(),
            d.resource_signatures.size() };
        d.cpu_resource_id = ggml_backend_cuda_moe_ffn_cpu_resource_register(d.cuda_backend, &resource);
        if (d.cpu_resource_id <= 0) return fail("CUDA-owned CPU resource registration");
        d.cpu_backend_transferred = true;
    }

    for (auto & d : devices) for (const auto & g : d.groups) {
        const auto & frozen_group = g.lease.snapshot();
        for (const int32_t il : frozen_group.group_layers) {
            llama_moe_cache_frozen_layer snapshot;
            ggml_moe_ffn_layout layout{};
            if (!cache.frozen_layer_snapshot(frozen_group, il, snapshot) ||
                    !layer_is_eligible(model, il, cache, snapshot, d.device, d.cuda_backend, layout)) continue;
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
        }
    }

    layer_by_index.assign(model.layers.size(), nullptr);
    size_t indexed_bindings = 0;
    for (const auto & d : devices) for (const auto & layer : d.layer_bindings) {
        const int32_t il = layer.snapshot.layer;
        if (il < 0 || (size_t) il >= layer_by_index.size() || layer_by_index[il])
            return fail("duplicate or invalid layer binding index");
        layer_by_index[il] = &layer;
        ++indexed_bindings;
    }
    if (indexed_bindings != eligible_count) return fail("incomplete layer binding index");
    // Binding vectors are frozen after publication, so these pointers stay stable until shutdown.
    ready = true;

    LLAMA_LOG_INFO("%s: devices=%zu groups=%zu compatible-layers=%zu unsupported-layers=%d resources=%zu threads/device=%d graph-context=%zu MiB\n",
            __func__, devices.size(), std::accumulate(devices.begin(), devices.end(), size_t(0),
                [](size_t n, const device_state & d) { return n + d.groups.size(); }),
            indexed_bindings, ineligible_count, devices.size(), cpu_threads,
            indexed_bindings * CPU_GRAPH_CONTEXT_BYTES / (1024 * 1024));
    return true;
#endif
}

void llama_moe_cache_ffn_state::shutdown() {
    ready = false;
    layer_by_index.clear();
#ifdef GGML_USE_CUDA
    for (auto & d : devices) if (d.cpu_resource_id > 0 &&
            !ggml_backend_cuda_moe_ffn_cpu_resource_stop(d.cuda_backend, d.cpu_resource_id)) std::abort();
    for (const auto & d : devices) {
        ggml_cuda_moe_ffn_stats semantic{};
        ggml_cuda_moe_ffn_cpu_resource_stats resource{};
        bool stats_ok = d.cpu_resource_id > 0 &&
                ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(d.cuda_backend, d.cpu_resource_id, &resource);
        for (const auto & layer : d.layer_bindings) {
            ggml_cuda_moe_ffn_stats current{};
            if (layer.binding_id <= 0 ||
                    !ggml_backend_cuda_moe_ffn_get_stats(d.cuda_backend, layer.binding_id, &current)) {
                stats_ok = false;
            } else {
                add_stats(semantic, current);
            }
        }
        if (stats_ok) {
            LLAMA_LOG_INFO("%s: device=%d layers=%zu calls=%llu all-hit=%llu any-miss=%llu gpu-hit-rows=%llu cpu-miss-rows=%llu fills=%llu/%llu failed=%llu fill-bytes=%llu pending-jobs=%llu pending-pins=%llu bindings=%llu\n",
                    __func__, d.device, d.layer_bindings.size(),
                    (unsigned long long) (semantic.all_hit_calls + semantic.any_miss_calls),
                    (unsigned long long) semantic.all_hit_calls,
                    (unsigned long long) semantic.any_miss_calls,
                    (unsigned long long) semantic.gpu_hit_draws,
                    (unsigned long long) semantic.cpu_miss_draws,
                    (unsigned long long) resource.fills_completed,
                    (unsigned long long) resource.fills_accepted,
                    (unsigned long long) resource.fills_failed,
                    (unsigned long long) resource.fill_bytes_completed,
                    (unsigned long long) resource.fill_outstanding,
                    (unsigned long long) resource.pending_pin_active,
                    (unsigned long long) resource.active_bindings);
        } else {
            LLAMA_LOG_WARN("%s: unable to collect final CUDA hybrid FFN counters for device %d\n", __func__, d.device);
        }
    }
    for (auto & d : devices) for (auto & layer : d.layer_bindings) if (layer.binding_id > 0) {
        if (!ggml_backend_cuda_moe_ffn_unregister(d.cuda_backend, layer.binding_id)) std::abort();
        layer.binding_id = 0;
    }
    for (auto & d : devices) {
        for (auto & layer : d.layer_bindings) {
            if (layer.cpu_buffer) { ggml_backend_buffer_free(layer.cpu_buffer); layer.cpu_buffer = nullptr; }
            if (layer.cpu_ctx) { ggml_free(layer.cpu_ctx); layer.cpu_ctx = nullptr; }
        }
        d.layer_bindings.clear();
        if (d.cpu_resource_id > 0) {
            if (!ggml_backend_cuda_moe_ffn_cpu_resource_unregister(d.cuda_backend, d.cpu_resource_id)) std::abort();
            d.cpu_resource_id = 0;
            d.cpu_backend = nullptr;
            d.cpu_backend_transferred = false;
        } else if (d.cpu_backend && !d.cpu_backend_transferred) {
            ggml_backend_free(d.cpu_backend);
            d.cpu_backend = nullptr;
        }
        for (auto & g : d.groups) if (g.lease) {
            if (!g.lease.release()) std::abort();
        }
        d.groups.clear();
    }
    LLAMA_LOG_INFO("%s: CUDA hybrid FFN teardown complete (devices=%zu, bindings=0, resources=0, leases=0)\n",
            __func__, devices.size());
#endif
    devices.clear();
    cache = nullptr;
}

const llama_moe_cache_ffn_state::layer_binding * llama_moe_cache_ffn_state::find_layer(int32_t il) const {
    if (!ready || il < 0 || (size_t) il >= layer_by_index.size()) return nullptr;
    return layer_by_index[il];
}

ggml_tensor * llama_moe_cache_ffn_state::build_node(ggml_context * ctx, ggml_backend_sched_t sched,
        const ggml_moe_ffn_layout & layout) {
#ifdef GGML_USE_CUDA
    const layer_binding * layer = find_layer(layout.cache_layer);
    if (!ready || !layer || !sched || !cache || layer->binding_id <= 0 ||
            !cache->frozen_snapshot_valid(layer->snapshot) || !ggml_moe_ffn_layout_supported(&layout) ||
            !ggml_moe_ffn_graph_supported(&layout) || layout.owner_backend != layer->cuda_backend ||
            layout.cache_owner != cache || layout.device != layer->layout.device ||
            layout.cache_group != layer->layout.cache_group || layout.cache_layer != layer->layout.cache_layer ||
            layout.n_experts != layer->layout.n_experts || layout.n_experts_used != layer->layout.n_experts_used ||
            layout.n_embd != layer->layout.n_embd || layout.n_ff != layer->layout.n_ff ||
            layout.n_slots != layer->layout.n_slots || layout.storage_topology != layer->layout.storage_topology ||
            layout.up != layer->layout.up || layout.gate != layer->layout.gate || layout.down != layer->layout.down ||
            layout.bank_up != layer->layout.bank_up || layout.bank_gate != layer->layout.bank_gate ||
            layout.bank_down != layer->layout.bank_down || layout.up_gate_storage != layer->layout.up_gate_storage ||
            layout.bank_up_gate_storage != layer->layout.bank_up_gate_storage ||
            layout.unary_op != layer->layout.unary_op ||
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
