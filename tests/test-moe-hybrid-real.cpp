#include "llama.h"
#include "llama-model.h"
#include "llama-moe-cache.h"

#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#include "ggml-cpp.h"
#include "ggml-moe-ffn.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <sys/resource.h>
#include <utility>
#include <vector>

namespace {

constexpr int32_t N_SELECTED = 10;
constexpr size_t CACHE_BYTES = size_t(8) << 30;

} // namespace

struct llama_moe_cache_test_access {
    static bool stale_release_retains_lease(llama_moe_cache::frozen_lease & lease,
            const llama_moe_cache_frozen_layer & valid_snapshot, const llama_moe_cache & cache) {
        auto stale = valid_snapshot;
        stale.generation ^= (UINT64_C(1) << 32);
        if (stale.generation == valid_snapshot.generation) ++stale.generation;
        lease.descriptor = stale;
        const bool released = lease.release();
        const bool retained = lease.owner != nullptr && lease.descriptor.generation == stale.generation;
        lease.descriptor = valid_snapshot;
        return !released && retained && cache.frozen_snapshot_valid(valid_snapshot) &&
                cache.get_slot_map(valid_snapshot.layer, 1, N_SELECTED) == nullptr;
    }
};

namespace {

struct error_metrics {
    double max_abs = 0;
    double rms = 0;
    size_t worst = 0;
};

error_metrics compare(const std::vector<float> & a, const std::vector<float> & b) {
    error_metrics result;
    if (a.size() != b.size() || a.empty()) {
        result.max_abs = INFINITY;
        result.rms = INFINITY;
        return result;
    }
    double sum_sq = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double delta = std::abs(double(a[i]) - b[i]);
        if (delta > result.max_abs) { result.max_abs = delta; result.worst = i; }
        sum_sq += delta * delta;
    }
    result.rms = std::sqrt(sum_sq / a.size());
    return result;
}

bool all_finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

void report_tensor(const char * part, const ggml_tensor * t) {
    std::printf("tensor part=%s name=%s type=%s shape=[%lld,%lld,%lld,%lld] stride=[%zu,%zu,%zu,%zu] host=%d weights=%d\n",
            part, t->name, ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1],
            (long long)t->ne[2], (long long)t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3],
            t->buffer && ggml_backend_buffer_is_host(t->buffer),
            t->buffer && ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
}

struct graph_fixture {
    ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * activation = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * routes = nullptr;
    ggml_tensor * slot_map = nullptr;
    ggml_tensor * mapped_ids = nullptr;
    ggml_tensor * output = nullptr;

    void clear() {
        if (buffer) ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        if (ctx) ggml_free(ctx);
        ctx = nullptr;
        graph = nullptr;
        activation = ids = routes = slot_map = mapped_ids = output = nullptr;
    }
};

struct real_fixture {
    llama_model * model = nullptr;
    ggml_backend_t cuda = nullptr;
    ggml_backend_t cpu = nullptr;
    bool cpu_owned_by_resource = false;
    bool backend_initialized = false;
    bool lease_restored = true;
    std::unique_ptr<llama_moe_cache> cache;
    llama_moe_cache::frozen_lease lease;
    int32_t cpu_resource_id = 0;
    int32_t binding_id = 0;
    graph_fixture canonical_cpu;
    graph_fixture cpu_single;
    graph_fixture cached_cuda;
    graph_fixture semantic_cuda;

    bool unregister_execution() {
        if (cpu_resource_id > 0 && cuda &&
                !ggml_backend_cuda_moe_ffn_cpu_resource_stop(cuda, cpu_resource_id)) return false;
        if (binding_id > 0 && cuda) {
            const int32_t stopped_binding = binding_id;
            if (!ggml_backend_cuda_moe_ffn_unregister(cuda, stopped_binding) ||
                    ggml_backend_cuda_moe_ffn_unregister(cuda, stopped_binding)) return false;
            binding_id = 0;
        }
        semantic_cuda.clear();
        cached_cuda.clear();
        cpu_single.clear();
        canonical_cpu.clear();
        if (cpu_resource_id > 0 && cuda) {
            if (!ggml_backend_cuda_moe_ffn_cpu_resource_unregister(cuda, cpu_resource_id)) return false;
            cpu_resource_id = 0;
            cpu = nullptr;
            cpu_owned_by_resource = false;
        }
        if (cpu && !cpu_owned_by_resource) {
            ggml_backend_free(cpu);
            cpu = nullptr;
        }
        return true;
    }

    void cleanup() {
        if (!unregister_execution()) {
            std::fprintf(stderr, "failed to quiesce/unregister semantic execution before owner teardown\n");
            std::abort();
        }
        if (lease) {
            const llama_moe_cache_frozen_layer frozen = lease.snapshot();
            lease_restored = lease.release();
            if (lease_restored && cache && cuda) {
                ggml_tensor * map = cache->get_slot_map(frozen.layer, 1, N_SELECTED);
                lease_restored = !cache->frozen_snapshot_valid(frozen) && map &&
                        cache->copy(cuda, map, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_QUERY);
            }
            std::printf("lease_release_restored=%d stale_snapshot_rejected=%d normal_query_restored=%d\n",
                    lease_restored ? 1 : 0,
                    cache && !cache->frozen_snapshot_valid(frozen) ? 1 : 0,
                    cache && cache->get_slot_map(frozen.layer, 1, N_SELECTED) != nullptr ? 1 : 0);
        }
        cache.reset();
        if (cuda) {
            ggml_backend_free(cuda);
            cuda = nullptr;
        }
        if (model) {
            llama_free_model(model);
            model = nullptr;
        }
        if (backend_initialized) {
            llama_backend_free();
            backend_initialized = false;
        }
    }

    ~real_fixture() { cleanup(); }
};

bool make_cpu_graph(graph_fixture & f, ggml_backend_t backend, const ggml_tensor * up,
        const ggml_tensor * gate, const ggml_tensor * down, int32_t n_embd, int32_t n_ff,
        int32_t n_selected) {
    f.ctx = ggml_init({ 32 * 1024 * 1024, nullptr, true });
    if (!f.ctx) return false;
    f.activation = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, n_embd, 1);
    f.ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, n_selected, 1);
    f.routes = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, 1, n_selected, 1);
    // GGML graph constructors take mutable pointers even though these source weights are read-only.
    ggml_tensor * activated = ggml_moe_up_gate(f.ctx, const_cast<ggml_tensor *>(up),
            const_cast<ggml_tensor *>(gate), f.activation, f.ids, GGML_UNARY_OP_SILU);
    ggml_tensor * projected = ggml_mul_mat_id(f.ctx, const_cast<ggml_tensor *>(down), activated, f.ids);
    f.output = ggml_mul_multi_add(f.ctx, projected, f.routes);
    if (f.output->ne[0] != n_embd || f.output->ne[1] != 1 || f.output->ne[2] != 1 || n_ff <= 0) return false;
    f.graph = ggml_new_graph_custom(f.ctx, 64, false);
    ggml_build_forward_expand(f.graph, f.output);
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, backend);
    return f.buffer != nullptr;
}

bool make_blocking_graph(graph_fixture & f, ggml_backend_t backend, llama_moe_cache & cache,
        int32_t layer_id, const llama_layer & layer, int32_t n_embd, int32_t n_ff) {
    const ggml_tensor * host_map = cache.get_slot_map(layer_id, 1, N_SELECTED);
    const ggml_tensor * up = cache.get_experts(layer.ffn_up_exps);
    const ggml_tensor * gate = cache.get_experts(layer.ffn_gate_exps);
    const ggml_tensor * down = cache.get_experts(layer.ffn_down_exps);
    if (!host_map || !up || !gate || !down) return false;
    f.ctx = ggml_init({ 32 * 1024 * 1024, nullptr, true });
    if (!f.ctx) return false;
    f.activation = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, n_embd, 1);
    f.ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, N_SELECTED, 1);
    f.routes = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, 1, N_SELECTED, 1);
    f.slot_map = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, host_map->ne[0], host_map->ne[1]);
    // Match the production cache lookup: GET_ROWS returns [1, top_k], then the
    // graph reshapes its slots to [top_k, n_tokens] for the MoE kernels.
    f.mapped_ids = ggml_reshape_2d(f.ctx, ggml_get_rows(f.ctx, f.slot_map, f.ids), N_SELECTED, 1);
    ggml_tensor * activated = ggml_moe_up_gate(f.ctx, const_cast<ggml_tensor *>(up),
            const_cast<ggml_tensor *>(gate), f.activation, f.mapped_ids, GGML_UNARY_OP_SILU);
    ggml_tensor * projected = ggml_mul_mat_id(f.ctx, const_cast<ggml_tensor *>(down), activated, f.mapped_ids);
    f.output = ggml_mul_multi_add(f.ctx, projected, f.routes);
    if (f.output->ne[0] != n_embd || f.output->ne[1] != 1 || f.output->ne[2] != 1 || n_ff <= 0) return false;
    f.graph = ggml_new_graph_custom(f.ctx, 64, false);
    ggml_build_forward_expand(f.graph, f.output);
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, backend);
    return f.buffer != nullptr;
}

bool make_callback_graph(graph_fixture & f, ggml_backend_t backend,
        const ggml_tensor * host_map, int32_t n_ids) {
    f.ctx = ggml_init({ 4 * 1024 * 1024, nullptr, true });
    if (!f.ctx) return false;
    f.slot_map = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, host_map->ne[0], host_map->ne[1]);
    f.ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, n_ids, 1);
    f.mapped_ids = ggml_get_rows(f.ctx, f.slot_map, f.ids);
    f.graph = ggml_new_graph_custom(f.ctx, 8, false);
    ggml_build_forward_expand(f.graph, f.mapped_ids);
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, backend);
    return f.buffer != nullptr;
}

bool make_semantic_graph(graph_fixture & f, ggml_backend_t backend, int32_t device,
        int32_t binding_id, uint64_t generation, int32_t n_embd, int32_t n_ff) {
    f.ctx = ggml_init({ 16 * 1024 * 1024, nullptr, true });
    if (!f.ctx) return false;
    f.activation = ggml_new_tensor_2d(f.ctx, GGML_TYPE_F32, n_embd, 1);
    f.ids = ggml_new_tensor_2d(f.ctx, GGML_TYPE_I32, N_SELECTED, 1);
    f.routes = ggml_new_tensor_3d(f.ctx, GGML_TYPE_F32, 1, N_SELECTED, 1);
    f.output = ggml_moe_ffn(f.ctx, f.activation, f.ids, f.routes, device, binding_id,
            generation, n_embd, n_ff, GGML_UNARY_OP_SILU, 0.0f);
    f.graph = ggml_new_graph_custom(f.ctx, 16, false);
    ggml_build_forward_expand(f.graph, f.output);
    f.buffer = ggml_backend_alloc_ctx_tensors(f.ctx, backend);
    return f.buffer != nullptr;
}

void set_inputs(graph_fixture & f, ggml_backend_t backend, const std::vector<float> & activation,
        const std::vector<int32_t> & ids, const std::vector<float> & routes) {
    ggml_backend_tensor_set(f.activation, activation.data(), 0, activation.size() * sizeof(float));
    ggml_backend_tensor_set(f.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
    ggml_backend_tensor_set(f.routes, routes.data(), 0, routes.size() * sizeof(float));
    (void) backend;
}

bool compute(graph_fixture & f, ggml_backend_t backend, const std::vector<float> & activation,
        const std::vector<int32_t> & ids, const std::vector<float> & routes,
        std::vector<float> & result) {
    set_inputs(f, backend, activation, ids, routes);
    if (ggml_backend_graph_compute(backend, f.graph) != GGML_STATUS_SUCCESS) return false;
    result.resize(ggml_nelements(f.output));
    ggml_backend_tensor_get(f.output, result.data(), 0, result.size() * sizeof(float));
    return all_finite(result);
}

std::vector<float> route_weights(int variant) {
    std::vector<float> values(N_SELECTED);
    float total = 0;
    for (int32_t i = 0; i < N_SELECTED; ++i) {
        values[i] = 0.25f + float((i * 7 + variant * 3) % 13) * 0.11f;
        total += values[i];
    }
    for (float & value : values) value /= total;
    return values;
}

bool valid_candidate(const llama_model & model, const llama_layer & layer) {
    const ggml_tensor * up = layer.ffn_up_exps;
    const ggml_tensor * gate = layer.ffn_gate_exps;
    const ggml_tensor * down = layer.ffn_down_exps;
    if (layer.ffn_up_gate_exps || !up || !gate || !down) return false;
    if (up->type != GGML_TYPE_IQ3_S || gate->type != GGML_TYPE_IQ3_S || down->type != GGML_TYPE_IQ4_NL) return false;
    if (up->ne[2] != (int64_t)model.hparams.n_expert || gate->ne[2] != up->ne[2] || down->ne[2] != up->ne[2]) return false;
    if (up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] || down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0]) return false;
    if (!up->data || !gate->data || !down->data || !up->buffer || !gate->buffer || !down->buffer) return false;
    for (const ggml_tensor * t : { up, gate, down }) {
        if (!ggml_backend_buffer_is_host(t->buffer) || ggml_backend_buffer_get_usage(t->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) return false;
        if (t->nb[2] != ggml_nbytes(t) / t->ne[2]) return false;
        if (model.hadamard_rotation(t)) return false;
    }
    return !layer.ffn_up_exps_b && !layer.ffn_gate_exps_b && !layer.ffn_down_exps_b && !layer.ffn_down_exps_s;
}

struct timing_samples {
    std::vector<double> us;
};

double process_cpu_seconds(const rusage & value) {
    return double(value.ru_utime.tv_sec) + double(value.ru_utime.tv_usec) / 1.0e6 +
            double(value.ru_stime.tv_sec) + double(value.ru_stime.tv_usec) / 1.0e6;
}

void print_timing_summary(const char * arm, const timing_samples & samples, double cpu_cores) {
    if (samples.us.empty()) return;
    std::vector<double> sorted = samples.us;
    std::sort(sorted.begin(), sorted.end());
    const double mean = std::accumulate(sorted.begin(), sorted.end(), 0.0) / sorted.size();
    const double median = sorted.size() % 2 ? sorted[sorted.size()/2] :
            (sorted[sorted.size()/2 - 1] + sorted[sorted.size()/2]) / 2.0;
    const size_t trim = sorted.size() / 10;
    const double trimmed = std::accumulate(sorted.begin() + trim, sorted.end() - trim, 0.0) /
            (sorted.size() - 2 * trim);
    const double p95 = sorted[std::min(sorted.size() - 1, size_t(std::ceil(sorted.size() * 0.95)) - 1)];
    double variance = 0;
    for (double x : sorted) variance += (x - mean) * (x - mean);
    const double stdev = std::sqrt(variance / sorted.size());
    std::printf("timing_summary,arm=%s,n=%zu,median_us=%.3f,mean_us=%.3f,trimmed_mean_us=%.3f,p95_us=%.3f,min_us=%.3f,stdev_us=%.3f,process_cpu_cores=%.2f\n",
            arm, sorted.size(), median, mean, trimmed, p95, sorted.front(), stdev, cpu_cores);
}

bool run_timing(real_fixture & f, const std::vector<float> & activation,
        const std::vector<int32_t> & hit_ids, const std::vector<int32_t> & hit_slots,
        const std::vector<int32_t> & miss_ids, const std::vector<int32_t> & five_miss_ids,
        const std::vector<int32_t> & all_miss_ids, const std::vector<float> & routes,
        const char * mode, const std::vector<int32_t> & cpu_thread_counts, int32_t samples) {
    if (hit_slots.size() != N_SELECTED) return false;
    const std::vector<int32_t> & slots = hit_slots;
    std::vector<float> cpu_reference;
    if (!compute(f.canonical_cpu, f.cpu, activation, miss_ids, routes, cpu_reference)) return false;
    std::vector<float> five_cpu_reference;
    if (!compute(f.canonical_cpu, f.cpu, activation, five_miss_ids, routes, five_cpu_reference)) return false;
    std::vector<float> all_cpu_reference;
    if (!compute(f.canonical_cpu, f.cpu, activation, all_miss_ids, routes, all_cpu_reference)) return false;
    std::vector<float> hit_reference;
    if (!compute(f.cached_cuda, f.cuda, activation, hit_slots, routes, hit_reference)) return false;
    std::vector<float> output(activation.size());
    const size_t bytes_activation = activation.size() * sizeof(float);
    const size_t bytes_ids = miss_ids.size() * sizeof(int32_t);
    const size_t bytes_routes = routes.size() * sizeof(float);
    std::vector<float> host_activation(activation.size());
    std::vector<int32_t> host_ids(miss_ids.size());
    std::vector<float> host_routes(routes.size());

    const auto time_graph = [&](const char * arm, graph_fixture & graph, ggml_backend_t backend,
            const std::vector<int32_t> & ids, int rep, int count, int threads = 0,
            const std::vector<float> * expected = nullptr) {
        timing_samples samples;
        for (int i = 0; i < 8; ++i) {
            set_inputs(graph, backend, activation, ids, routes);
            if (ggml_backend_graph_compute(backend, graph.graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_synchronize(backend);
        }
        rusage cpu_before{};
        rusage cpu_after{};
        getrusage(RUSAGE_SELF, &cpu_before);
        const auto cpu_wall_start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) {
            set_inputs(graph, backend, activation, ids, routes);
            const auto start = std::chrono::steady_clock::now();
            const ggml_status status = ggml_backend_graph_compute(backend, graph.graph);
            ggml_backend_synchronize(backend);
            const auto stop = std::chrono::steady_clock::now();
            if (status != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(graph.output, output.data(), 0, bytes_activation);
            if (!all_finite(output)) return false;
            const bool semantic_miss = graph.output == f.semantic_cuda.output && ids != hit_ids;
            const std::vector<float> & reference = expected ? *expected :
                    (semantic_miss ? cpu_reference : hit_reference);
            const error_metrics error = compare(output, reference);
            if (error.max_abs > (graph.output == f.semantic_cuda.output ? 1.0e-4 : 1.0e-6)) return false;
            const double us = std::chrono::duration<double, std::micro>(stop - start).count();
            samples.us.push_back(us);
            std::printf("timing_sample,arm=%s,rep=%d,index=%d,threads=%d,us=%.3f\n", arm, rep, i, threads, us);
        }
        const auto cpu_wall_stop = std::chrono::steady_clock::now();
        getrusage(RUSAGE_SELF, &cpu_after);
        const double wall_seconds = std::chrono::duration<double>(cpu_wall_stop - cpu_wall_start).count();
        print_timing_summary(arm, samples, (process_cpu_seconds(cpu_after) - process_cpu_seconds(cpu_before)) / wall_seconds);
        return true;
    };

    const auto time_transfer_cpu = [&](const char * arm, const std::vector<int32_t> & ids,
            const std::vector<float> & reference, int rep, int count, int threads = 0) {
        timing_samples samples;
        for (int i = 0; i < 8; ++i) {
            ggml_backend_tensor_get_async(f.cuda, f.semantic_cuda.activation, host_activation.data(), 0, bytes_activation);
            ggml_backend_tensor_get_async(f.cuda, f.semantic_cuda.ids, host_ids.data(), 0, bytes_ids);
            ggml_backend_tensor_get_async(f.cuda, f.semantic_cuda.routes, host_routes.data(), 0, bytes_routes);
            ggml_backend_synchronize(f.cuda);
            ggml_backend_tensor_set(f.canonical_cpu.activation, host_activation.data(), 0, bytes_activation);
            ggml_backend_tensor_set(f.canonical_cpu.ids, host_ids.data(), 0, bytes_ids);
            ggml_backend_tensor_set(f.canonical_cpu.routes, host_routes.data(), 0, bytes_routes);
            if (ggml_backend_graph_compute(f.cpu, f.canonical_cpu.graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(f.canonical_cpu.output, output.data(), 0, bytes_activation);
            ggml_backend_tensor_set_async(f.cuda, f.semantic_cuda.output, output.data(), 0, bytes_activation);
            ggml_backend_synchronize(f.cuda);
        }
        rusage cpu_before{};
        rusage cpu_after{};
        getrusage(RUSAGE_SELF, &cpu_before);
        const auto cpu_wall_start = std::chrono::steady_clock::now();
        for (int i = 0; i < count; ++i) {
            set_inputs(f.semantic_cuda, f.cuda, activation, ids, routes);
            const auto start = std::chrono::steady_clock::now();
            ggml_backend_tensor_get_async(f.cuda, f.semantic_cuda.activation, host_activation.data(), 0, bytes_activation);
            ggml_backend_tensor_get_async(f.cuda, f.semantic_cuda.ids, host_ids.data(), 0, bytes_ids);
            ggml_backend_tensor_get_async(f.cuda, f.semantic_cuda.routes, host_routes.data(), 0, bytes_routes);
            ggml_backend_synchronize(f.cuda);
            ggml_backend_tensor_set(f.canonical_cpu.activation, host_activation.data(), 0, bytes_activation);
            ggml_backend_tensor_set(f.canonical_cpu.ids, host_ids.data(), 0, bytes_ids);
            ggml_backend_tensor_set(f.canonical_cpu.routes, host_routes.data(), 0, bytes_routes);
            if (ggml_backend_graph_compute(f.cpu, f.canonical_cpu.graph) != GGML_STATUS_SUCCESS) return false;
            ggml_backend_tensor_get(f.canonical_cpu.output, output.data(), 0, bytes_activation);
            const auto cpu_done = std::chrono::steady_clock::now();
            ggml_backend_tensor_set_async(f.cuda, f.semantic_cuda.output, output.data(), 0, bytes_activation);
            ggml_backend_synchronize(f.cuda);
            const auto stop = std::chrono::steady_clock::now();
            const error_metrics error = compare(output, reference);
            if (!all_finite(output) || error.max_abs != 0.0) return false;
            const double us = std::chrono::duration<double, std::micro>(stop - start).count();
            samples.us.push_back(us);
            std::printf("timing_sample,arm=%s,rep=%d,index=%d,threads=%d,us=%.3f,cpu_done_us=%.3f,d2h_bytes=%zu,h2d_bytes=%zu\n",
                    arm, rep, i, threads, us, std::chrono::duration<double, std::micro>(cpu_done - start).count(),
                    bytes_activation + bytes_ids + bytes_routes, bytes_activation);
        }
        const auto cpu_wall_stop = std::chrono::steady_clock::now();
        getrusage(RUSAGE_SELF, &cpu_after);
        const double wall_seconds = std::chrono::duration<double>(cpu_wall_stop - cpu_wall_start).count();
        print_timing_summary(arm, samples, (process_cpu_seconds(cpu_after) - process_cpu_seconds(cpu_before)) / wall_seconds);
        return true;
    };

    const bool disabled = std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr;
    std::printf("timing_mode=%s,cuda_graphs_disabled=%d,logical_ids_same=1,cache_slots_mapped_for_stock=1,completion=backend_synchronize,transfer_d2h_bytes=%zu,transfer_h2d_bytes=%zu\n",
            mode, disabled ? 1 : 0, bytes_activation + bytes_ids + bytes_routes, bytes_activation);
    if (std::strcmp(mode, "primary") == 0) {
        return time_graph("stock_capture_A1", f.cached_cuda, f.cuda, slots, 1, 24) &&
                time_graph("semantic_allhit_C1", f.semantic_cuda, f.cuda, hit_ids, 1, 24) &&
                time_graph("semantic_allhit_C2", f.semantic_cuda, f.cuda, hit_ids, 2, 24) &&
                time_graph("stock_capture_A2", f.cached_cuda, f.cuda, slots, 2, 24);
    }
    if (std::strcmp(mode, "nocapture") == 0) {
        if (!disabled) { std::fprintf(stderr, "nocapture mode requires GGML_CUDA_DISABLE_GRAPHS=1\n"); return false; }
        return time_graph("stock_no_capture_B1", f.cached_cuda, f.cuda, slots, 1, 24) &&
                time_graph("stock_no_capture_B2", f.cached_cuda, f.cuda, slots, 2, 24);
    }
    if (std::strcmp(mode, "miss") == 0) {
        return time_graph("semantic_partition_1miss_D1", f.semantic_cuda, f.cuda, miss_ids, 1, 20, 0, &cpu_reference) &&
                time_graph("semantic_partition_1miss_D2", f.semantic_cuda, f.cuda, miss_ids, 2, 20, 0, &cpu_reference) &&
                time_graph("semantic_partition_5miss_F1", f.semantic_cuda, f.cuda, five_miss_ids, 1, 20, 0, &five_cpu_reference) &&
                time_graph("semantic_partition_5miss_F2", f.semantic_cuda, f.cuda, five_miss_ids, 2, 20, 0, &five_cpu_reference) &&
                time_graph("semantic_allmiss_G1", f.semantic_cuda, f.cuda, all_miss_ids, 1, 20, 0, &all_cpu_reference) &&
                time_graph("semantic_allmiss_G2", f.semantic_cuda, f.cuda, all_miss_ids, 2, 20, 0, &all_cpu_reference) &&
                time_transfer_cpu("transfer_cpu_1miss_E1", miss_ids, cpu_reference, 1, 20) &&
                time_transfer_cpu("transfer_cpu_1miss_E2", miss_ids, cpu_reference, 2, 20) &&
                time_transfer_cpu("transfer_cpu_5miss_E3", five_miss_ids, five_cpu_reference, 3, 20) &&
                time_transfer_cpu("transfer_cpu_5miss_E4", five_miss_ids, five_cpu_reference, 4, 20) &&
                time_transfer_cpu("transfer_cpu_allmiss_E5", all_miss_ids, all_cpu_reference, 5, 20) &&
                time_transfer_cpu("transfer_cpu_allmiss_E6", all_miss_ids, all_cpu_reference, 6, 20);
    }
    if (std::strcmp(mode, "trace") == 0) {
        return time_graph("stock_capture_trace_A", f.cached_cuda, f.cuda, slots, 1, 4) &&
                time_graph("semantic_allhit_trace_C", f.semantic_cuda, f.cuda, hit_ids, 1, 4);
    }
    if (std::strcmp(mode, "scaling") == 0 || std::strcmp(mode, "scaling-abba") == 0) {
        if (cpu_thread_counts.empty()) return false;
        const int scale_samples = samples;
        if (std::strcmp(mode, "scaling") == 0) {
            if (time_graph("stock_capture_A1", f.cached_cuda, f.cuda, slots, 1, 24) &&
                time_graph("semantic_allhit_C1", f.semantic_cuda, f.cuda, hit_ids, 1, 24) &&
                time_graph("semantic_allhit_C2", f.semantic_cuda, f.cuda, hit_ids, 2, 24) &&
                time_graph("stock_capture_A2", f.cached_cuda, f.cuda, slots, 2, 24)) {
                // Continue with each CPU resource setting only after the all-hit control blocks settle.
            } else {
                return false;
            }
            for (size_t i = 0; i < cpu_thread_counts.size(); ++i) {
                const int32_t threads = cpu_thread_counts[i];
                if (threads <= 0) return false;
                ggml_backend_cpu_set_n_threads(f.cpu, threads);
                std::printf("cpu_scale_config,threads=%d,max_parallel_calls_required=1,affinity=inherited\n", threads);
                const std::string sem = "semantic_anymiss_t" + std::to_string(threads);
                const std::string xfer = "transfer_cpu_t" + std::to_string(threads);
            if (!time_graph(sem.c_str(), f.semantic_cuda, f.cuda, miss_ids, (int)i, scale_samples, threads, &cpu_reference) ||
                    !time_transfer_cpu(xfer.c_str(), miss_ids, cpu_reference, (int)i, scale_samples, threads)) return false;
            }
            return true;
        }
        if (cpu_thread_counts.size() != 2) return false;
        const std::array<int32_t, 4> order = { cpu_thread_counts[0], cpu_thread_counts[1],
                cpu_thread_counts[1], cpu_thread_counts[0] };
        const std::array<const char *, 4> labels = { "A1", "B1", "B2", "A2" };
        for (size_t i = 0; i < order.size(); ++i) {
            const int32_t threads = order[i];
            if (threads <= 0) return false;
            ggml_backend_cpu_set_n_threads(f.cpu, threads);
            std::printf("cpu_scale_config,block=%s,threads=%d,max_parallel_calls_required=1,affinity=inherited\n",
                    labels[i], threads);
            const std::string sem = std::string("semantic_anymiss_") + labels[i];
            const std::string xfer = std::string("transfer_cpu_") + labels[i];
            if (!time_graph(sem.c_str(), f.semantic_cuda, f.cuda, miss_ids, (int)i, samples, threads, &cpu_reference) ||
                !time_transfer_cpu(xfer.c_str(), miss_ids, cpu_reference, (int)i, samples, threads)) return false;
        }
        return true;
    }
    if (std::strcmp(mode, "thread-curve") == 0 || std::strcmp(mode, "thread-abba") == 0) {
        std::vector<int32_t> order = cpu_thread_counts;
        std::vector<const char *> labels(order.size(), nullptr);
        if (std::strcmp(mode, "thread-curve") == 0) {
            if (order.empty()) return false;
        } else {
            if (order.size() != 2) return false;
            order = { cpu_thread_counts[0], cpu_thread_counts[1], cpu_thread_counts[1], cpu_thread_counts[0] };
            labels = { "A1", "B1", "B2", "A2" };
        }
        for (size_t i = 0; i < order.size(); ++i) {
            const int32_t threads = order[i];
            if (threads <= 0) return false;
            // f.cpu is the same backend registered with the CUDA-owned resource.
            ggml_backend_cpu_set_n_threads(f.cpu, threads);
            const std::string suffix = labels[i] ? labels[i] : "t" + std::to_string(threads);
            std::printf("cpu_scale_config,block=%s,threads=%d,max_parallel_calls_required=1,affinity=inherited,registered_backend=1\n",
                    suffix.c_str(), threads);
            const std::string all_hit = "semantic_allhit_" + suffix;
            const std::string one_miss = "semantic_partition_1miss_" + suffix;
            const std::string five_miss = "semantic_partition_5miss_" + suffix;
            const std::string all_miss = "semantic_complete_cpu_allmiss_" + suffix;
            const int rep = (int) i;
            if (!time_graph(all_hit.c_str(), f.semantic_cuda, f.cuda, hit_ids, rep, samples,
                            threads, &hit_reference) ||
                !time_graph(one_miss.c_str(), f.semantic_cuda, f.cuda, miss_ids, rep, samples,
                            threads, &cpu_reference) ||
                !time_graph(five_miss.c_str(), f.semantic_cuda, f.cuda, five_miss_ids, rep, samples,
                            threads, &five_cpu_reference) ||
                !time_graph(all_miss.c_str(), f.semantic_cuda, f.cuda, all_miss_ids, rep, samples,
                            threads, &all_cpu_reference)) return false;
        }
        return true;
    }
    return std::strcmp(mode, "correctness") == 0;
}

std::vector<int32_t> blocking_ids(int32_t miss_count, int32_t sample, int32_t resident_count) {
    std::vector<int32_t> ids;
    ids.reserve(N_SELECTED);
    for (int32_t i = 0; i < resident_count; ++i) ids.push_back(i);
    const int32_t cold_base = 32 + sample * miss_count;
    for (int32_t i = 0; i < miss_count; ++i) ids.push_back(cold_base + i);
    return ids;
}

bool run_blocking_cache(real_fixture & f, int32_t layer_id, int32_t n_embd, int32_t n_ff,
        int32_t miss_count, int32_t samples, bool graphs_disabled) {
    if (miss_count < 1 || miss_count > N_SELECTED || samples < 1) return false;
    if (f.cache->backend(layer_id) != f.cuda) return false;
    const llama_layer & layer = f.model->layers[layer_id];
    const int32_t resident_count = N_SELECTED - miss_count;
    const ggml_tensor * source_map = f.cache->get_slot_map(layer_id, 1, N_SELECTED);
    if (!source_map || !make_blocking_graph(f.cached_cuda, f.cuda, *f.cache, layer_id, layer, n_embd, n_ff) ||
            !make_cpu_graph(f.canonical_cpu, f.cpu, layer.ffn_up_exps, layer.ffn_gate_exps,
                    layer.ffn_down_exps, n_embd, n_ff, N_SELECTED)) return false;
    graph_fixture seed_graph;
    if (resident_count > 0 && !make_callback_graph(seed_graph, f.cuda, source_map, resident_count)) return false;
    if (!f.cache->copy(f.cuda, source_map, f.cached_cuda.slot_map, f.cached_cuda.graph,
                GGML_BACKEND_SCHED_COPY_PHASE_QUERY)) return false;

    const std::vector<float> activation = [&] {
        std::vector<float> v(n_embd);
        for (int32_t i = 0; i < n_embd; ++i) v[i] = 0.25f * std::sin(float(i + 17) * 0.007f);
        return v;
    }();
    const std::vector<float> routes = route_weights(0);
    const size_t expert_bytes = layer.ffn_up_exps->nb[2] + layer.ffn_gate_exps->nb[2] +
            layer.ffn_down_exps->nb[2];
    size_t seed_bytes = 0;
    if (resident_count > 0) {
        std::vector<int32_t> seed_ids(resident_count);
        std::iota(seed_ids.begin(), seed_ids.end(), 0);
        ggml_backend_tensor_set(seed_graph.ids, seed_ids.data(), 0, ggml_nbytes(seed_graph.ids));
        if (!f.cache->copy(f.cuda, source_map, seed_graph.slot_map, seed_graph.graph,
                    GGML_BACKEND_SCHED_COPY_PHASE_EXECUTE)) return false;
        ggml_backend_synchronize(f.cuda);
        seed_bytes = resident_count * expert_bytes;
    }
    std::printf("blocking_setup,miss_count=%d,resident_seed_experts=%d,seed_bytes=%zu,cache_slot_map_shape=[%lld,%lld],callback=llama_moe_cache::copy_execute,capture_setting=%s\n",
            miss_count, resident_count, seed_bytes, (long long)source_map->ne[0],
            (long long)source_map->ne[1], graphs_disabled ? "disabled" : "normal");

    const auto run_one = [&](const char * phase, int sample) {
        const std::vector<int32_t> ids = blocking_ids(miss_count, sample, resident_count);
        if (ids.size() != N_SELECTED || std::set<int32_t>(ids.begin(), ids.end()).size() != N_SELECTED) return false;
        std::vector<float> expected;
        if (!compute(f.canonical_cpu, f.cpu, activation, ids, routes, expected)) return false;
        set_inputs(f.cached_cuda, f.cuda, activation, ids, routes);

        const int32_t * before = (const int32_t *)source_map->data;
        int32_t hit_count = 0;
        std::vector<int32_t> misses;
        for (int32_t expert : ids) {
            if (before[expert] >= 0) ++hit_count;
            else misses.push_back(expert);
        }
        if ((int)misses.size() != miss_count) {
            std::fprintf(stderr, "blocking route did not contain the requested cold miss count\n");
            return false;
        }

        const auto start = std::chrono::steady_clock::now();
        if (!f.cache->copy(f.cuda, source_map, f.cached_cuda.slot_map, f.cached_cuda.graph,
                    GGML_BACKEND_SCHED_COPY_PHASE_EXECUTE)) return false;
        if (ggml_backend_graph_compute(f.cuda, f.cached_cuda.graph) != GGML_STATUS_SUCCESS) return false;
        ggml_backend_synchronize(f.cuda);
        const auto stop = std::chrono::steady_clock::now();

        std::vector<float> output(n_embd);
        ggml_backend_tensor_get(f.cached_cuda.output, output.data(), 0, output.size() * sizeof(float));
        const error_metrics error = compare(output, expected);
        if (!all_finite(output) || error.max_abs > 1.0e-4 ||
                !std::all_of(ids.begin(), ids.end(), [&](int32_t id) { return ((const int32_t *)source_map->data)[id] >= 0; })) {
            std::fprintf(stderr, "blocking cache output/mapping validation failed: max_abs=%.9g rms=%.9g\n",
                    error.max_abs, error.rms);
            return false;
        }

        std::sort(misses.begin(), misses.end());
        const int32_t * after = (const int32_t *)source_map->data;
        size_t upload_runs = 0;
        for (size_t i = 0; i < misses.size(); ++i) {
            if (i == 0 || misses[i] != misses[i - 1] + 1 || after[misses[i]] != after[misses[i - 1]] + 1) {
                ++upload_runs;
            }
        }
        const size_t part_count = 3; // validated separate up, gate and down banks
        const size_t expert_h2d_calls = upload_runs * part_count;
        const size_t expert_h2d_bytes = misses.size() * expert_bytes;
        const double us = std::chrono::duration<double, std::micro>(stop - start).count();
        std::printf("blocking_sample,phase=%s,miss_count=%d,index=%d,selected=%zu,hits=%d,misses=%zu,expert_upload_calls=%zu,expert_upload_bytes=%zu,slot_map_h2d_calls=1,slot_map_h2d_bytes=%zu,callback_id_d2h_bytes=%zu,callback_explicit_syncs=1,completion_syncs=1,capture_setting=%s,us=%.3f,max_abs=%.9g,rms=%.9g\n",
                phase, miss_count, sample, ids.size(), hit_count, misses.size(), expert_h2d_calls,
                expert_h2d_bytes, ggml_nbytes(source_map), ggml_nbytes(f.cached_cuda.ids),
                graphs_disabled ? "disabled" : "normal", us, error.max_abs, error.rms);
        return true;
    };

    // These real callback invocations warm vector capacity, transfer submission and graph capture.
    // Their cold expert IDs are disjoint from all timed route sets.
    for (int i = 0; i < 8; ++i) if (!run_one("warmup", i)) return false;
    for (int i = 0; i < samples; ++i) if (!run_one("measured", i + 8)) return false;
    std::printf("blocking_result,miss_count=%d,measured_samples=%d,distinct_cold_experts=%d,eviction_expected=0,full_weight_copy=0\n",
            miss_count, samples, miss_count * (samples + 8));
    return true;
}

bool run_seed_fill_isolation(real_fixture & f, int32_t layer_a, int32_t n_experts) {
    ggml_tensor * const ordinary_map = f.cache->get_slot_map(layer_a, 1, N_SELECTED);
    if (!ordinary_map) return false;
    llama_moe_cache::frozen_lease lease_a;
    if (!f.cache->freeze_group_for_layer(layer_a, f.cuda, { 0 }, lease_a)) return false;
    const auto seeded_a = lease_a.snapshot();
    ggml_cuda_moe_ffn_resolution before{};
    const int32_t expert = 0;
    if (!f.cache->resolve_and_pin(seeded_a.device_group, seeded_a.group, layer_a, seeded_a.generation,
                &expert, 1, &before) || before.n_draws != 1 || before.n_pins != 1 || before.draw_slots[0] < 0) return false;
    const int32_t slot = before.draw_slots[0];
    const auto owner_before = before.pins[0];
    if (owner_before.key != layer_a * n_experts + expert || owner_before.slot != slot ||
            !f.cache->release_pins(before.pins, before.n_pins)) return false;

    ggml_backend_synchronize(f.cuda);
    const std::array<const ggml_tensor *, 3> banks_before = { seeded_a.bank_up, seeded_a.bank_gate, seeded_a.bank_down };
    std::array<std::vector<uint8_t>, 3> contents_before;
    for (size_t part = 0; part < banks_before.size(); ++part) {
        const size_t bytes = banks_before[part]->nb[2];
        contents_before[part].resize(bytes);
        ggml_backend_tensor_get(banks_before[part], contents_before[part].data(), (size_t) slot * bytes, bytes);
    }
    if (!lease_a.release()) return false;

    int32_t layer_b = -1;
    for (const int32_t candidate : seeded_a.group_layers) {
        if (candidate != layer_a && valid_candidate(*f.model, f.model->layers[candidate])) {
            layer_b = candidate;
            break;
        }
    }
    if (layer_b < 0) return false;
    llama_moe_cache::frozen_lease lease_b;
    if (!f.cache->freeze_group_for_layer(layer_b, f.cuda, {}, lease_b)) return false;
    const auto empty_b = lease_b.snapshot();
    llama_moe_cache_frozen_layer seeded_a_after;
    ggml_cuda_moe_ffn_resolution after{};
    const bool snapshot_ok = empty_b.group == seeded_a.group && empty_b.seed_bytes == 0 &&
            f.cache->frozen_layer_snapshot(empty_b, layer_a, seeded_a_after);
    const bool resolve_ok = snapshot_ok && f.cache->resolve_and_pin(empty_b.device_group,
            empty_b.group, layer_a, empty_b.generation, &expert, 1, &after);
    const bool same_owner = resolve_ok && after.n_draws == 1 && after.n_pins == 1 &&
            after.draw_slots[0] == slot && after.pins[0].key == owner_before.key &&
            after.pins[0].slot == owner_before.slot && after.pins[0].owner == owner_before.owner &&
            after.pins[0].device == owner_before.device && after.pins[0].group == owner_before.group;
    const bool released_pins = after.n_pins == 0 || f.cache->release_pins(after.pins, after.n_pins);
    bool same_contents = snapshot_ok;
    ggml_backend_synchronize(f.cuda);
    const std::array<const ggml_tensor *, 3> banks_after = {
        seeded_a_after.bank_up, seeded_a_after.bank_gate, seeded_a_after.bank_down
    };
    for (size_t part = 0; same_contents && part < banks_after.size(); ++part) {
        std::vector<uint8_t> contents(banks_after[part]->nb[2]);
        ggml_backend_tensor_get(banks_after[part], contents.data(),
                (size_t) slot * contents.size(), contents.size());
        same_contents = contents == contents_before[part];
    }
    const bool released = lease_b.release();
    const bool restored_map = f.cache->get_slot_map(layer_a, 1, N_SELECTED) == ordinary_map;
    std::printf("seed_fill_isolation layer_a=%d layer_b=%d group=%d expert=%d slot=%d seed_a_bytes=%zu empty_b_bytes=%zu owner_unchanged=%d bank_contents_unchanged=%d blocking_map_restored=%d\n",
            layer_a, layer_b, seeded_a.group, expert, slot, seeded_a.seed_bytes, empty_b.seed_bytes,
            same_owner && released_pins ? 1 : 0, same_contents ? 1 : 0, restored_map ? 1 : 0);
    return released && same_owner && released_pins && same_contents && restored_map;
}

bool run(const std::string & model_path, int32_t device, const char * timing_mode,
        const std::vector<int32_t> & cpu_thread_counts, int32_t samples, int32_t miss_count) {
    real_fixture f;
    llama_backend_init();
    f.backend_initialized = true;
    f.cuda = ggml_backend_cuda_init(device, nullptr, nullptr);
    if (!f.cuda) { std::fprintf(stderr, "failed to initialize CUDA device %d\n", device); return false; }

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;
    model_params.max_gpu = 1;
    model_params.main_gpu = device;
    model_params.split_mode = LLAMA_SPLIT_MODE_NONE;
    model_params.ncmoe = 999; // -cmoe semantics: keep expert weights canonical on the host.
    model_params.use_mmap = true;
    model_params.fit = false;
    model_params.merge_up_gate_exps = false;
    f.model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!f.model) { std::fprintf(stderr, "model load failed: %s\n", model_path.c_str()); return false; }
    if (f.model->hparams.n_expert_used != N_SELECTED) {
        std::fprintf(stderr, "expected top-%d model, got top-%u\n", N_SELECTED, f.model->hparams.n_expert_used);
        return false;
    }

    int32_t layer_id = -1;
    for (size_t i = 0; i < f.model->layers.size(); ++i) {
        if (valid_candidate(*f.model, f.model->layers[i])) { layer_id = (int32_t)i; break; }
    }
    if (layer_id < 0) { std::fprintf(stderr, "no host-resident separate IQ3_S/IQ3_S/IQ4_NL layer without bias/scale/Hadamard\n"); return false; }
    const llama_layer & layer = f.model->layers[layer_id];
    const int32_t n_embd = (int32_t) layer.ffn_up_exps->ne[0];
    const int32_t n_ff = (int32_t) layer.ffn_up_exps->ne[1];
    const int32_t n_experts = (int32_t) layer.ffn_up_exps->ne[2];
    std::printf("model=%s layer=%d experts=%d top_k=%d n_embd=%d n_ff=%d\n",
            model_path.c_str(), layer_id, n_experts, N_SELECTED, n_embd, n_ff);
    report_tensor("up", layer.ffn_up_exps);
    report_tensor("gate", layer.ffn_gate_exps);
    report_tensor("down", layer.ffn_down_exps);

    const std::vector<ggml_backend *> backends{ f.cuda };
    const std::vector<ggml_backend_buffer_type *> bufts{ ggml_backend_cuda_buffer_type(device) };
    f.cache = std::make_unique<llama_moe_cache>(*f.model, backends, bufts, CACHE_BYTES);
    if (std::strcmp(timing_mode, "seed-isolation") == 0)
        return run_seed_fill_isolation(f, layer_id, n_experts);
    if (std::strcmp(timing_mode, "blocking") == 0) {
        f.cpu = ggml_backend_cpu_init();
        if (!f.cpu) return false;
        ggml_backend_cpu_set_n_threads(f.cpu, 1);
        return run_blocking_cache(f, layer_id, n_embd, n_ff, miss_count, samples,
                std::getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);
    }
    if (f.cache->backend(layer_id) != f.cuda) {
        std::fprintf(stderr, "selected layer is not owned by the requested one-device cache backend\n");
        return false;
    }
    ggml_tensor * old_slot_map = f.cache->get_slot_map(layer_id, 1, N_SELECTED);
    if (!old_slot_map) { std::fprintf(stderr, "selected cache group cannot hold the routed set\n"); return false; }

    std::vector<std::pair<int32_t, ggml_tensor *>> old_group_maps;
    for (int32_t il = 0; il < (int32_t) f.model->layers.size(); ++il) {
        if (il == layer_id) continue;
        if (f.cache->backend(il) == f.cuda) {
            ggml_tensor * map = f.cache->get_slot_map(il, 1, N_SELECTED);
            if (map && layer_id >= 0) old_group_maps.emplace_back(il, map);
        }
    }
    const bool replacement_mode = std::strcmp(timing_mode, "replacement") == 0;
    std::vector<int32_t> seed_experts{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    size_t setup_seed_bytes = 0;
    size_t setup_seed_count = seed_experts.size();
    int32_t group_slot_count = 0;
    llama_moe_cache::frozen_lease lease;
    if (replacement_mode) {
        if (!f.cache->freeze_group_for_layer(layer_id, f.cuda, seed_experts, lease)) return false;
        const llama_moe_cache_frozen_layer probe = lease.snapshot();
        group_slot_count = probe.n_slots;
        setup_seed_bytes = probe.seed_bytes;
        std::vector<int32_t> group_layers = probe.group_layers;
        if (group_slot_count < (int32_t) seed_experts.size() || group_slot_count >= n_experts * (int32_t)group_layers.size()) {
            std::fprintf(stderr, "replacement capacity probe invalid slots=%d seed=%zu experts=%d members=%zu\n",
                    group_slot_count, seed_experts.size(), n_experts, group_layers.size());
            return false;
        }
        if (!lease.release()) { std::fprintf(stderr, "replacement capacity probe lease release failed\n"); return false; }
        group_layers.erase(std::remove(group_layers.begin(), group_layers.end(), layer_id), group_layers.end());
        group_layers.insert(group_layers.begin(), layer_id);
        seed_experts.clear();
        int32_t remaining = group_slot_count;
        for (const int32_t member : group_layers) {
            const int32_t count = std::min<int32_t>(member == layer_id ? n_experts - 2 : n_experts, remaining);
            if (count <= 0) break;
            std::vector<int32_t> member_seed(count);
            for (int32_t expert = 0; expert < count; ++expert) member_seed[expert] = expert;
            if (!f.cache->freeze_group_for_layer(member, f.cuda, member_seed, lease)) {
                std::fprintf(stderr, "replacement group seed failed layer=%d count=%d remaining=%d\n",
                        member, count, remaining);
                return false;
            }
            const auto member_snapshot = lease.snapshot();
            if (member_snapshot.group != probe.group || member_snapshot.n_slots != group_slot_count ||
                    member_snapshot.layer != member || member_snapshot.seed_bytes == 0) return false;
            setup_seed_bytes += member_snapshot.seed_bytes;
            if (member == layer_id) seed_experts = member_seed;
            if (!lease.release()) return false;
            remaining -= count;
        }
        if (remaining != 0 || seed_experts.size() != (size_t)(n_experts - 2)) {
            std::fprintf(stderr, "replacement group seed incomplete remaining=%d target_seed=%zu capacity=%d\n",
                    remaining, seed_experts.size(), group_slot_count);
            return false;
        }
        setup_seed_count = group_slot_count;
        std::fprintf(stderr, "replacement group pool seeded slots=%d members=%zu target_experts=%zu\n",
                group_slot_count, group_layers.size(), seed_experts.size());
    }
    const std::vector<int32_t> lease_seed_experts = replacement_mode ? std::vector<int32_t>{ 0 } : seed_experts;
    if (!f.cache->freeze_group_for_layer(layer_id, f.cuda, lease_seed_experts, lease)) {
        std::fprintf(stderr, "cache could not validate/seed/freeze the selected layer\n");
        return false;
    }
    f.lease = std::move(lease);
    const llama_moe_cache_frozen_layer & snapshot = f.lease.snapshot();
    if (snapshot.n_slots < (int32_t)seed_experts.size()) {
        std::fprintf(stderr, "frozen cache snapshot does not cover the deterministic seed set\n");
        return false;
    }
    setup_seed_bytes += snapshot.seed_bytes;
    const size_t expected_seed_bytes = setup_seed_count *
            (layer.ffn_up_exps->nb[2] + layer.ffn_gate_exps->nb[2] + layer.ffn_down_exps->nb[2]);
    if (setup_seed_bytes != expected_seed_bytes) {
        std::fprintf(stderr, "seed byte accounting mismatch got=%zu expected=%zu\n", setup_seed_bytes, expected_seed_bytes);
        return false;
    }
    std::fprintf(stderr, "replacement final seed slots=%d experts=%zu setup_bytes=%zu\n",
            snapshot.n_slots, setup_seed_count, setup_seed_bytes);
    const int32_t duplicate_ids[N_SELECTED] = { 0, 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    ggml_cuda_moe_ffn_resolution pin_result{};
    const bool rejected_device = f.cache->resolve_and_pin(device + 1, snapshot.group, layer_id, snapshot.generation,
                duplicate_ids, N_SELECTED, &pin_result) ||
            f.cache->resolve_and_pin(device, snapshot.group + 1, layer_id, snapshot.generation,
                duplicate_ids, N_SELECTED, &pin_result) ||
            !f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                duplicate_ids, N_SELECTED, &pin_result);
    if (rejected_device || pin_result.n_draws != N_SELECTED || pin_result.n_pins != N_SELECTED - 1) {
        std::fprintf(stderr, "cache pin setup failed device_reject=%d draws=%d pins=%d expected=%d\n",
                rejected_device ? 1 : 0, pin_result.n_draws, pin_result.n_pins, N_SELECTED - 1);
        return false;
    }
    const auto original_pin = pin_result.pins[0];
    auto invalid_pin = original_pin;
    ++invalid_pin.device;
    if (f.cache->release_pins(&invalid_pin, 1)) return false;
    invalid_pin = original_pin; ++invalid_pin.group;
    if (f.cache->release_pins(&invalid_pin, 1)) return false;
    invalid_pin = original_pin; ++invalid_pin.key;
    if (f.cache->release_pins(&invalid_pin, 1)) return false;
    invalid_pin = original_pin; ++invalid_pin.generation;
    if (f.cache->release_pins(&invalid_pin, 1)) return false;
    invalid_pin = original_pin; ++invalid_pin.owner_generation;
    if (f.cache->release_pins(&invalid_pin, 1)) return false;
    if (!f.cache->release_pins(pin_result.pins, pin_result.n_pins) ||
            f.cache->release_pins(pin_result.pins, pin_result.n_pins)) return false;
    ggml_cuda_moe_ffn_resolution repinned{};
    if (!f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                duplicate_ids, N_SELECTED, &repinned) ||
            repinned.pins[0].generation == original_pin.generation ||
            f.cache->release_pins(pin_result.pins, pin_result.n_pins) ||
            !f.cache->release_pins(repinned.pins, repinned.n_pins)) return false;
    std::printf("cache_pin_contract draws=%d unique_slots=%d wrong_device_group_key_generation_rejected=1 repeated_stale_rejected=1 repin_generation_advanced=1\n",
            N_SELECTED, N_SELECTED - 1);
    std::printf("freeze layer=%d group=%d device_group=%d generation=%llu slots=%d group_members=%zu seed_experts=%zu seed_bytes=%zu bank_identity=%p,%p,%p\n",
            snapshot.layer, snapshot.group, snapshot.device_group, (unsigned long long)snapshot.generation,
            snapshot.n_slots, snapshot.group_layers.size(), seed_experts.size(), snapshot.seed_bytes,
            (const void *)snapshot.bank_up, (const void *)snapshot.bank_gate, (const void *)snapshot.bank_down);

    size_t callback_rejects = 0;
    size_t unaffected_groups = 0;
    for (const auto & entry : old_group_maps) {
        const bool frozen_member = std::find(snapshot.group_layers.begin(), snapshot.group_layers.end(), entry.first) !=
                snapshot.group_layers.end();
        ggml_tensor * current_map = f.cache->get_slot_map(entry.first, 1, N_SELECTED);
        if (frozen_member) {
            if (current_map != nullptr) { std::fprintf(stderr, "frozen member %d retained a slot map\n", entry.first); return false; }
            if (f.cache->copy(f.cuda, entry.second, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_QUERY)) {
                std::fprintf(stderr, "frozen member %d accepted a callback query\n", entry.first); return false;
            }
            if (f.cache->copy(f.cuda, entry.second, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_EXECUTE)) {
                std::fprintf(stderr, "frozen member %d accepted callback execute\n", entry.first); return false;
            }
            callback_rejects += 2;
        } else {
            if (current_map != entry.second ||
            !f.cache->copy(f.cuda, entry.second, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_QUERY)) {
                std::fprintf(stderr, "unfrozen layer %d lost its slot map/callback query\n", entry.first); return false;
            }
            ++unaffected_groups;
        }
    }
    if (f.cache->get_slot_map(layer_id, 1, N_SELECTED) != nullptr ||
            f.cache->copy(f.cuda, old_slot_map, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_QUERY) ||
            f.cache->copy(f.cuda, old_slot_map, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_EXECUTE)) {
        std::fprintf(stderr, "selected frozen layer accepted a slot-map/callback query or execute\n"); return false;
    }
    callback_rejects += 2;
    if (!f.cache->frozen_snapshot_valid(snapshot)) return false;
    std::printf("frozen_callback group_members=%zu rejected_stale_queries_and_executes=%zu unaffected_eligible_layers=%zu snapshot_valid=1\n",
            snapshot.group_layers.size(), callback_rejects, unaffected_groups);

    f.cpu = ggml_backend_cpu_init();
    if (!f.cpu) return false;
    const int32_t resource_threads = cpu_thread_counts.empty() ? 24 : cpu_thread_counts.front();
    ggml_backend_cpu_set_n_threads(f.cpu, resource_threads);
    std::printf("registered_cpu_resource,n_threads=%d,max_parallel_calls=1\n", resource_threads);
    std::vector<float> activation(n_embd);
    for (int32_t i = 0; i < n_embd; ++i) activation[i] = 0.25f * std::sin(float(i + 17) * 0.007f);
    const std::vector<int32_t> seed_ids{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<int32_t> miss_ids{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 11 };
    std::vector<int32_t> five_miss_ids{ 0, 1, 2, 3, 4, 11, 12, 13, 14, 15 };
    std::vector<int32_t> all_miss_ids{ 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 };
    std::vector<int32_t> final_ids{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    if (replacement_mode) {
        const int32_t miss_expert = n_experts - 2;
        miss_ids[9] = miss_expert;
        for (int32_t i = 0; i < 5; ++i) five_miss_ids[5 + i] = miss_expert + i;
        for (int32_t i = 0; i < 10; ++i) all_miss_ids[i] = miss_expert + i;
        final_ids = miss_ids;
    }
    const std::vector<float> routes0 = route_weights(0);
    const std::vector<float> routes1 = route_weights(1);
    const std::vector<float> routes2 = route_weights(2);
    std::vector<int32_t> seed_slots(seed_experts.size(), -1);
    const auto capture_slots = [&](const std::vector<int32_t> & ids) {
        ggml_cuda_moe_ffn_resolution resolved{};
        if (!f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                    ids.data(), ids.size(), &resolved) || resolved.n_draws != (int32_t) ids.size()) return false;
        bool valid = true;
        for (int32_t i = 0; i < resolved.n_draws; ++i) {
            const auto it = std::find(seed_experts.begin(), seed_experts.end(), ids[i]);
            if (resolved.draw_slots[i] < 0 || it == seed_experts.end()) {
                valid = false;
            } else {
                seed_slots[it - seed_experts.begin()] = resolved.draw_slots[i];
            }
        }
        return f.cache->release_pins(resolved.pins, resolved.n_pins) && valid;
    };
    if (!capture_slots(seed_ids)) {
        std::fprintf(stderr, "failed to capture seed CUDA slots after freeze\n");
        return false;
    }
    int32_t replaced_expert = -1;
    int32_t replaced_slot = -1;
    std::vector<int32_t> evicted_seed_experts;
    const auto slot_for_seed = [&](int32_t expert) {
        if (expert == replaced_expert) return replaced_slot;
        if (std::find(evicted_seed_experts.begin(), evicted_seed_experts.end(), expert) != evicted_seed_experts.end()) return -1;
        const auto it = std::find(seed_experts.begin(), seed_experts.end(), expert);
        return it == seed_experts.end() ? -1 : seed_slots[it - seed_experts.begin()];
    };
    std::vector<int32_t> hit_slots(N_SELECTED);
    for (int32_t i = 0; i < N_SELECTED; ++i) {
        hit_slots[i] = slot_for_seed(seed_ids[i]);
        if (hit_slots[i] < 0) return false;
    }
    if (!make_cpu_graph(f.canonical_cpu, f.cpu, snapshot.host_up, snapshot.host_gate,
                snapshot.host_down, n_embd, n_ff, N_SELECTED)) return false;
    set_inputs(f.canonical_cpu, f.cpu, activation, seed_ids, routes0);
    if (!make_cpu_graph(f.cpu_single, f.cpu, snapshot.host_up, snapshot.host_gate,
                snapshot.host_down, n_embd, n_ff, 1)) return false;
    set_inputs(f.cpu_single, f.cpu, activation, { seed_ids.front() }, { routes0.front() });

    ggml_cuda_moe_ffn_cpu_resource cpu_resource{ f.cpu, n_embd, n_ff, N_SELECTED, resource_threads };
    f.cpu_resource_id = ggml_backend_cuda_moe_ffn_cpu_resource_register(f.cuda, &cpu_resource);
    if (f.cpu_resource_id <= 0) return false;
    f.cpu_owned_by_resource = true;

    ggml_cuda_moe_ffn_binding binding{};
    binding.generation = snapshot.generation;
    binding.device = device;
    binding.n_experts = n_experts;
    binding.n_slots = snapshot.n_slots;
    binding.n_experts_used = N_SELECTED;
    binding.n_embd = n_embd;
    binding.n_ff = n_ff;
    binding.up_type = snapshot.host_up->type;
    binding.gate_type = snapshot.host_gate->type;
    binding.down_type = snapshot.host_down->type;
    binding.unary_op = GGML_UNARY_OP_SILU;
    binding.limit = 0.0f;
    binding.host_up = snapshot.host_up;
    binding.host_gate = snapshot.host_gate;
    binding.host_down = snapshot.host_down;
    binding.cuda_up = snapshot.cached_up;
    binding.cuda_gate = snapshot.cached_gate;
    binding.cuda_down = snapshot.cached_down;
    binding.cache_callbacks = llama_moe_cache_get_cuda_callbacks();
    binding.cache_context = f.cache.get();
    binding.cache_group = snapshot.group;
    binding.cache_layer = snapshot.layer;
    binding.cpu_resource_id = f.cpu_resource_id;
    binding.cpu_graph = f.cpu_single.graph;
    binding.cpu_activations = f.cpu_single.activation;
    binding.cpu_ids = f.cpu_single.ids;
    binding.cpu_route_weights = f.cpu_single.routes;
    binding.cpu_output = f.cpu_single.output;
    f.binding_id = ggml_backend_cuda_moe_ffn_register(f.cuda, &binding);
    if (f.binding_id <= 0) return false;

    if (!make_cpu_graph(f.cached_cuda, f.cuda, snapshot.cached_up, snapshot.cached_gate,
                snapshot.cached_down, n_embd, n_ff, N_SELECTED)) return false;
    if (!make_semantic_graph(f.semantic_cuda, f.cuda, device, f.binding_id,
                binding.generation, n_embd, n_ff)) return false;
    if (!ggml_backend_cuda_moe_ffn_test_hold_fill_worker(f.cuda, f.cpu_resource_id, true)) return false;
    std::printf("fill_worker_test_gate=held_before_transport\n");

    const auto run_step = [&](const char * name, const std::vector<int32_t> & experts,
            const std::vector<float> & routes) {
        std::vector<float> canonical;
        if (!compute(f.canonical_cpu, f.cpu, activation, experts, routes, canonical)) return false;
        std::vector<float> semantic;
        if (!compute(f.semantic_cuda, f.cuda, activation, experts, routes, semantic)) return false;
        int32_t hits = 0;
        for (const int32_t expert : experts) hits += slot_for_seed(expert) >= 0;
        if (hits == N_SELECTED) {
            std::vector<int32_t> slots(N_SELECTED);
            for (int32_t i = 0; i < N_SELECTED; ++i) {
                const int32_t slot = slot_for_seed(experts[i]);
                if (slot < 0) return false;
                slots[i] = slot;
            }
            std::vector<float> cached;
            if (!compute(f.cached_cuda, f.cuda, activation, slots, routes, cached)) return false;
            const error_metrics semantic_vs_cache = compare(semantic, cached);
            const error_metrics cache_vs_cpu = compare(cached, canonical);
            const error_metrics semantic_vs_cpu = compare(semantic, canonical);
            std::printf("operation=%s branch=all-hit semantic_vs_cached_max=%.9g_rms=%.9g_worst=%zu cached_vs_cpu_max=%.9g_rms=%.9g semantic_vs_cpu_max=%.9g_rms=%.9g\n",
                    name, semantic_vs_cache.max_abs, semantic_vs_cache.rms, semantic_vs_cache.worst,
                    cache_vs_cpu.max_abs, cache_vs_cpu.rms, semantic_vs_cpu.max_abs, semantic_vs_cpu.rms);
            return all_finite(cached) && semantic_vs_cache.max_abs <= 1.0e-4 &&
                    semantic_vs_cpu.max_abs <= 1.0e-4 && f.cache->frozen_snapshot_valid(snapshot);
        }
        const error_metrics semantic_vs_cpu = compare(semantic, canonical);
        std::printf("operation=%s branch=%s gpu_hit_draws=%d cpu_miss_draws=%d semantic_vs_canonical_cpu_max=%.9g_rms=%.9g_worst=%zu\n",
                name, hits == 0 ? "all-miss" : "partitioned", hits, N_SELECTED - hits,
                semantic_vs_cpu.max_abs, semantic_vs_cpu.rms, semantic_vs_cpu.worst);
        return all_finite(semantic) && semantic_vs_cpu.max_abs <= 1.0e-4 &&
                f.cache->frozen_snapshot_valid(snapshot);
    };

    if (replacement_mode) {
        if ((int32_t) setup_seed_count != snapshot.n_slots ||
                !run_step("full-pool-all-hit", seed_ids, routes0) ||
                !run_step("replacement-miss-copying", miss_ids, routes1)) return false;

        ggml_cuda_moe_ffn_cpu_resource_stats queued{};
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &queued)) return false;
        if (queued.fills_accepted != 1 || queued.fills_completed != 0 || queued.fill_outstanding != 1) return false;
        const int32_t copying_ids[2] = { 10, miss_ids[9] };
        ggml_cuda_moe_ffn_resolution copying{};
        if (!f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                    copying_ids, 2, &copying) || copying.draw_slots[0] >= 0 ||
                copying.draw_slots[1] >= 0 || copying.n_pins != 0) return false;
        std::printf("replacement_queued capacity=%d accepted=%llu outstanding=%llu old_owner=10_absent requested_expert=%d_absent cpu_miss_draws=1\n",
                snapshot.n_slots, (unsigned long long)queued.fills_accepted,
                (unsigned long long)queued.fill_outstanding, miss_ids[9]);

        if (!ggml_backend_cuda_moe_ffn_test_hold_fill_worker(f.cuda, f.cpu_resource_id, false) ||
                !ggml_backend_cuda_moe_ffn_test_wait_fill_idle(f.cuda, f.cpu_resource_id, 30000)) return false;
        ggml_cuda_moe_ffn_cpu_resource_stats published{};
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &published) ||
                published.fills_accepted != 1 || published.fills_completed != 1 ||
                published.fills_failed != 0 || published.fill_outstanding != 0 ||
                published.fill_parts_completed != GGML_CUDA_MOE_FFN_MAX_FILL_PARTS) return false;
        replaced_expert = miss_ids[9];
        ggml_cuda_moe_ffn_resolution published_expert{};
        if (!f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                    &replaced_expert, 1, &published_expert) || published_expert.draw_slots[0] < 0 ||
                !f.cache->release_pins(published_expert.pins, published_expert.n_pins)) return false;
        replaced_slot = published_expert.draw_slots[0];
        evicted_seed_experts.push_back(10);
        if (replaced_slot < 0 || !run_step("later-ready-hit", miss_ids, routes2) ||
                !run_step("final-all-hit-control", miss_ids, routes0)) return false;

        const int32_t failed_expert = miss_ids[9] + 1;
        std::vector<int32_t> failed_ids{ 0, 1, 2, 3, 4, 5, 6, 7, 8, failed_expert };
        if (!ggml_backend_cuda_moe_ffn_test_fail_next_fill_after_part(f.cuda, f.cpu_resource_id) ||
                !run_step("partial-copy-failure-miss", failed_ids, routes1) ||
                !ggml_backend_cuda_moe_ffn_test_wait_fill_idle(f.cuda, f.cpu_resource_id, 30000)) return false;
        evicted_seed_experts.push_back(11);
        ggml_cuda_moe_ffn_cpu_resource_stats failed_fill{};
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &failed_fill) ||
                failed_fill.fills_accepted != 2 || failed_fill.fills_completed != 1 ||
                failed_fill.fills_failed != 1 || failed_fill.fill_outstanding != 0 ||
                failed_fill.fill_parts_completed != GGML_CUDA_MOE_FFN_MAX_FILL_PARTS) return false;
        const int32_t failed_lookup_ids[2] = { 11, failed_expert };
        ggml_cuda_moe_ffn_resolution failed_lookup{};
        if (!f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                    failed_lookup_ids, 2, &failed_lookup) ||
                failed_lookup.draw_slots[0] >= 0 || failed_lookup.draw_slots[1] >= 0 ||
                failed_lookup.n_pins != 0) return false;
        std::printf("replacement_partial_failure accepted=%llu completed=%llu failed=%llu outstanding=%llu copied_parts_total=%llu successful_fill_bytes=%llu failed_first_part_attempt_bytes=%zu overwritten_owner_not_resurrected=1\n",
                (unsigned long long)failed_fill.fills_accepted, (unsigned long long)failed_fill.fills_completed,
                (unsigned long long)failed_fill.fills_failed, (unsigned long long)failed_fill.fill_outstanding,
                (unsigned long long)failed_fill.fill_parts_completed,
                (unsigned long long)failed_fill.fill_bytes_completed, layer.ffn_up_exps->nb[2]);

        ggml_cuda_moe_ffn_stats operation_stats{};
        if (!ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &operation_stats) ||
                operation_stats.all_hit_calls != 3 || operation_stats.any_miss_calls != 2 ||
                operation_stats.partitioned_calls != 2 || operation_stats.gpu_hit_draws != 48 ||
                operation_stats.cpu_miss_draws != 2 || operation_stats.output_commits != 5 ||
                operation_stats.full_expert_weight_copy_bytes != 0 ||
                !f.cache->frozen_snapshot_valid(snapshot)) return false;
        std::printf("replacement_published expert=%d slot=%d fills_accepted=%llu completed=%llu failed=%llu parts=%llu intentional_fill_bytes=%llu failed_partial_part_bytes=%zu legacy_full_expert_weight_copy_bytes=%llu owner_snapshot_unchanged=1 overwritten_owner_not_resurrected=1\n",
                replaced_expert, replaced_slot, (unsigned long long)published.fills_accepted,
                (unsigned long long)published.fills_completed, (unsigned long long)published.fills_failed,
                (unsigned long long)published.fill_parts_completed,
                (unsigned long long)published.fill_bytes_completed,
                layer.ffn_up_exps->nb[2],
                (unsigned long long)operation_stats.full_expert_weight_copy_bytes);

        if (!ggml_backend_cuda_moe_ffn_cpu_resource_stop(f.cuda, f.cpu_resource_id)) {
            std::fprintf(stderr, "replacement teardown failed to stop transport\n");
            return false;
        }
        ggml_cuda_moe_ffn_cpu_resource_stats stopped{};
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &stopped)) {
            std::fprintf(stderr, "replacement teardown could not read stopped resource\n");
            return false;
        }
        if (stopped.pending_pin_active != 0 || stopped.fill_outstanding != 0) return false;
        if (!f.unregister_execution()) { std::fprintf(stderr, "replacement unregister_execution failed after explicit stop\n"); return false; }
        const int32_t frozen_layer = snapshot.layer;
        if (!f.lease.release()) { std::fprintf(stderr, "replacement frozen lease release failed after unregister\n"); return false; }
        if (f.cache->frozen_snapshot_valid(snapshot) || !f.cache->get_slot_map(frozen_layer, 1, N_SELECTED)) {
            std::fprintf(stderr, "replacement cache query was not restored after lease release\n");
            return false;
        }
        f.lease_restored = true;
        f.cleanup();
        std::printf("replacement_teardown pending_pins=%llu transport_outstanding=%llu lease_released=1 ordinary_query_restored=1\n",
                (unsigned long long)stopped.pending_pin_active,
                (unsigned long long)stopped.fill_outstanding);
        return f.lease_restored;
    }

    if (!run_step("all-hit-before", seed_ids, routes0) ||
            !run_step("one-miss", miss_ids, routes1) ||
            !run_step("five-miss", five_miss_ids, routes2) ||
            !run_step("all-miss", all_miss_ids, routes0) ||
            !run_step("all-hit-after", final_ids, routes1)) return false;

    if (std::strcmp(timing_mode, "correctness") != 0 &&
            !run_timing(f, activation, seed_ids, hit_slots, miss_ids, five_miss_ids,
                    all_miss_ids, routes0, timing_mode, cpu_thread_counts, samples)) return false;

    ggml_cuda_moe_ffn_stats stats{};
    ggml_cuda_moe_ffn_cpu_resource_stats cpu_stats{};
    if (!ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &stats) ||
            !ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &cpu_stats)) return false;
    const uint64_t thread_curve_blocks = std::strcmp(timing_mode, "thread-curve") == 0 ?
            cpu_thread_counts.size() : std::strcmp(timing_mode, "thread-abba") == 0 ? 4 : 0;
    const uint64_t timing_all_hits = std::strcmp(timing_mode, "primary") == 0 ||
            std::strcmp(timing_mode, "scaling") == 0 ? 64 :
            std::strcmp(timing_mode, "trace") == 0 ? 12 :
            thread_curve_blocks * (8 + samples);
    const uint64_t timing_misses = std::strcmp(timing_mode, "miss") == 0 ? 6 * (8 + 20) :
            std::strcmp(timing_mode, "scaling") == 0 ? cpu_thread_counts.size() * (8 + samples) :
            std::strcmp(timing_mode, "scaling-abba") == 0 ? 4 * (8 + samples) :
            thread_curve_blocks * 3 * (8 + samples);
    const uint64_t timing_partitioned = std::strcmp(timing_mode, "miss") == 0 ? 4 * (8 + 20) :
            std::strcmp(timing_mode, "scaling") == 0 ? timing_misses :
            std::strcmp(timing_mode, "scaling-abba") == 0 ? timing_misses :
            thread_curve_blocks * 2 * (8 + samples);
    const uint64_t timing_complete_cpu = std::strcmp(timing_mode, "miss") == 0 ? 2 * (8 + 20) :
            thread_curve_blocks * (8 + samples);
    const uint64_t timing_cpu_draws = std::strcmp(timing_mode, "miss") == 0 ? 2 * (8 + 20) * 1 +
            2 * (8 + 20) * 5 + 2 * (8 + 20) * 10 :
            thread_curve_blocks ? thread_curve_blocks * (8 + samples) * (1 + 5 + 10) : timing_misses;
    const uint64_t timing_gpu_draws = std::strcmp(timing_mode, "miss") == 0 ? 2 * (8 + 20) * 9 +
            2 * (8 + 20) * 5 : std::strcmp(timing_mode, "scaling") == 0 ||
            std::strcmp(timing_mode, "scaling-abba") == 0 ? timing_misses * 9 :
            thread_curve_blocks ? thread_curve_blocks * (8 + samples) * (9 + 5) + timing_all_hits * N_SELECTED :
            timing_all_hits * N_SELECTED;
    if (stats.all_hit_calls != 2 + timing_all_hits || stats.any_miss_calls != 3 + timing_misses ||
            stats.output_commits != 5 + timing_all_hits + timing_misses ||
            stats.gpu_hit_draws != 34 + timing_gpu_draws ||
            stats.cpu_miss_draws != 16 + timing_cpu_draws ||
            stats.cpu_expert_calls != 16 + timing_cpu_draws ||
            stats.partitioned_calls != 2 + timing_partitioned ||
            stats.complete_cpu_fallback_calls != 1 + timing_complete_cpu ||
            stats.full_expert_weight_copy_bytes != 0 || cpu_stats.active_bindings != 1 ||
            cpu_stats.max_parallel_calls != 1) return false;

    if (std::strcmp(timing_mode, "correctness") == 0) {
        auto get_fill_stats = [&]() {
            ggml_cuda_moe_ffn_cpu_resource_stats result{};
            return ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &result) ? result :
                    ggml_cuda_moe_ffn_cpu_resource_stats{};
        };
        const auto fills = get_fill_stats();
        if (fills.fills_accepted != 0 || fills.fills_completed != 0 || fills.fills_failed != 0 ||
                fills.fill_outstanding != 0 || fills.fills_rejected == 0) {
            std::fprintf(stderr, "frozen real cache group admitted a fill or failed to reject misses\n");
            return false;
        }
        std::printf("frozen_group_fill_isolation accepted=%llu completed=%llu failed=%llu rejected=%llu outstanding=%llu\n",
                (unsigned long long)fills.fills_accepted, (unsigned long long)fills.fills_completed,
                (unsigned long long)fills.fills_failed, (unsigned long long)fills.fills_rejected,
                (unsigned long long)fills.fill_outstanding);
    }

    std::printf("execution branches=all-hit,one-miss,five-miss,all-miss,all-hit commits=%llu gpu_hit_draws=%llu cpu_miss_draws=%llu partitioned_calls=%llu complete_cpu_fallbacks=%llu cpu_expert_calls=%llu gpu_pending_at_cpu_start=%llu gpu_completed_during_cpu=%llu seed_bytes=%zu post_freeze_expert_weight_write_bytes=%llu callback_rejects=%zu cache_generation=%llu cpu_resource_calls=%llu active_bindings=%llu\n",
            (unsigned long long)stats.output_commits,
            (unsigned long long)stats.gpu_hit_draws, (unsigned long long)stats.cpu_miss_draws,
            (unsigned long long)stats.partitioned_calls, (unsigned long long)stats.complete_cpu_fallback_calls,
            (unsigned long long)stats.cpu_expert_calls, (unsigned long long)stats.gpu_pending_at_cpu_start,
            (unsigned long long)stats.gpu_completed_during_cpu,
            snapshot.seed_bytes,
            (unsigned long long)stats.full_expert_weight_copy_bytes, callback_rejects,
            (unsigned long long)snapshot.generation, (unsigned long long)cpu_stats.cpu_calls,
            (unsigned long long)cpu_stats.active_bindings);
    const llama_moe_cache_frozen_layer first_snapshot = snapshot;
    ggml_cuda_moe_ffn_cpu_resource_stats before_lease_stats{};
    if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &before_lease_stats)) return false;
    const uint64_t pin_releases_before_rejected_lease = before_lease_stats.pin_releases;
    const bool released_while_pinned = f.lease.release();
    ggml_cuda_moe_ffn_cpu_resource_stats rejected_release_stats{};
    if (released_while_pinned || !f.lease || f.lease.snapshot().generation != first_snapshot.generation ||
            f.lease.snapshot().bank_up != first_snapshot.bank_up ||
            !f.cache->frozen_snapshot_valid(first_snapshot) ||
            f.cache->get_slot_map(first_snapshot.layer, 1, N_SELECTED) != nullptr ||
            f.cache->copy(f.cuda, old_slot_map, nullptr, nullptr, GGML_BACKEND_SCHED_COPY_PHASE_QUERY) ||
            !ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &rejected_release_stats) ||
            rejected_release_stats.pending_pin_active != 1 ||
            rejected_release_stats.pin_acquisitions != before_lease_stats.pin_acquisitions ||
            rejected_release_stats.pin_releases != pin_releases_before_rejected_lease) {
        std::fprintf(stderr, "pinned frozen-group release changed lease, owner or blocking isolation state\n");
        return false;
    }
    std::printf("lease_release_with_pending_bank_read=rejected lease_retained=1 snapshot_valid=1 pending_pin_retained=1 blocking_group_isolated=1\n");
    if (std::strcmp(timing_mode, "correctness") == 0) {
        const int32_t active_binding = f.binding_id;
        ggml_cuda_moe_ffn_stats still_registered{};
        if (ggml_backend_cuda_moe_ffn_unregister(f.cuda, active_binding) ||
                !ggml_backend_cuda_moe_ffn_get_stats(f.cuda, active_binding, &still_registered) ||
                !run_step("binding-usable-after-unregister-rejected", seed_ids, routes0)) {
            std::fprintf(stderr, "binding unregister before transport stop did not reject while preserving use\n");
            return false;
        }
        std::printf("binding_unregister_before_stop=rejected_binding_still_usable=1\n");
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_stop(f.cuda, f.cpu_resource_id)) return false;
        ggml_cuda_moe_ffn_cpu_resource_stats drained{};
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &drained) ||
                drained.fills_accepted != 0 || drained.fills_completed != 0 || drained.fills_failed != 0 ||
                drained.fill_outstanding != 0 || drained.fill_parts_completed != 0 ||
                drained.fill_bytes_completed != 0 || drained.fills_rejected == 0 ||
                drained.pending_pin_active != 0) return false;
        std::printf("transport_stop_frozen_group accepted=%llu completed=%llu failed=%llu rejected=%llu outstanding=%llu parts=%llu bytes=%llu pending_pins=%llu\n",
                (unsigned long long)drained.fills_accepted, (unsigned long long)drained.fills_completed,
                (unsigned long long)drained.fills_failed, (unsigned long long)drained.fills_rejected,
                (unsigned long long)drained.fill_outstanding,
                (unsigned long long)drained.fill_parts_completed, (unsigned long long)drained.fill_bytes_completed,
                (unsigned long long)drained.pending_pin_active);
    }
    if (!f.unregister_execution()) {
        std::fprintf(stderr, "final pending pins were not released before lease tests\n");
        return false;
    }
    std::printf("teardown_binding_and_resource_unregister_released_final_pending=1\n");
    if (std::strcmp(timing_mode, "correctness") == 0) {
        const int32_t absent_id = 11;
        ggml_cuda_moe_ffn_resolution absent{};
        if (!f.cache->resolve_and_pin(device, snapshot.group, layer_id, snapshot.generation,
                    &absent_id, 1, &absent) || absent.draw_slots[0] >= 0 || absent.n_pins != 0) {
            std::fprintf(stderr, "frozen-group miss became resident during teardown\n");
            return false;
        }
        std::printf("teardown_frozen_group_miss_remains_absent=1\n");
    }
    const bool retry_release = f.lease.release();
    ggml_tensor * restored_map = f.cache->get_slot_map(first_snapshot.layer, 1, N_SELECTED);
    const bool retry_snapshot_valid = f.cache->frozen_snapshot_valid(first_snapshot);
    const bool retry_callback_query = restored_map && f.cache->copy(f.cuda, restored_map, nullptr, nullptr,
            GGML_BACKEND_SCHED_COPY_PHASE_QUERY);
    std::printf("lease_retry release=%d retained=%d snapshot_valid=%d map=%d callback_query=%d\n",
            retry_release ? 1 : 0, f.lease ? 1 : 0, retry_snapshot_valid ? 1 : 0,
            restored_map ? 1 : 0, retry_callback_query ? 1 : 0);
    if (!retry_release || f.lease || retry_snapshot_valid || !restored_map || !retry_callback_query) {
        std::fprintf(stderr, "lease retry after pending-pin unregister did not restore the blocking cache\n");
        return false;
    }
    f.lease_restored = true;
    std::printf("lease_lifecycle pinned_release_rejected=1 pending_unregister_release=1 retry_release=1 snapshot_invalidated=1 blocking_query_restored=1\n");
    f.cleanup();
    return f.lease_restored;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 4 || std::strcmp(argv[2], "--device") != 0) {
        std::fprintf(stderr, "usage: %s <Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf> --device <ordinal> [--timing-mode correctness|replacement|primary|nocapture|miss|trace|scaling|scaling-abba|thread-curve|thread-abba|seed-isolation] [--cpu-thread-counts list] [--samples N] [--miss-count 1|5|10]\n", argv[0]);
        return 2;
    }
    const int device = std::stoi(argv[3]);
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) return 2;
    std::string mode = "correctness";
    std::vector<int32_t> thread_counts;
    int32_t samples = 24;
    int32_t miss_count = 1;
    for (int i = 4; i < argc; i += 2) {
        if (i + 1 >= argc) return 2;
        const std::string option = argv[i];
        const std::string value = argv[i + 1];
        if (option == "--timing-mode") {
            mode = value;
        } else if (option == "--cpu-thread-counts") {
            size_t start = 0;
            while (start < value.size()) {
                const size_t end = value.find(',', start);
                thread_counts.push_back(std::stoi(value.substr(start, end - start)));
                if (end == std::string::npos) break;
                start = end + 1;
            }
        } else if (option == "--samples") {
            samples = std::stoi(value);
        } else if (option == "--miss-count") {
            miss_count = std::stoi(value);
        } else {
            std::fprintf(stderr, "unknown option: %s\n", option.c_str());
            return 2;
        }
    }
    std::printf("harness_config,mode=%s,cpu_thread_counts=", mode.c_str());
    for (size_t i = 0; i < thread_counts.size(); ++i) std::printf("%s%d", i ? "," : "", thread_counts[i]);
    std::printf(",samples=%d,miss_count=%d,affinity=inherited,max_parallel_calls=1\n", samples, miss_count);
    return run(argv[1], device, mode.c_str(), thread_counts, samples, miss_count) ? 0 : 1;
}
