//

// Copyright (C) 2023-2024 The ggml authors
// Copyright (C) 2024 Iwan Kawrakow
// MIT license
// SPDX-License-Identifier: MIT
//
#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#include "ggml.h"
#include "ggml-backend-impl.h"
//#include "ggml-impl.h"
#include "ggml-utils.h"

#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/dmmv.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/graph.cuh"
#include "ggml-cuda/mmq_id.cuh"
#include "ggml-cuda/quantize_id.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/multiadd.cuh"
#include "ggml-cuda/hadamard.cuh"
#include "ggml-cuda/reduce.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/delta-net.cuh"
#include "ggml-cuda/kda.cuh"
#include "ggml-cuda/sinkhorn.cuh"
#include "ggml-cuda/latent_attn.cuh"
#include "ggml-cuda/blend.cuh"
#include "ggml-cuda/indexer_topk.cuh"
#include "ggml-cuda/ds4_comp.cuh"
#include "ggml-moe-ffn.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <climits>
#include <mutex>
#include <thread>

static constexpr int32_t MOE_CACHE_TRANSPORT_STREAM = 1;

struct ggml_cuda_moe_ffn_cpu_resource_state {
    struct fill_request {
        ggml_cuda_moe_ffn_fill_job owner_job{};
        std::vector<ggml_cuda_moe_ffn_fill_part> parts;
        const ggml_cuda_moe_ffn_cache_callbacks * callbacks = nullptr;
        void * cache_context = nullptr;
        int32_t binding_id = -1;
        uint64_t binding_generation = 0;
    };

    struct pending_pin_record {
        const ggml_cuda_moe_ffn_cache_callbacks * callbacks = nullptr;
        void * context = nullptr;
        std::vector<ggml_cuda_moe_ffn_pin_token> pins;
        bool active = false;
    };

    ggml_cuda_moe_ffn_cpu_resource resource{};
    std::vector<ggml_cuda_moe_ffn_resource_signature> signatures;
    int32_t device = -1;
    std::mutex execution_mutex;
    std::mutex fill_mutex;
    std::condition_variable fill_cv;
    std::thread fill_worker;
    ggml_backend_cuda_context * cuda_context = nullptr;
    cudaStream_t fill_stream = nullptr;
    fill_request queued_fill{};
    fill_request stuck_fill{};
    bool fill_queued = false;
    bool fill_active = false;
    bool fill_outstanding = false;
    bool fill_stuck = false;
    bool fill_stopping = false;
    bool transport_stopped = false;
    bool accepting_fills = true;
    uint64_t fills_accepted = 0;
    uint64_t fills_completed = 0;
    uint64_t fills_rejected = 0;
    uint64_t fills_failed = 0;
    uint64_t fill_parts_completed = 0;
    uint64_t fill_bytes_completed = 0;
    void * host_stage = nullptr;
    int32_t * host_ids = nullptr;
    int32_t * host_slots = nullptr;
    int32_t * host_hit_slots = nullptr;
    float * host_activations = nullptr;
    float * host_routes = nullptr;
    float * host_hit_routes = nullptr;
    float * host_output = nullptr;
    int32_t * device_slots = nullptr;
    float * device_routes = nullptr;
    void * device_up_q8 = nullptr;
    void * device_down_q8 = nullptr;
    float * device_hidden = nullptr;
    float * device_down = nullptr;
    float * device_output = nullptr;
    cudaEvent_t hit_kernels_done = nullptr;
    size_t up_q8_bytes = 0;
    size_t down_q8_bytes = 0;
    uint64_t cpu_calls = 0;
    uint64_t max_parallel_calls = 0;
    uint64_t active_calls = 0;
    uint64_t pin_resolve_calls = 0;
    uint64_t pin_acquisitions = 0;
    uint64_t pin_releases = 0;
    uint64_t pending_pin_releases = 0;
    uint64_t pin_release_failures = 0;
    std::atomic<bool> accepting_bindings{true};
    pending_pin_record pending_pins{};
    std::vector<ggml_cuda_moe_ffn_pin_token> resolved_pins;
    std::vector<int32_t> fill_candidates;

    ~ggml_cuda_moe_ffn_cpu_resource_state() {
        GGML_ASSERT(!pending_pins.active && "destroying CUDA MoE resource with live cache pins");
        GGML_ASSERT(!fill_worker.joinable() && !fill_outstanding && "destroying CUDA MoE resource with live fill work");
        int previous_device = 0;
        cudaGetDevice(&previous_device);
        if (device >= 0) cudaSetDevice(device);
        if (hit_kernels_done) cudaEventDestroy(hit_kernels_done);
        if (device_slots) cudaFree(device_slots);
        if (device_routes) cudaFree(device_routes);
        if (device_up_q8) cudaFree(device_up_q8);
        if (device_down_q8) cudaFree(device_down_q8);
        if (device_hidden) cudaFree(device_hidden);
        if (device_down) cudaFree(device_down);
        if (device_output) cudaFree(device_output);
        if (host_stage) cudaFreeHost(host_stage);
        cudaSetDevice(previous_device);
        if (resource.cpu_backend) ggml_backend_free(resource.cpu_backend);
    }
};

struct ggml_cuda_moe_ffn_binding_state {
    ggml_cuda_moe_ffn_binding binding{};
    ggml_cuda_moe_ffn_stats stats{};
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> cpu_resource;
    bool active = true; // Protected by the shared CPU-resource execution mutex.
};

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <float.h>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <stdint.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sstream>

static size_t ggml_cuda_moe_ffn_physical_part_count(const ggml_moe_ffn_layout & layout) {
    switch (layout.storage_topology) {
        case GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN: return 3;
        case GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN: return 2;
        default: return 0;
    }
}

static std::pair<const ggml_tensor *, const ggml_tensor *> ggml_cuda_moe_ffn_physical_part(
        const ggml_moe_ffn_layout & layout, size_t index) {
    if (layout.storage_topology == GGML_MOE_FFN_STORAGE_SEPARATE_UP_GATE_DOWN) {
        switch (index) {
            case 0: return { layout.up, layout.bank_up };
            case 1: return { layout.gate, layout.bank_gate };
            case 2: return { layout.down, layout.bank_down };
            default: break;
        }
    } else if (layout.storage_topology == GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN) {
        switch (index) {
            case 0: return { layout.up_gate_storage, layout.bank_up_gate_storage };
            case 1: return { layout.down, layout.bank_down };
            default: break;
        }
    }
    return { nullptr, nullptr };
}

static bool ggml_cuda_moe_ffn_validate_fill_job(const ggml_cuda_moe_ffn_binding & binding,
        const ggml_cuda_moe_ffn_fill_job & job, int32_t expected_expert,
        const ggml_cuda_moe_ffn_fill_part * parts, size_t part_count) {
    const auto & layout = binding.layout;
    const size_t expected_parts = ggml_cuda_moe_ffn_physical_part_count(layout);
    if (binding.layout.n_experts <= 0 || binding.layout.n_slots <= 0 || binding.layout.cache_layer < 0 ||
            expected_expert < 0 || expected_expert >= binding.layout.n_experts ||
            job.owner != binding.cache_context || job.device != binding.layout.device ||
            job.group != binding.layout.cache_group || job.layer != binding.layout.cache_layer ||
            job.owner_generation != binding.generation || job.slot < 0 || job.slot >= binding.layout.n_slots ||
            !parts || expected_parts == 0 || part_count != expected_parts || job.generation == 0 ||
            job.key < 0 || job.key / binding.layout.n_experts != binding.layout.cache_layer ||
            job.key % binding.layout.n_experts != expected_expert) return false;

    size_t total = 0;
    for (size_t i = 0; i < expected_parts; ++i) {
        if (layout.storage_topology == GGML_MOE_FFN_STORAGE_MERGED_UP_GATE_DOWN && i == 0) {
            const ggml_tensor * host = layout.up_gate_storage;
            const ggml_tensor * bank = layout.bank_up_gate_storage;
            const ggml_tensor * host_root = host && host->view_src ? host->view_src : host;
            const ggml_tensor * bank_root = bank && bank->view_src ? bank->view_src : bank;
            const ggml_tensor * views[2] = { layout.gate, layout.up };
            const ggml_tensor * bank_views[2] = { layout.bank_gate, layout.bank_up };
            if (!host || !host->data || !bank || !bank->data) return false;
            for (size_t v = 0; v < 2; ++v) {
                const ggml_tensor * source_view = views[v];
                const ggml_tensor * destination_view = bank_views[v];
                if (!source_view || !destination_view || source_view->view_src != host_root ||
                        destination_view->view_src != bank_root ||
                        source_view->view_offs < host->view_offs ||
                        destination_view->view_offs < bank->view_offs ||
                        source_view->ne[2] <= expected_expert ||
                        destination_view->ne[2] <= job.slot || source_view->nb[2] == 0 ||
                        destination_view->nb[2] == 0) return false;
                const size_t rows = ggml_row_size(source_view->type, source_view->ne[0]);
                if (!rows || rows > SIZE_MAX / (size_t) source_view->ne[1]) return false;
                const size_t bytes = rows * (size_t) source_view->ne[1];
                if (source_view->type != destination_view->type || bytes == 0 ||
                        (size_t) expected_expert > SIZE_MAX / source_view->nb[2] ||
                        (size_t) job.slot > SIZE_MAX / destination_view->nb[2]) return false;
                const size_t source_step = (size_t) expected_expert * source_view->nb[2];
                const size_t destination_step = (size_t) job.slot * destination_view->nb[2];
                const size_t source_base = source_view->view_offs - host->view_offs;
                const size_t destination_base = destination_view->view_offs - bank->view_offs;
                if (source_base > SIZE_MAX - source_step ||
                        destination_base > SIZE_MAX - destination_step) return false;
                const size_t source_offset = source_base + source_step;
                const size_t destination_offset = destination_base + destination_step;
                const size_t source_size = ggml_nbytes(host);
                const size_t destination_size = ggml_nbytes(bank);
                const void * expected_source = (const uint8_t *) host->data + source_offset;
                void * expected_destination = (uint8_t *) bank->data + destination_offset;
                const size_t descriptor_bytes = v == 0 ? parts[i].bytes : parts[i].bytes2;
                const void * descriptor_source = v == 0 ? parts[i].source : parts[i].source2;
                void * descriptor_destination = v == 0 ? parts[i].destination : parts[i].destination2;
                if (source_offset > source_size || bytes > source_size - source_offset ||
                        destination_offset > destination_size || bytes > destination_size - destination_offset ||
                        descriptor_bytes != bytes || descriptor_source != expected_source ||
                        descriptor_destination != expected_destination || total > SIZE_MAX - bytes) return false;
                total += bytes;
            }
            continue;
        }
        const auto expected = ggml_cuda_moe_ffn_physical_part(layout, i);
        const ggml_tensor * host = expected.first;
        const ggml_tensor * bank = expected.second;
        if (!host || !host->data || !bank || !bank->data ||
                host->ne[2] <= expected_expert || bank->ne[2] <= job.slot ||
                host->nb[2] == 0 || host->nb[2] != bank->nb[2]) return false;
        const size_t bytes = host->nb[2];
        if (parts[i].bytes != bytes || parts[i].source2 || parts[i].destination2 || parts[i].bytes2 != 0 ||
                (size_t) expected_expert > SIZE_MAX / bytes || (size_t) job.slot > SIZE_MAX / bytes) return false;
        const size_t source_offset = (size_t) expected_expert * bytes;
        const size_t destination_offset = (size_t) job.slot * bytes;
        const size_t source_size = ggml_nbytes(host);
        const size_t destination_size = ggml_nbytes(bank);
        if (source_offset > source_size || bytes > source_size - source_offset ||
                destination_offset > destination_size || bytes > destination_size - destination_offset ||
                parts[i].source != (const uint8_t *) host->data + source_offset ||
                parts[i].destination != (uint8_t *) bank->data + destination_offset ||
                total > SIZE_MAX - bytes) return false;
        total += bytes;
    }
    return total != 0 && job.total_bytes == total;
}

static int ggml_cuda_moe_ffn_copy_fill_request(
        ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_cuda_moe_ffn_cpu_resource_state::fill_request & request) {
    const auto & job = request.owner_job;
    if (!request.callbacks || !request.callbacks->finish_fill || !request.cache_context) return 0;
    auto rollback = [&]() { return request.callbacks->finish_fill(request.cache_context, &job, false) ? 1 : 0; };
    if (!resource.fill_stream || !resource.cuda_context || job.device != resource.device) return rollback();
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> binding_state;
    {
        std::lock_guard<std::mutex> lock(resource.cuda_context->moe_ffn_mutex);
        auto it = resource.cuda_context->moe_ffn_bindings.find(request.binding_id);
        if (it != resource.cuda_context->moe_ffn_bindings.end() &&
                it->second->binding.generation == request.binding_generation &&
                it->second->cpu_resource.get() == &resource) binding_state = it->second;
    }
    if (!binding_state || binding_state->binding.layout.n_experts <= 0 || job.key < 0) return rollback();
    const int32_t expert = job.key % binding_state->binding.layout.n_experts;
    if (!ggml_cuda_moe_ffn_validate_fill_job(binding_state->binding, job, expert,
                request.parts.data(), request.parts.size())) return rollback();

    if (cudaSetDevice(resource.device) != cudaSuccess) return 0;
    bool copied = true;
    for (size_t i = 0; i < request.parts.size(); ++i) {
        const auto & part = request.parts[i];
        // Canonical model weights are persistent host buffers. CUDA may stage
        // pageable sources synchronously; correctness does not assume overlap.
        if (cudaMemcpyAsync(part.destination, part.source, part.bytes,
                    cudaMemcpyHostToDevice, resource.fill_stream) != cudaSuccess ||
                (part.bytes2 && cudaMemcpyAsync(part.destination2, part.source2, part.bytes2,
                    cudaMemcpyHostToDevice, resource.fill_stream) != cudaSuccess)) {
            copied = false;
            break;
        }
    }
    const cudaError_t drained = cudaStreamSynchronize(resource.fill_stream);
    if (drained != cudaSuccess) {
        return 0;
    }
    if (!request.callbacks->finish_fill(request.cache_context, &job, copied)) return 0;
    return copied ? 2 : 1;
}

static void ggml_cuda_moe_ffn_fill_worker(ggml_cuda_moe_ffn_cpu_resource_state * resource) {
    for (;;) {
        ggml_cuda_moe_ffn_cpu_resource_state::fill_request request{};
        {
            std::unique_lock<std::mutex> lock(resource->fill_mutex);
            resource->fill_cv.wait(lock, [&]() {
                return resource->fill_stopping || resource->fill_queued;
            });
            if (!resource->fill_queued) {
                if (resource->fill_stopping) return;
                continue;
            }
            request = std::move(resource->queued_fill);
            resource->queued_fill = {};
            resource->fill_queued = false;
            resource->fill_active = true;
        }

        const int result = ggml_cuda_moe_ffn_copy_fill_request(*resource, request);
        {
            std::lock_guard<std::mutex> lock(resource->fill_mutex);
            resource->fill_active = false;
            if (result == 0) {
                resource->fill_stuck = true;
                resource->stuck_fill = request;
            } else {
                if (result == 2) {
                    ++resource->fills_completed;
                    resource->fill_parts_completed += request.parts.size();
                    resource->fill_bytes_completed += request.owner_job.total_bytes;
                }
                else ++resource->fills_failed;
                resource->fill_outstanding = false;
            }
        }
        resource->fill_cv.notify_all();
        if (result == 0) return;
    }
}

static bool ggml_cuda_moe_ffn_enqueue_fill(ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_cuda_moe_ffn_binding & binding, int32_t binding_id, int32_t expert) {
    const auto * callbacks = binding.cache_callbacks;
    if (!callbacks || !callbacks->reserve_fill || !callbacks->finish_fill) return false;
    {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        if (!resource.accepting_fills || resource.fill_stopping || resource.fill_outstanding) {
            ++resource.fills_rejected;
            return false;
        }
    }

    ggml_cuda_moe_ffn_fill_job job{};
    const size_t part_count = ggml_cuda_moe_ffn_physical_part_count(binding.layout);
    if (part_count == 0) {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        ++resource.fills_rejected;
        return false;
    }
    std::vector<ggml_cuda_moe_ffn_fill_part> parts;
    try {
        parts.resize(part_count);
    } catch (...) {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        ++resource.fills_rejected;
        return false;
    }
    if (!callbacks->reserve_fill(binding.cache_context, binding.layout.device, binding.layout.cache_group,
                binding.layout.cache_layer, binding.generation, expert, &job, parts.data(), parts.size())) {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        ++resource.fills_rejected;
        return false;
    }
    if (!ggml_cuda_moe_ffn_validate_fill_job(binding, job, expert, parts.data(), parts.size())) {
        if (!callbacks->finish_fill(binding.cache_context, &job, false)) {
            GGML_ABORT("failed to roll back invalid MoE cache fill reservation");
        }
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        ++resource.fills_rejected;
        return false;
    }

    ggml_cuda_moe_ffn_cpu_resource_state::fill_request request{};
    request.owner_job = job;
    request.parts = std::move(parts);
    request.callbacks = callbacks;
    request.cache_context = binding.cache_context;
    request.binding_id = binding_id;
    request.binding_generation = binding.generation;
    bool accepted = false;
    {
        std::lock_guard<std::mutex> lock(resource.fill_mutex);
        if (resource.accepting_fills && !resource.fill_stopping && !resource.fill_outstanding) {
            resource.queued_fill = std::move(request);
            resource.fill_queued = true;
            resource.fill_outstanding = true;
            ++resource.fills_accepted;
            accepted = true;
        } else {
            ++resource.fills_rejected;
        }
    }
    if (!accepted) {
        if (!callbacks->finish_fill(binding.cache_context, &job, false)) {
            GGML_ABORT("failed to roll back rejected MoE cache fill reservation");
        }
        return false;
    }
    resource.fill_cv.notify_one();
    return true;
}

#ifdef __linux__
#include <sys/mman.h>
#endif

#define IK_PRINT_TIMING 0

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void ggml_cuda_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    GGML_UNUSED(level);
    GGML_UNUSED(user_data);
    fprintf(stderr, "%s", msg);
}

ggml_log_callback ggml_cuda_log_callback = ggml_cuda_default_log_callback;
void * ggml_cuda_log_user_data = NULL;

GGML_CALL void ggml_backend_cuda_log_set_callback(ggml_log_callback log_callback, void * user_data) {
    ggml_cuda_log_callback = log_callback;
    ggml_cuda_log_user_data = user_data;
}

#define GGML_CUDA_LOG_INFO(...) ggml_cuda_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_CUDA_LOG_WARN(...) ggml_cuda_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_CUDA_LOG_ERROR(...) ggml_cuda_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define GGML_CUDA_LOG_DEBUG(...) ggml_cuda_log(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_cuda_log(enum ggml_log_level level, const char * format, ...) {
    if (ggml_cuda_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_cuda_log_callback(level, buffer, ggml_cuda_log_user_data);
        } else {
            std::vector<char> buffer2(len + 1);  // vsnprintf adds a null terminator
            va_end(args);
            va_start(args, format);
            vsnprintf(&buffer2[0], buffer2.size(), format, args);
            ggml_cuda_log_callback(level, buffer2.data(), ggml_cuda_log_user_data);
        }
        va_end(args);
    }
}

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    cudaGetDevice(&id);

    GGML_CUDA_LOG_ERROR("CUDA error: %s\n", msg);
    GGML_CUDA_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_CUDA_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ASSERT to get a stack trace
    GGML_ABORT("CUDA error");
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
#if defined(GGML_USE_HIPBLAS) && defined(GGML_HIP_UMA)
    auto res = hipMallocManaged(ptr, size);
    if (res == hipSuccess) {
        // if error we "need" to know why...
        CUDA_CHECK(hipMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device));
    }
    return res;
#else

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr)
    {
        err = cudaMallocManaged(ptr, size);
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
#else
    return cudaMalloc(ptr, size);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)

#endif
}

static ggml_cuda_device_info ggml_cuda_init() {
#ifdef __HIP_PLATFORM_AMD__
    // Workaround for a rocBLAS bug when using multiple graphics cards:
    // https://github.com/ROCmSoftwarePlatform/rocBLAS/issues/1346
    rocblas_initialize();
    CUDA_CHECK(cudaDeviceSynchronize());
#endif

    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.device_count);
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.device_count <= GGML_CUDA_MAX_DEVICES);

    int64_t total_vram = 0;
#ifdef GGML_CUDA_FORCE_MMQ
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_MMQ (instead of CUBLAS):    yes\n", __func__);
#endif // GGML_CUDA_FORCE_MMQ
#ifdef GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: GGML_CUDA_FORCE_CUBLAS (Instead of MMQ):    yes\n", __func__);
#endif // GGML_CUDA_FORCE_CUBLAS
    GGML_CUDA_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices:\n", __func__, info.device_count);
    for (int id = 0; id < info.device_count; ++id) {
        int device_vmm = 0;

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        GGML_CUDA_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n", id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                prop.totalGlobalMem/(1024*1024));

        info.default_tensor_split[id] = total_vram;
        total_vram += prop.totalGlobalMem;

        info.devices[id].nsm   = prop.multiProcessorCount;
        info.devices[id].smpb  = prop.sharedMemPerBlock;
        info.devices[id].integrated = prop.integrated;
#if defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
        info.devices[id].smpbo = prop.sharedMemPerBlock;
        info.devices[id].cc = 100*prop.major + 10*prop.minor + CC_OFFSET_AMD;
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
#endif // defined(GGML_USE_HIPBLAS) && defined(__HIP_PLATFORM_AMD__)
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

#ifdef GGML_USE_NCCL
    info.have_nccl = false;
    if (info.device_count > 1) {
        int gpu_list[GGML_CUDA_MAX_DEVICES];
        for(int i = 0; i < info.device_count; ++i) gpu_list[i] = i;
        auto status = ncclCommInitAll(info.nccl_coms, info.device_count, gpu_list);
        if (status != ncclSuccess) {
            printf("=============================== NCCL initialization failed with status %d\n", int(status));
        } else {
            printf("=============================== NCCL main communicator initialized\n");
            info.have_nccl = true;
            auto com = info.nccl_coms + info.device_count;
            if (info.device_count == 4) {
                int devs[8] = {0,1, 2,3, 0,2, 1,3};
                auto com = info.nccl_coms + info.device_count;
                for (int ip = 0; ip < 4; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            } else if (info.device_count == 3) {
                int devs[4] = {0,1, 0,2};
                for (int ip = 0; ip < 2; ++ip) {
                    if (auto status = ncclCommInitAll(com+2*ip, 2, devs+2*ip); status != ncclSuccess) {
                        printf("=============================== NCCL initialization of pair %d failed with status %d\n", ip, int(status));
                        GGML_ABORT("Fatal error");
                    }
                }
                printf("=============================== NCCL pair communicators for %d GPUs initialized\n", info.device_count);
            }
        }
    }
#endif
    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

/* ---------- hot-swap: invalidate all cached CUDA graphs ---------- */
extern "C" void ggml_backend_cuda_invalidate_graphs(const void * model) {
#ifdef USE_CUDA_GRAPH
    auto & info = const_cast<ggml_cuda_device_info &>(ggml_cuda_info());
    if (auto it = info.all_ctx.find(model); it != info.all_ctx.end()) {
        for (auto ctx : it->second) {
            if (ctx) {
                ctx->cuda_graphs.clear();
            }
        }
    } else {
        fprintf(stderr, "================================= %s: did not find entry for model at %p\n", __func__, model);
    }
#else
    GGML_UNUSED(model);
#endif // USE_CUDA_GRAPH
    //for (int i = 0; i < info.device_count; ++i) {
    //    if (info.all_ctx[i]) {
    //        info.all_ctx[i]->cuda_graphs.clear();
    //    }
    //}
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
            }
        }
        GGML_ASSERT(pool_size == 0);
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        CUDA_CHECK(ggml_cuda_device_malloc(&ptr, look_ahead_size, device));
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_CUDA_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_CUDA_LOG_WARN("Cuda buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CU_CHECK(cuMemMap(pool_addr + pool_size, reserve_size, 0, handle, 0));

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // set access
            CUmemAccessDesc access = {};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = device;
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            CU_CHECK(cuMemSetAccess(pool_addr + pool_size, reserve_size, &access, 1));

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) (pool_addr + pool_used);
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) (pool_addr + pool_used));
    }
};
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int device) {
#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_CUDA_NO_VMM) && !defined(GGML_USE_MUSA)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
//static std::atomic<int> ggml_cuda_lock_counter;
static int ggml_cuda_lock_counter = 0;

ggml_backend_cuda_context::ggml_backend_cuda_context(int device, const void * model) :
    device(device), name(GGML_CUDA_NAME + std::to_string(device)), model(model) {
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    auto & all_ctx = info->all_ctx[model];
    if (all_ctx[device]) {
        GGML_CUDA_LOG_WARN("%s: a context for device %d already exists?\n", __func__, device);
    } else{
        all_ctx[device] = this;
    }
}

ggml_backend_cuda_context::~ggml_backend_cuda_context() {

    ggml_cuda_set_device(device);
    for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
        if (streams[device][j] != nullptr) CUDA_CHECK(cudaStreamSynchronize(streams[device][j]));
    }
    decltype(moe_ffn_bindings) moe_ffn_bindings_to_release;
    decltype(moe_ffn_cpu_resources) moe_ffn_resources_to_release;
    {
        std::lock_guard<std::mutex> binding_lock(moe_ffn_mutex);
        moe_ffn_bindings_to_release.swap(moe_ffn_bindings);
        moe_ffn_resources_to_release.swap(moe_ffn_cpu_resources);
    }
    moe_ffn_bindings_to_release.clear();
    moe_ffn_resources_to_release.clear();

#ifdef USE_CUDA_GRAPH
    // Let's leave this debug log in for now, so we have a trace in case
    // number of CUDA graphs goes crazy
    GGML_CUDA_LOG_INFO("%s: have %d graphs\n", __func__, int(cuda_graphs.size()));
#endif

    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    if (compute_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(compute_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
        }
        if (cublas_handles[i] != nullptr) {
            CUBLAS_CHECK(cublasDestroy(cublas_handles[i]));
        }
    }
    auto info = const_cast<ggml_cuda_device_info*>(&ggml_cuda_info());
    if (auto it = info->all_ctx.find(model); it != info->all_ctx.end() && it->second[device] == this) {
        it->second[device] = nullptr;
    }
    //if (info->all_ctx[device] == this) {
    //    info->all_ctx[device] = nullptr;
    //}

}

// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

GGML_CALL static const char * ggml_backend_cuda_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->name.c_str();
}

GGML_CALL static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_buffer_get_name;
}

GGML_CALL static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

GGML_CALL static void ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        size_t original_size = ggml_nbytes(tensor);
        size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
}

GGML_CALL static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *)tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

GGML_CALL static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
#ifndef NDEBUG
    printf("%s(%s -> %s)\n", __func__, src->name, dst->name);
#endif
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        if (src_ctx->device == dst_ctx->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_ctx->device, src->data, src_ctx->device, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(ctx->dev_ptr, value, buffer->size));
    CUDA_CHECK(cudaDeviceSynchronize());
}

static ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

GGML_CALL static const char * ggml_backend_cuda_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    size = std::max(size, (size_t)1); // cudaMalloc returns null for size 0

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < GGML_CUDA_MAX_DEVICES; i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// cuda split buffer

struct ggml_backend_cuda_split_buffer_type_context {
    //std::array<float, GGML_CUDA_MAX_DEVICES> tensor_split;
};

struct ggml_backend_cuda_split_buffer_context {
    ~ggml_backend_cuda_split_buffer_context() {
        //for (ggml_tensor_extra_gpu * extra : tensor_extras) {
        //    for (int id = 0; id < GGML_CUDA_MAX_DEVICES; ++id) {
        //        for (int64_t is = 0; is < GGML_CUDA_MAX_STREAMS; ++is) {
        //            if (extra->events[id][is] != nullptr) {
        //                CUDA_CHECK(cudaEventDestroy(extra->events[id][is]));
        //            }
        //        }
        //        if (extra->data_device[id] != nullptr) {
        //            CUDA_CHECK(cudaFree(extra->data_device[id]));
        //        }
        //    }
        //    delete extra;
        //}
    }

    std::vector<ggml_tensor_extra_gpu *> tensor_extras;
};

GGML_CALL static const char * ggml_backend_cuda_split_buffer_get_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buffer);
}

static bool ggml_backend_buffer_is_cuda_split(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_cuda_split_buffer_get_name;
    GGML_UNUSED(ggml_backend_buffer_is_cuda_split); // only used in debug builds currently, avoid unused function warning in release builds
}

GGML_CALL static void ggml_backend_cuda_split_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_split_buffer_context * ctx = (ggml_backend_cuda_split_buffer_context *)buffer->context;
    delete ctx;
}

GGML_CALL static void * ggml_backend_cuda_split_buffer_get_base(ggml_backend_buffer_t buffer) {
    // the pointers are stored in the tensor extras, this is just a dummy address and never dereferenced
    return (void *)0x1000;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cuda_split_buffer_init_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    if (!tensor->extra) return;
    //printf("%s(%s, %p)\n", __func__, tensor->name, tensor->extra);
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());
    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        auto split = extra->splits[i];
        auto ne0 = split->ne[0];
        auto size = ggml_nbytes(split);
        auto padded_size = size;
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            int nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            auto row_size = ggml_row_size(split->type, ne0);
            padded_size += padded_row_size - row_size;
        }
        ggml_cuda_set_device(i);
        char * buf;
        CUDA_CHECK(ggml_cuda_device_malloc((void**)&buf, padded_size, i));
        if (padded_size > size) {
            CUDA_CHECK(cudaMemset(buf + size, 0, padded_size - size));
        }
        //printf("    allocated %zu bytes for tensor %s of type %s, dim = %ld x %ld x %ld. padding: %zu\n", padded_size, split->name, ggml_type_name(split->type),
        //        split->ne[0], split->ne[1], split->ne[2], padded_size - size);
        //printf("DEBUG init_tensor: dev=%d split_ne0=%ld type=%s ggml_nbytes=%zu padded=%zu data_ptr=%p\n",
        //       i, (long)ne0, ggml_type_name(split->type), size, padded_size, (void*)buf);
        //fflush(stdout);
        split->data = buf;
        auto ctx = new ggml_backend_cuda_buffer_context(i, buf);
        auto buft = ggml_backend_cuda_buffer_type(i);
        split->buffer = ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, padded_size);
        ggml_backend_buffer_set_usage(split->buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    }
    return;

}

GGML_CALL static void ggml_backend_cuda_split_buffer_set_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (!tensor->extra && tensor->view_src && tensor->view_src->extra) {
        // OK, this is an ugly hack, but I don't really see a way to trick the machine into correctly
        // loading non-contiguous merged split tensors.
        auto view_src = tensor->view_src;
        auto extra = (ggml_split_tensor_t *)view_src->extra;
        void * extra_ptr;
        memcpy(&extra_ptr, view_src->op_params, sizeof(extra_ptr));
        if (extra_ptr) {
            std::string merged_name = view_src->name;
            if (auto pos = merged_name.find("ffn_gate_up_exps.weight"); pos != std::string::npos) {
                std::string name = tensor->name;
                auto pos_u = name.find("ffn_up_exps.weight");
                auto pos_g = name.find("ffn_gate_exps.weight");
                if (pos_u != std::string::npos || pos_g != std::string::npos) {
                    GGML_ASSERT(extra->split_dim == 1);
                    auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                    int ne = 0;
                    for (int is = 0; is < int(ranges.size()); ++is) {
                        auto & r = ranges[is];
                        GGML_ASSERT((extra->splits[is] && !r.empty()) || (!extra->splits[is] && r.empty()));
                        if (r.empty()) continue;
                        GGML_ASSERT(r.size() == 2);
                        auto split = extra->splits[is];
                        ggml_cuda_set_device(is);
                        int ir = pos_g != std::string::npos ? 0 : 1;
                        auto p = r[ir];
                        size_t offset = 0;
                        if (ir == 1) {
                            p.first -= tensor->ne[1];
                            GGML_ASSERT(p.first >= 0);
                            offset = split->ne[1]/2 * split->nb[1];
                        }
                        for (int i02 = 0; i02 < split->ne[2]; ++i02) {
                            auto dst = (char *)split->data + i02*split->nb[2] + offset;
                            auto src = (const char *)data + i02*tensor->nb[2] + ne*tensor->nb[1];
                            CUDA_CHECK(cudaMemcpyAsync(dst, src, p.second*tensor->nb[1], cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        ne += p.second;
                        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
                return;
            }
            if (auto pos = merged_name.find("ffn_gate_up_exps.bias"); pos != std::string::npos) {
                std::string name = tensor->name;
                auto pos_u = name.find("ffn_up_exps.bias");
                auto pos_g = name.find("ffn_gate_exps.bias");
                if (pos_u != std::string::npos || pos_g != std::string::npos) {
                    GGML_ASSERT(extra->split_dim == 0);
                    auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                    int ne = 0;
                    for (int is = 0; is < int(ranges.size()); ++is) {
                        auto & r = ranges[is];
                        GGML_ASSERT((extra->splits[is] && !r.empty()) || (!extra->splits[is] && r.empty()));
                        if (r.empty()) continue;
                        GGML_ASSERT(r.size() == 2);
                        auto split = extra->splits[is];
                        ggml_cuda_set_device(is);
                        int ir = pos_g != std::string::npos ? 0 : 1;
                        auto p = r[ir];
                        size_t offset = 0;
                        if (ir == 1) {
                            p.first -= tensor->ne[0];
                            GGML_ASSERT(p.first >= 0);
                            offset = split->ne[0]/2 * split->nb[0];
                        }
                        for (int i01 = 0; i01 < split->ne[1]; ++i01) {
                            auto dst = (char *)split->data + i01*split->nb[1] + offset;
                            auto src = (const char *)data + i01*tensor->nb[1] + ne*tensor->nb[0];
                            CUDA_CHECK(cudaMemcpyAsync(dst, src, p.second*tensor->nb[0], cudaMemcpyHostToDevice, cudaStreamPerThread));
                        }
                        ne += p.second;
                        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                    }
                }
                return;
            }
        }
    }
    if (!tensor->extra) return;
    static std::map<ggml_type, int> k_map = {
        { GGML_TYPE_Q4_0_R8   , 8},
        { GGML_TYPE_Q5_0_R4   , 4},
        { GGML_TYPE_Q8_0_R8   , 8},
        { GGML_TYPE_Q2_K_R4   , 4},
        { GGML_TYPE_Q3_K_R4   , 4},
        { GGML_TYPE_Q4_K_R4   , 4},
        { GGML_TYPE_Q5_K_R4   , 4},
        { GGML_TYPE_Q6_K_R4   , 4},
        { GGML_TYPE_IQ2_XXS_R4, 4},
        { GGML_TYPE_IQ2_XS_R4 , 4},
        { GGML_TYPE_IQ3_XXS_R4, 4},
        { GGML_TYPE_IQ1_S_R4  , 4},
        { GGML_TYPE_IQ4_NL_R4 , 4},
        { GGML_TYPE_IQ3_S_R4  , 4},
        { GGML_TYPE_IQ2_S_R4  , 4},
        { GGML_TYPE_IQ4_XS_R8 , 8},
        { GGML_TYPE_IQ1_M_R4  , 4},
        { GGML_TYPE_BF16_R16  , 16},
        { GGML_TYPE_IQ3_KS_R16 , 16},
        { GGML_TYPE_Q6_0_R4   , 4},
        { GGML_TYPE_IQ2_BN_R4 , 4},
        { GGML_TYPE_IQ2_K_R4  , 4},
        { GGML_TYPE_IQ3_K_R4  , 4},
        { GGML_TYPE_IQ4_K_R4  , 4},
        { GGML_TYPE_IQ5_K_R4  , 4},
        { GGML_TYPE_IQ4_KS_R4 , 4},
        { GGML_TYPE_IQ5_KS_R4 , 4},
        { GGML_TYPE_Q8_K_R16  , 4},
        { GGML_TYPE_Q8_KV_R8  , 4},
        { GGML_TYPE_Q8_K_R8   , 8},
        { GGML_TYPE_IQ4_KS_R16, 16},
        { GGML_TYPE_IQ5_KS_R16, 16},
    };

    // split tensors must always be set in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    if (extra->split_dim < 0) {
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto nbytes = ggml_nbytes(tensor);
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(ggml_are_same_shape(tensor, split));
            GGML_ASSERT(ggml_nbytes(split) == nbytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(split->data, data, nbytes, cudaMemcpyHostToDevice, cudaStreamPerThread));
        }
    }
    else if (extra->split_dim == 0) {
        int n_interleave = 1;
        if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
        auto tt = ggml_internal_get_type_traits(tensor->type);
        std::vector<char> host_buffer;
        GGML_ASSERT(ggml_is_contiguous(tensor));
        int nrows = ggml_nrows(tensor);
        auto bs = tt.blck_size;
        auto ts = tt.type_size;
        void * extra_ptr;
        memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
        if (extra_ptr) {
            auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
            GGML_ASSERT(extra->n_device == int(ranges.size()));
            GGML_ASSERT(tensor->ne[2]*tensor->ne[3] == 1);
            GGML_ASSERT(n_interleave == 1);
            GGML_ASSERT(tt.row_meta_size == 0);
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) {
                    GGML_ASSERT(ranges[i].empty());
                    continue;
                }
                GGML_ASSERT(!ranges[i].empty());
                GGML_ASSERT((int)ggml_nrows(split) == nrows);
                auto split_row_size = ggml_row_size(split->type, split->ne[0]);
                if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
                auto dst = host_buffer.data();
                for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                    for (auto & p : ranges[i]) {
                        GGML_ASSERT(p.first  % bs == 0);
                        GGML_ASSERT(p.second % bs == 0);
                        auto src = (const char *)data + i01*tensor->nb[1] + (p.first/bs)*ts;
                        auto size = (p.second/bs)*ts;
                        memcpy(dst, src, size);
                        dst += size;
                    }
                }
                ggml_cuda_set_device(i);
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            }
        } else {
            int ne = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                GGML_ASSERT(split->ne[1]%n_interleave == 0);
                ggml_cuda_set_device(i);
                GGML_ASSERT(split->type == tensor->type);
                GGML_ASSERT((int)ggml_nrows(split) == nrows);
                GGML_ASSERT(ne % bs == 0);
                auto source_offset = n_interleave*(tt.row_meta_size + (ne / bs) * ts);
                auto split_row_size = ggml_row_size(split->type, split->ne[0]);
                if (host_buffer.size() < nrows*split_row_size) host_buffer.resize(nrows*split_row_size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    for (int64_t i01 = 0; i01 < split->ne[1]; i01 += n_interleave) {
                        auto dst = host_buffer.data() + (i02*split->ne[1] + i01)*split_row_size;
                        auto src = (const char *)data + i02*tensor->nb[2] + i01*tensor->nb[1];
                        if (tt.row_meta_size > 0) {
                            memcpy(dst, src, tt.row_meta_size*n_interleave);
                        }
                        memcpy(dst + tt.row_meta_size*n_interleave, src + source_offset, n_interleave*(split_row_size - tt.row_meta_size));
                    }
                }
                //printf("DEBUG set_tensor: dev=%d split_ne0=%ld nrows=%d split_row_size=%zu total=%zu "
                //       "split_data=%p host_data=%p host_capacity=%zu source_offset=%zu\n",
                //       i, (long)split->ne[0], nrows, split_row_size, nrows*split_row_size,
                //       (void*)split->data, (void*)host_buffer.data(), host_buffer.size(),
                //       (size_t)source_offset);
                //fflush(stdout);
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), nrows*split_row_size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne += split->ne[0];
            }
        }
    }
    else if (extra->split_dim == 1) {
        void * extra_ptr;
        memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
        if (tensor->ne[2] > 1) {
            auto row_size = ggml_row_size(tensor->type, tensor->ne[0]);
            std::vector<char> host_buffer;
            int ne1 = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                auto size = ggml_nbytes(split);
                if (host_buffer.size() < size) host_buffer.resize(size);
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    auto dst = host_buffer.data() + i02*split->ne[1]*row_size;
                    if (extra_ptr) {
                        auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                        for (auto & p : ranges[i]) {
                            auto this_src = (const char *)data + i02*tensor->nb[2] + p.first*tensor->nb[1];
                            auto this_size = p.second*tensor->nb[1];
                            memcpy(dst, this_src, this_size);
                            dst += this_size;
                        }
                    } else {
                        auto src = (const char *)data + i02*tensor->nb[2] + ne1*tensor->nb[1];
                        memcpy(dst, src, split->ne[1]*row_size);
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(split->data, host_buffer.data(), size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                ne1 += split->ne[1];
            }
        } else {
            int n_interleave = 1;
            if (auto it = k_map.find(tensor->type); it != k_map.end()) n_interleave = it->second;
            if (extra_ptr) {
                auto & ranges = *(const std::vector<std::vector<std::pair<int,int>>> *)extra_ptr;
                GGML_ASSERT(extra->n_device == int(ranges.size()));
                GGML_ASSERT(tensor->ne[2]*tensor->ne[3] == 1);
                GGML_ASSERT(n_interleave == 1);
                for (int i = 0; i < extra->n_device; ++i) {
                    auto split = extra->splits[i];
                    if (!split) {
                        GGML_ASSERT(ranges[i].empty());
                        continue;
                    }
                    GGML_ASSERT(!ranges[i].empty());
                    ggml_cuda_set_device(i);
                    auto dst = (char *)split->data;
                    for (auto & p : ranges[i]) {
                        GGML_ASSERT(p.first  >= 0 && p.first < tensor->ne[1]);
                        GGML_ASSERT(p.second >= 0 && p.first + p.second <= tensor->ne[1]);
                        auto src = (const char *)data + p.first*tensor->nb[1];
                        auto size = p.second*tensor->nb[1];
                        CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                        dst += size;
                    }
                }
            } else {
                size_t cur_offset = 0;
                for (int i = 0; i < extra->n_device; ++i) {
                    auto split = extra->splits[i];
                    if (!split) continue;
                    GGML_ASSERT(split->ne[1]%n_interleave == 0);
                    ggml_cuda_set_device(i);
                    auto size = ggml_nbytes(split);
                    const char * buf_host = (const char *)data + cur_offset;
                    CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
                    cur_offset += size;
                }
            }
        }
    }
    else if (extra->split_dim == 2) {
        size_t cur_offset = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            ggml_cuda_set_device(i);
            auto size = ggml_nbytes(split);
            const char * buf_host = (const char *)data + cur_offset;
            CUDA_CHECK(cudaMemcpyAsync(split->data, buf_host, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
            cur_offset += size;
        }
    }
    else {
        fprintf(stderr, "%s: not implemented for split dim %d\n", __func__, extra->split_dim);
        GGML_ABORT("fatal error");
    }

    for (int i = 0; i < extra->n_device; ++i) {
        if (!extra->splits[i]) continue;
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    }

}

GGML_CALL static void ggml_backend_cuda_split_buffer_get_tensor([[maybe_unused]] ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
        void * data, size_t offset, size_t size) {
    // split tensors must always be read in their entirety at once
    GGML_ASSERT(offset == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    if (!tensor->extra) return;

    // Inverse of split_buffer_set_tensor; refuses paths with no defined inverse.
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    // Repacked types are block-de-interleaved by set_tensor; no runtime inverse.
    {
        const ggml_type t = tensor->type;
        const bool is_repacked =
            t == GGML_TYPE_Q4_0_R8  || t == GGML_TYPE_Q5_0_R4  || t == GGML_TYPE_Q8_0_R8  ||
            t == GGML_TYPE_Q2_K_R4  || t == GGML_TYPE_Q3_K_R4  || t == GGML_TYPE_Q4_K_R4  ||
            t == GGML_TYPE_Q5_K_R4  || t == GGML_TYPE_Q6_K_R4  || t == GGML_TYPE_IQ4_NL_R4 ||
            t == GGML_TYPE_IQ4_XS_R8 || t == GGML_TYPE_Q6_0_R4;
        if (is_repacked) {
            GGML_ABORT("%s: get_tensor of repacked type %s is not invertible",
                       __func__, ggml_type_name(t));
        }
    }

    // Explicit-ranges form (non-contiguous expert assignments) is not invertible.
    void * extra_ptr = nullptr;
    memcpy(&extra_ptr, tensor->op_params, sizeof(extra_ptr));
    if (extra_ptr) {
        GGML_ABORT("%s: get_tensor with explicit ranges is not implemented", __func__);
    }

    if (extra->split_dim < 0) {
        // Replicated: read from first present device.
        GGML_ASSERT(ggml_is_contiguous(tensor));
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(data, split->data, ggml_nbytes(tensor),
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            return;
        }
        GGML_ABORT("%s: no device holds a copy of the replicated tensor", __func__);
    }
    else if (extra->split_dim == 0) {
        // Row-split (concat along ne[0]).
        GGML_ASSERT(ggml_is_contiguous(tensor));
        auto tt = ggml_internal_get_type_traits(tensor->type);
        GGML_ASSERT(tt.row_meta_size == 0);
        std::vector<char> host_buffer;
        int64_t ne0_acc = 0;
        for (int i = 0; i < extra->n_device; ++i) {
            auto split = extra->splits[i];
            if (!split) continue;
            GGML_ASSERT(split->type == tensor->type);
            GGML_ASSERT(split->ne[0] % tt.blck_size == 0);
            const size_t split_row_size = ggml_row_size(split->type, split->ne[0]);
            const size_t dev_bytes      = (size_t)ggml_nrows(split) * split_row_size;
            if (host_buffer.size() < dev_bytes) host_buffer.resize(dev_bytes);
            ggml_cuda_set_device(i);
            CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), split->data, dev_bytes,
                                       cudaMemcpyDeviceToHost, cudaStreamPerThread));
            CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            const size_t source_offset = (ne0_acc / tt.blck_size) * tt.type_size;
            for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                for (int64_t i01 = 0; i01 < split->ne[1]; ++i01) {
                    const char * src = host_buffer.data() + (i02*split->ne[1] + i01) * split_row_size;
                    char       * dst = (char *)data + i02*tensor->nb[2] + i01*tensor->nb[1] + source_offset;
                    memcpy(dst, src, split_row_size);
                }
            }
            ne0_acc += split->ne[0];
        }
    }
    else if (extra->split_dim == 1) {
        // Column/ne[1] split.
        const size_t row_size = ggml_row_size(tensor->type, tensor->ne[0]);
        if (tensor->ne[2] > 1) {
            std::vector<char> host_buffer;
            int64_t ne1_acc = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                const size_t dev_bytes = ggml_nbytes(split);
                if (host_buffer.size() < dev_bytes) host_buffer.resize(dev_bytes);
                ggml_cuda_set_device(i);
                CUDA_CHECK(cudaMemcpyAsync(host_buffer.data(), split->data, dev_bytes,
                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
                for (int64_t i02 = 0; i02 < split->ne[2]; ++i02) {
                    const char * src = host_buffer.data() + i02 * split->ne[1] * row_size;
                    char       * dst = (char *)data + i02*tensor->nb[2] + ne1_acc*tensor->nb[1];
                    memcpy(dst, src, split->ne[1] * row_size);
                }
                ne1_acc += split->ne[1];
            }
        } else {
            size_t cur_offset = 0;
            for (int i = 0; i < extra->n_device; ++i) {
                auto split = extra->splits[i];
                if (!split) continue;
                ggml_cuda_set_device(i);
                const size_t dev_bytes = ggml_nbytes(split);
                CUDA_CHECK(cudaMemcpyAsync((char *)data + cur_offset, split->data, dev_bytes,
                                           cudaMemcpyDeviceToHost, cudaStreamPerThread));
                cur_offset += dev_bytes;
            }
            for (int i = 0; i < extra->n_device; ++i) {
                if (!extra->splits[i]) continue;
                CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
            }
        }
    }
    else {
        GGML_ABORT("%s: not implemented for split_dim %d", __func__, extra->split_dim);
    }
}

GGML_CALL static void ggml_backend_cuda_split_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_UNUSED(buffer);
    GGML_UNUSED(value);
}

static struct ggml_backend_buffer_i ggml_backend_cuda_split_buffer_interface = {
    /* .get_name        = */ ggml_backend_cuda_split_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_cuda_split_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_split_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_split_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_cuda_split_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_split_buffer_get_tensor,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_cuda_split_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda split buffer type

GGML_CALL static const char * ggml_backend_cuda_split_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Split";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_split(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_split_buffer_type_name;
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_split_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // since we don't know the exact split after rounding, we cannot allocate the device buffers at this point
    // instead, we allocate them for each tensor separately in init_tensor
    // however, the size still represents the maximum cumulative size of all the device buffers after the tensors are allocated,
    // as returned by get_alloc_size. this limit is enforced during tensor allocation by ggml-alloc, so it must be correct.
    ggml_backend_cuda_split_buffer_context * ctx = new ggml_backend_cuda_split_buffer_context();

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_split_buffer_interface, ctx, size);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_cuda_split_buffer_type_get_alloc_size([[maybe_unused]] ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    if (!tensor->extra) return 0;
    auto extra = (ggml_split_tensor_t *)tensor->extra;
    GGML_ASSERT(extra->n_device <= ggml_backend_cuda_get_device_count());

    size_t total_size = 0;
    for (int i = 0; i < extra->n_device; ++i) {
        auto split = extra->splits[i];
        if (!split) continue;
        total_size += ggml_nbytes(split);
        auto ne0 = split->ne[0];
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            auto nblock = (ne0 + MATRIX_ROW_PADDING - 1)/MATRIX_ROW_PADDING;
            auto row_size = ggml_row_size(split->type, ne0);
            auto padded_row_size = ggml_row_size(split->type, nblock*MATRIX_ROW_PADDING);
            total_size += padded_row_size - row_size;
        }
    }
    return total_size;

}

GGML_CALL static bool ggml_backend_cuda_split_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_i ggml_backend_cuda_split_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_split_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_split_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_split_buffer_type_get_alignment,
    /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
    /* .get_alloc_size   = */ ggml_backend_cuda_split_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_cuda_split_buffer_type_is_host,
};

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(const float * /*tensor_split*/) {
    static ggml_backend_buffer_type buft {
        /* .iface   = */ ggml_backend_cuda_split_buffer_type_interface,
        /* .context = */ new ggml_backend_cuda_split_buffer_type_context{}, //{tensor_split_arr},
    };
    return &buft;
}

// host buffer type

GGML_CALL static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cuda_host_buffer_name(ggml_backend_buffer_t buffer) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buffer);
}

#ifdef __linux__
GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaHostUnregister(buffer->context));
    munmap(buffer->context, buffer->size);
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    constexpr double k_warn_limit = 8.0;
    double size_GiB = size/(1024.*1024.*1024.);
    auto tim1 = ggml_time_us();
    if (size_GiB > k_warn_limit) {
        GGML_CUDA_LOG_INFO("\n\nAllocating %.2f GiB of pinned host memory, this may take a while.\n", size_GiB);
        GGML_CUDA_LOG_INFO("Using pinned host memory improves PP performance by a significant margin.\n");
        GGML_CUDA_LOG_INFO("But if it takes too long for your model and amount of patience, kill the process and run using\n\n");
        GGML_CUDA_LOG_INFO("GGML_CUDA_NO_PINNED=1 your_command_goes_here\n");
    }

    void * ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) {
        GGML_CUDA_LOG_WARN("%s: mmap of %.2f MiB failed\n", __func__, size/1024.0/1024.0);
        return nullptr;
    }

    // Whether to request the kernel to attempt to defragment memory to back the region with 2M hugepages.
    // Otherwise dependent on kernel settings:
    //   * enabled="always":  Hand over whatever 2M pages it has on hand and the rest will be 4k
    //   * enabled="madvise": 4k pages
    //   * enabled="never":   4k pages
    // Potluck on performance. If there's not much defragmentation to do, then you win. Otherwise come back in an hour.
    // Defaults to disabled unless GGML_CUDA_HOST_MALLOC_THP is set.
    if (getenv("GGML_CUDA_HOST_MALLOC_THP") != nullptr) {
        madvise(ptr, size, MADV_HUGEPAGE);
    }

    // prefault the whole region. If the kernel knows how to do this then let it do so.
    // Might be worth spawning threads to speed up this process on huge allocations.
    int needs_manual_prefault = 1;
#ifdef MADV_POPULATE_WRITE
    needs_manual_prefault = madvise(ptr, size, MADV_POPULATE_WRITE);
#endif
    if (needs_manual_prefault) {
        char * p = (char *) ptr;
        for (size_t off = 0; off < size; off += 4096) {
            p[off] = 0;
        }
    }

    cudaError_t err = cudaHostRegister(ptr, size, cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        cudaGetLastError(); // clear the error
        GGML_CUDA_LOG_WARN("%s: cudaHostRegister of %.2f MiB failed: %s\n", __func__,
                           size/1024.0/1024.0, cudaGetErrorString(err));
        munmap(ptr, size);
        return nullptr;
    }

    if (size_GiB > k_warn_limit) {
        auto tim2 = ggml_time_us();
        GGML_CUDA_LOG_INFO("    done allocating %.2f GiB in %.1f ms\n\n", size_GiB, 1e-3*(tim2-tim1));
    }
    return ptr;
}

#else // !__linux__
// WDDM limits the size of a single pinned allocation and the total pinned memory of a process.
// GGML_CUDA_PINNED_CAP_GIB caps the total; requests beyond the cap fall back to pageable memory.
static size_t g_cuda_pinned_bytes = 0;
static int    g_cuda_pinned_count = 0;

static size_t ggml_cuda_pinned_cap() {
    static const size_t cap = [] {
        const char * e = getenv("GGML_CUDA_PINNED_CAP_GIB");
        const long long v = e ? atoll(e) : 0;
        return v > 0 ? (size_t) v << 30 : SIZE_MAX;
    }();
    return cap;
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }
    if (size > ggml_cuda_pinned_cap() - g_cuda_pinned_bytes) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_CUDA_LOG_INFO("%s: pinned cap %zu GiB: %.2f MiB request refused with %.2f GiB pinned "
                               "in %d buffers; requests that do not fit stay pageable\n", __func__,
                               ggml_cuda_pinned_cap() >> 30, size / 1024.0 / 1024.0,
                               g_cuda_pinned_bytes/(1024.*1024.*1024.), g_cuda_pinned_count);
        }
        return nullptr;
    }
    constexpr double k_warn_limit = 8.0;

    void * ptr = nullptr;
    double size_GiB = size/(1024.*1024.*1024.);
    auto tim1 = ggml_time_us();
    if (size_GiB > k_warn_limit) {
        GGML_CUDA_LOG_INFO("\n\nAllocating %.2f GiB of pinned host memory, this may take a while.\n", size_GiB);
        GGML_CUDA_LOG_INFO("Using pinned host memory improves PP performance by a significant margin.\n");
        GGML_CUDA_LOG_INFO("But if it takes too long for your model and amount of patience, kill the process and run using\n\n");
        GGML_CUDA_LOG_INFO("GGML_CUDA_NO_PINNED=1 your_command_goes_here\n");
    }
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (size_GiB > k_warn_limit) {
        auto tim2 = ggml_time_us();
        GGML_CUDA_LOG_INFO("    done allocating %.2f GiB in %.1f ms\n\n", size_GiB, 1e-3*(tim2-tim1));
    }
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
        GGML_CUDA_LOG_WARN("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    g_cuda_pinned_bytes += size;
    g_cuda_pinned_count += 1;
    if (size >= ((size_t) 1 << 30)) {
        GGML_CUDA_LOG_INFO("%s: pinned %.2f MiB (buffer %d, total %.2f GiB)\n", __func__,
                           size / 1024.0 / 1024.0, g_cuda_pinned_count,
                           g_cuda_pinned_bytes/(1024.*1024.*1024.));
    }
    return ptr;
}

GGML_CALL static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
    g_cuda_pinned_bytes -= std::min(g_cuda_pinned_bytes, buffer->size);
    g_cuda_pinned_count -= g_cuda_pinned_count > 0 ? 1 : 0;
}

#endif // __linux__

// GGML_CUDA_HOST_CHUNK_GIB: split host weight buffers into pieces of at most this size,
// so that each piece can be pinned on its own
GGML_CALL static size_t ggml_backend_cuda_host_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    static const size_t chunk = [] {
        const char * e = getenv("GGML_CUDA_HOST_CHUNK_GIB");
        const long long v = e ? atoll(e) : 0;
        return v > 0 ? (size_t) v << 30 : SIZE_MAX;
    }();
    return chunk;

    GGML_UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
        // a chunked buffer must not look CPU-typed: ~llama_model() calls get_base() on those
        if (buffer != nullptr && ggml_backend_cuda_host_buffer_type_get_max_size(buft) != SIZE_MAX) {
            buffer->buft = buft;
        }
        return buffer;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cuda_host_buffer_name;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ ggml_backend_cuda_host_buffer_type_get_max_size,
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

#ifndef GGML_CUDA_PEER_MAX_BATCH_SIZE
#define GGML_CUDA_PEER_MAX_BATCH_SIZE 128
#endif // GGML_CUDA_PEER_MAX_BATCH_SIZE

#define MUL_MAT_SRC1_COL_STRIDE 128

static __global__ void mul_mat_p021_f16_f32(
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst,
    const int ncols_x, const int nrows_x, const int nchannels_x, const int nchannels_y) {

    const half * x = (const half *) vx;

    const int row_x = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / (nchannels_y / nchannels_x);

    const int nrows_y = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst = row_x;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        // x is transposed and permuted
        const int ix = row_x*nchannels_x*ncols_x + channel_x*ncols_x + col_x;
        const float xi = __half2float(x[ix]);

        const int row_y = col_x;

        // y is not transposed but permuted
        const int iy = channel*nrows_y + row_y;

        tmp += xi * y[iy];
    }

    // dst is not transposed and not permuted
    const int idst = channel*nrows_dst + row_dst;

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static __global__ void mul_mat_vec_nc_f16_f32( // nc == non-contiguous
    const void * __restrict__ vx, const float * __restrict__ y, float * __restrict__ dst, const int ncols_x, const int nrows_x,
    const int row_stride_x, const int channel_stride_x, const int channel_x_divisor) {

    const half * x = (const half *) vx;

    const int row_x     = blockDim.y*blockIdx.y + threadIdx.y;
    const int channel   = blockDim.z*blockIdx.z + threadIdx.z;
    const int channel_x = channel / channel_x_divisor;

    const int nrows_y   = ncols_x;
    const int nrows_dst = nrows_x;
    const int row_dst   = row_x;

    const int idst = channel*nrows_dst + row_dst;

    float tmp = 0.0f;

    for (int col_x0 = 0; col_x0 < ncols_x; col_x0 += blockDim.x) {
        const int col_x = col_x0 + threadIdx.x;

        if (col_x >= ncols_x) {
            break;
        }

        const int row_y = col_x;

        const int ix = channel_x*channel_stride_x + row_x*row_stride_x + col_x;
        const int iy = channel*nrows_y + row_y;

        const float xi = __half2float(x[ix]);

        tmp += xi * y[iy];
    }

    // sum up partial sums and write back result
    tmp = warp_reduce_sum(tmp);

    if (threadIdx.x == 0) {
        dst[idst] = tmp;
    }
}

static void ggml_mul_mat_p021_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x,
    const int nchannels_x, const int nchannels_y, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_p021_f16_f32<<<block_nums, block_dims, 0, stream>>>(vx, y, dst, ncols_x, nrows_x, nchannels_x, nchannels_y);
}

static void ggml_mul_mat_vec_nc_f16_f32_cuda(
    const void * vx, const float * y, float * dst, const int ncols_x, const int nrows_x, const int row_stride_x,
    const int nchannels_x, const int nchannels_y, const int channel_stride_x, cudaStream_t stream) {

    const dim3 block_nums(1, nrows_x, nchannels_y);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    mul_mat_vec_nc_f16_f32<<<block_nums, block_dims, 0, stream>>>
        (vx, y, dst, ncols_x, nrows_x, row_stride_x, channel_stride_x, nchannels_y/nchannels_x);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    GGML_ASSERT(ggml_backend_buffer_is_cuda(src->buffer));
    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t rs = ggml_row_size(type, ne0);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == rs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        // TODO: this only works if the row does not contain meta data
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*rs);
            // pretend the row is a matrix with cols=1
            // TODO: this only works if the row does not contain meta data
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

static void ggml_cuda_op_mul_mat_cublas(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0_dd_i  != nullptr);
    GGML_ASSERT(src1_ddf_i != nullptr);
    GGML_ASSERT(dst_dd_i   != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne10 = src1->ne[0];

    const int64_t ne0 = dst->ne[0];

    const int64_t row_diff = row_high - row_low;

    int id = ggml_cuda_get_device();

    // the main device has a larger memory buffer to hold the results from all GPUs
    // ldc == nrows of the matrix that cuBLAS writes into
    int64_t ldc = id == ctx.device ? ne0 : row_diff;

    const int compute_capability = ggml_cuda_info().devices[id].cc;

    if (src0->type == GGML_TYPE_BF16 && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {

        ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
        if (src1->type != GGML_TYPE_BF16) {
            const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
            GGML_ASSERT(to_bf16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_bf16.alloc(ne);
            to_bf16_cuda(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
        }
        const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
        const nv_bfloat16 * src0_ptr = (const nv_bfloat16 *)src0_dd_i;
        ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

        const float alpha_f32 = 1.0f;
        const float beta_f32  = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                                 src1_ptr,       CUDA_R_16BF, ne10,
                    &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
        to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
        return;
    }

#ifdef GGML_CUDA_IQK_FORCE_BF16
    if (ggml_is_quantized(src0->type) && ggml_is_contiguous(src0) && row_diff == src0->ne[1]) {
        to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src0->type);
        to_bf16_cuda_t to_bf16_cuda_1 = src1->type != GGML_TYPE_BF16 ? ggml_get_to_bf16_cuda(src1->type) : nullptr;
        if (to_bf16_cuda && (src1->type == GGML_TYPE_BF16 || to_bf16_cuda_1)) {
            size_t ne = row_diff*ne00;
            ggml_cuda_pool_alloc<nv_bfloat16> src0_as_bf16(ctx.pool(id), ne);
            to_bf16_cuda(src0_dd_i, src0_as_bf16.get(), row_diff, ne00, stream);

            ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
            if (src1->type != GGML_TYPE_BF16) {
                size_t ne = src1_ncols*ne10;
                src1_as_bf16.alloc(ne);
                to_bf16_cuda_1(src1_ddf_i, src1_as_bf16.get(), src1_ncols, ne10, stream);
            }
            const nv_bfloat16 * src1_ptr = src1->type == GGML_TYPE_BF16 ? (const nv_bfloat16 *) src1_ddf_i : src1_as_bf16.get();
            const nv_bfloat16 * src0_ptr = src0_as_bf16.get();

            ggml_cuda_pool_alloc<nv_bfloat16> dst_bf16(ctx.pool(id), row_diff*src1_ncols);

            const float alpha_f32 = 1.0f;
            const float beta_f32  = 0.0f;

            CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
            CUBLAS_CHECK(
                    cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                        row_diff, src1_ncols, ne10,
                        &alpha_f32,  src0_ptr,       CUDA_R_16BF, ne00,
                        src1_ptr,       CUDA_R_16BF, ne10,
                        &beta_f32,   dst_bf16.get(), CUDA_R_16BF, ldc,
                        CUBLAS_COMPUTE_32F,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
            to_fp32_cuda(dst_bf16.get(), dst_dd_i, row_diff, src1_ncols, stream);
            return;
        }
    }
#endif

    if (compute_capability >= CC_VOLTA && (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || ggml_is_quantized(src0->type)) && ggml_is_contiguous(src0) && row_diff == src0->ne[1] && dst->op_params[0] == GGML_PREC_DEFAULT) {
        // convert src0 and src1 to fp16, multiply as fp16, convert dst to fp32
        ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
        if (src0->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = row_diff*ne00;
            src0_as_f16.alloc(ne);
            to_fp16_cuda(src0_dd_i, src0_as_f16.get(), row_diff, ne00, stream);
        }
        const half * src0_ptr = src0->type == GGML_TYPE_F16 ? (const half *) src0_dd_i : src0_as_f16.get();

        ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = src1_ncols*ne10;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_ddf_i, src1_as_f16.get(), src1_ncols, ne10, stream);
        }
        const half * src1_ptr = src1->type == GGML_TYPE_F16 ? (const half *) src1_ddf_i : src1_as_f16.get();

        ggml_cuda_pool_alloc<half> dst_f16(ctx.pool(id), row_diff*src1_ncols);

        const half alpha_f16 = 1.0f;
        const half beta_f16 = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasGemmEx(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha_f16, src0_ptr,       CUDA_R_16F, ne00,
                                src1_ptr,       CUDA_R_16F, ne10,
                    &beta_f16,   dst_f16.get(), CUDA_R_16F, ldc,
                    CUBLAS_COMPUTE_16F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));

        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        to_fp32_cuda(dst_f16.get(), dst_dd_i, row_diff, src1_ncols, stream);
    } else {
        ggml_cuda_pool_alloc<float> src0_ddq_as_f32(ctx.pool(id));
        ggml_cuda_pool_alloc<float> src1_ddq_as_f32(ctx.pool(id));

        if (src0->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src0_ddq_as_f32.alloc(row_diff*ne00);
            to_fp32_cuda(src0_dd_i, src0_ddq_as_f32.get(), row_diff, ne00, stream);
        }
        if (src1->type != GGML_TYPE_F32) {
            const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src1->type);
            GGML_ASSERT(to_fp32_cuda != nullptr);
            src1_ddq_as_f32.alloc(src1_ncols*ne10);
            to_fp32_cuda(src1_ddf_i, src1_ddq_as_f32.get(), src1_ncols, ne10, stream);
        }

        const float * src0_ddf_i = src0->type == GGML_TYPE_F32 ? (const float *) src0_dd_i : src0_ddq_as_f32.get();
        const float * src1_ddf1_i = src1->type == GGML_TYPE_F32 ? (const float *) src1_ddf_i : src1_ddq_as_f32.get();

        const float alpha = 1.0f;
        const float beta = 0.0f;

        CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
        CUBLAS_CHECK(
            cublasSgemm(ctx.cublas_handle(id), CUBLAS_OP_T, CUBLAS_OP_N,
                    row_diff, src1_ncols, ne10,
                    &alpha, src0_ddf_i,  ne00,
                            src1_ddf1_i, ne10,
                    &beta,  dst_dd_i,    ldc));
    }

    GGML_UNUSED(dst);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(src1_padded_row_size);
}

static bool ggml_cuda_set_peer_access(int main_device) {
    ggml_cuda_set_device(main_device);

    bool all_enabled = true;
    for (int id_other = 0; id_other < ggml_backend_cuda_get_device_count(); ++id_other) {
        if (main_device == id_other) {
            continue;
        }

        int can_access_peer;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, main_device, id_other));
        if (can_access_peer) {
//~ #ifdef NDEBUG
            GGML_CUDA_LOG_INFO(" =========================== %s: Enabling Peer Access between Devices %d->%d\n", __func__, main_device, id_other);
//~ #endif //NDEBUG
            cudaError_t err = cudaDeviceEnablePeerAccess(id_other, 0);
            if (err != cudaErrorPeerAccessAlreadyEnabled) {
                CUDA_CHECK(err);
            } else {
                // reset the error
                (void)cudaGetLastError();
            }
        } else {
            all_enabled = false;
        }
    }
    return all_enabled;
}

static cudaError_t ggml_cuda_Memcpy2DPeerAsync(
    void * dst, int dstDevice, size_t dpitch, void * src, int srcDevice, size_t spitch, size_t width, size_t height, cudaStream_t stream) {

#if !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
    // cudaMemcpy2DAsync may fail with copies between vmm pools of different devices
    cudaMemcpy3DPeerParms p = {};
    p.dstDevice = dstDevice;
    p.dstPtr = make_cudaPitchedPtr(dst, dpitch, dpitch, height);
    p.srcDevice = srcDevice;
    p.srcPtr = make_cudaPitchedPtr(src, spitch, spitch, height);
    p.extent = make_cudaExtent(width, height, 1);
    return cudaMemcpy3DPeerAsync(&p, stream);
#else
    // HIP does not support cudaMemcpy3DPeerAsync or vmm pools
    GGML_UNUSED(dstDevice);
    GGML_UNUSED(srcDevice);
    return cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, stream);
#endif // !defined(GGML_USE_HIPBLAS) && !defined(GGML_USE_MUSA)
}

static void ggml_cuda_op_mul_mat(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_cuda_op_mul_mat_t op,
    quantize_cuda_t quantize_src1) {

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne03 = src0->ne[3];

    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    const int64_t nrows1 = ggml_nrows(src1);

    GGML_ASSERT(ne03 == ne13);

    const int64_t ne0 = dst->ne[0];
    const int64_t ne1 = dst->ne[1];

    const int64_t nb2 = dst->nb[2];
    const int64_t nb3 = dst->nb[3];

    GGML_ASSERT(ggml_backend_buffer_is_cuda(dst->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src1->buffer));
    ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
    ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;

    GGML_ASSERT(src1->type == GGML_TYPE_F32 || (src1->ne[2] == 1 && src1->ne[3] == 1));

    GGML_ASSERT(ne12 >= ne02 && ne12 % ne02 == 0);

    const int64_t i02_divisor = ne12 / ne02;

    const size_t src0_rs = ggml_row_size(src0->type, ne00);
    const size_t q8_1_ts = sizeof(block_q8_1);
    const size_t q8_1_bs = QK8_1;

    const bool src0_is_contiguous = ggml_is_contiguous(src0);
    const bool src1_is_contiguous = ggml_is_contiguous(src1);

    const int64_t src1_padded_col_size = GGML_PAD(ne10, MATRIX_ROW_PADDING);

    struct dev_data {
        int cc;

        ggml_cuda_pool_alloc<char>   src0_dd_alloc;
        ggml_cuda_pool_alloc<float> src1_ddf_alloc;
        ggml_cuda_pool_alloc<char>  src1_ddq_alloc;
        ggml_cuda_pool_alloc<float>   dst_dd_alloc;

        char  *  src0_dd = nullptr;
        float * src1_ddf = nullptr; // float
        char  * src1_ddq = nullptr; // q8_1
        float *   dst_dd = nullptr;

        int64_t  row_low;
        int64_t row_high;
    };

    dev_data dev[GGML_CUDA_MAX_DEVICES];

    int used_devices = 0;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        dev[id].cc = ggml_cuda_info().devices[id].cc;

        // by default, use all rows
        dev[id].row_low  = 0;
        dev[id].row_high = ne01;

    }

    bool quantization_done = false;

    for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
        if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
            continue;
        }

        used_devices++;

        const bool src1_on_device = id == src1_ctx->device;
        const bool  dst_on_device = id == dst_ctx->device;

        ggml_cuda_set_device(id);
        cudaStream_t stream = ctx.stream(id, 0);

        if (src0_is_contiguous) {
            dev[id].src0_dd = (char *) src0->data;
        } else {
            // If src0 is not contiguous it will be copied to a temporary buffer, it may then be necessary to clear padding.
            const size_t nbytes_data    = ggml_nbytes(src0);
            const size_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            dev[id].src0_dd = dev[id].src0_dd_alloc.alloc(ctx.pool(id), nbytes_data + nbytes_padding);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd, 0, nbytes_data + nbytes_padding, stream));
        }

        // If src0 is on a temporary compute buffer (partial offloading) there may be some padding that needs to be cleared:
        if (ne00 % MATRIX_ROW_PADDING != 0 && ggml_is_quantized(src0->type) && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE && src0->view_src == nullptr) {
            const int64_t nbytes_data    = ggml_nbytes(src0);
            const int64_t nbytes_padding = ggml_row_size(src0->type, MATRIX_ROW_PADDING - ne00 % MATRIX_ROW_PADDING);
            CUDA_CHECK(cudaMemsetAsync(dev[id].src0_dd + nbytes_data , 0, nbytes_padding, stream));
        }

        if (src1_on_device && src1_is_contiguous) {
            dev[id].src1_ddf = (float *) src1->data;
        } else {
            dev[id].src1_ddf = dev[id].src1_ddf_alloc.alloc(ctx.pool(id), ggml_nelements(src1));
        }

        if (quantize_src1) {
            size_t src_1_ddq_size = nrows1*src1_padded_col_size*q8_1_ts/q8_1_bs;
            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                src_1_ddq_size += get_mmq_x_max_host(dev[id].cc)*sizeof(block_q8_1_mmq);
            }
            dev[id].src1_ddq = dev[id].src1_ddq_alloc.alloc(ctx.pool(id), src_1_ddq_size);

            if (src1_on_device && (src1_is_contiguous || (src1->ne[1] == 1 && src1->ne[3] == 1 && src1->nb[0] == sizeof(float)))) {
                if (src1_is_contiguous) {
                    quantize_src1(dev[id].src1_ddf, dev[id].src1_ddq, ne10, ne11, ne12*ne13, src1_padded_col_size, src0->type, stream);
                } else {
                    //printf("Calling quantize_tensor_q8_1_cuda for %s\n", src0->name);
                    quantize_tensor_q8_1_cuda(src1, dev[id].src1_ddq, src0->type, stream);
                }
                CUDA_CHECK(cudaGetLastError());
                quantization_done = true;
            }
        }

        if (dst_on_device) {
            dev[id].dst_dd = (float *) dst->data;
        } else {
            const size_t size_dst_ddf = ggml_nelements(dst);
            dev[id].dst_dd = dev[id].dst_dd_alloc.alloc(ctx.pool(id), size_dst_ddf);
        }
    }

    const int64_t src1_col_stride = ne11;
    // split-buffer src0 has data == NULL; per-device dispatch happens in the slow path below.
    const bool src0_is_split = src0->buffer && ggml_backend_buft_is_cuda_split(src0->buffer->buft);
    if (quantization_done && ne11 == 1 && ne12 > 1 && ne13 == 1 && ne02 == ne12 && ne02 == dst->ne[2] && !src0_is_split) {
        int id = ctx.device;
        char  *  src0_dd_i =  dev[id].src0_dd;
        float * src1_ddf_i = dev[id].src1_ddf;
        char  * src1_ddq_i = dev[id].src1_ddq;
        float *   dst_dd_i =   dev[id].dst_dd;
        cudaStream_t stream = ctx.stream(id, 0);
        ggml_cuda_op_mul_mat_vec_q_3D(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                dev[id].row_low, dev[id].row_high, ne11, src1_padded_col_size, stream);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    for (int64_t src1_col_0 = 0; src1_col_0 < ne11; src1_col_0 += src1_col_stride) {
        const int64_t is = 0;
        const int64_t src1_ncols = src1_col_0 + src1_col_stride > ne11 ? ne11 - src1_col_0 : src1_col_stride;

        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            if (id != ctx.device || dev[id].row_low == dev[id].row_high) {
                continue;
            }

            const bool src1_on_device = id == src1_ctx->device;
            const bool  dst_on_device = id == dst_ctx->device;
            const int64_t row_diff = dev[id].row_high - dev[id].row_low;

            ggml_cuda_set_device(id);
            cudaStream_t stream = ctx.stream(id, is);

            for (int64_t i0 = 0; i0 < ne13*ne12; ++i0) {
                const int64_t i03 = i0 / ne12;
                const int64_t i02 = i0 % ne12;

                size_t src1_ddq_i_offset = i0*ne11 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                    src1_ddq_i_offset += src1_col_0 * sizeof(block_q8_1_mmq);
                } else {
                    src1_ddq_i_offset += src1_col_0 * src1_padded_col_size*q8_1_ts/q8_1_bs;
                }

                // for split tensors the data begins at i0 == i0_offset_low
                char  *  src0_dd_i =  dev[id].src0_dd + (i0/i02_divisor) * ne01*src0_rs;
                float * src1_ddf_i = dev[id].src1_ddf + (i0*ne11 + src1_col_0) * ne10;
                char  * src1_ddq_i = dev[id].src1_ddq +  src1_ddq_i_offset;
                float *   dst_dd_i =   dev[id].dst_dd + (i0*ne1  + src1_col_0) * (dst_on_device ? ne0 : row_diff);

                // the main device memory buffer can be on VRAM scratch, with space for all partial results
                // in that case an offset on dst_ddf_i is needed
                if (id == ctx.device) {
                    dst_dd_i += dev[id].row_low; // offset is 0 if no tensor split
                }

                // copy src0, src1 to device if necessary
                if (src1_is_contiguous) {
                    if (id != ctx.device) {
                        if (quantize_src1) {
                            char * src1_ddq_i_source = dev[ctx.device].src1_ddq + src1_ddq_i_offset;
                            if (quantize_src1 == quantize_mmq_q8_1_cuda) {
                                const size_t pitch = ne11*sizeof(block_q8_1_mmq);
                                const size_t width = src1_ncols*sizeof(block_q8_1_mmq);
                                const size_t height = src1_padded_col_size/(4*QK8_1);
                                CUDA_CHECK(ggml_cuda_Memcpy2DPeerAsync(src1_ddq_i, id, pitch, src1_ddq_i_source, ctx.device, pitch, width, height, stream));
                            } else {
                                CUDA_CHECK(cudaMemcpyPeerAsync(
                                    src1_ddq_i, id, src1_ddq_i_source, ctx.device, src1_ncols*src1_padded_col_size*q8_1_ts/q8_1_bs, stream));
                            }
                        } else {
                            float * src1_ddf_i_source = (float *) src1->data;
                            src1_ddf_i_source += (i0*ne11 + src1_col_0) * ne10;
                            CUDA_CHECK(cudaMemcpyPeerAsync(src1_ddf_i, id, src1_ddf_i_source, ctx.device,
                                                            src1_ncols*ne10*sizeof(float), stream));
                        }
                    }
                } else if (src1_on_device && !src1_is_contiguous) {
                    if (!quantization_done) {
                        //printf("Copying %s\n", src1->name);
                        CUDA_CHECK(ggml_cuda_cpy_tensor_2d(
                                    src1_ddf_i, src1, i03, i02, src1_col_0, src1_col_0+src1_ncols, stream));
                    }
                } else {
                    GGML_ABORT("fatal error");
                }

                if (quantize_src1 && !src1_is_contiguous && !quantization_done) {
                    //printf("Quantizing %s\n", src1->name);
                    quantize_src1(src1_ddf_i, src1_ddq_i, ne10, src1_ncols, 1, src1_padded_col_size, src0->type, stream);
                    CUDA_CHECK(cudaGetLastError());
                }

                if (src1_col_0 == 0 && !src0_is_contiguous && i02 % i02_divisor == 0) {
                    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(src0_dd_i, src0, i03, i02/i02_divisor, dev[id].row_low, dev[id].row_high, stream));
                }

                // do the computation
                op(ctx, src0, src1, dst, src0_dd_i, src1_ddf_i, src1_ddq_i, dst_dd_i,
                    dev[id].row_low, dev[id].row_high, src1_ncols, src1_padded_col_size, stream);
                CUDA_CHECK(cudaGetLastError());

                // copy dst to host or other device if necessary
                if (!dst_on_device) {
                    void * dst_off_device = dst->data;
                    float * dhf_dst_i = (float *) ((char *) dst_off_device + i02*nb2 + i03*nb3);
                    GGML_ASSERT(dst->nb[1] == ne0*sizeof(float));
                    dhf_dst_i += src1_col_0*ne0;
                    CUDA_CHECK(cudaMemcpyAsync(dhf_dst_i, dst_dd_i, src1_ncols*ne0*sizeof(float), cudaMemcpyDeviceToDevice, stream));
                }

            }
        }
    }
}

static void ggml_cuda_mul_mat_vec_p021(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_permuted(src0) && ggml_is_permuted(src1));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->nb[0] <= src0->nb[1] && src0->nb[2] <= src0->nb[3]); // 0213 permutation
    GGML_ASSERT(src1->nb[0] <= src1->nb[1] && src1->nb[2] <= src1->nb[3]); // 0213 permutation
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    ggml_mul_mat_p021_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, ne02, ne12, main_stream);
}

static void ggml_cuda_mul_mat_vec_nc(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(!ggml_is_permuted(src0));
    GGML_ASSERT(ggml_backend_buffer_is_cuda(src0->buffer));
    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];

    const int64_t nb01 = src0->nb[1];
    const int64_t nb02 = src0->nb[2];

    const int64_t ne12 = src1->ne[2];

    cudaStream_t main_stream = ctx.stream();

    void  * src0_ddq = src0->data;
    float * src1_ddf = (float *) src1->data;
    float * dst_ddf  = (float *) dst->data;

    const int64_t row_stride_x = nb01 / sizeof(half);
    const int64_t channel_stride_x = nb02 / sizeof(half);

    ggml_mul_mat_vec_nc_f16_f32_cuda(src0_ddq, src1_ddf, dst_ddf, ne00, ne01, row_stride_x, ne02, ne12, channel_stride_x, main_stream);
}

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto get_nc_converter(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

template<ggml_type src0_type>
static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<src0_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(!ggml_is_transposed(src0));
    GGML_ASSERT(!ggml_is_transposed(src1));
    GGML_ASSERT(src0->type == src0_type);
    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(), main_stream));

    float * dst_ddf = (float *) dst->data;
    const size_t ts_src1 = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == ts_src1);
    int64_t s11 = nb11 / ts_src1;
    int64_t s12 = nb12 / ts_src1;
    int64_t s13 = nb13 / ts_src1;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    // Handle src0
    src0_ptr = (const cuda_t *) src0->data;

    // Handle src1 - convert if necessary
    if (src1->type == src0_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else {
        // Convert src1 to target type using traits conversion functions
        const int64_t ne_src1 = ggml_nelements(src1);
        src1_alloc.alloc(ne_src1);

        const auto convert_func = traits::get_nc_converter(src1->type);
        GGML_ASSERT(convert_func != nullptr);
        convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
        src1_ptr = src1_alloc.get();
        s11 = ne10;
        s12 = ne11*s11;
        s13 = ne12*s12;

        is_src1_cont_2 = true;
    }

    // Setup destination buffer
    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_t;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();
    const float alpha_f32 = 1.0f;
    const float beta_f32 = 0.0f;

    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
        if constexpr (src0_type == GGML_TYPE_F32) {
            dst_t = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_t = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    } else {
        dst_t = (char *) dst_ddf;
        cu_compute_type = CUBLAS_COMPUTE_32F;
        cu_data_type = CUDA_R_32F;
        alpha = &alpha_f32;
        beta = &beta_f32;
    }

    int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        //printf("Using cublasGemmStridedBatchedEx for %s\n", dst->name);
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? nb03/nb00 : nb02/nb00;
        //const int64_t smb = ne12 == 1 ? s13       : s12;
        const int64_t smb = ne12 == 1 ? nb13/nb10 : nb12/nb10;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, nb01/nb00, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11,       smb,     // strideB
                beta,     dst_t, cu_data_type,   ne0,       ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        //printf("Using cublasGemmBatchedEx for %s\n", dst->name);
        //printf("    src0: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n",src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        //printf("    src1: %ld x %ld x %ld x %ld; %zu x %zu x %zu x %zu\n",src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3], src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        size_t src1_stride_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        dim3 block_dims(threads_x, threads_y);

        dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_t,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                nb02, nb03,
                (src1->type == src0_type) ? nb12 : s12*src1_stride_size,
                (src1->type == src0_type) ? nb13 : s13*src1_stride_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(ctx.cublas_handle(), CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, nb01/nb00,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (dst->op_params[0] == GGML_PREC_DEFAULT && cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, 1, main_stream);
    }
}

static void ggml_cuda_mul_mat_batched_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_batched_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("Unsupported type");
    }
}

static int ggml_cuda_mul_mat_q(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n, bool is_gemv) {

    auto stream = ctx.stream();

    auto fusion = ctx.fusion && src1->ne[1] == 1;

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*ggml_nrows(src1);
    if (!is_gemv) {
        quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    if (is_gemv) {
        quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], src1->ne[2], ne10_padded,
                src0->type, stream);
        CUDA_CHECK(cudaGetLastError());

        // The code below handles the case when Q, K, V have a bias applied after the respective matrix multiplication.
        // In that case the graph contains mul_mat(Q) -> mul_mat(K) -> mul_mat(V) -> add(Q) -> add(K) -> add(V)
        if (fusion && cgraph && node_n + 5 < cgraph->n_nodes &&
            cgraph->nodes[node_n+1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[node_n+2]->op == GGML_OP_MUL_MAT &&
            ggml_is_quantized(cgraph->nodes[node_n+1]->src[0]->type) &&
            ggml_is_quantized(cgraph->nodes[node_n+2]->src[0]->type) &&
            cgraph->nodes[node_n+3]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+4]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+5]->op == GGML_OP_ADD &&
            cgraph->nodes[node_n+0] == cgraph->nodes[node_n+3]->src[0] &&
            cgraph->nodes[node_n+1] == cgraph->nodes[node_n+4]->src[0] &&
            cgraph->nodes[node_n+2] == cgraph->nodes[node_n+5]->src[0]) {
            for (int i = 0; i < 3; ++i) {
                auto src0_i = cgraph->nodes[node_n+i]->src[0];
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, src0_i, src1, cgraph->nodes[node_n+i], cgraph->nodes[node_n+i+3]->src[1],
                        (const char *)src0_i->data, nullptr, src1_quantized.get(), (float *)cgraph->nodes[node_n+i]->data,
                        0, src0_i->ne[1], src1->ne[1], ne10_padded, stream);
                CUDA_CHECK(cudaGetLastError());
            }
            node_n += 5;
        } else if (fusion && cgraph && node_n + 1 < cgraph->n_nodes &&
                   cgraph->nodes[node_n+1]->op == GGML_OP_ADD &&
                   dst == cgraph->nodes[node_n+1]->src[0] &&
                   dst->ne[0] == cgraph->nodes[node_n+1]->src[1]->ne[0] &&
                   cgraph->nodes[node_n+1]->src[1]->type == GGML_TYPE_F32 &&
                   ggml_nrows(cgraph->nodes[node_n+1]->src[1]) == 1) {
            // We have a bias applied after the matrix multiplication and we can fuse it
            ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+1], cgraph->nodes[node_n+1]->src[1],
                 (const char *)dst->src[0]->data, nullptr, src1_quantized.get(), (float *)cgraph->nodes[node_n+1]->data,
                 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            ++node_n;
        } else {
            ggml_cuda_op_mul_mat_vec_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                    0, src0->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded, src0->type, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_q(ctx, src0, src1, dst, (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    if (!cgraph) return node_n;

    while (node_n + 1 < cgraph->n_nodes) {
        dst = cgraph->nodes[node_n+1];
        if (ggml_is_empty(dst) || dst->op == GGML_OP_RESHAPE || dst->op == GGML_OP_TRANSPOSE || dst->op == GGML_OP_VIEW
                               || dst->op == GGML_OP_PERMUTE || dst->op == GGML_OP_NONE) {
            ++node_n; continue;
        }
        if (dst->op != GGML_OP_MUL_MAT || dst->src[1] != src1 || !ggml_is_quantized(dst->src[0]->type) ||
                !ggml_cuda_should_use_mmq(dst->src[0]->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[1])) break;
        if (!is_gemv && mmq_get_q8_1_ds_layout(src0->type) != mmq_get_q8_1_ds_layout(dst->src[0]->type)) break;
        if (is_gemv) {
            if (fusion && node_n + 2 < cgraph->n_nodes &&
                cgraph->nodes[node_n+2]->op == GGML_OP_ADD &&
                dst == cgraph->nodes[node_n+2]->src[0] &&
                dst->ne[0] == cgraph->nodes[node_n+2]->src[1]->ne[0] &&
                cgraph->nodes[node_n+2]->src[1]->type == GGML_TYPE_F32 &&
                ggml_nrows(cgraph->nodes[node_n+2]->src[1]) == 1) {
                // We have a bias applied after the matrix multiplication and we can fuse it
                ggml_cuda_op_mul_mat_vec_q_biased(ctx, dst->src[0], src1, cgraph->nodes[node_n+2], cgraph->nodes[node_n+2]->src[1],
                        (const char *)dst->src[0]->data, nullptr, src1_quantized.get(), (float *)cgraph->nodes[node_n+2]->data,
                        0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
                ++node_n;
            } else {
                ggml_cuda_op_mul_mat_vec_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, src1_quantized.get(),
                        (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
            }
        } else {
            ggml_cuda_op_mul_mat_q(ctx, dst->src[0], src1, dst, (const char *)dst->src[0]->data, nullptr, src1_quantized.get(),
                    (float *)dst->data, 0, dst->src[0]->ne[1], src1->ne[1], ne10_padded, stream);
        }
        CUDA_CHECK(cudaGetLastError());
        ++node_n;
    }

    return node_n;

}

template <typename src_t, int block_size = 256>
static __global__ void mul_mat_row(int n, const src_t * x, const float * y, float * z) {
    float sum = 0;
    for (int i = threadIdx.x; i < n; i += block_size) {
        float xi = ggml_cuda_cast<float, src_t>(x[i]);
        sum += xi * y[i];
    }
    sum = warp_reduce_sum(sum);
    if constexpr (block_size > WARP_SIZE) {
        __shared__ float tmp[block_size/WARP_SIZE];
        if (threadIdx.x % WARP_SIZE == 0) {
            tmp[threadIdx.x / WARP_SIZE] = sum;
        }
        __syncthreads();
        sum = threadIdx.x < block_size / WARP_SIZE ? tmp[threadIdx.x] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    if (threadIdx.x == 0) {
        z[0] = sum;
    }
}

static void mul_mat_1row(const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, ggml_backend_cuda_context & ctx) {
    constexpr int kBlockSize = 256;
    GGML_ASSERT(src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    if (src0->type == GGML_TYPE_F16) {
        mul_mat_row<<<1, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const half *)src0->data, (const float *)src1->data, (float *)dst->data);
    }
    else if (src0->type == GGML_TYPE_BF16) {
        mul_mat_row<<<1, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const nv_bfloat16 *)src0->data, (const float *)src1->data, (float *)dst->data);
    }
    else if (src0->type == GGML_TYPE_F32) {
        mul_mat_row<<<1, kBlockSize, 0, ctx.stream()>>>((int)src0->ne[0], (const float *)src0->data, (const float *)src1->data, (float *)dst->data);
    }
    else {
        GGML_ABORT("Fatal error");
    }
}

static __global__ void k_simple_gemm_f32(int n, int nx, const float * x, const float * y, float * z, size_t nb01, size_t nb11) {
    int ix = blockIdx.x;
    int iy = blockIdx.y;
    x += nb01 * ix;
    y += nb11 * iy;
    float sum = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        sum += x[j]*y[j];
    }
    sum = warp_reduce_sum(sum);
    __shared__ float tmp[32];
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane_id == 0) tmp[warp_id] = sum;
    __syncthreads();
    sum = tmp[lane_id];
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        z[iy*nx + ix] = sum;
    }
}

static int ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cgraph * cgraph, int node_n) {

    if (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[0] >= 1024 && src0->ne[2]*src0->ne[3] == 1 && src1->ne[2]*src1->ne[3] == 1 && src1->ne[1] <= 8) {
        auto nsm = ggml_cuda_info().devices[ctx.device].nsm;
        if (src0->ne[1] * src1->ne[1] <= nsm) {
            dim3 grid(src0->ne[1], src1->ne[1], 1);
            k_simple_gemm_f32<<<grid, 1024, 0, ctx.stream()>>>(src0->ne[0], src0->ne[1], (const float *)src0->data, (const float *)src1->data, (float *)dst->data,
                    src0->nb[1]/sizeof(float), src1->nb[1]/sizeof(float));
            return node_n;
        }
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_q
    // (mul_mat_vec_q does not read past the end of a row).
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS instead of mul_mat_q.
    const size_t src0_nbytes = ggml_nbytes(src0);
    const size_t src0_alloc  = ggml_backend_buffer_get_alloc_size(src0->buffer, src0);
    const bool   src0_padded = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && src0_nbytes != src0_alloc;
    const bool bad_padding_clear = src0_padded && src0->view_src;

    const int cc = ggml_cuda_info().devices[ctx.device].cc;

    bool use_dequantize_mul_mat_vec = ggml_cuda_dmmv_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src0->ne[0] % (GGML_CUDA_DMMV_X*2) == 0 && src1->ne[1] == 1;
    bool          use_mul_mat_vec_q =  ggml_is_quantized(src0->type)
        && ggml_cuda_mmvq_type_supported(src0->type)
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32
        && src1->ne[1] <= ggml_cuda_mmvq_max_batch_size(cc);
    bool              use_mul_mat_q =  ggml_is_quantized(src0->type) && !bad_padding_clear
        && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    // if mmvq is available it's a better choice than dmmv:
#ifndef GGML_CUDA_FORCE_DMMV
    use_dequantize_mul_mat_vec = use_dequantize_mul_mat_vec && !use_mul_mat_vec_q;
#endif // GGML_CUDA_FORCE_DMMV

    bool any_gpus_with_slow_fp16 = false;

    use_mul_mat_q           = use_mul_mat_q           && ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[1]);
    any_gpus_with_slow_fp16 = any_gpus_with_slow_fp16 || !fast_fp16_available(cc);

    if ((use_mul_mat_vec_q || use_mul_mat_q) && src1->ne[2]*src1->ne[3] == 1) {
        // This return does not go through ggml_cuda_op_mul_mat, which is where the padding is otherwise cleared.
        if (src0_padded && !use_mul_mat_vec_q) {
            CUDA_CHECK(cudaMemsetAsync((char *) src0->data + src0_nbytes, 0, src0_alloc - src0_nbytes, ctx.stream()));
        }
        return ggml_cuda_mul_mat_q(ctx, src0, src1, dst, cgraph, node_n, use_mul_mat_vec_q);
    }

    if (ggml_nrows(src0) == 1 && ggml_nrows(src1) == 1) { // && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32) {
        mul_mat_1row(src0, src1, dst, ctx);
        return node_n;
    }
    bool debug = false; //src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_F32;

    // debug helpers
    //printf("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
    //printf("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
    //printf("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
    //printf("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
    //printf("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);

    if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && ggml_is_permuted(src0) && ggml_is_permuted(src1) && src1->ne[1] == 1) {
        if (debug) printf("%s(%s): using ggml_cuda_mul_mat_vec_p021\n", __func__, dst->name);
        // FP32 precision KQ single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_p021(ctx, src0, src1, dst);
    } else if (any_gpus_with_slow_fp16 && src0->type == GGML_TYPE_F16 && !ggml_is_contiguous(src0) && !ggml_is_transposed(src1) && src1->ne[1] == 1) {
        if (debug) printf("%s(%s): using ggml_cuda_mul_mat_vec_nc\n", __func__, dst->name);
        // FP32 precision KQV single-batch for batch size 1 without FlashAttention
        ggml_cuda_mul_mat_vec_nc(ctx, src0, src1, dst);
    } else if ((src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32) && (src1->type == src0->type || !any_gpus_with_slow_fp16)
               && !ggml_is_transposed(src0) && !ggml_is_transposed(src1) && src1->ne[2]*src1->ne[3] > 1) {
        if (debug) printf("%s(%s): ggml_cuda_mul_mat_batched_cublas\n", __func__, dst->name);
        // KQ + KQV multi-batch without FlashAttention
        ggml_cuda_mul_mat_batched_cublas(ctx, src0, src1, dst);
    } else if (use_dequantize_mul_mat_vec) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_dequantize_mul_mat_vec)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_dequantize_mul_mat_vec, nullptr);
    } else if (use_mul_mat_vec_q) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_vec_q)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_vec_q, quantize_row_q8_1_cuda);
    } else if (use_mul_mat_q) {
        if (debug) printf("%s(%s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_q)\n", __func__, dst->name);
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_q, quantize_mmq_q8_1_cuda);
    } else {
        if (debug) printf("%s(%s, %s): ggml_cuda_op_mul_mat(ggml_cuda_op_mul_mat_cublas)\n", __func__, dst->name, ggml_type_name(src0->type));
        ggml_cuda_op_mul_mat(ctx, src0, src1, dst, ggml_cuda_op_mul_mat_cublas, nullptr);
    }
    return node_n;
}

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

template <typename data_t = float>
static __global__ void k_copy_src_to_contiguous(const char * __restrict__ src_original, char * __restrict__ src_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne10, int64_t ne11, size_t nb11, size_t nb12) {
    int32_t i = blockIdx.x;

    const int32_t i11 = row_mapping[i].i1 % ne11;
    const int32_t i12 = row_mapping[i].i2;

    data_t * src_row_contiguous = (data_t *)(src_contiguous + i*nb11);
    const data_t * src_row_original = (const data_t *)(src_original + i11*nb11 + i12*nb12);

    for (int j = threadIdx.x; j < ne10; j += blockDim.x) {
        src_row_contiguous[j] = src_row_original[j];
    }
}

static __global__ void k_copy_dst_from_contiguous(char * __restrict__ dst_original, const char * __restrict__ dst_contiguous,
                                                  const mmid_row_mapping * __restrict__ row_mapping,
                                                  int64_t ne0,
                                                  size_t nb1, size_t nb2) {
    int32_t i = blockIdx.x;

    const int32_t i1 = row_mapping[i].i1;
    const int32_t i2 = row_mapping[i].i2;

    const float * dst_row_contiguous = (const float *)(dst_contiguous + i*nb1);
    float * dst_row_original = (float *)(dst_original + i1*nb1 + i2*nb2);

    for (int j = threadIdx.x; j < ne0; j += blockDim.x) {
        dst_row_original[j] = dst_row_contiguous[j];
    }
}

//static __global__ void k_quick_add(uint32_t n, uint32_t n_per_row, const float * src1, const float * src2, float * dst) {
//
//    for (uint32_t j = threadIdx.x; j < n; j += blockDim.x) {
//        dst[j] = src1[j] + src2[j % n_per_row];
//    }
//}

static __global__ void k_quick_add(uint32_t n_per_row, const float * src1, const float * src2, float * dst) {

    uint32_t row = blockIdx.x;
    const float * src1_row = src1 + row*n_per_row;
    float * dst_row = dst + row*n_per_row;

    for (uint32_t j = threadIdx.x; j < n_per_row; j += blockDim.x) {
        dst_row[j] = src1_row[j] + src2[j];
    }
}

static inline bool prepare_row_mappigs(ggml_backend_cuda_context& ctx, int64_t n_as, int64_t n_ids,
        const ggml_tensor * ids, std::vector<int>& moe_counts, std::vector<int>& cum_moe_counts,
        ggml_cuda_pool_alloc<mmid_row_mapping>& dev_row_mapping) {

    GGML_ASSERT(moe_counts.empty() && cum_moe_counts.empty());

    auto stream = ctx.stream();

    std::vector<char> ids_host(ggml_nbytes(ids));
    const char * ids_dev = (const char *) ids->data;
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids_dev, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<mmid_row_mapping> rmapping(ids->ne[1]*n_ids);
    moe_counts.resize(n_as, 0);
    cum_moe_counts.resize(n_as + 1);

    bool is_ser = false;
    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) ++moe_counts[row_id_i];
            else is_ser = true;
        }
    }
    cum_moe_counts[0] = 0;
    for (int i = 0; i < (int)n_as; ++i) {
        cum_moe_counts[i+1] = cum_moe_counts[i] + moe_counts[i];
    }

    dev_row_mapping.alloc(cum_moe_counts[n_as]);

    for (int64_t iid1 = 0; iid1 < ids->ne[1]; iid1++) {
        for (int64_t id = 0; id < n_ids; id++) {
            const int32_t row_id_i = *(const int32_t *) (ids_host.data() + iid1*ids->nb[1] + id*ids->nb[0]);
            if (row_id_i >= 0 && row_id_i < n_as) {
                rmapping[cum_moe_counts[row_id_i]++] = {(int)id, (int)iid1};
            }
        }
    }

    for (int i = 0; i < (int)n_as; ++i) cum_moe_counts[i] -= moe_counts[i];

    CUDA_CHECK(cudaMemcpyAsync(dev_row_mapping.get(), rmapping.data(),
                cum_moe_counts[n_as]*sizeof(mmid_row_mapping), cudaMemcpyHostToDevice, stream));
    //CUDA_CHECK(cudaStreamSynchronize(stream));

    return is_ser;
}

static bool ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * next) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    CUDA_CHECK(cudaMemsetAsync((char *)dst->data, 0, ggml_nbytes(dst), ctx.stream()));

    if (src1->ne[1] <= MMVQ_MAX_BATCH_SIZE && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        ggml_is_quantized(src0->type) &&
        ggml_backend_buffer_is_cuda(src0->buffer) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_ctx = (ggml_backend_cuda_buffer_context *) src0->buffer->context;
        ggml_backend_cuda_buffer_context * src1_ctx = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx  = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_ctx->device == device_id &&
            src1_ctx->device == device_id &&
            dst_ctx->device  == device_id) {
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[2] = local_dst.nb[1];

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto local_src1 = *src1;
            local_src1.ne[1] = 1;
            local_src1.nb[1] = src_1_ddq_size;
            local_src1.nb[2] = src1->ne[1] > 1 ? src_1_ddq_size : 0;
            local_src1.nb[3] = local_src1.nb[2];

            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            local_src1.data = src1_quantized.alloc(src_1_ddq_size*src1->ne[1]);
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    src1_padded_col_size, src0->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_vec_q_id(ctx, src0, &local_src1, ids, &local_dst, nullptr,
                (const char *)src0->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0->ne[1], 1, src1_padded_col_size, stream);
            CUDA_CHECK(cudaGetLastError());

            if (next && next->op == GGML_OP_MUL_MAT_ID && next->src[0]->type == src0->type && src1 == next->src[1] &&
                ggml_are_same_shape(src0, next->src[0]) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ggml_backend_buffer_is_cuda(next->buffer)) {
                ggml_backend_cuda_buffer_context * next_src0_ctx = (ggml_backend_cuda_buffer_context *) next->src[0]->buffer->context;
                ggml_backend_cuda_buffer_context * next_dst_ctx  = (ggml_backend_cuda_buffer_context *) next->buffer->context;
                if (next_src0_ctx->device == device_id &&
                    next_dst_ctx->device  == device_id) {
                    local_dst.data = next->data;
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, next->src[0], &local_src1, ids, &local_dst, nullptr,
                        (const char *)next->src[0]->data, nullptr, src1_quantized.get(), (float *)next->data,
                        0, src0->ne[1], 1, src1_padded_col_size, stream);
                    CUDA_CHECK(cudaGetLastError());
                    return true;
                }
            }

            return false;
        }
    }

    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0->type) && ggml_cuda_can_use_mmq_id(src0->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
        ggml_cuda_mul_mat_q_id(ctx, src0, src1, ids, dst, nullptr, nullptr);
        return false;
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor src0_row = *src0;
    ggml_tensor src1_row = *src1;
    ggml_tensor dst_row  = *dst;

    char * src0_original = (char *) src0->data;
    char * src1_original = (char *) src1->data;
    char * dst_original  = (char *)  dst->data;

    src0_row.ne[2] = 1;
    src0_row.ne[3] = 1;
    src0_row.nb[3] = nb02;
    src0_row.view_src = dst->src[0];

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;


    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;
    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
    }

    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(src1));
    ggml_cuda_pool_alloc<char>  dst_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

    src1_row.data = src1_contiguous.get();
    dst_row.data  =  dst_contiguous.get();

    for (int64_t i02 = 0; i02 < n_as; i02++) {

        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) {
            continue;
        }

        size_t mapping_offset = cum_moe_counts[i02];

        {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
        }

        src0_row.data = src0_original + i02*nb02;

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = nb11;
        src1_row.nb[2] = num_src1_rows*nb11;
        src1_row.nb[3] = num_src1_rows*nb11;

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1;
        dst_row.nb[2] = num_src1_rows*nb1;
        dst_row.nb[3] = num_src1_rows*nb1;

        if (ggml_is_quantized(src0->type) &&
            ggml_cuda_should_use_mmq(src0->type, ggml_cuda_info().devices[ctx.device].cc, num_src1_rows)) {
            auto src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            auto src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            auto src1_quantized_size  = src1_padded_row_size*num_src1_rows;
            if (true || num_src1_rows > MMVQ_MAX_BATCH_SIZE) {
                src1_quantized_size += get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_mmq_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_mul_mat_q_id(ctx, &src0_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());

                CUDA_CHECK(cudaGetLastError());
            } else {
                ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), src1_quantized_size);
                quantize_row_q8_1_cuda((const float *)src1_contiguous.get(), src1_quantized.get(), ne00, num_src1_rows, 1,
                        src1_padded_num_cols, src0->type, stream);
                src1_row.nb[1] = src1_padded_row_size;
                src1_row.nb[2] = src1_row.nb[3] = src1_row.nb[1]*num_src1_rows;
                ggml_cuda_op_mul_mat_vec_q(ctx, &src0_row, &src1_row, &dst_row, (const char *)src0_row.data, nullptr,
                        src1_quantized.get(), (float *)dst_row.data,
                        0, src0_row.ne[1], num_src1_rows, src1_padded_num_cols, stream);
                CUDA_CHECK(cudaGetLastError());
            }
        } else {
            ggml_cuda_mul_mat(ctx, &src0_row, &src1_row, &dst_row, nullptr, 0);
        }

        {
            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return false;
}

static int ggml_cuda_moe_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_cgraph * graph, int i) {
    ggml_tensor * next = i + 1 < graph->n_nodes ? graph->nodes[i+1] : nullptr;

    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src0 = src0_1;
    const ggml_tensor * src1 = dst->src[2];
    const ggml_tensor * ids  = dst->src[3];

    if (src1->ne[1] == 1 && src1->ne[2] <= 8 && src1->ne[3] == 1 &&
        ggml_is_quantized(src0_1->type) &&
        (!src0_2 || ggml_is_quantized(src0_2->type)) &&
        ggml_backend_buffer_is_cuda(src0_1->buffer) &&
        (!src0_2 || ggml_backend_buffer_is_cuda(src0_2->buffer)) &&
        ggml_backend_buffer_is_cuda(src1->buffer) &&
        ggml_backend_buffer_is_cuda(dst->buffer) &&
        src1->type == GGML_TYPE_F32) {
        int device_id = ctx.device;
        ggml_backend_cuda_buffer_context * src0_1_ctx = (ggml_backend_cuda_buffer_context *) src0_1->buffer->context;
        ggml_backend_cuda_buffer_context * src0_2_ctx = src0_2 ? (ggml_backend_cuda_buffer_context *) src0_2->buffer->context : nullptr;
        ggml_backend_cuda_buffer_context * src1_ctx   = (ggml_backend_cuda_buffer_context *) src1->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx    = (ggml_backend_cuda_buffer_context *) dst->buffer->context;
        if (src0_1_ctx->device == device_id &&
            (!src0_2_ctx || src0_2_ctx->device == device_id) &&
            src1_ctx->device   == device_id &&
            dst_ctx->device    == device_id) {
            // Fast TG path
            const int64_t n_ids = ids->ne[0];
            auto stream = ctx.stream(device_id, 0);

            auto local_dst = *dst;
            local_dst.ne[2] = n_ids;
            local_dst.ne[1] = local_dst.ne[3] = 1;
            local_dst.nb[1] = local_dst.nb[2] = local_dst.nb[3] = local_dst.ne[0]*sizeof(float);

            auto local_src1 = *src1;
            local_src1.nb[2] = local_src1.nb[3] = 0;
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;

            int Ny = src1->ne[2];

            const int64_t src1_padded_col_size = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
            GGML_ASSERT(src1->ne[0] % QK8_1 == 0);
            auto src_1_ddq_size = src1_padded_col_size*sizeof(block_q8_1)/QK8_1;
            local_src1.data = src1_quantized.alloc(src_1_ddq_size * Ny);
            // Note: no use is currently made of the quantization type passed into quantize_row_q8_1_cuda.
            //       If that were to change, we would need to adjust the code to handle src0_1->type != src0_2->type
            quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], Ny, 1, src1_padded_col_size,
                    src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_src1.nb[1] = src_1_ddq_size;

            bool fuse_next = next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
                ggml_backend_buffer_is_cuda(next->src[0]->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->src[0]->buffer->context)->device == device_id &&
                ggml_backend_buffer_is_cuda(next->buffer) &&
                ((ggml_backend_cuda_buffer_context *)next->buffer->context)->device == device_id;

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            float limit = *(const float *)(dst->op_params + 1);

            auto local_ids = *ids;
            local_ids.ne[1] = 1;

            for (int iy = 0; iy < Ny; ++iy) {
                local_src1.data = src1_quantized.get() + iy*src_1_ddq_size;
                local_ids.data  = (char *)ids->data + iy*ids->nb[1];
                local_dst.data  = (char *)dst->data + iy*dst->nb[2];
                if (src0_2) {
                    ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, &local_src1, &local_ids, &local_dst,
                            dst->src[4], dst->src[5],
                            (const char *)src0_1->data, (const char *)src0_2->data,
                            (const float *)src1->data, (const char *)local_src1.data,
                            (float *)local_dst.data, 0, src0_1->ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                } else {
                    auto local_src0_1 = *src0_1;
                    local_src0_1.ne[1] /= 2;
                    auto local_src0_2 = local_src0_1;
                    local_src0_2.data = (char *)local_src0_1.data + local_src0_1.ne[1]*local_src0_1.nb[1];
                    if (!dst->src[4]) {
                        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, &local_ids, &local_dst,
                                nullptr, nullptr,
                                (const char *)local_src0_2.data, (const char *)local_src0_1.data,
                                (const float *)src1->data, (const char *)local_src1.data,
                                (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                    } else {
                        GGML_ASSERT(!dst->src[5]);
                        auto local_bias_1 = *dst->src[4];
                        local_bias_1.ne[0] /= 2;
                        auto local_bias_2 = local_bias_1;
                        local_bias_2.data = (char *)local_bias_1.data + local_bias_1.ne[0]*local_bias_1.nb[0];
                        ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, &local_src0_1, &local_src1, &local_ids, &local_dst,
                                &local_bias_2, &local_bias_1,
                                (const char *)local_src0_2.data, (const char *)local_src0_1.data,
                                (const float *)src1->data, (const char *)local_src1.data,
                                (float *)local_dst.data, 0, local_src0_1.ne[1], 1, src1_padded_col_size, unary_op, limit, stream);
                    }
                }
                CUDA_CHECK(cudaGetLastError());
            }

            if (!fuse_next) return i;

            const int64_t dst_padded_col_size = GGML_PAD(dst->ne[0], MATRIX_ROW_PADDING);
            GGML_ASSERT(dst->ne[0] % QK8_1 == 0);
            auto dst_row_size = dst_padded_col_size*sizeof(block_q8_1)/QK8_1;
            auto dst_ddq_size = n_ids*dst_row_size;
            ggml_cuda_pool_alloc<char> dst_quantized(ctx.pool(), dst_ddq_size*Ny);
            quantize_row_q8_1_cuda((const float *)dst->data, (void *)dst_quantized.get(), dst->ne[0], n_ids*Ny, 1,
                    dst_padded_col_size, next->src[0]->type, stream);
            CUDA_CHECK(cudaGetLastError());

            local_dst.ne[2] = 1;

            auto local_next = *next;
            local_next.ne[2] = local_next.ne[1];
            local_next.ne[1] = local_next.ne[3] = 1;
            local_next.nb[2] = local_next.nb[1];

            local_src1 = *next->src[1];
            local_src1.ne[1] = local_src1.ne[2] = local_src1.ne[3] = 1;
            local_src1.nb[1] = local_src1.nb[2] = local_src1.nb[3] = dst_row_size;

            auto local_src0 = *next->src[0];
            local_src0.ne[2] = local_src0.ne[3] = 1;

            int result = i + 1;

            //printf("next: %ld x %ld x %ld x %ld,    %zu x %zu x %zu x %zu\n", next->ne[0], next->ne[1], next->ne[2], next->ne[3], next->nb[0], next->nb[1], next->nb[2], next->nb[3]);

            for (int iy = 0; iy < Ny; ++iy) {
                local_ids.data  = (char *)ids->data + iy*ids->nb[1];
                auto this_dst_quantized = dst_quantized.get() + iy*dst_ddq_size;
                if (i+2 < graph->n_nodes &&
                    graph->nodes[i+2]->op == GGML_OP_ADD_ID &&
                    graph->nodes[i+2]->src[0] == next &&
                    graph->nodes[i+2]->src[2] == ids) {
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, &local_ids, &local_next, graph->nodes[i+2]->src[1],
                            (const char *)next->src[0]->data, nullptr, this_dst_quantized, (float *)graph->nodes[i+2]->data + iy*next->ne[0]*n_ids,
                            0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                    if (iy == 0) {
                        ++result;
                    }
                } else {
                    ggml_cuda_op_mul_mat_vec_q_id(ctx, &local_src0, &local_src1, &local_ids, &local_next, nullptr,
                            (const char *)next->src[0]->data, nullptr, this_dst_quantized, (float *)next->data + iy*next->ne[0]*n_ids,
                            0, next->src[0]->ne[1], 1, dst_padded_col_size, stream);
                }
                CUDA_CHECK(cudaGetLastError());
            }

            return result;
        }
    }

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();

    const int64_t n_as = ne02;
    const int64_t n_ids = ids->ne[0];

    ggml_tensor dst_row = *dst;

    // The heuristics src1->ne[2] <= 32*src0->ne[2] to use the mul_mat_id implementation instead of the original version
    // is derived from
    //    * DeepSeek-Lite:  64 total, 6 active experts
    //    * GPT-OSS-20B  :  32 total, 4 active experts
    //    * Qwen3-30B-A3B: 128 total, 8 active experts
    // My original hypothesis was that it is dependent on the total/active experts ratio, but from these 3 it
    // looks like it really depends just on the total number of experts.
    // TODO: verify with more models, or perhaps make the magic constant '32' to be defined via a compile time define.
    if (src1->ne[2] <= ctx.mmq_id_thresh*src0->ne[2] &&
        ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1 &&
        ggml_cuda_can_use_mmq_id(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {

        const int64_t ne_get_rows = ne12 * n_ids;
        ggml_cuda_pool_alloc<int32_t> ids_device(ctx.pool(), ne_get_rows + ne_get_rows + n_as + 1);
        auto ids_src1 = ids_device.get();
        auto ids_dst  = ids_src1 + ne_get_rows;
        auto expert_bounds = ids_dst + ne_get_rows;

        compute_row_ids((const int32_t *)ids->data, ids_src1, ids_dst, expert_bounds,
                ne02, ne12, n_ids, ne11, nb11, nb12, ids->nb[1], stream);

        const int64_t ne11_flat = ne12*n_ids;
        const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
        size_t nbytes_src1_q8_1 = ne11_flat*ne10_padded * sizeof(block_q8_1)/QK8_1 +
                get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
        ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), nbytes_src1_q8_1);

        size_t ts_src1 = ggml_type_size(src1->type);
        quantize_mmq_q8_1_cuda_id((const float *)src1->data, ids_src1, src1_quantized.get(),
                src0_1->type, ne10, src1->nb[1] / ts_src1, src1->nb[2] / ts_src1, src1->nb[2] / ts_src1,
                ne10_padded, ne11_flat, 1, 1, stream);

        if (src0_2) {
        ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
        ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));

        dst_row.data = dst_up_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[4]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        dst_row.data = dst_gate_contiguous.get();
        ggml_cuda_mul_mat_q_id(ctx, src0_2, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
        if (dst->src[5]) {
            ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[5]->data, (const int32_t *)ids->data,
                    (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                    dst_row.nb[1], dst_row.nb[2], dst->src[5]->nb[1], ids->nb[1], stream);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
            ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst->data, ggml_nelements(dst), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
        } else {
            float limit = *((const float *)(dst->op_params + 1));
            //printf("%s: using limit = %g\n", __func__, limit);
            ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                    (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                    (float *)dst->data, limit);
        }
        } else {

            ggml_cuda_pool_alloc<char> dst_up_gate_contiguous(ctx.pool(), 2*sizeof(float)*ggml_nelements(dst));
            ggml_cuda_pool_alloc<char> dst_gate_contiguous(ctx.pool(), sizeof(float)*ggml_nelements(dst));
            dst_row.ne[0] *= 2;
            dst_row.nb[1] *= 2;
            dst_row.nb[2] *= 2;
            dst_row.nb[3] *= 2;
            dst_row.data = dst_up_gate_contiguous.get();
            ggml_cuda_mul_mat_q_id(ctx, src0_1, src1, ids, &dst_row, (char *)ids_device.get(), src1_quantized.get());
            if (dst->src[4]) {
                GGML_ASSERT(!dst->src[5]);
                ggml_cuda_add_id((const float *)dst_row.data, (const float *)dst->src[4]->data, (const int32_t *)ids->data,
                        (float *)dst_row.data, dst_row.ne[0], dst_row.ne[1], dst_row.ne[2], dst_row.ne[0], dst_row.ne[1],
                        dst_row.nb[1], dst_row.nb[2], dst->src[4]->nb[1], ids->nb[1], stream);
                CUDA_CHECK(cudaGetLastError());
            }

            auto unary_op = (ggml_unary_op)dst->op_params[0];
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_gate_contiguous.get(), (const float *)dst_up_gate_contiguous.get() + dst->ne[0],
                        (float *)dst->data, ggml_nelements(dst), dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                float limit = *((const float *)(dst->op_params + 1));
                //printf("%s: using limit = %g\n", __func__, limit);
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(dst), dst->ne[0],
                        (const float *)dst_up_gate_contiguous.get(), (float *)dst->data, limit);
            }
        }
        CUDA_CHECK(cudaGetLastError());

        if (next && next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type) &&
            ggml_cuda_should_use_mmq(next->src[0]->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            //ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, (char *)ids_device.get(), nullptr);
            ggml_cuda_mul_mat_q_id(ctx, next->src[0], dst, ids, next, nullptr, nullptr);
            return i+1;
        }

        return i;
    }

    ggml_tensor src0_1_row = *src0_1;
    ggml_tensor src0_2_row; if (src0_2) src0_2_row = *src0_2;
    ggml_tensor src1_row   = *src1;
    ggml_tensor final_dst;
    ggml_tensor final_src;

    char * src0_1_original = (char *) src0_1->data;
    char * src0_2_original = src0_2 ? (char *) src0_2->data : nullptr;
    char * src1_original   = (char *) src1->data;
    char * dst_original    = (char *)  dst->data;

    src0_1_row.ne[2] = 1;
    src0_1_row.ne[3] = 1;
    src0_1_row.nb[3] = nb02;
    src0_1_row.view_src = dst->src[0];
    if (src0_2) {
        src0_2_row.ne[2] = 1;
        src0_2_row.ne[3] = 1;
        src0_2_row.nb[3] = nb02;
        src0_2_row.view_src = dst->src[1];
    }

    src1_row.ne[1] = 1;
    src1_row.ne[2] = 1;
    src1_row.ne[3] = 1;
    src1_row.nb[2] = nb11;
    src1_row.nb[3] = nb11;

    dst_row.ne[1] = 1;
    dst_row.ne[2] = 1;
    dst_row.ne[3] = 1;
    dst_row.nb[2] = nb1;
    dst_row.nb[3] = nb1;

    bool fuse_down = next && next->op == GGML_OP_MUL_MAT_ID;
    // The down result of an expert is written to next before the up/gate data of the remaining experts is read.
    // If the up/gate experts are copies of host tensors, next may have been allocated on top of them.
    for (int j = 0; fuse_down && j < GGML_MAX_SRC; ++j) {
        auto src = dst->src[j];
        fuse_down = !src || (const char *)src->data >= (const char *)next->data + ggml_nbytes(next) ||
                            (const char *)next->data >= (const char *)src->data + ggml_nbytes(src);
    }
    if (fuse_down) {
        final_dst = *next;
        final_dst.ne[1] = final_dst.ne[2] = final_dst.ne[3] = 1;
        final_dst.nb[2] = final_dst.nb[3] = final_dst.nb[1];
        final_src = *next->src[0];
        final_src.ne[2] = final_src.ne[3] = 1;
        final_src.nb[3] = final_src.nb[2];
        final_src.view_src = next->src[0];
    }

    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool());
    bool use_quantized_src1 = false;
    int64_t src1_padded_num_cols = 0, src1_padded_row_size = 0, src1_quantized_size = 0;
    if (ggml_is_quantized(src0_1->type) && (!src0_2 || src0_1->type == src0_2->type) && src1->ne[1] == 1 && src1->ne[3] == 1) {
        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[2])) {
            src1_padded_num_cols = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
            src1_padded_row_size = src1_padded_num_cols/ggml_blck_size(GGML_TYPE_Q8_1)*ggml_type_size(GGML_TYPE_Q8_1);
            src1_quantized_size  = src1_padded_row_size*src1->ne[2] + get_mmq_x_max_host(ggml_cuda_info().devices[ctx.device].cc)*sizeof(block_q8_1_mmq);
            src1_quantized.alloc(src1_quantized_size);
            use_quantized_src1 = true;
        }
    }
    ggml_cuda_pool_alloc<char> src1_contiguous(ctx.pool());
    if (!use_quantized_src1) {
        src1_contiguous.alloc(sizeof(float)*ggml_nelements(src1));
    }
    ggml_cuda_pool_alloc<char> dst_up_contiguous(ctx.pool()), dst_gate_contiguous(ctx.pool());
    if (src0_2) {
        dst_up_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    } else {
        dst_up_contiguous.alloc(2*sizeof(float)*ggml_nelements(dst));
        dst_gate_contiguous.alloc(sizeof(float)*ggml_nelements(dst));
    }
    src1_row.data = src1_contiguous.get();

    ggml_cuda_pool_alloc<mmid_row_mapping> dev_row_mapping(ctx.pool());
    std::vector<int> moe_counts, cum_moe_counts;

    bool is_ser = prepare_row_mappigs(ctx, n_as, n_ids, ids, moe_counts, cum_moe_counts, dev_row_mapping);
    if (is_ser) {
        if (fuse_down) {
            CUDA_CHECK(cudaMemsetAsync(next->data, 0, ggml_nbytes(next), stream));
        } else {
            CUDA_CHECK(cudaMemsetAsync(dst->data, 0, ggml_nbytes(dst), stream));
        }
    }

    ggml_cuda_pool_alloc<char> final_dst_contiguous(ctx.pool());
    if (fuse_down) {
        const int max_rows = *std::max_element(moe_counts.begin(), moe_counts.end());
        final_dst.data = final_dst_contiguous.alloc(sizeof(float)*next->ne[0]*max_rows);
        final_dst.src[1] = &dst_row;
    }

    for (int64_t i02 = 0; i02 < n_as; i02++) {
        int64_t num_src1_rows = moe_counts[i02];

        if (num_src1_rows == 0) continue;
        size_t mapping_offset = cum_moe_counts[i02];

        if (use_quantized_src1) {
            quantize_mmq_q8_1_id_cuda((const float *)src1->data, src1_quantized.get(), (const char *)(dev_row_mapping.get() + mapping_offset),
                    src1->ne[0], num_src1_rows, src1_padded_num_cols, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_quantized.get();
        }
        else {
            dim3 block_dims(std::min((unsigned int)ne10, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_src_to_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    src1_original, src1_contiguous.get(), dev_row_mapping.get() + mapping_offset, ne10, ne11, nb11, nb12);
            CUDA_CHECK(cudaGetLastError());
            src1_row.data = src1_contiguous.get();
        }

        src0_1_row.data = src0_1_original + i02*nb02;
        if (src0_2_original) src0_2_row.data = src0_2_original + i02*nb02;

        GGML_ASSERT(nb11 == sizeof(float)*ne10);
        GGML_ASSERT(nb1 == sizeof(float)*ne0);

        auto nb1l = nb1;
        if (!src0_2) {
            nb1l = nb1*2;
            dst_row.ne[0] = dst->ne[0] * 2;
        }

        src1_row.ne[1] = num_src1_rows;
        src1_row.nb[1] = use_quantized_src1 ? src1_padded_row_size : nb11;
        src1_row.nb[2] = num_src1_rows*src1_row.nb[1];
        src1_row.nb[3] = num_src1_rows*src1_row.nb[1];

        dst_row.ne[1] = num_src1_rows;
        dst_row.nb[1] = nb1l;
        dst_row.nb[2] = num_src1_rows*nb1l;
        dst_row.nb[3] = num_src1_rows*nb1l;

        dst_row.data  =  dst_up_contiguous.get();
        if (use_quantized_src1) {
            ggml_cuda_mul_mat_q_id(ctx, &src0_1_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
        } else {
            ggml_cuda_mul_mat(ctx, &src0_1_row, &src1_row, &dst_row, nullptr, 0);
        }
        CUDA_CHECK(cudaGetLastError());

        if (dst->src[4]) {
            GGML_ASSERT(dst_row.ne[0] == dst->src[4]->ne[0]);
            dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
            dim3 grid_dims(num_src1_rows);
            k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                    (const float *)((const char *)dst->src[4]->data + i02*dst->src[4]->nb[1]), (float *)dst_row.data);
            CUDA_CHECK(cudaGetLastError());
        }

        auto unary_op = (ggml_unary_op)dst->op_params[0];
        float limit = *(const float *)(dst->op_params + 1);
        //printf("%s: using limit = %g\n", __func__, limit);
        if (src0_2) {
            dst_row.data  = dst_gate_contiguous.get();
            if (use_quantized_src1) {
                ggml_cuda_mul_mat_q_id(ctx, &src0_2_row, &src1_row, nullptr, &dst_row, nullptr, src1_quantized.get());
            } else {
                ggml_cuda_mul_mat(ctx, &src0_2_row, &src1_row, &dst_row, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            if (dst->src[5]) {
                dim3 block_dims(std::min(uint32_t(dst_row.ne[0]), 768u));
                dim3 grid_dims(num_src1_rows);
                k_quick_add<<<grid_dims, block_dims, 0, stream>>>(dst_row.ne[0], (const float *)dst_row.data,
                        (const float *)((const char *)dst->src[5]->data + i02*dst->src[5]->nb[1]), (float *)dst_row.data);
                CUDA_CHECK(cudaGetLastError());
            }
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row), dst_row.ne[0],  dst_row.ne[0],  dst_row.ne[0],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row),
                        (const float *)dst_gate_contiguous.get(), (const float *)dst_up_contiguous.get(),
                        (float *)dst_gate_contiguous.get(), limit);
            }
        } else {
            if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
                ggml_swiglu_oai_cuda_f32((const float *)dst_up_contiguous.get(), (const float *)dst_up_contiguous.get() + dst->ne[0],
                        (float *)dst_gate_contiguous.get(), ggml_nelements(&dst_row)/2, dst->ne[0], src0_1->ne[1], src0_1->ne[1],
                        1.702f, 7.0f, stream);
            } else {
                ggml_fused_mul_unary(ctx, (ggml_unary_op)dst->op_params[0], ggml_nelements(&dst_row)/2, dst->ne[0],
                        (const float *)dst_up_contiguous.get(), (float *)dst_gate_contiguous.get(), limit);
            }
            dst_row.data = dst_gate_contiguous.get();
            dst_row.ne[0] /= 2;
            dst_row.nb[1] /= 2;
            dst_row.nb[2] /= 2;
            dst_row.nb[3] /= 2;
        }
        CUDA_CHECK(cudaGetLastError());

        if (fuse_down) {

            final_dst.ne[1] = num_src1_rows;
            final_dst.nb[1] = final_dst.ne[0]*sizeof(float);
            final_dst.nb[2] = final_dst.nb[3] = num_src1_rows*final_dst.nb[1];
            final_src.data = (char *)next->src[0]->data + i02*next->src[0]->nb[2];
            if (ggml_is_quantized(next->src[0]->type) &&
                ggml_cuda_should_use_mmq(final_src.type, ggml_cuda_info().devices[ctx.device].cc, dst_row.ne[1])) {
                ggml_cuda_mul_mat_q_id(ctx, &final_src, &dst_row, nullptr, &final_dst, nullptr, nullptr);
            } else {
                ggml_cuda_mul_mat(ctx, &final_src, &dst_row, &final_dst, nullptr, 0);
            }
            CUDA_CHECK(cudaGetLastError());

            dim3 block_dims(std::min((unsigned int)next->ne[0], 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    (char *)next->data, final_dst_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    next->ne[0],
                    next->nb[1], next->nb[2]);
            CUDA_CHECK(cudaGetLastError());

        }
        else {

            dim3 block_dims(std::min((unsigned int)ne0, 768u));
            dim3 grid_dims(num_src1_rows);
            k_copy_dst_from_contiguous<<<grid_dims, block_dims, 0, stream>>>(
                    dst_original, dst_gate_contiguous.get(),
                    dev_row_mapping.get() + mapping_offset,
                    ne0,
                    nb1, nb2);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return fuse_down ? i+1 : i;
}

static void ggml_cuda_up_gate_unary(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0_1 = dst->src[0];
    const ggml_tensor * src0_2 = dst->src[1];
    const ggml_tensor * src1 = dst->src[2];

    GGML_ASSERT(ggml_is_quantized(src0_1->type));
    GGML_ASSERT(src0_1->type == src0_2->type);
    GGML_ASSERT(src1->ne[2] == 1);
    GGML_ASSERT(src1->ne[3] == 1);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);

    auto stream = ctx.stream();

    float limit = *(const float *)(dst->op_params + 1);

    auto ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
    auto nb10_padded = ne10_padded*sizeof(block_q8_1)/QK8_1;
    auto quantized_size = nb10_padded*src1->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const bool use_mmvq = src1->ne[1] <= ggml_cuda_mmvq_max_batch_size(cc);
    if (!use_mmvq) {
        quantized_size += get_mmq_x_max_host(cc)*sizeof(block_q8_1_mmq);
    }
    ggml_cuda_pool_alloc<float> dst_up(ctx.pool(), ggml_nelements(dst));
    ggml_cuda_pool_alloc<char> src1_quantized(ctx.pool(), quantized_size);
    if (use_mmvq) {
        quantize_row_q8_1_cuda((const float *)src1->data, (void *)src1_quantized.get(), src1->ne[0], src1->ne[1], 1, ne10_padded,
                src0_1->type, stream);
        CUDA_CHECK(cudaGetLastError());

        if (src1->ne[1] == 1 && src0_1->type == src0_2->type) {
            ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, src0_1, src1, nullptr, dst,
                    dst->src[4], dst->src[5],
                    (const char *)src0_1->data, (const char *)src0_2->data, (const float *)src1->data, src1_quantized.get(),
                    (float *)dst->data, 0, src0_1->ne[1], 1, ne10_padded,
                    (ggml_unary_op)dst->op_params[0], limit, stream);
            return;
        }

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, src1_quantized.get(), dst_up.get(),
                0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());

        ggml_cuda_op_mul_mat_vec_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, src1_quantized.get(), (float *)dst->data,
                0, src0_2->ne[1], src1->ne[1], ne10_padded, stream);
        CUDA_CHECK(cudaGetLastError());
    } else {

        if (ggml_cuda_should_use_mmq(src0_1->type, ggml_cuda_info().devices[ctx.device].cc, src1->ne[1])) {
            quantize_mmq_q8_1_cuda((const float *)src1->data, src1_quantized.get(), src1->ne[0], src1->ne[1], 1,
                    ne10_padded, src0_1->type, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_1, src1, dst, (const char *)src0_1->data, nullptr, src1_quantized.get(), dst_up.get(),
                    0, src0_1->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());

            ggml_cuda_op_mul_mat_q(ctx, src0_2, src1, dst, (const char *)src0_2->data, nullptr, src1_quantized.get(), (float *)dst->data,
                    0, src0_2->ne[1], src1->ne[1], ne10_padded, stream);
            CUDA_CHECK(cudaGetLastError());
        } else {
            auto local_dst = *dst;
            local_dst.data = dst_up.get();
            ggml_cuda_mul_mat(ctx, src0_1, src1, &local_dst, nullptr, 0);
            ggml_cuda_mul_mat(ctx, src0_2, src1, dst, nullptr, 0);
        }
    }

    auto unary_op = (ggml_unary_op)dst->op_params[0];
    if (unary_op == GGML_UNARY_OP_SWIGLU_OAI) {
        ggml_swiglu_oai_cuda_f32((const float *)dst->data, (const float *)dst_up.get(),
                (float *)dst->data, ggml_nelements(dst), dst->ne[0],  dst->ne[0],  dst->ne[0],
                1.702f, 7.0f, stream);
    } else {
        //printf("%s: using limit = %g\n", __func__, limit);
        ggml_fused_mul_unary(ctx, unary_op, ggml_nelements(dst),
                (const float *)dst->data, dst_up.get(), (float *)dst->data, limit);
    }
    CUDA_CHECK(cudaGetLastError());

}

static inline bool ops_are_same_device(const ggml_cgraph * cgraph, int first, int last) {
    if (last <= first) return true;
    int device = ((const ggml_backend_cuda_buffer_context *)cgraph->nodes[first]->buffer->context)->device;
    for (int i = first; i <= last; ++i) {
        auto node = cgraph->nodes[i];
        if (((const ggml_backend_cuda_buffer_context *)node->buffer->context)->device != device) return false;
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (!node->src[j] || !node->src[j]->buffer) continue;
            if (((const ggml_backend_cuda_buffer_context *)node->src[j]->buffer->context)->device != device) return false;
        }
    }
    return true;
}

static std::shared_ptr<ggml_cuda_moe_ffn_binding_state> ggml_cuda_moe_ffn_lookup(
        ggml_backend_cuda_context & ctx, int32_t binding_id, uint64_t generation) {
    std::lock_guard<std::mutex> lock(ctx.moe_ffn_mutex);
    auto it = ctx.moe_ffn_bindings.find(binding_id);
    if (it == ctx.moe_ffn_bindings.end() || it->second->binding.generation != generation) {
        return {};
    }
    return it->second;
}

static bool ggml_cuda_moe_ffn_op_params(const ggml_tensor * op, ggml_moe_ffn_op_params & params) {
    if (!op || op->op != GGML_OP_MOE_FFN) return false;
    memcpy(&params, op->op_params, sizeof(params));
    return params.device >= 0 && params.binding_id > 0 && params.generation > 0 &&
        params.n_embd > 0 && params.n_ff > 0;
}

static bool ggml_cuda_moe_ffn_tensor_on_device(const ggml_tensor * tensor, int device) {
    if (!tensor || !tensor->buffer || !ggml_backend_buffer_is_cuda(tensor->buffer)) return false;
    auto * buffer = (ggml_backend_cuda_buffer_context *) tensor->buffer->context;
    return buffer->device == device;
}

static bool ggml_cuda_moe_ffn_shape_matches(const ggml_backend_cuda_context & ctx,
        const ggml_tensor * op, const ggml_moe_ffn_op_params & params,
        const ggml_cuda_moe_ffn_binding & b) {
    ggml_moe_ffn_layout graph = b.layout;
    graph.activation = op->src[0];
    graph.selected_ids = op->src[1];
    graph.route_weights = op->src[2];
    return params.device == ctx.device && b.layout.device == ctx.device &&
        params.n_embd == b.layout.n_embd && params.n_ff == b.layout.n_ff &&
        params.unary_op == (int32_t) b.layout.unary_op && params.limit == b.layout.limit &&
        op->type == GGML_TYPE_F32 && op->ne[0] == b.layout.n_embd && op->ne[1] == 1 &&
        op->ne[2] == 1 && op->ne[3] == 1 && ggml_moe_ffn_graph_inputs_supported(&graph) &&
        ggml_is_contiguous(op);
}

static bool ggml_cuda_moe_ffn_resource_supports_signature(
        const ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_moe_ffn_layout & layout) {
    if (layout.n_embd > resource.resource.n_embd || layout.n_ff > resource.resource.n_ff ||
            layout.n_experts_used > resource.resource.n_experts_used) return false;
    return std::any_of(resource.signatures.begin(), resource.signatures.end(), [&](const auto & signature) {
        return signature.n_embd == layout.n_embd && signature.n_ff == layout.n_ff &&
            signature.n_experts_used == layout.n_experts_used;
    });
}

static bool ggml_cuda_moe_ffn_supports_op(ggml_backend_cuda_context & ctx, const ggml_tensor * op) {
    ggml_moe_ffn_op_params params{};
    if (!ggml_cuda_moe_ffn_op_params(op, params) || params.device != ctx.device ||
        !op->src[0] || !op->src[1] || !op->src[2]) return false;
    auto state = ggml_cuda_moe_ffn_lookup(ctx, params.binding_id, params.generation);
    if (!state) return false;
    std::lock_guard<std::mutex> request_lock(state->cpu_resource->execution_mutex);
    return state->active && ggml_cuda_moe_ffn_shape_matches(ctx, op, params, state->binding);
}

static bool ggml_cuda_moe_ffn_validate_op(ggml_backend_cuda_context & ctx, const ggml_tensor * op) {
    ggml_moe_ffn_op_params params{};
    if (!ggml_cuda_moe_ffn_op_params(op, params) || params.device != ctx.device ||
        !op->src[0] || !op->src[1] || !op->src[2]) return false;
    auto state = ggml_cuda_moe_ffn_lookup(ctx, params.binding_id, params.generation);
    if (!state) return false;
    std::lock_guard<std::mutex> request_lock(state->cpu_resource->execution_mutex);
    const auto & b = state->binding;
    return state->active && ggml_cuda_moe_ffn_shape_matches(ctx, op, params, b) &&
        ggml_cuda_moe_ffn_tensor_on_device(op->src[0], ctx.device) &&
        ggml_cuda_moe_ffn_tensor_on_device(op->src[1], ctx.device) &&
        ggml_cuda_moe_ffn_tensor_on_device(op->src[2], ctx.device) &&
        ggml_cuda_moe_ffn_tensor_on_device(op, ctx.device);
}

static bool ggml_cuda_moe_ffn_cpu_draws(const ggml_cuda_moe_ffn_binding & b,
        ggml_cuda_moe_ffn_cpu_resource_state & resource, ggml_cuda_moe_ffn_stats & stats,
        bool misses_only, size_t activation_bytes) {
    auto * total = resource.host_output;
    const size_t output_bytes = size_t(b.layout.n_embd) * sizeof(float);
    memset(total, 0, output_bytes);
    memcpy(b.cpu_activations->data, resource.host_activations, activation_bytes);
    ++stats.cpu_copy_calls;

    ++resource.active_calls;
    resource.max_parallel_calls = std::max(resource.max_parallel_calls, resource.active_calls);
    bool success = true;
    for (int32_t i = 0; i < b.layout.n_experts_used; ++i) {
        if (misses_only && resource.host_slots[i] >= 0) continue;
        *(int32_t *) b.cpu_ids->data = resource.host_ids[i];
        *(float *) b.cpu_route_weights->data = resource.host_routes[i];
        stats.cpu_copy_calls += 2;
        if (ggml_backend_graph_compute(resource.resource.cpu_backend, b.cpu_graph) != GGML_STATUS_SUCCESS) {
            success = false;
            break;
        }
        ++resource.cpu_calls;
        ++stats.cpu_expert_calls;
        const float * contribution = (const float *) b.cpu_output->data;
        for (int32_t j = 0; j < b.layout.n_embd; ++j) total[j] += contribution[j];
    }
    --resource.active_calls;
    return success;
}

static bool ggml_cuda_moe_ffn_enqueue_hit_rows(ggml_backend_cuda_context & ctx,
        const ggml_tensor * op, const ggml_cuda_moe_ffn_binding & b,
        ggml_cuda_moe_ffn_cpu_resource_state & resource, ggml_cuda_moe_ffn_stats & stats,
        int32_t n_hits, bool include_cpu_partial, bool & kernels_started) {
    kernels_started = false;
    if (n_hits <= 0 || n_hits > b.layout.n_experts_used) return false;
    cudaStream_t stream = ctx.stream();
    const size_t slots_bytes = size_t(n_hits) * sizeof(int32_t);
    if (cudaMemcpyAsync(resource.device_slots, resource.host_hit_slots, slots_bytes,
                cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
    ++stats.cuda_copy_calls;
    stats.slot_map_upload_bytes += slots_bytes;

    if (include_cpu_partial) {
        const size_t route_bytes = size_t(n_hits + 1) * sizeof(float);
        resource.host_hit_routes[n_hits] = 1.0f;
        if (cudaMemcpyAsync(resource.device_routes, resource.host_hit_routes, route_bytes,
                    cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        ++stats.cuda_copy_calls;
    }

    const int64_t up_padded = GGML_PAD(b.layout.n_embd, MATRIX_ROW_PADDING);
    quantize_row_q8_1_cuda((const float *) op->src[0]->data, resource.device_up_q8,
            b.layout.n_embd, 1, 1, up_padded, b.layout.up->type, stream);
    kernels_started = true;
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor ids = *op->src[1];
    ids.ne[0] = n_hits;
    ids.ne[1] = ids.ne[2] = ids.ne[3] = 1;
    ids.nb[0] = sizeof(int32_t);
    ids.data = resource.device_slots;
    ggml_tensor activation = *op->src[0];
    activation.ne[1] = activation.ne[2] = activation.ne[3] = 1;
    activation.nb[1] = activation.nb[2] = activation.nb[3] = 0;
    ggml_tensor hidden{};
    hidden.type = GGML_TYPE_F32;
    hidden.ne[0] = b.layout.n_ff; hidden.ne[1] = 1; hidden.ne[2] = n_hits; hidden.ne[3] = 1;
    hidden.nb[0] = sizeof(float); hidden.nb[1] = b.layout.n_ff * sizeof(float);
    hidden.nb[2] = b.layout.n_ff * sizeof(float); hidden.nb[3] = b.layout.n_ff * n_hits * sizeof(float);
    hidden.data = resource.device_hidden;
    ggml_cuda_op_fused_mul_mat_vec_q_id(ctx, b.layout.bank_up, &activation, &ids, &hidden, nullptr, nullptr,
            (const char *) b.layout.bank_up->data, (const char *) b.layout.bank_gate->data, nullptr,
            (const char *) resource.device_up_q8, (float *) resource.device_hidden, 0, b.layout.n_ff, 1,
            up_padded, b.layout.unary_op, b.layout.limit, stream);
    CUDA_CHECK(cudaGetLastError());

    const int64_t down_padded = GGML_PAD(b.layout.n_ff, MATRIX_ROW_PADDING);
    const size_t q8_per_expert = (size_t) GGML_PAD((int64_t) b.layout.n_ff, MATRIX_ROW_PADDING) /
        QK8_1 * sizeof(block_q8_1);
    quantize_row_q8_1_cuda((const float *) resource.device_hidden, resource.device_down_q8,
            b.layout.n_ff, n_hits, 1, down_padded, b.layout.down->type, stream);
    CUDA_CHECK(cudaGetLastError());
    ggml_tensor down_input = hidden;
    down_input.ne[1] = 1;
    down_input.ne[2] = 1;
    down_input.nb[1] = q8_per_expert;
    down_input.nb[2] = q8_per_expert;
    ggml_tensor down_output{};
    down_output.type = GGML_TYPE_F32;
    down_output.ne[0] = b.layout.n_embd; down_output.ne[1] = 1;
    down_output.ne[2] = n_hits; down_output.ne[3] = 1;
    down_output.nb[0] = sizeof(float); down_output.nb[1] = b.layout.n_embd * sizeof(float);
    down_output.nb[2] = b.layout.n_embd * sizeof(float); down_output.nb[3] = b.layout.n_embd * down_output.ne[2] * sizeof(float);
    down_output.data = resource.device_down;
    ggml_cuda_op_mul_mat_vec_q_id(ctx, b.layout.bank_down, &down_input, &ids, &down_output, nullptr,
            (const char *) b.layout.bank_down->data, nullptr, (const char *) resource.device_down_q8,
            resource.device_down, 0, b.layout.n_embd, 1, down_padded, stream);
    CUDA_CHECK(cudaGetLastError());

    if (include_cpu_partial &&
            cudaEventRecord(resource.hit_kernels_done, stream) != cudaSuccess) return false;
    return true;
}

static bool ggml_cuda_moe_ffn_reduce_commit(ggml_backend_cuda_context & ctx, ggml_tensor * op,
        const ggml_cuda_moe_ffn_binding & b, ggml_cuda_moe_ffn_cpu_resource_state & resource,
        ggml_cuda_moe_ffn_stats & stats, int32_t n_hits, bool include_cpu_partial) {
    cudaStream_t stream = ctx.stream();
    const int32_t n_rows = n_hits + (include_cpu_partial ? 1 : 0);
    if (include_cpu_partial) {
        const size_t offset = size_t(n_hits) * b.layout.n_embd * sizeof(float);
        const size_t bytes = size_t(b.layout.n_embd) * sizeof(float);
        if (cudaMemcpyAsync((char *) resource.device_down + offset, resource.host_output,
                    bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        ++stats.cuda_copy_calls;
        stats.output_copy_bytes += bytes;
    }
    ggml_tensor reduce_input{};
    reduce_input.type = GGML_TYPE_F32;
    reduce_input.ne[0] = b.layout.n_embd; reduce_input.ne[1] = n_rows;
    reduce_input.ne[2] = reduce_input.ne[3] = 1;
    reduce_input.nb[0] = sizeof(float); reduce_input.nb[1] = b.layout.n_embd * sizeof(float);
    reduce_input.nb[2] = b.layout.n_embd * n_rows * sizeof(float); reduce_input.nb[3] = reduce_input.nb[2];
    reduce_input.data = resource.device_down;
    ggml_tensor routes = *op->src[2];
    if (include_cpu_partial) {
        routes.ne[0] = 1; routes.ne[1] = n_rows; routes.ne[2] = routes.ne[3] = 1;
        routes.nb[0] = sizeof(float); routes.nb[1] = sizeof(float);
        routes.nb[2] = routes.nb[3] = n_rows * sizeof(float);
        routes.data = resource.device_routes;
    }
    ggml_tensor reduce_output{};
    reduce_output.type = GGML_TYPE_F32;
    reduce_output.ne[0] = b.layout.n_embd; reduce_output.ne[1] = 1;
    reduce_output.ne[2] = reduce_output.ne[3] = 1;
    reduce_output.nb[0] = sizeof(float); reduce_output.nb[1] = b.layout.n_embd * sizeof(float);
    reduce_output.nb[2] = reduce_output.nb[3] = reduce_output.nb[1];
    reduce_output.data = resource.device_output;
    reduce_output.src[0] = &reduce_input;
    reduce_output.src[1] = &routes;
    ggml_cuda_op_mul_multi_add(ctx, &reduce_output);
    CUDA_CHECK(cudaGetLastError());
    const size_t output_bytes = size_t(b.layout.n_embd) * sizeof(float);
    if (cudaMemcpyAsync(op->data, resource.device_output, output_bytes,
                cudaMemcpyDeviceToDevice, stream) != cudaSuccess) return false;
    ++stats.cuda_copy_calls;
    stats.output_copy_bytes += output_bytes;
    ++stats.cuda_calls;
    ++stats.output_commits;
    return true;
}

static bool ggml_cuda_moe_ffn_release_pin_set(const ggml_cuda_moe_ffn_cache_callbacks * callbacks,
        void * context, const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins,
        ggml_cuda_moe_ffn_cpu_resource_state & resource, bool pending) {
    if (n_pins == 0) return true;
    if (!callbacks || !callbacks->release_pins || !callbacks->release_pins(context, pins, n_pins)) {
        ++resource.pin_release_failures;
        return false;
    }
    resource.pin_releases += n_pins;
    if (pending) ++resource.pending_pin_releases;
    return true;
}

static bool ggml_cuda_moe_ffn_release_pending_pins(ggml_cuda_moe_ffn_cpu_resource_state & resource) {
    auto & pending = resource.pending_pins;
    if (!pending.active) return true;
    if (!ggml_cuda_moe_ffn_release_pin_set(pending.callbacks, pending.context, pending.pins.data(),
                (int32_t) pending.pins.size(), resource, true)) {
        return false;
    }
    pending = {};
    return true;
}

static bool ggml_cuda_moe_ffn_validate_resolution(const ggml_cuda_moe_ffn_binding & binding,
        const ggml_cuda_moe_ffn_resolution & resolved) {
    if (resolved.n_draws != binding.layout.n_experts_used || resolved.n_draws < 0 ||
            !resolved.draw_slots || resolved.draw_capacity < (size_t) resolved.n_draws ||
            resolved.n_pins < 0 || resolved.n_pins > resolved.n_draws || !resolved.pins ||
            resolved.pin_capacity < (size_t) resolved.n_pins) return false;
    int32_t unique_hits = 0;
    for (int32_t i = 0; i < resolved.n_draws; ++i) {
        const int32_t slot = resolved.draw_slots[i];
        if (slot < -1 || slot >= binding.layout.n_slots) return false;
        if (slot < 0) continue;
        bool seen = false;
        for (int32_t j = 0; j < i; ++j) seen |= resolved.draw_slots[j] == slot;
        if (!seen) ++unique_hits;
    }
    if (unique_hits != resolved.n_pins) return false;
    for (int32_t i = 0; i < resolved.n_pins; ++i) {
        const auto & pin = resolved.pins[i];
        if (!pin.owner || pin.device != binding.layout.device || pin.group != binding.layout.cache_group ||
                pin.layer != binding.layout.cache_layer || pin.owner_generation != binding.generation ||
                pin.slot < 0 || pin.slot >= binding.layout.n_slots ||
                pin.generation == 0) return false;
        for (int32_t j = 0; j < i; ++j) {
            if (resolved.pins[j].slot == pin.slot) return false;
        }
        bool used = false;
        for (int32_t j = 0; j < resolved.n_draws; ++j) used |= resolved.draw_slots[j] == pin.slot;
        if (!used) return false;
    }
    return true;
}

static void ggml_cuda_moe_ffn_store_pending_pins(ggml_cuda_moe_ffn_cpu_resource_state & resource,
        const ggml_cuda_moe_ffn_binding & binding, const ggml_cuda_moe_ffn_resolution & resolved) {
    auto & pending = resource.pending_pins;
    GGML_ASSERT(!pending.active);
    if (resolved.n_pins == 0) return;
    pending.callbacks = binding.cache_callbacks;
    pending.context = binding.cache_context;
    pending.pins.assign(resolved.pins, resolved.pins + resolved.n_pins);
    pending.active = true;
}

static bool ggml_cuda_moe_ffn_compute(ggml_backend_cuda_context & ctx, ggml_tensor * op) {
    ggml_moe_ffn_op_params params{};
    if (!ggml_cuda_moe_ffn_op_params(op, params) || params.device != ctx.device) return false;
    auto state = ggml_cuda_moe_ffn_lookup(ctx, params.binding_id, params.generation);
    if (!state) return false;
    auto & resource = *state->cpu_resource;
    std::unique_lock<std::mutex> request_lock(resource.execution_mutex);
    auto & b = state->binding;
    auto & stats = state->stats;
    if (!state->active || !ggml_cuda_moe_ffn_shape_matches(ctx, op, params, b)) return false;
    ++stats.capture_bypasses;

    cudaStream_t stream = ctx.stream();
    const size_t ids_bytes = size_t(b.layout.n_experts_used) * sizeof(int32_t);
    const size_t activation_bytes = size_t(b.layout.n_embd) * sizeof(float);
    const size_t route_bytes = size_t(b.layout.n_experts_used) * sizeof(float);
    const size_t output_bytes = size_t(b.layout.n_embd) * sizeof(float);

    if (cudaMemcpyAsync(resource.host_ids, op->src[1]->data, ids_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        ++stats.rejected_calls;
        return false;
    }
    ++stats.cuda_copy_calls;
    stats.id_readback_bytes += ids_bytes;
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        ++stats.rejected_calls;
        return false;
    }
    ++stats.synchronizations;

    if (!ggml_cuda_moe_ffn_release_pending_pins(resource)) {
        ++stats.rejected_calls;
        return false;
    }
    ggml_cuda_moe_ffn_resolution resolved{};
    resolved.draw_slots = resource.host_slots;
    resolved.draw_capacity = (size_t) b.layout.n_experts_used;
    resource.resolved_pins.resize((size_t) b.layout.n_experts_used);
    resolved.pins = resource.resolved_pins.data();
    resolved.pin_capacity = resource.resolved_pins.size();
    const auto * const draw_storage = resolved.draw_slots;
    auto * const pin_storage = resolved.pins;
    const size_t draw_capacity = resolved.draw_capacity;
    const size_t pin_capacity = resolved.pin_capacity;
    const bool resolved_ok = b.cache_callbacks->resolve_and_pin(b.cache_context, b.layout.device,
            b.layout.cache_group, b.layout.cache_layer, b.generation, resource.host_ids,
            b.layout.n_experts_used, &resolved);
    // A C-style owner callback may fill only caller-owned storage. If it reports
    // malformed counts, release every token that could have fit in that storage.
    if (!resolved_ok || resolved.draw_slots != draw_storage || resolved.pins != pin_storage ||
            resolved.draw_capacity != draw_capacity || resolved.pin_capacity != pin_capacity) {
        const int32_t acquired = std::clamp<int32_t>(resolved.n_pins, 0, (int32_t) pin_capacity);
        resource.pin_acquisitions += (uint64_t) acquired;
        if (acquired > 0 && !ggml_cuda_moe_ffn_release_pin_set(b.cache_callbacks,
                    b.cache_context, pin_storage, acquired, resource, false)) {
            GGML_ABORT("failed to release malformed MoE cache pin result");
        }
        ++stats.rejected_calls;
        return false;
    }
    const auto release_current_pins = [&]() {
        return ggml_cuda_moe_ffn_release_pin_set(b.cache_callbacks, b.cache_context,
                resolved.pins, resolved.n_pins, resource, false);
    };
    if (!ggml_cuda_moe_ffn_validate_resolution(b, resolved)) {
        const int32_t acquired = std::clamp<int32_t>(resolved.n_pins, 0, (int32_t) pin_capacity);
        resource.pin_acquisitions += (uint64_t) acquired;
        if (acquired > 0 && !ggml_cuda_moe_ffn_release_pin_set(b.cache_callbacks,
                    b.cache_context, pin_storage, acquired, resource, false))
            GGML_ABORT("failed to release invalid MoE cache pin result");
        ++stats.rejected_calls;
        return false;
    }
    ++resource.pin_resolve_calls;
    resource.pin_acquisitions += resolved.n_pins;
    int32_t n_hits = 0;
    for (int32_t i = 0; i < b.layout.n_experts_used; ++i) {
        const int32_t slot = resolved.draw_slots[i];
        resource.host_slots[i] = slot;
        if (slot >= 0) resource.host_hit_slots[n_hits++] = slot;
    }
    stats.selected_ids += b.layout.n_experts_used;

    const int32_t n_misses = b.layout.n_experts_used - n_hits;
    resource.fill_candidates.clear();
    int32_t n_fill_candidates = 0;
    for (int32_t i = 0; i < b.layout.n_experts_used; ++i) {
        if (resource.host_slots[i] >= 0) continue;
        const int32_t expert = resource.host_ids[i];
        bool duplicate = false;
        for (int32_t j = 0; j < n_fill_candidates; ++j) duplicate |= resource.fill_candidates[j] == expert;
        if (!duplicate) { resource.fill_candidates.push_back(expert); ++n_fill_candidates; }
    }
    for (int32_t i = 0; i < n_fill_candidates; ++i) {
        ggml_cuda_moe_ffn_enqueue_fill(resource, b, params.binding_id, resource.fill_candidates[i]);
    }
    bool cpu_inputs_ready = false;
    const auto read_cpu_inputs = [&]() {
        if (cudaMemcpyAsync(resource.host_activations, op->src[0]->data, activation_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
            cudaMemcpyAsync(resource.host_routes, op->src[2]->data, route_bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
            return false;
        }
        stats.input_copy_bytes += activation_bytes + route_bytes;
        stats.cuda_copy_calls += 2;
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            return false;
        }
        ++stats.synchronizations;
        cpu_inputs_ready = true;
        return true;
    };
    if (n_misses > 0 && !read_cpu_inputs()) {
        if (!release_current_pins()) GGML_ABORT("failed to release MoE cache pins before hit dispatch");
        ++stats.rejected_calls;
        return false;
    }
    if (n_misses > 0) {
        int32_t packed_hits = 0;
        for (int32_t i = 0; i < b.layout.n_experts_used; ++i) {
            if (resource.host_slots[i] >= 0) {
                resource.host_hit_routes[packed_hits++] = resource.host_routes[i];
            }
        }
        if (packed_hits != n_hits) {
            if (!release_current_pins()) GGML_ABORT("failed to release malformed MoE cache hit pins");
            return false;
        }
    }

    if (n_misses == 0) {
        ++stats.all_hit_calls;
    } else {
        ++stats.any_miss_calls;
    }
    stats.gpu_hit_draws += n_hits;
    stats.cpu_miss_draws += n_misses;

    const auto complete_cpu_fallback = [&]() {
        if (!cpu_inputs_ready) {
            if (!read_cpu_inputs()) return false;
        }
        ++stats.complete_cpu_fallback_calls;
        ++stats.cpu_calls;
        if (!ggml_cuda_moe_ffn_cpu_draws(b, resource, stats, false, activation_bytes)) return false;
        if (cudaMemcpyAsync(op->data, resource.host_output, output_bytes,
                    cudaMemcpyHostToDevice, stream) != cudaSuccess) return false;
        ++stats.cuda_copy_calls;
        stats.output_copy_bytes += output_bytes;
        ++stats.output_commits;
        return true;
    };

    if (n_hits == 0) {
        if (!release_current_pins()) GGML_ABORT("failed to release pins for all-miss MoE operation");
        if (!complete_cpu_fallback()) { ++stats.rejected_calls; return false; }
        return true;
    }

    bool kernels_started = false;
    bool dispatch_ok = ggml_cuda_moe_ffn_enqueue_hit_rows(ctx, op, b, resource, stats, n_hits,
            n_misses > 0, kernels_started);
    if (kernels_started) {
        ggml_cuda_moe_ffn_store_pending_pins(resource, b, resolved);
    } else if (!release_current_pins()) {
        GGML_ABORT("failed to release MoE cache pins after pre-dispatch rejection");
    }
    if (!dispatch_ok) {
        if (kernels_started && cudaStreamSynchronize(stream) != cudaSuccess) {
            ++stats.rejected_calls;
            return false;
        }
        if (!complete_cpu_fallback()) { ++stats.rejected_calls; return false; }
        return true;
    }

    if (n_misses == 0) {
        if (!ggml_cuda_moe_ffn_reduce_commit(ctx, op, b, resource, stats, n_hits, false)) {
            if (cudaStreamSynchronize(stream) != cudaSuccess || !complete_cpu_fallback()) {
                ++stats.rejected_calls;
                return false;
            }
        }
        return true;
    }

    const cudaError_t hit_state = cudaEventQuery(resource.hit_kernels_done);
    const bool hit_pending_at_cpu_start = hit_state == cudaErrorNotReady;
    if (hit_state != cudaSuccess && !hit_pending_at_cpu_start) {
        ++stats.rejected_calls;
        return false;
    }
    if (hit_pending_at_cpu_start) ++stats.gpu_pending_at_cpu_start;

    ++stats.cpu_calls;
    if (!ggml_cuda_moe_ffn_cpu_draws(b, resource, stats, true, activation_bytes)) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            ++stats.rejected_calls;
            return false;
        }
        --stats.cpu_calls;
        if (!complete_cpu_fallback()) { ++stats.rejected_calls; return false; }
        return true;
    }
    if (hit_pending_at_cpu_start && cudaEventQuery(resource.hit_kernels_done) == cudaSuccess) {
        ++stats.gpu_completed_during_cpu;
    }
    ++stats.partitioned_calls;
    if (!ggml_cuda_moe_ffn_reduce_commit(ctx, op, b, resource, stats, n_hits, true)) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) {
            ++stats.rejected_calls;
            return false;
        }
        --stats.cpu_calls;
        if (!complete_cpu_fallback()) { ++stats.rejected_calls; return false; }
    }
    return true;
}

static bool ggml_cuda_moe_ffn_validate_graph(ggml_backend_cuda_context & ctx, const ggml_cgraph * graph) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_MOE_FFN && !ggml_cuda_moe_ffn_validate_op(ctx, node)) return false;
    }
    return true;
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst, const ggml_cgraph * cgraph, int & i) {

#if IK_PRINT_TIMING
    int64_t tim1 = ggml_time_us();
#endif

    if (ggml_is_noop(dst)) {
        return true;
    }

    // In case we forget to do that in some kernel.
    ggml_cuda_set_device(ctx.device);

    auto next = i < cgraph->n_nodes - 1 ? cgraph->nodes[i+1] : nullptr;

    auto fusion = ctx.fusion;

    //printf("%4d %s(%s) on device %d. time = %ld\n", i, ggml_op_name(dst->op), dst->name, ctx.device, ggml_time_us());
    switch (dst->op) {
        case GGML_OP_REDUCE:
            ggml_cuda_op_reduce(ctx, dst);
            break;
        case GGML_OP_BLEND:
            ggml_cuda_op_blend(ctx, dst);
            break;
        case GGML_OP_FAKE_CPY:
            break;
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_HADAMARD:
            ggml_cuda_op_hadamard(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_cpy_2(ctx, dst->src[0], cgraph->nodes[i+2]->src[0], dst->src[1], cgraph->nodes[i+2]->src[1])) {
                i += 2;
            } else {
                ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            }
            break;
        case GGML_OP_CONT:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i+2]->op == GGML_OP_TRANSPOSE &&
                dst->src[0]->op == GGML_OP_TRANSPOSE) {
                ggml_cuda_op_sum_rows_nc(ctx, cgraph->nodes[i+1]);
                i += 2;
            } else {
                ggml_cuda_dup(ctx, dst);
            }
            break;
        case GGML_OP_ADD:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ADD &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+2]->op_params[0] < 2 &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                dst->src[0]->type == GGML_TYPE_F32 &&               // with split mode "attn" we can end up having f16
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                dst == cgraph->nodes[i+1]->src[0] &&
                ggml_is_contiguous(cgraph->nodes[i+1]->src[1]) &&
                ggml_are_same_shape(dst, cgraph->nodes[i+1]->src[1]) &&
                cgraph->nodes[i+1] == cgraph->nodes[i+2]->src[0] &&
                ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_fused_add_add_rms_norm(ctx, dst, cgraph->nodes[i+1], cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_SOFTPLUS &&
                cgraph->nodes[i+2]->op == GGML_OP_MUL &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+1]->src[0] == cgraph->nodes[i] &&
                ggml_nrows(cgraph->nodes[i+0]->src[1]) == 1 &&
                ggml_nrows(cgraph->nodes[i+2]->src[1]) == 1) {
                ggml_cuda_fused_softplus(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            else if (false && fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+1]->op_params[0] < 2 &&
                ggml_is_contiguous(dst->src[0]) &&
                ggml_is_contiguous(dst->src[1]) &&
                ggml_are_same_shape(dst->src[0], dst->src[1]) &&
                dst == cgraph->nodes[i+1]->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_fused_add_rms_norm(ctx, dst, cgraph->nodes[i+1]);
                ++i;
            } else {
                ggml_cuda_op_add(ctx, dst);
            }
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_MULTI_ADD:
            ggml_cuda_op_multi_add(ctx, dst);
            break;
        case GGML_OP_MUL_MULTI_ADD:
            ggml_cuda_op_mul_multi_add(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                cgraph->nodes[i+2]->op == GGML_OP_MUL &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_EXP &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[1] == dst->src[1]) {
                ggml_cuda_fused_mul_exp_mul(ctx, cgraph->nodes[i+2]);
                i += 2;
                //printf("mul(%s) -> exp(%s) -> mul(%s), %d, %d, %zu, %zu; %ld x %ld x %ld x %ld - %ld x %ld x %ld x %ld\n", dst->name, cgraph->nodes[i+1]->name, cgraph->nodes[i+2]->name,
                //        ggml_is_contiguous(dst->src[0]), ggml_is_contiguous(dst->src[1]), ggml_nelements(dst->src[0]), ggml_nelements(dst->src[1]),
                //        dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3],
                //        dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3]);
            } else {
                //printf("mul(%s): %d, %d, %d, %ld x %ld x %ld x %ld * %ld x %ld x %ld x %ld\n", dst->name, ggml_is_contiguous(dst->src[0]), ggml_is_contiguous(dst->src[1]), ggml_is_contiguous(dst),
                //        dst->src[0]->ne[0], dst->src[0]->ne[1], dst->src[0]->ne[2], dst->src[0]->ne[3],
                //        dst->src[1]->ne[0], dst->src[1]->ne[1], dst->src[1]->ne[2], dst->src[1]->ne[3]);
                ggml_cuda_op_mul(ctx, dst);
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY:
            ggml_cuda_op_fused_mul_unary(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_UNARY:
            //printf("unary(%s, %s)\n", dst->name, ggml_unary_op_name((ggml_unary_op)dst->op_params[0]));
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    if (fusion && i + 5 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                        cgraph->nodes[i+4]->op == GGML_OP_VIEW &&
                        cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+5)) {
                        cuda_glm45moe_experts(ctx, cgraph->nodes[i+5], cgraph->nodes[i+4]);
                        i += 5;
                    }
                    else if (fusion && i + 4 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                        cgraph->nodes[i+3]->op == GGML_OP_GROUPED_TOPK &&
                        cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS && ops_are_same_device(cgraph, i, i+4)) {
                        cuda_bailingmoev2_experts(ctx, cgraph->nodes[i+4], cgraph->nodes[i+3]);
                        i += 4;
                    } else if (fusion && i + 2 < cgraph->n_nodes &&
                        cgraph->nodes[i+1]->op == GGML_OP_RESHAPE &&
                        cgraph->nodes[i+2]->op == GGML_OP_ADD && ops_are_same_device(cgraph, i, i+2)) {
                        ggml_cuda_op_biased_sigmoid(ctx, cgraph->nodes[i+2]);
                        i += 2;
                    } else {
                        ggml_cuda_op_sigmoid(ctx, dst);
                    }
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                case GGML_UNARY_OP_SQRT_SOFTPLUS:
                    ggml_cuda_op_sqrt_softplus(ctx, dst);
                    break;
                default:
                    return -1;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_CPY &&
                ggml_cuda_concat_cpy(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            } else {
                ggml_cuda_op_concat(ctx, dst);
            }
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_FUSED_RMS_NORM:
            if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+2]->op_params[1] < 2 &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+3], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (false && fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST &&
                cgraph->nodes[i+2]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+3]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+3]->op_params[1] < 2 &&
                cgraph->nodes[i+4]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rms_rope_fast(ctx, cgraph->nodes[i+1], cgraph->nodes[i+4])) {
                i += 4;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                (cgraph->nodes[i+1]->op == GGML_OP_VIEW || cgraph->nodes[i+1]->op == GGML_OP_RESHAPE) &&
                cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                cgraph->nodes[i+2]->op_params[1] < 2 &&
                dst->ne[2] == 1 && cgraph->nodes[i+2]->ne[2] == 1) {
                ggml_cuda_op_fused_rms_rms_norm(ctx, dst, cgraph->nodes[i+2]);
                i += 2;
            }
            else {
                int inow = i;
                // First try rms -> add -> rms
                // This doesn't always work because the second rms result may get allocated on top of the
                // first rms source
                if (fusion && i + 2 < cgraph->n_nodes &&
                    cgraph->nodes[i+1]->op == GGML_OP_ADD &&
                    cgraph->nodes[i+2]->op == GGML_OP_FUSED_RMS_NORM &&
                    dst->src[0]->ne[1] == 1 &&
                    dst->src[0]->type != GGML_TYPE_Q8_0 && // In case someone has decided to use Q8_0 as the graph reduce type
                    cgraph->nodes[i+1]->src[0] == dst &&
                    cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                    ggml_are_same_shape(dst, cgraph->nodes[i+1]->src[1]) &&
                    cgraph->nodes[i+0]->op_params[1] < 2 &&
                    cgraph->nodes[i+2]->op_params[1] < 2) {
                    auto src0 = (const char *)dst->src[0]->data;
                    auto src0_end = src0 + ggml_nbytes(dst->src[0]);
                    auto add1 = (const char *)cgraph->nodes[i+1]->data;
                    auto rms2 = (const char *)cgraph->nodes[i+2]->data;
                    auto nbytes = ggml_nbytes(dst);
                    bool overlap1 = add1 > src0 && add1 < src0_end;
                    bool overlap2 = add1 + nbytes > src0 && add1 + nbytes < src0_end;
                    bool overlap3 = add1 <= src0 && add1 + nbytes >= src0_end && !(add1 == src0 && add1 + nbytes == src0_end);
                    bool overlap4 = rms2 > src0 && rms2 < src0_end;
                    bool overlap5 = rms2 + nbytes > src0 && rms2 + nbytes < src0_end;
                    bool overlap6 = rms2 <= src0 && rms2 + nbytes >= src0_end && !(rms2 == src0 && rms2 + nbytes == src0_end);
                    if (!overlap1 && !overlap2 && !overlap3 && !overlap4 && !overlap5 && !overlap6) {
                        ggml_cuda_op_fused_rms_add_rms(ctx, cgraph->nodes[i+2]);
                        i += 2;
                    }
                }
                // If that did not work, try rms -> add
                if (fusion && inow == i && i + 1 < cgraph->n_nodes &&
                    cgraph->nodes[i+0]->op_params[1] < 2 &&
                    dst->src[0]->type != GGML_TYPE_Q8_0 && // In case someone has decided to use Q8_0 as the graph reduce type
                    cgraph->nodes[i+1]->op == GGML_OP_ADD &&
                    cgraph->nodes[i+1]->src[0] == dst &&
                    ggml_are_same_shape(dst, cgraph->nodes[i+1]->src[1])) {
                    ggml_cuda_op_fused_rms_add(ctx, cgraph->nodes[i+1]);
                    i += 1;
                }
                if (inow == i) {
                    ggml_cuda_op_fused_rms_norm(ctx, dst);
                }
            }
            break;
        case GGML_OP_FUSED_RMS_RMS_ADD:
            ggml_cuda_op_fused_rms_rms_add(ctx, dst);
            break;
        case GGML_OP_FUSED_NORM:
            ggml_cuda_op_fused_rms_norm(ctx, dst, true);
            break;
        case GGML_OP_MUL_MAT:
            if (dst->src[0]->ne[3] != dst->src[1]->ne[3]) {
                GGML_CUDA_LOG_ERROR("%s: cannot compute %s: src0->ne[3] = %" PRId64 ", src1->ne[3] = %" PRId64 " - fallback to CPU\n", __func__, dst->name, dst->src[0]->ne[3], dst->src[1]->ne[3]);
                return -1;
            } else {
                i = ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst, cgraph, i);
            }
            break;
        case GGML_OP_MUL_MAT_ID:
            if (ggml_cuda_mul_mat_id(ctx, dst, next)) ++i;
            break;
        case GGML_OP_MOE_FUSED_UP_GATE:
            i = ggml_cuda_moe_up_gate_unary(ctx, dst, cgraph, i);
            break;
        case GGML_OP_MOE_FFN:
            if (!ggml_cuda_moe_ffn_compute(ctx, dst)) return false;
            break;
        case GGML_OP_FUSED_UP_GATE:
            ggml_cuda_up_gate_unary(ctx, dst);
            break;
        case GGML_OP_SCALE:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                cgraph->nodes[i+2]->op == GGML_OP_SCALE &&
                cgraph->nodes[i+1]->src[0] == cgraph->nodes[i+0] &&
                cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+1] &&
                (ggml_unary_op)cgraph->nodes[i+1]->op_params[0] == GGML_UNARY_OP_SOFTPLUS) {
                ggml_cuda_op_scaled_softplus(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_UNARY &&
                cgraph->nodes[i+1]->src[0] == dst &&
                ggml_cuda_op_scale_unary(ctx, cgraph->nodes[i+1])) {
                i += 1;
            } else {
                ggml_cuda_op_scale(ctx, dst);
            }
            break;
        case GGML_OP_SOFTCAP:
            ggml_cuda_op_softcap(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            if (fusion && i + 8 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ADD &&
                cgraph->nodes[i+3]->op == GGML_OP_ARGSORT &&
                cgraph->nodes[i+4]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+5]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+6]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+7]->op == GGML_OP_SUM_ROWS &&
                cgraph->nodes[i+8]->op == GGML_OP_DIV) {
                ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+8], cgraph->nodes[i+4], cgraph->nodes[i+2]->src[1]);
                i += 8;
            }
            else if (fusion && i + 4 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_RESHAPE  &&
                cgraph->nodes[i+2]->op == GGML_OP_ARGSORT  &&
                cgraph->nodes[i+3]->op == GGML_OP_VIEW     &&
                cgraph->nodes[i+4]->op == GGML_OP_GET_ROWS &&
                ggml_cuda_should_use_topk_moe(cgraph->nodes[i], cgraph->nodes[i+4]) &&
                ops_are_same_device(cgraph, i, i+4)) {
                if (i + 7 < cgraph->n_nodes &&
                    cgraph->nodes[i+5]->op == GGML_OP_RESHAPE  &&
                    cgraph->nodes[i+6]->op == GGML_OP_SUM_ROWS &&
                    cgraph->nodes[i+7]->op == GGML_OP_DIV) {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+7], cgraph->nodes[i+3]);
                    i += 7;
                } else {
                    ggml_cuda_op_topk_moe(ctx, cgraph->nodes[i], cgraph->nodes[i+4], cgraph->nodes[i+3]);
                    i += 4;
                }
            } else {
                ggml_cuda_op_soft_max(ctx, dst);
            }
            break;
        case GGML_OP_SOFT_CAP_MAX:
            ggml_cuda_op_soft_cap_max(ctx, dst);
            break;
        case GGML_OP_ROPE:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE &&
                ggml_cuda_op_rope_rope(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            } else {
                ggml_cuda_op_rope(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROPE_FAST:
            if (fusion && i + 3 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
               (cgraph->nodes[i+2]->op == GGML_OP_RESHAPE || cgraph->nodes[i+2]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+3]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+3])) {
                i += 3;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
               (cgraph->nodes[i+1]->op == GGML_OP_RESHAPE || cgraph->nodes[i+1]->op == GGML_OP_VIEW) &&
                cgraph->nodes[i+2]->op == GGML_OP_ROPE_FAST &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+2])) {
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_ROPE_FAST   &&
                ggml_cuda_op_fused_rope_fast(ctx, dst, cgraph->nodes[i+1])) {
                i += 1;
            }
            else {
                ggml_cuda_op_rope_fast(ctx, dst);
            }
            break;
        case GGML_OP_ROPE_CACHE:
            ggml_cuda_op_rope_cache(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            if (fusion && i + 2 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_SCALE &&
                cgraph->nodes[i+2]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[0] == dst &&
                cgraph->nodes[i+2]->src[1] == cgraph->nodes[i+1] &&
                cgraph->nodes[i+2]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+2)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+2]);
                i += 2;
            }
            else if (fusion && i + 1 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_DIV &&
                cgraph->nodes[i+1]->src[1] == dst &&
                cgraph->nodes[i+1]->src[0] == dst->src[0] && ops_are_same_device(cgraph, i, i+1)) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+1]);
                ++i;
            }
            else if (fusion && i + 2 < cgraph->n_nodes &&
                    cgraph->nodes[i+1]->op == GGML_OP_CLAMP &&
                    cgraph->nodes[i+2]->op == GGML_OP_DIV &&
                    cgraph->nodes[i+1]->src[0] == cgraph->nodes[i+0] &&
                    cgraph->nodes[i+2]->src[1] == cgraph->nodes[i+1] &&
                    cgraph->nodes[i+2]->src[0] == cgraph->nodes[i+0]->src[0]) {
                ggml_cuda_op_sum_rows_div(ctx, cgraph->nodes[i+2]);
                i += 2;
            } else {
                ggml_cuda_op_sum_rows(ctx, dst);
            }
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            if (fusion && i + 5 < cgraph->n_nodes &&
                cgraph->nodes[i+1]->op == GGML_OP_VIEW &&
                cgraph->nodes[i+2]->op == GGML_OP_GET_ROWS &&
                cgraph->nodes[i+3]->op == GGML_OP_RESHAPE &&
                cgraph->nodes[i+4]->op == GGML_OP_SOFT_MAX &&
                cgraph->nodes[i+5]->op == GGML_OP_RESHAPE && ops_are_same_device(cgraph, i, i+4)) {
                cuda_openai_experts(ctx, dst, cgraph->nodes[i+4]);
                i += 5;
            } else {
                ggml_cuda_op_argsort(ctx, dst);
            }
            break;
        case GGML_OP_ARGSORT_THRESH:
            ggml_cuda_op_argsort_thresh(ctx, dst);
            break;
        case GGML_OP_GROUPED_TOPK:
            ggml_cuda_op_grouped_topk(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_DELTA_NET: {
            const auto op_delta_net = dst->src[3]->ne[1] == 1 ? ggml_cuda_op_delta_net : ggml_cuda_op_kda;
            const int j = fusion ? ggml_delta_net_find_state_cpy(cgraph, i) : -1;
            if (j >= 0) {
                ggml_tensor fused = *dst;
                fused.src[7] = cgraph->nodes[j]->src[1];
                op_delta_net(ctx, &fused);
#ifdef USE_CUDA_GRAPH
                // claim the entry of the copy that is not going to be launched
                if (ctx.cur_graph && ctx.cur_graph->use_cpy_indirection) {
                    ctx.cur_graph->graph_cpynode_index++;
                }
#endif
                i = j;
            } else {
                op_delta_net(ctx, dst);
            }
        } break;
        case GGML_OP_SINKHORN:
            ggml_cuda_op_sinkhorn(ctx, dst);
            break;
        case GGML_OP_LATENT_ATTN:
            ggml_cuda_op_latent_attn(ctx, dst);
            break;
        case GGML_OP_HC_PRE:
            ggml_cuda_op_hc_pre(ctx, dst);
            break;
        case GGML_OP_HC_POST:
            ggml_cuda_op_hc_post(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_INDEXER_TOPK:
            ggml_cuda_op_indexer_topk(ctx, dst);
            break;
        case GGML_OP_MASK_TOPK:
            ggml_cuda_op_indexer_mask(ctx, dst);
            break;
        case GGML_OP_MASK_TO_IDX:
            ggml_cuda_op_mask_to_index(ctx, dst);
            break;
        case GGML_OP_DS4_COMP:
            ggml_cuda_op_ds4_comp(ctx, dst);
            break;
        default:
            return false;
    }

#if 0
    if (auto err = cudaStreamSynchronize(ctx.stream()); err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }
#endif

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

#if IK_PRINT_TIMING
    if (auto err = cudaStreamSynchronize(ctx.stream()); err != cudaSuccess) {
        GGML_CUDA_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }
    int64_t tim2 = ggml_time_us();
    printf("%s(%s): %d us\n", ggml_op_name(dst->op), dst->name, (int)(tim2 - tim1));
#endif

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

GGML_CALL static const char * ggml_backend_cuda_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

GGML_CALL static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cuda_get_default_buffer_type(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return ggml_backend_cuda_buffer_type(cuda_ctx->device);
}

GGML_CALL static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *)tensor->data + offset, data, size, cudaMemcpyHostToDevice, cuda_ctx->stream()));
}

GGML_CALL static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *)tensor->data + offset, size, cudaMemcpyDeviceToHost, cuda_ctx->stream()));
}

GGML_CALL static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(src->buffer) || !ggml_backend_buffer_is_cuda(dst->buffer)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *)backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *)backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *)buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *)buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_CUDA_LOG_WARN("%s: backend and buffer devices do not match\n", __func__);
#endif
        return false;
    }

    if (backend_src != backend_dst) {
        ggml_cuda_pool_alloc<half> tmp_src(cuda_ctx_src->pool());
        ggml_cuda_pool_alloc<half> tmp_dst(cuda_ctx_dst->pool());
        bool needs_f16_f32_copy = false;
        // copy on src stream
        if (cuda_ctx_src->device == cuda_ctx_dst->device) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            if (false && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && dst->ne[1] >= 32) {
                //
                // The goal here is to reduce traffic between GPU's, which is entirely non-negligible
                // for prompt processing.
                // We cast the tensor to be copied to f16, copy the f16 data peer-to-peer
                // and then cast back to f32 on the destination side.
                // The cost for converting to/from f16 is much lower than the cost of copying
                // two times more data over PCI-E (well, at least the 30 GB/s PCI-E I have).
                // But for some reason the following is slower.
                // Can somebody tell me why?
                //

                ggml_cuda_set_device(cuda_ctx_dst->device);
                tmp_dst.alloc(ggml_nelements(dst));

                ggml_cuda_set_device(cuda_ctx_src->device);
                tmp_src.alloc(ggml_nelements(src));

                auto src_f16 = *src;
                src_f16.type = GGML_TYPE_F16;
                for (int i = 0; i < 4; ++i) src_f16.nb[i] /= 2;
                src_f16.data = tmp_src.get();

                ggml_cuda_cpy(*cuda_ctx_src, src, &src_f16, true);

                CUDA_CHECK(cudaMemcpyPeerAsync(tmp_dst.ptr, cuda_ctx_dst->device, src_f16.data, cuda_ctx_src->device, ggml_nbytes(&src_f16), cuda_ctx_src->stream()));

                needs_f16_f32_copy = true;

            } else {
                ggml_cuda_set_device(cuda_ctx_src->device);
                CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, cuda_ctx_dst->device, src->data, cuda_ctx_src->device, ggml_nbytes(dst), cuda_ctx_src->stream()));
            }
#endif
        }

        // record event on src stream after the copy
        ggml_cuda_set_device(cuda_ctx_src->device);
        if (!cuda_ctx_src->copy_event) {
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        ggml_cuda_set_device(cuda_ctx_dst->device);
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
        if (needs_f16_f32_copy) {
            auto dst_f16 = *dst;
            dst_f16.type = GGML_TYPE_F16;
            for (int i = 0; i < 4; ++i) dst_f16.nb[i] /= 2;
            dst_f16.data = tmp_dst.get();
            ggml_cuda_cpy(*cuda_ctx_dst, &dst_f16, dst, true);
        }
    } else {
        // src and dst are on the same backend
        // printf("Why is this being invoked?\n");
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

GGML_CALL static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

#ifdef USE_CUDA_GRAPH

static inline const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph) {
    return cgraph->nodes[0];
}

static inline ggml_cuda_graph * ggml_cuda_get_graph(ggml_backend_cuda_context & ctx, const void * key) {
    auto & graph = ctx.cuda_graphs[key];
    if (!graph) {
        graph = std::make_unique<ggml_cuda_graph>();
    }
    if (ctx.max_graphs > 0) {
        graph->last_used_time = ggml_time_us();
        if (ctx.cuda_graphs.size() > size_t(ctx.max_graphs)) {
            auto lru = ctx.cuda_graphs.begin();
            for (auto it = ctx.cuda_graphs.begin(); it != ctx.cuda_graphs.end(); ++it) {
                if (it->second->last_used_time < lru->second->last_used_time) {
                    lru = it;
                }
            }
            ctx.cuda_graphs.erase(lru);
        }
    }
    return graph.get();
}

static bool check_node_graph_compatibility_and_refresh_copy_ops(ggml_cuda_graph * graph, ggml_cgraph * cgraph,
    bool use_cuda_graph, cudaStream_t stream) {

    // Loop over nodes in GGML graph to obtain info needed for CUDA graph
    graph->cpy_dest_ptrs.clear();

    const std::string gemma3n_per_layer_proj_src0_name = "inp_per_layer_selected";
    const std::string gemma3n_per_layer_proj_src1_name = "per_layer_proj";
    const std::string ffn_moe_gate_bias_prefix = "ffn_moe_gate_biased";
    const std::string ffn_moe_up_bias_prefix = "ffn_moe_up_biased";
    const std::string ffn_moe_down_bias_prefix = "ffn_moe_down_biased";

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_is_noop(node)) continue;

        if (node->op == GGML_OP_REDUCE) {
            use_cuda_graph = false;
            break;
        }

        if (node->op == GGML_OP_MUL_MAT_ID && (node->ne[2] != 1 || node->src[2]->ne[0] != 1)) {
            use_cuda_graph = false; // This node type is not supported by CUDA graph capture
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s(%s): disabling CUDA graphs due to unsupported node type %ld %ld\n",
                    __func__, node->src[0]->name, node->ne[2], node->src[2]->ne[0]);
#endif
        }
        if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
            auto src0_1 = node->src[0];
            auto src0_2 = node->src[1];
            auto src1   = node->src[2];
            if (src1->ne[1] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1 || src1->type != GGML_TYPE_F32 ||
                !ggml_is_quantized(src0_1->type) || (src0_2 && !ggml_is_quantized(src0_2->type))) {
                use_cuda_graph = false;
            } else {
                if (i < cgraph->n_nodes-1) {
                    auto next = cgraph->nodes[i+1];
                    if (next->op == GGML_OP_MUL_MAT_ID && ggml_is_quantized(next->src[0]->type)) {
                        ++i;
                    }
                }
            }
        }

        // Why was this needed? Leaving it in place but disabled in case it is actually needed.
        if (false && node->op == GGML_OP_ADD &&
            node->src[1] && node->src[1]->ne[1] > 1 &&
            (node->src[0] ? node->src[0]->name != gemma3n_per_layer_proj_src0_name : true) &&
            (node->src[1] ? node->src[1]->name != gemma3n_per_layer_proj_src1_name : true) &&
            strncmp(node->name, ffn_moe_gate_bias_prefix.c_str(), ffn_moe_gate_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_up_bias_prefix.c_str(), ffn_moe_up_bias_prefix.size()) != 0 &&
            strncmp(node->name, ffn_moe_down_bias_prefix.c_str(), ffn_moe_down_bias_prefix.size()) != 0) {
            // disable CUDA graphs for batch size > 1 for now while excluding the matrix-matrix addition as part of Gemma3n's `project_per_layer_input` operation
            // by means of matching node names. See
            // https://github.com/ggml-org/llama.cpp/blob/f9a31eea06a859e34cecb88b4d020c7f03d86cc4/src/llama-model.cpp#L10199-L10241 and
            // https://github.com/huggingface/transformers/blob/bda75b4011239d065de84aa3e744b67ebfa7b245/src/transformers/models/gemma3n/modeling_gemma3n.py#L1773,
            // Generally, changes in batch size or context size can cause changes to the grid size of some kernels.
            use_cuda_graph = false;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to batch size > 1 [%s] [%ld %ld %ld %ld]\n", __func__, node->name, node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
#endif
        }

        if (node->op == GGML_OP_CPY) {

            // Store the pointers which are updated for each token, such that these can be sent
            // to the device and accessed using indirection from CUDA graph
            graph->cpy_dest_ptrs.push_back((char *) node->src[1]->data);

            // store a pointer to each copy op CUDA kernel to identify it later
            void * ptr = ggml_cuda_cpy_fn(node->src[0], node->src[1]);
            if (!ptr) {
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported copy op\n", __func__);
#endif
            }
        }
        if (!use_cuda_graph) {
            break;
        }
    }

    if (use_cuda_graph) {
        graph->use_cpy_indirection = true;
        // copy pointers to GPU so they can be accessed via indirection within CUDA graph
        ggml_cuda_cpy_dest_ptrs_copy(graph, graph->cpy_dest_ptrs.data(), graph->cpy_dest_ptrs.size(), stream);
    } else {
        graph->use_cpy_indirection = false;
    }

    return use_cuda_graph;
}

static void set_ggml_graph_node_properties(ggml_tensor * node, ggml_graph_node_properties * graph_node_properties) {
    graph_node_properties->node_address = node->data;
    graph_node_properties->node_op = node->op;
    graph_node_properties->type = node->type;
    graph_node_properties->view_src = node->view_src;
    graph_node_properties->view_offs = node->view_offs;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        graph_node_properties->ne[i] = node->ne[i];
        graph_node_properties->nb[i] = node->nb[i];
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        graph_node_properties->src_address[i] = node->src[i] ? node->src[i]->data : nullptr;
    }
    memcpy(graph_node_properties->op_params, node->op_params, GGML_MAX_OP_PARAMS);
}

static bool is_cuda_graph_update_required(ggml_cuda_graph * graph, ggml_cgraph * cgraph) {

    if (cgraph->uid != 0 && graph->uid == cgraph->uid) {
        GGML_ASSERT(graph->ggml_graph_properties.size() == (size_t)cgraph->n_nodes);
        return false;
    }

    bool cuda_graph_update_required = false;

    if (graph->instance == nullptr) {
        cuda_graph_update_required = true;
    }

    // Check if the graph size has changed
    if (graph->ggml_graph_properties.size() != (size_t)cgraph->n_nodes) {
        cuda_graph_update_required = true;
        graph->ggml_graph_properties.resize(cgraph->n_nodes);
    }

    // Loop over nodes in GGML graph to determine if CUDA graph update is required
    // and store properties to allow this comparison for the next token
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_graph_node_properties new_props;
        set_ggml_graph_node_properties(cgraph->nodes[i], &new_props);
        if (memcmp(&graph->ggml_graph_properties[i], &new_props, sizeof(new_props)) != 0) {
            cuda_graph_update_required = true;
            memcpy(&graph->ggml_graph_properties[i], &new_props, sizeof(new_props));
        }
    }

    graph->uid = cgraph->uid;

    return cuda_graph_update_required;
}

static void update_cuda_graph_executable(ggml_cuda_graph * graph) {

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_CUDA_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif

static bool evaluate_and_capture_cuda_graph(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph,
    bool & graph_evaluated_or_captured, bool & use_cuda_graph, bool & cuda_graph_update_required) {
    // flag used to determine whether it is an integrated_gpu
    // TODO
    [[maybe_unused]] const bool integrated = false; //ggml_cuda_info().devices[cuda_ctx->device].integrated;

#ifdef USE_CUDA_GRAPH
    auto graph = use_cuda_graph ? ggml_cuda_get_graph(*cuda_ctx, ggml_cuda_graph_get_key(cgraph)) : nullptr;
#endif

#if IK_PRINT_TIMING
    printf("======================== %s: graph with %d nodes on device %d. time = %ld\n", __func__, cgraph->n_nodes, cuda_ctx->device, ggml_time_us());
#endif
    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {

            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];

                if (ggml_is_noop(node)) continue;

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node, cgraph, i);
                if (!ok) {
                    if (node->op == GGML_OP_MOE_FFN) return false;
                    GGML_CUDA_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
            }
        }
#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (--ggml_cuda_lock_counter == 0) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            update_cuda_graph_executable(graph);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
    }
#else
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    return true;
}

GGML_CALL static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
    if (!ggml_cuda_moe_ffn_validate_graph(*cuda_ctx, cgraph)) return GGML_STATUS_FAILED;
    bool split_has_moe_ffn = false;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        split_has_moe_ffn = split_has_moe_ffn || cgraph->nodes[i]->op == GGML_OP_MOE_FFN;
    }

#ifdef USE_CUDA_GRAPH
    cuda_ctx->cur_graph = nullptr;

    static const bool disable_cuda_graphs_due_to_env = (getenv("GGML_CUDA_DISABLE_GRAPHS") != nullptr);

    // Disable CUDA graphs in presence of env var, old GPU, use-case which is changing too rapidly,
    // or previous graph capture failure.
    // Also disable for multi-gpu for now. TO DO investigate
    bool use_cuda_graph = !split_has_moe_ffn && !disable_cuda_graphs_due_to_env && cuda_ctx->use_cuda_graph;

    ggml_cuda_graph * graph = nullptr;
    if (use_cuda_graph) {
        auto graph_key = ggml_cuda_graph_get_key(cgraph);
        graph = ggml_cuda_get_graph(*cuda_ctx, graph_key);
    }
    cuda_ctx->cur_graph = graph;

    bool cuda_graph_update_required = false;

    if (use_cuda_graph && graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < CC_AMPERE) {
            graph->disable_due_to_gpu_arch = true;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && (
        graph->disable_due_to_gpu_arch ||
        graph->disable_due_to_too_many_updates ||
        graph->disable_due_to_failed_graph_capture)) {
        use_cuda_graph = false;
    }

    if (use_cuda_graph) {
        cuda_graph_update_required = is_cuda_graph_update_required(graph, cgraph);

        use_cuda_graph = check_node_graph_compatibility_and_refresh_copy_ops(graph, cgraph, use_cuda_graph, cuda_ctx->stream());

        // Disable CUDA graphs (from the next token) if the use-case is demanding too many consecutive graph updates.
        if (use_cuda_graph) {
            if (cuda_graph_update_required) {
                graph->number_consecutive_updates++;
            } else {
                graph->number_consecutive_updates = 0;
            }
        }

        if (graph->number_consecutive_updates >= 4) {
            graph->disable_due_to_too_many_updates = true;
            use_cuda_graph = false;
            cuda_ctx->cur_graph = nullptr;
#ifndef NDEBUG
            GGML_CUDA_LOG_DEBUG("%s: disabling CUDA graphs due to too many consecutive updates\n", __func__);
#endif
        }
    }

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        // Why are we protecting an atomic_int with a mutex?
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ++ggml_cuda_lock_counter;
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    if (graph && !use_cuda_graph) {
        graph->use_cpy_indirection = false;
    }

#else
    bool use_cuda_graph = false;
    bool cuda_graph_update_required = false;
#endif // USE_CUDA_GRAPH

    bool graph_evaluated_or_captured = false;

    if (!evaluate_and_capture_cuda_graph(cuda_ctx, cgraph, graph_evaluated_or_captured, use_cuda_graph, cuda_graph_update_required)) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

GGML_CALL static bool ggml_backend_cuda_supports_op(ggml_backend_t backend, const ggml_tensor * op) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    // Non-mul_mat ops can't read a split-buffer parent (no data ptr); let the scheduler fall back to CPU.
    if (op->op != GGML_OP_MUL_MAT && op->op != GGML_OP_MUL_MAT_ID) {
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda_split(op->src[i]->buffer->buft)) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_SWIGLU:
                case GGML_UNARY_OP_SWIGLU_OAI:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_SQRT_SOFTPLUS:
                case GGML_UNARY_OP_NEG:
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_FUSED_MUL_UNARY: return ggml_is_contiguous(op->src[0]);
        case GGML_OP_MOE_FFN:
            return ggml_cuda_moe_ffn_supports_op(*cuda_ctx, op);
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_MOE_FUSED_UP_GATE:
        case GGML_OP_FUSED_UP_GATE:
            {
                bool is_fused_up_gate = op->op == GGML_OP_MOE_FUSED_UP_GATE || op->op == GGML_OP_FUSED_UP_GATE;
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = is_fused_up_gate ? op->src[2] : op->src[1];
                if (is_fused_up_gate && op->src[1] && a->type != op->src[1]->type) {
                    fprintf(stderr, "%s: returning false for GGML_OP_MOE_FUSED_UP_GATE because src0->type != src1->type\n", __func__);
                    return false;
                }
                //==================================================================
                //if (ggml_is_quantized(a->type) && ggml_is_quantized(b->type)) {
                //    return false;
                //}
                //==================================================================
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16 && !ggml_is_quantized(a->type)) {
                    printf("%s: returning false for op %d because (case 1)\n", __func__, (int)op->op);
                    return false;
                }
                if (op->op == GGML_OP_MUL_MAT && a->ne[3] != b->ne[3]) {
                    return false;
                }
                switch (a->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q6_0:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_IQ2_KL:
                    case GGML_TYPE_IQ3_KS:
                    case GGML_TYPE_IQ4_KS:
                    case GGML_TYPE_IQ4_KSS:
                    case GGML_TYPE_IQ5_KS:
                    case GGML_TYPE_IQ2_K:
                    case GGML_TYPE_IQ2_KS:
                    case GGML_TYPE_IQ1_KT:
                    case GGML_TYPE_IQ2_KT:
                    case GGML_TYPE_IQ3_KT:
                    case GGML_TYPE_IQ4_KT:
                    case GGML_TYPE_IQ3_K:
                    case GGML_TYPE_IQ4_K:
                    case GGML_TYPE_IQ5_K:
                    case GGML_TYPE_IQ6_K:
                    case GGML_TYPE_IQ1_BN:
                    case GGML_TYPE_IQ2_BN:
                    case GGML_TYPE_PQ2_0:
                    case GGML_TYPE_IQ2_K_R4:
                    case GGML_TYPE_IQ3_K_R4:
                    case GGML_TYPE_IQ4_K_R4:
                    case GGML_TYPE_IQ4_KS_R4:
                    case GGML_TYPE_IQ5_K_R4:
                    case GGML_TYPE_IQ5_KS_R4:
                    case GGML_TYPE_IQ1_S_R4:
                    case GGML_TYPE_IQ1_M_R4:
                    case GGML_TYPE_IQ3_KS_R16:
                    case GGML_TYPE_IQ4_KS_R16:
                    case GGML_TYPE_IQ5_KS_R16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS:
            {
                if (op->type == op->src[0]->type) {
                    return true;
                }
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    case GGML_TYPE_I32:
                        return op->src[0]->type == op->type;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                       op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                       op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL) &&
                       op->src[0]->type == GGML_TYPE_F32 &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_BF16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q6_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F16) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F16 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    // DSA lightning-indexer top_k indices (I32) copy/cont.
                    return true;
                }
                if (ggml_is_quantized(src0_type) && (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_F32)) {
                    return true;
                }
                if (ggml_is_contiguous(op->src[0]) && ggml_are_same_shape(op->src[0], op->src[1])) {
                    if (src1_type == GGML_TYPE_F16 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F32) {
                        return true;
                    }
                }
                if (ggml_are_same_shape(op->src[0], op->src[1]) && op->src[0]->type == GGML_TYPE_Q8_0 && op->src[1]->type == GGML_TYPE_Q8_0) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_REDUCE:
        case GGML_OP_BLEND:
        case GGML_OP_FAKE_CPY:
        case GGML_OP_ARGMAX:
            return true;
        case GGML_OP_HADAMARD: {
            if (!(op->op_params[0] == 64 || op->op_params[0] == 128 || op->op_params[0] == 256 || op->op_params[0] == 512 ||
                  op->op_params[0] == 1024 || op->op_params[0] == 2048 || op->op_params[0] == 4096)) return false;
            if (op->ne[0] % op->op_params[0] != 0) return false;
            if (op->type != GGML_TYPE_F32) return false;
            switch (op->src[0]->type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q6_0:
                case GGML_TYPE_IQ4_NL:
                    return true;
                default:
                    return false;
            }
        }
        case GGML_OP_DUP:
        case GGML_OP_REPEAT:
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                return src0_type != GGML_TYPE_I32 && src0_type != GGML_TYPE_I16;
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return true;
        case GGML_OP_L2_NORM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]) && op->ne[0] % WARP_SIZE == 0;
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD:
        case GGML_OP_ADD_ID:
        case GGML_OP_MULTI_ADD:
        case GGML_OP_MUL_MULTI_ADD:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_SUB:
        case GGML_OP_FUSED_RMS_NORM:
        case GGML_OP_FUSED_RMS_RMS_ADD:
        case GGML_OP_SCALE:
        case GGML_OP_SOFTCAP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_CLAMP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_DIAG:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_SOFT_CAP_MAX:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
        case GGML_OP_ROPE_FAST:
        case GGML_OP_ROPE_CACHE:
            return true;
        case GGML_OP_FUSED_NORM:
            return ggml_is_contiguous(op->src[0]);
        //case GGML_OP_ROPE:
        //    return ggml_is_contiguous(op->src[0]);
        case GGML_OP_ARGSORT:
            return true;
        case GGML_OP_ARGSORT_THRESH:
            // The CUDA bitonic argsort launches one thread per (padded) column, so the
            // row width rounded up to a power of 2 must fit in a single CUDA block (<=1024
            // threads). Wider rows (e.g. the DSA lightning-indexer scoring over a large
            // n_kv) would fail at launch with "invalid configuration argument"; report them
            // as unsupported so the scheduler falls back to the CPU argsort (no size limit).
            {
                int64_t ncols = op->src[0]->ne[0];
                int64_t ncols_pad = 1;
                while (ncols_pad < ncols) ncols_pad *= 2;
                return ncols_pad <= 1024;
            }
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_GROUPED_TOPK:
        case GGML_OP_ACC:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return true;
        case GGML_OP_MEAN:
        case GGML_OP_CUMSUM:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_TRI:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   op->src[0]->type == op->type;
        case GGML_OP_FILL:
            return ggml_is_contiguous(op) && (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16);
        case GGML_OP_SOLVE_TRI:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   ggml_is_contiguous(op) &&
                   op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->ne[0] == op->src[0]->ne[1] &&
                   op->src[0]->ne[1] == op->src[1]->ne[1] &&
                   op->src[0]->ne[2] == op->src[1]->ne[2] &&
                   op->src[0]->ne[3] == op->src[1]->ne[3];
        case GGML_OP_SSM_CONV:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->src[2]->type == GGML_TYPE_F32 &&
                   op->src[3]->type == GGML_TYPE_I32 &&
                   op->type == GGML_TYPE_F32 &&
                   op->src[0]->nb[0] == sizeof(float) &&
                   op->src[1]->nb[0] == sizeof(float) &&
                   op->src[2]->nb[0] == sizeof(float) &&
                   op->src[3]->nb[0] == sizeof(int32_t) &&
                   op->src[2]->ne[0] == op->src[0]->ne[0] + 1 &&
                   op->src[2]->ne[1] == op->src[0]->ne[1] &&
                   op->src[1]->ne[0] == op->src[0]->ne[1] &&
                   op->src[3]->ne[0] == op->src[0]->ne[2];
        case GGML_OP_DELTA_NET:
        case GGML_OP_INDEXER_TOPK:
        case GGML_OP_MASK_TOPK:
        case GGML_OP_MASK_TO_IDX:
        case GGML_OP_DS4_COMP:
            return true;
        case GGML_OP_HC_PRE:
        case GGML_OP_HC_POST:
            return true;
        case GGML_OP_SINKHORN: {
            const int sink_s = op->op_params[0];
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 &&
                   sink_s >= 1 && sink_s <= 8 && op->src[0]->ne[0] == (int64_t) sink_s*sink_s;
        }
        case GGML_OP_LATENT_ATTN:
            return ggml_cuda_latent_attn_is_supported(op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_fattn_is_supported(*cuda_ctx, op);
        default:
            return false;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cuda_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    //printf("%s(%s, %s): %d, %d\n", __func__, ggml_backend_name(backend), ggml_backend_buft_name(buft), ggml_backend_buft_is_cuda_split(buft), ggml_backend_buft_is_cuda(buft));
    if (ggml_backend_buft_is_cuda_split(buft)) {
        return true;
    }

    if (ggml_backend_buft_is_cuda(buft)) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;
        ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;
        return buft_ctx->device == cuda_ctx->device;
    }

    return false;
}

GGML_CALL static bool ggml_backend_cuda_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    auto ctx = (ggml_backend_cuda_context *)backend->context;
    int min_batch_size = ctx->offload_batch_size; // originally: GGML_CUDA_MIN_BATCH_OFFLOAD;

    // Why do we want to do this? The heuristics that the batch must have more than min_batch_size tokens to be worth it
    // offloading the required model weights comes from dense models. For MoE models, the average number of tokens
    // each expert deals with in a batch is (active_experts / total_experts) * batch_size. Hence, according to the
    // learned heuristics, we need (active_experts / total_experts) * batch_size >= min_batch_size.
    // Rearranging we get
    //
    //           batch_size * active_experts >= min_batch_size * total_experts
    //
    // as the condition for offloading model weights residing in RAM to the GPU.
    // In this case, the number of tokens is not as usual in op->ne[1] but rather in op->ne[2].
    if (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE) {
        if (ctx->offload_batch_size_per_byte >= 0) {
            auto src0 = op->src[0];
            auto row_size = ggml_row_size(src0->type, src0->ne[0]);
            min_batch_size = int(1.*ctx->offload_batch_size_per_byte*row_size/src0->ne[0]);
        }
        auto ids = op->op == GGML_OP_MUL_MAT_ID ? op->src[2] : op->src[3];
        int64_t batch_size = op->ne[2];
        if (batch_size < min_batch_size) return false;
        int64_t n_experts_tot    = op->src[0]->ne[2];
        int64_t n_experts_active = ids->ne[0];
        bool should_offload = batch_size*n_experts_active >= min_batch_size*n_experts_tot;
        return should_offload;
    }

    return op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS;

    // Original:
    //return (op->ne[1] >= min_batch_size && op->op != GGML_OP_GET_ROWS) ||
    //       (op->ne[2] >= min_batch_size && (op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_MOE_FUSED_UP_GATE));

    GGML_UNUSED(backend);
}

static ggml_backend_event_t ggml_backend_cuda_event_new(ggml_backend_t backend) {
#ifdef GGML_CUDA_NO_PEER_COPY
    return nullptr;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    ggml_cuda_set_device(cuda_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .backend = */ backend,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_event_free(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));

    delete event;
}

static void ggml_backend_cuda_event_record(ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)event->backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(event->backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

static void ggml_backend_cuda_event_synchronize(ggml_backend_event_t event) {
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .get_default_buffer_type = */ ggml_backend_cuda_get_default_buffer_type,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .supports_op             = */ ggml_backend_cuda_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_offload_op,
    /* .event_new               = */ ggml_backend_cuda_event_new,
    /* .event_free              = */ ggml_backend_cuda_event_free,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .event_synchronize       = */ ggml_backend_cuda_event_synchronize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

struct cuda_params {
    int  fusion = GGML_CUDA_FUSION;
    int  offload_batch_size = GGML_CUDA_MIN_BATCH_OFFLOAD;
    int  offload_batch_size_per_byte = -1;
    int  mmq_id_thresh = 32;
    float fa_offset = 0.6931f;
    bool  fabsum = false;
#ifdef USE_CUDA_GRAPH
    bool use_cuda_graph = true;
    int  max_graphs = 0;
#else
    bool use_cuda_graph = false;
#endif
    bool enable_p2p = true;
};

static std::vector<std::string> string_split(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> parts;
    size_t start = 0;
    size_t end = str.find(delimiter);

    while (end != std::string::npos) {
        parts.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
        end = str.find(delimiter, start);
    }

    parts.push_back(str.substr(start));

    return parts;
}

template <typename T> bool read_value(const std::string& val, T& result) {
    std::istringstream str(val);
    T tmp; str >> tmp;
    if (!str.fail()) {
        result = tmp;
        return true;
    }
    return false;
}

static cuda_params ggml_cuda_parse_params(const char * params_string) {
    cuda_params params{};
    if (!params_string) return params;
    auto values = string_split(std::string{params_string}, ",");
    if (values.empty()) return params;
    for (auto& value : values) {
        auto parsed = string_split(value, "=");
        bool is_good = false;
        if (parsed.size() == 2) {
            if (parsed[0] == "fusion") {
                is_good = read_value(parsed[1], params.fusion);
            }
            else if (parsed[0] == "offload-batch-size") {
                int tmp = 0;
                is_good = read_value(parsed[1], tmp);
                if (is_good) {
                    params.offload_batch_size = tmp;
                }
            }
            else if (parsed[0] == "offload-batch-size-per-byte") {
                is_good = read_value(parsed[1], params.offload_batch_size_per_byte);
            }
            else if (parsed[0] == "mmq-id-size") {
                is_good = read_value(parsed[1], params.mmq_id_thresh);
            }
            else if (parsed[0] == "fa-fabsum") {
                is_good = read_value(parsed[1], params.fabsum);
            }
            else if (parsed[0] == "enable-p2p") {
                is_good = read_value(parsed[1], params.enable_p2p);
            }
            else if (parsed[0] == "fa-offset") {
                float tmp;
                is_good = read_value(parsed[1], tmp);
                if (is_good) {
                    if (tmp < 0.0f || tmp > 3.0f) {
                        GGML_CUDA_LOG_WARN("%s: bad value for %s. It is %g, but must be in [0...3]\n", __func__, parsed[0].c_str(), tmp);
                    } else {
                        params.fa_offset = tmp;
                    }
                }
            }
#ifdef USE_CUDA_GRAPH
            else if (parsed[0] == "graphs") {
                is_good = read_value(parsed[1], params.use_cuda_graph);
            }
            else if (parsed[0] == "max-graphs") {
                is_good = read_value(parsed[1], params.max_graphs);
            }
#endif
        }
        if (!is_good) {
            GGML_CUDA_LOG_WARN("%s: invalid parameter %s (%d) -> ignored\n", __func__, value.c_str(), (int)parsed.size());
        }
    }
    return params;
}

GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, [[maybe_unused]] const void * param_string, const void * model) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_CUDA_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device, model);
    if (ctx == nullptr) {
        GGML_CUDA_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid      = */ ggml_backend_cuda_guid(),
        /* .interface = */ ggml_backend_cuda_interface,
        /* .context   = */ ctx
    };

    bool enable_p2p = true;
    if (param_string) {
        [[maybe_unused]] auto params = ggml_cuda_parse_params((const char *)param_string);
        if (params.fusion != ctx->fusion) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fusion to %d\n", __func__, params.fusion);
            ctx->fusion             = params.fusion;
        }
        if (params.offload_batch_size != ctx->offload_batch_size) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size to %d\n", __func__, params.offload_batch_size);
            ctx->offload_batch_size = params.offload_batch_size;
        }
        if (params.offload_batch_size_per_byte != ctx->offload_batch_size_per_byte) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting offload_batch_size_per_byte to %d\n", __func__, params.offload_batch_size_per_byte);
            ctx->offload_batch_size_per_byte = params.offload_batch_size_per_byte;
        }
        if (params.mmq_id_thresh != ctx->mmq_id_thresh) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting mmq_id_thresh to %d\n", __func__, params.mmq_id_thresh);
            ctx->mmq_id_thresh      = params.mmq_id_thresh;
        }
        if (params.fa_offset != ctx->fa_offset) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fa_offset to %g\n", __func__, params.fa_offset);
            ctx->fa_offset = params.fa_offset;
        }
        if (params.fabsum != ctx->fabsum) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting fabsum to %d\n", __func__, params.fabsum);
            ctx->fabsum = params.fabsum;
        }
        enable_p2p = params.enable_p2p;
#ifdef USE_CUDA_GRAPH
        if (params.use_cuda_graph != ctx->use_cuda_graph) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting use_cuda_graph to %d\n", __func__, params.use_cuda_graph);
            ctx->use_cuda_graph = params.use_cuda_graph;
        }
        if (params.max_graphs != ctx->max_graphs) {
            GGML_CUDA_LOG_INFO(" =========================== %s: setting max_graphs to %d\n", __func__, params.max_graphs);
            ctx->max_graphs = params.max_graphs;
        }
#endif
    }

#ifdef GGML_USE_NCCL
    if (!enable_p2p) {
        printf("================== P2P disabled, but needed for NCCL\n");
        enable_p2p = true;
    }
#endif

#if !defined(GGML_CUDA_NO_PEER_COPY)
    if (enable_p2p) {
        ctx->p2p_enabled = ggml_cuda_set_peer_access(device);
    }
#endif

    return cuda_backend;
}

GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

static bool ggml_cuda_moe_ffn_host_tensor(const ggml_tensor * tensor) {
    return tensor && tensor->data && (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer));
}

static bool ggml_cuda_moe_ffn_device_tensor(const ggml_tensor * tensor, int device) {
    return tensor && tensor->data && ggml_cuda_moe_ffn_tensor_on_device(tensor, device);
}

struct ggml_cuda_moe_ffn_workspace_sizes {
    size_t ids, activation, route, output, host_stage;
    size_t up_q8, down_q8, hidden, down, device_routes;
};

static bool ggml_cuda_moe_ffn_checked_mul(size_t a, size_t b, size_t & result) {
    if (b && a > SIZE_MAX / b) return false;
    result = a * b;
    return true;
}

static bool ggml_cuda_moe_ffn_checked_add(size_t a, size_t b, size_t & result) {
    if (a > SIZE_MAX - b) return false;
    result = a + b;
    return true;
}

static bool ggml_cuda_moe_ffn_get_workspace_sizes(int32_t n_embd, int32_t n_ff,
        int32_t n_experts_used, ggml_cuda_moe_ffn_workspace_sizes & sizes) {
    if (n_embd <= 0 || n_ff <= 0 || n_experts_used <= 0 ||
            n_embd % QK8_1 != 0 || n_ff % QK8_1 != 0) return false;

    const int64_t up_padded = GGML_PAD((int64_t) n_embd, MATRIX_ROW_PADDING);
    const int64_t down_padded = GGML_PAD((int64_t) n_ff, MATRIX_ROW_PADDING);
    if (up_padded > INT_MAX || down_padded > INT_MAX) return false;

    size_t up_q8_row = 0, down_q8_row = 0;
    if (!ggml_cuda_moe_ffn_checked_mul((size_t) up_padded / QK8_1,
                sizeof(block_q8_1), up_q8_row) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) down_padded / QK8_1,
                sizeof(block_q8_1), down_q8_row) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_experts_used, sizeof(int32_t), sizes.ids) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_embd, sizeof(float), sizes.activation)) return false;
    if (!ggml_cuda_moe_ffn_checked_mul((size_t) n_experts_used, sizeof(float), sizes.route)) return false;
    sizes.output = sizes.activation;
    size_t ids_staging = 0, routes_staging = 0, host_base = 0;
    if (!ggml_cuda_moe_ffn_checked_mul(sizes.ids, 3, ids_staging) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.route, 2, routes_staging) ||
        !ggml_cuda_moe_ffn_checked_add(ids_staging, sizes.activation, host_base) ||
        !ggml_cuda_moe_ffn_checked_add(host_base, routes_staging, host_base) ||
        !ggml_cuda_moe_ffn_checked_add(host_base, sizes.output, sizes.host_stage) ||
        !ggml_cuda_moe_ffn_checked_mul(down_q8_row, (size_t) n_experts_used, sizes.down_q8) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_ff, (size_t) n_experts_used, sizes.hidden) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.hidden, sizeof(float), sizes.hidden) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_embd, (size_t) n_experts_used + 1, sizes.down) ||
        !ggml_cuda_moe_ffn_checked_mul(sizes.down, sizeof(float), sizes.down) ||
        !ggml_cuda_moe_ffn_checked_mul((size_t) n_experts_used + 1, sizeof(float), sizes.device_routes)) return false;
    sizes.up_q8 = up_q8_row;
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_layout_supported(ggml_backend_t backend,
        const ggml_moe_ffn_layout * layout) {
    if (!ggml_backend_is_cuda(backend) || !layout) return false;
    const auto * ctx = (const ggml_backend_cuda_context *) backend->context;
    ggml_cuda_moe_ffn_workspace_sizes sizes{};
    return layout->owner_backend == backend && (layout->device < 0 || layout->device == ctx->device) &&
        ggml_moe_ffn_layout_supported(layout) && layout->up->type == layout->gate->type &&
        ggml_cuda_mmvq_type_supported(layout->up->type) &&
        ggml_cuda_mmvq_type_supported(layout->down->type) &&
        ggml_cuda_moe_ffn_get_workspace_sizes(layout->n_embd, layout->n_ff,
                layout->n_experts_used, sizes);
}

GGML_CALL int32_t ggml_backend_cuda_moe_ffn_register(ggml_backend_t backend,
        const ggml_cuda_moe_ffn_binding * binding) {
    if (!ggml_backend_is_cuda(backend) || !binding) return -1;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    const ggml_moe_ffn_layout & layout = binding->layout;
    if (layout.device != ctx->device || layout.owner_backend != backend ||
        layout.cache_owner != binding->cache_context || binding->generation == 0 ||
        !ggml_backend_cuda_moe_ffn_layout_supported(backend, &layout) || layout.n_slots <= 0 ||
        !binding->cache_callbacks || !binding->cache_callbacks->resolve_and_pin ||
        !binding->cache_callbacks->release_pins || !binding->cache_context ||
        binding->cpu_resource_id <= 0 || !binding->cpu_graph ||
        !binding->cpu_activations || !binding->cpu_ids || !binding->cpu_route_weights || !binding->cpu_output) return -1;

    if (!ggml_cuda_moe_ffn_host_tensor(layout.up) || !ggml_cuda_moe_ffn_host_tensor(layout.gate) ||
        !ggml_cuda_moe_ffn_host_tensor(layout.down) || !ggml_cuda_moe_ffn_device_tensor(layout.bank_up, ctx->device) ||
        !ggml_cuda_moe_ffn_device_tensor(layout.bank_gate, ctx->device) ||
        !ggml_cuda_moe_ffn_device_tensor(layout.bank_down, ctx->device)) return -1;

    if (!binding->cpu_activations->data || !binding->cpu_ids->data || !binding->cpu_route_weights->data || !binding->cpu_output->data ||
        binding->cpu_activations->type != GGML_TYPE_F32 || binding->cpu_activations->ne[0] != layout.n_embd ||
        binding->cpu_ids->type != GGML_TYPE_I32 || binding->cpu_ids->ne[0] != 1 ||
        binding->cpu_route_weights->type != GGML_TYPE_F32 || binding->cpu_route_weights->ne[0] != 1 ||
        binding->cpu_route_weights->ne[1] != 1 ||
        binding->cpu_output->type != GGML_TYPE_F32 || binding->cpu_output->ne[0] != layout.n_embd) return -1;

    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> resource;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(binding->cpu_resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return -1;
        resource = it->second;
    }
    if (resource->device != ctx->device || !ggml_cuda_moe_ffn_resource_supports_signature(*resource, layout)) return -1;

    {
        std::lock_guard<std::mutex> execution_lock(resource->execution_mutex);
        if (!resource->accepting_bindings.load(std::memory_order_acquire) ||
            ggml_backend_graph_compute(resource->resource.cpu_backend, binding->cpu_graph) != GGML_STATUS_SUCCESS) return -1;
    }

    auto state = std::make_shared<ggml_cuda_moe_ffn_binding_state>();
    state->binding = *binding;
    state->cpu_resource = resource;
    std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
    auto resource_it = ctx->moe_ffn_cpu_resources.find(binding->cpu_resource_id);
    if (resource_it == ctx->moe_ffn_cpu_resources.end() || resource_it->second != resource ||
        !resource->accepting_bindings.load(std::memory_order_acquire) ||
        ctx->moe_ffn_next_binding_id <= 0) return -1;
    const int32_t binding_id = ctx->moe_ffn_next_binding_id++;
    ctx->moe_ffn_bindings.emplace(binding_id, std::move(state));
    return binding_id;
}

GGML_CALL int32_t ggml_backend_cuda_moe_ffn_cpu_resource_register(ggml_backend_t backend,
        const ggml_cuda_moe_ffn_cpu_resource * resource) {
    if (!ggml_backend_is_cuda(backend) || !resource || !ggml_backend_is_cpu(resource->cpu_backend) ||
        resource->n_threads <= 0 || !resource->signatures || resource->n_signatures == 0) return -1;
    ggml_cuda_moe_ffn_workspace_sizes sizes{};
    if (!ggml_cuda_moe_ffn_get_workspace_sizes(resource->n_embd, resource->n_ff,
                resource->n_experts_used, sizes)) return -1;
    for (size_t i = 0; i < resource->n_signatures; ++i) {
        const auto & signature = resource->signatures[i];
        ggml_cuda_moe_ffn_workspace_sizes signature_sizes{};
        if (signature.n_embd <= 0 || signature.n_ff <= 0 || signature.n_experts_used <= 0 ||
                signature.n_embd > resource->n_embd || signature.n_ff > resource->n_ff ||
                signature.n_experts_used > resource->n_experts_used ||
                !ggml_cuda_moe_ffn_get_workspace_sizes(signature.n_embd, signature.n_ff,
                    signature.n_experts_used, signature_sizes)) return -1;
        for (size_t j = 0; j < i; ++j) {
            if (resource->signatures[j].n_embd == signature.n_embd &&
                    resource->signatures[j].n_ff == signature.n_ff &&
                    resource->signatures[j].n_experts_used == signature.n_experts_used) return -1;
        }
    }
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        if (!ctx->moe_ffn_cpu_resources.empty()) return -1;
    }

    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    try {
        state = std::make_shared<ggml_cuda_moe_ffn_cpu_resource_state>();
        state->signatures.assign(resource->signatures, resource->signatures + resource->n_signatures);
        state->resolved_pins.resize((size_t) resource->n_experts_used);
        state->pending_pins.pins.reserve((size_t) resource->n_experts_used);
        state->fill_candidates.reserve((size_t) resource->n_experts_used);
    } catch (...) {
        return -1;
    }
    state->device = ctx->device;
    state->cuda_context = ctx;
    const size_t ids_bytes = sizes.ids;
    const size_t activation_bytes = sizes.activation;
    const size_t route_bytes = sizes.route;
    const size_t output_bytes = sizes.output;
    const size_t host_bytes = sizes.host_stage;
    ggml_cuda_set_device(ctx->device);
    if (cudaHostAlloc(&state->host_stage, host_bytes, cudaHostAllocPortable) != cudaSuccess) return -1;
    auto * host = (uint8_t *) state->host_stage;
    state->host_ids = (int32_t *) host;
    state->host_slots = (int32_t *) (host + ids_bytes);
    state->host_hit_slots = (int32_t *) (host + 2 * ids_bytes);
    state->host_activations = (float *) (host + 3 * ids_bytes);
    state->host_routes = (float *) (host + 3 * ids_bytes + activation_bytes);
    state->host_hit_routes = (float *) (host + 3 * ids_bytes + activation_bytes + route_bytes);
    state->host_output = (float *) (host + 3 * ids_bytes + activation_bytes + 2 * route_bytes);
    state->up_q8_bytes = sizes.up_q8;
    state->down_q8_bytes = sizes.down_q8;
    const size_t hidden_bytes = sizes.hidden;
    const size_t down_bytes = sizes.down;
    auto alloc_device = [](void ** ptr, size_t bytes) { return cudaMalloc(ptr, bytes) == cudaSuccess; };
    if (!alloc_device((void **) &state->device_slots, ids_bytes) ||
        !alloc_device((void **) &state->device_routes, sizes.device_routes) ||
        !alloc_device(&state->device_up_q8, state->up_q8_bytes) ||
        !alloc_device(&state->device_down_q8, state->down_q8_bytes) ||
        !alloc_device((void **) &state->device_hidden, hidden_bytes) ||
        !alloc_device((void **) &state->device_down, down_bytes) ||
        !alloc_device((void **) &state->device_output, output_bytes) ||
        cudaEventCreateWithFlags(&state->hit_kernels_done, cudaEventDisableTiming) != cudaSuccess) return -1;

    if (MOE_CACHE_TRANSPORT_STREAM <= 0 || MOE_CACHE_TRANSPORT_STREAM >= GGML_CUDA_MAX_STREAMS) return -1;
    if (ctx->streams[ctx->device][MOE_CACHE_TRANSPORT_STREAM] == nullptr &&
            cudaStreamCreateWithFlags(&ctx->streams[ctx->device][MOE_CACHE_TRANSPORT_STREAM],
                    cudaStreamNonBlocking) != cudaSuccess) return -1;
    state->fill_stream = ctx->streams[ctx->device][MOE_CACHE_TRANSPORT_STREAM];

    ggml_backend_cpu_set_n_threads(resource->cpu_backend, resource->n_threads);
    state->resource = *resource; // Ownership transfers only after successful warm allocation.
    state->resource.signatures = state->signatures.data();
    int32_t resource_id = -1;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        if (!ctx->moe_ffn_cpu_resources.empty() || ctx->moe_ffn_next_cpu_resource_id <= 0) {
            state->resource.cpu_backend = nullptr;
            return -1;
        }
        resource_id = ctx->moe_ffn_next_cpu_resource_id++;
        ctx->moe_ffn_cpu_resources.emplace(resource_id, state);
    }
    try {
        state->fill_worker = std::thread(ggml_cuda_moe_ffn_fill_worker, state.get());
    } catch (...) {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        ctx->moe_ffn_cpu_resources.erase(resource_id);
        state->resource.cpu_backend = nullptr;
        return -1;
    }
    return resource_id;
}

static bool ggml_cuda_moe_ffn_stop_transport(ggml_backend_cuda_context * ctx,
        ggml_cuda_moe_ffn_cpu_resource_state & state) {
    {
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        state.accepting_fills = false;
        state.fill_stopping = true;
    }
    state.fill_cv.notify_all();
    if (state.fill_worker.joinable()) state.fill_worker.join();

    if (cudaSetDevice(ctx->device) != cudaSuccess) return false;
    if (state.fill_stream && cudaStreamSynchronize(state.fill_stream) != cudaSuccess) return false;
    if (state.fill_stuck) {
        const auto stuck = state.stuck_fill;
        if (!stuck.callbacks || !stuck.callbacks->finish_fill || !stuck.cache_context ||
                !stuck.callbacks->finish_fill(stuck.cache_context, &stuck.owner_job, false)) return false;
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        state.fill_stuck = false;
        state.fill_outstanding = false;
        state.stuck_fill = {};
    }
    {
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        if (state.fill_outstanding || state.fill_queued || state.fill_active) return false;
    }

    std::lock_guard<std::mutex> execution_lock(state.execution_mutex);
    const cudaStream_t stream = ctx->streams[ctx->device][0];
    if (stream && cudaStreamSynchronize(stream) != cudaSuccess) return false;
    if (!ggml_cuda_moe_ffn_release_pending_pins(state)) return false;
    {
        std::lock_guard<std::mutex> lock(state.fill_mutex);
        state.transport_stopped = true;
    }
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_stop(ggml_backend_t backend, int32_t resource_id) {
    if (!ggml_backend_is_cuda(backend) || resource_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return false;
        state = it->second;
    }
    return ggml_cuda_moe_ffn_stop_transport(ctx, *state);
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_unregister(ggml_backend_t backend, int32_t resource_id) {
    if (!ggml_backend_is_cuda(backend) || resource_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return false;
        for (const auto & entry : ctx->moe_ffn_bindings) {
            if (entry.second->binding.cpu_resource_id == resource_id) return false;
        }
        state = it->second;
        state->accepting_bindings.store(false, std::memory_order_release);
    }
    if (!ggml_cuda_moe_ffn_stop_transport(ctx, *state)) {
        state->accepting_bindings.store(true, std::memory_order_release);
        return false;
    }
    std::lock_guard<std::mutex> registry_lock(ctx->moe_ffn_mutex);
    auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
    if (it == ctx->moe_ffn_cpu_resources.end() || it->second != state) return false;
    for (const auto & entry : ctx->moe_ffn_bindings) {
        if (entry.second->binding.cpu_resource_id == resource_id) {
            state->accepting_bindings.store(true, std::memory_order_release);
            return false;
        }
    }
    ctx->moe_ffn_cpu_resources.erase(it);
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(ggml_backend_t backend,
        int32_t resource_id, ggml_cuda_moe_ffn_cpu_resource_stats * stats) {
    if (!ggml_backend_is_cuda(backend) || !stats || resource_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_cpu_resource_state> state;
    uint64_t active_bindings = 0;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_cpu_resources.find(resource_id);
        if (it == ctx->moe_ffn_cpu_resources.end()) return false;
        state = it->second;
        for (const auto & entry : ctx->moe_ffn_bindings) {
            active_bindings += entry.second->binding.cpu_resource_id == resource_id;
        }
    }
    std::lock_guard<std::mutex> lock(state->execution_mutex);
    stats->cpu_calls = state->cpu_calls;
    stats->max_parallel_calls = state->max_parallel_calls;
    stats->active_bindings = active_bindings;
    stats->pin_resolve_calls = state->pin_resolve_calls;
    stats->pin_acquisitions = state->pin_acquisitions;
    stats->pin_releases = state->pin_releases;
    stats->pending_pin_releases = state->pending_pin_releases;
    stats->pin_release_failures = state->pin_release_failures;
    stats->pending_pin_active = state->pending_pins.active ? 1 : 0;
    {
        std::lock_guard<std::mutex> fill_lock(state->fill_mutex);
        stats->fills_accepted = state->fills_accepted;
        stats->fills_completed = state->fills_completed;
        stats->fills_rejected = state->fills_rejected;
        stats->fills_failed = state->fills_failed;
        stats->fill_outstanding = state->fill_outstanding ? 1 : 0;
        stats->fill_parts_completed = state->fill_parts_completed;
        stats->fill_bytes_completed = state->fill_bytes_completed;
    }
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_unregister(ggml_backend_t backend, int32_t binding_id) {
    if (!ggml_backend_is_cuda(backend) || binding_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_bindings.find(binding_id);
        if (it == ctx->moe_ffn_bindings.end()) return false;
        state = it->second;
    }
    {
        std::lock_guard<std::mutex> fill_lock(state->cpu_resource->fill_mutex);
        if (!state->cpu_resource->transport_stopped) return false;
    }
    std::lock_guard<std::mutex> request_lock(state->cpu_resource->execution_mutex);
    ggml_cuda_set_device(ctx->device);
    for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
        if (ctx->streams[ctx->device][j] != nullptr && cudaStreamSynchronize(ctx->streams[ctx->device][j]) != cudaSuccess) return false;
    }
    if (!ggml_cuda_moe_ffn_release_pending_pins(*state->cpu_resource)) return false;
    state->active = false;
    std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
    auto it = ctx->moe_ffn_bindings.find(binding_id);
    if (it == ctx->moe_ffn_bindings.end() || it->second != state) return false;
    ctx->moe_ffn_bindings.erase(it);
    return true;
}

GGML_CALL bool ggml_backend_cuda_moe_ffn_get_stats(ggml_backend_t backend, int32_t binding_id,
        ggml_cuda_moe_ffn_stats * stats) {
    if (!ggml_backend_is_cuda(backend) || !stats || binding_id <= 0) return false;
    auto * ctx = (ggml_backend_cuda_context *) backend->context;
    std::shared_ptr<ggml_cuda_moe_ffn_binding_state> state;
    {
        std::lock_guard<std::mutex> lock(ctx->moe_ffn_mutex);
        auto it = ctx->moe_ffn_bindings.find(binding_id);
        if (it == ctx->moe_ffn_bindings.end()) return false;
        state = it->second;
    }
    std::lock_guard<std::mutex> request_lock(state->cpu_resource->execution_mutex);
    *stats = state->stats;
    return true;
}

GGML_CALL int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

GGML_CALL void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    snprintf(description, description_size, "%s", prop.name);
}

GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));
}

GGML_CALL bool ggml_backend_cuda_is_integrated(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        return false;
    }
    return ggml_cuda_info().devices[device].integrated;
}

GGML_CALL bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11100 || defined(GGML_USE_MUSA)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();

        GGML_CUDA_LOG_WARN("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    return false;
#endif
}

GGML_CALL void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        cudaGetLastError();
    }
}

// backend registry
GGML_CALL static ggml_backend_t ggml_backend_reg_cuda_init(const char * params, void * user_data) {
    ggml_backend_t cuda_backend = ggml_backend_cuda_init((int) (intptr_t) user_data, nullptr, nullptr);
    return cuda_backend;

    GGML_UNUSED(params);
}

extern "C" GGML_CALL int ggml_backend_cuda_reg_devices();

GGML_CALL int ggml_backend_cuda_reg_devices() {
    int device_count = ggml_backend_cuda_get_device_count();
    //int device_count = 1; // DEBUG: some tools require delaying CUDA initialization
    for (int i = 0; i < device_count; i++) {
        char name[128];
        snprintf(name, sizeof(name), "%s%d", GGML_CUDA_NAME, i);
        ggml_backend_register(name, ggml_backend_reg_cuda_init, ggml_backend_cuda_buffer_type(i), (void *) (intptr_t) i);
    }
    return device_count;
}
