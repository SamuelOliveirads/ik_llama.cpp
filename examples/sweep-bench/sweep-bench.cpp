#include "ggml.h"
#include "llama.h"
#include "common.h"
#include "speculative.h"
#include "llama-vocab.h"
#include "llama-context.h"
#include "llama-moe-ffn-prototype.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#endif

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/resource.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

static double get_rss_hwm_mib() {
#ifdef _WIN32
    return -1.0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return -1.0;
    }
#ifdef __APPLE__
    return usage.ru_maxrss / (1024.0 * 1024.0);
#else
    return usage.ru_maxrss / 1024.0;
#endif
#endif
}

struct sweep_vram_tracker {
    std::vector<size_t> baseline;

    void start() {
#ifdef GGML_USE_CUDA
        const int count = ggml_backend_cuda_get_device_count();
        baseline.resize(count);
        for (int device = 0; device < count; ++device) {
            size_t free;
            size_t total;
            ggml_backend_cuda_get_device_memory(device, &free, &total);
            baseline[device] = free;
        }
#endif
    }

    double sample() {
#ifdef GGML_USE_CUDA
        if (baseline.empty()) {
            return -1.0;
        }
        size_t used = 0;
        for (int device = 0; device < (int) baseline.size(); ++device) {
            size_t free;
            size_t total;
            ggml_backend_cuda_get_device_memory(device, &free, &total);
            used += baseline[device] > free ? baseline[device] - free : 0;
        }
        return used / (1024.0 * 1024.0);
#else
        return -1.0;
#endif
    }
};

static std::string format_mib(double value, int precision, const char * missing) {
    if (value < 0.0) {
        return missing;
    }
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
    return buffer;
}

static void llama_selective_log_callback(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    const char * skip_patterns[] = {
        "Setting default device in layer",
        "llama_model_loader: Dumping metadata",
        "llama_model_loader: - kv  ",
        "llama_model_loader: - type ",
        "validate_override:",
        "load: printing all EOG",
        "load:   - ",
        "load: special tokens cache",
        "load: token to piece cache",
        "llm_load_print_meta:",
        "print_info:",
        "------------------- Layer sizes",
        "Layer ",
        "llm_load_tensors:",
        "==========================",
        "merging up/gate in layer",
        "repacking up/gate experts weight in layer",
    };
    for (const char * pat : skip_patterns) {
        if (strstr(text, pat) != nullptr) {
            return;
        }
    }
    // Skip incomplete/continuation lines
    int i = 0;
    while (text[i] == ' ' || text[i] == '\t') {
        i++;
    }
    if (text[i] == ',' || text[i] == '(' || text[i] == ')'|| (text[i] >= '0' && text[i] <= '9')) {
        return;
    }
    LOG_TEE("%s", text);
}

static void print_usage(int argc, char ** argv) {
    gpt_params params;
    params.sweep_bench = true;
    gpt_params_print_usage(argc, argv, params);

    LOG_TEE("\nsweep-bench specific options:\n\n");
    LOG_TEE("  -nrep, --n-repetitions N        number of repetitions for each context size (default: 1)\n");
    LOG_TEE("         --sweep-stride N         measure every Nth sweep row (default: 1)\n");
    LOG_TEE("         --sweep-memory           report RSS high-water and sampled VRAM delta\n");
    LOG_TEE("  -wb,   --warmup-batch           run a warmup batch before measurement\n");
    LOG_TEE("  -f,    --file FNAME             feed the tokens of a text file instead of random tokens\n");
    LOG_TEE("         --output-format FORMAT    output format: table (default) or jsonl\n");
    LOG_TEE("\nexample usage:\n");
    LOG_TEE("\n    %s -m model.gguf -c 8192 -b 2048 -ub 512\n", argv[0]);
    LOG_TEE("\n");
}

struct f5a_device_stats_snapshot {
    int device;
    ggml_cuda_moe_ffn_stats stats;
    ggml_cuda_moe_ffn_cpu_resource_stats resource;
};

struct f5a_stats_checkpoint {
    unsigned int n_kv;
    ggml_cuda_moe_ffn_stats aggregate;
    ggml_cuda_moe_ffn_cpu_resource_stats aggregate_resource;
    std::vector<f5a_device_stats_snapshot> devices;
};

static bool collect_f5a_stats(llama_context * ctx, unsigned int n_kv, f5a_stats_checkpoint & checkpoint) {
    if (ctx == nullptr || !ctx->moe_ffn_prototype) return false;
    checkpoint.n_kv = n_kv;
    if (!ctx->moe_ffn_prototype->collect_stats(checkpoint.aggregate) ||
            !ctx->moe_ffn_prototype->collect_resource_stats(checkpoint.aggregate_resource)) return false;
    checkpoint.devices.clear();
    checkpoint.devices.reserve(ctx->moe_ffn_prototype->devices.size());
    for (const auto & device : ctx->moe_ffn_prototype->devices) {
        f5a_device_stats_snapshot snapshot{};
        snapshot.device = device.device;
        if (!ctx->moe_ffn_prototype->collect_device_stats(device.cuda_backend, snapshot.stats) ||
                !ctx->moe_ffn_prototype->collect_device_resource_stats(device.cuda_backend, snapshot.resource)) return false;
        checkpoint.devices.push_back(snapshot);
    }
    return true;
}

static uint64_t f5a_delta(uint64_t current, uint64_t previous) {
    return current >= previous ? current - previous : 0;
}

static void print_f5a_stats(const char * scope, unsigned int n_kv, int device,
        const ggml_cuda_moe_ffn_stats & s, const ggml_cuda_moe_ffn_cpu_resource_stats & r) {
    const uint64_t rows = s.gpu_hit_draws + s.cpu_miss_draws;
    const double row_hit_rate = rows == 0 ? 0.0 : 100.0 * (double) s.gpu_hit_draws / (double) rows;
    const uint64_t calls = s.all_hit_calls + s.any_miss_calls;
    const double all_hit_rate = calls == 0 ? 0.0 : 100.0 * (double) s.all_hit_calls / (double) calls;
    LOG_TEE("f5a_stats scope=%s n_kv=%u device=%d all_hit_calls=%llu any_miss_calls=%llu rejected_calls=%llu output_commits=%llu "
            "all_hit_call_rate=%.4f selected_ids=%llu id_readback_bytes=%llu slot_map_upload_bytes=%llu input_copy_bytes=%llu output_copy_bytes=%llu "
            "cuda_copy_calls=%llu cpu_copy_calls=%llu synchronizations=%llu capture_bypasses=%llu cuda_calls=%llu cpu_calls=%llu cpu_expert_calls=%llu "
            "partitioned_calls=%llu complete_cpu_fallback_calls=%llu gpu_pending_at_cpu_start=%llu gpu_completed_during_cpu=%llu "
            "full_expert_weight_copy_bytes=%llu eligible_rows=%llu gpu_hit_rows=%llu cpu_miss_rows=%llu row_hit_rate=%.4f "
            "resource_cpu_calls=%llu max_parallel_calls=%llu active_bindings=%llu pin_resolve_calls=%llu pin_acquisitions=%llu pin_releases=%llu "
            "pending_pin_releases=%llu pin_release_failures=%llu pending_pins=%llu fills_accepted=%llu fills_completed=%llu fills_rejected=%llu "
            "fills_failed=%llu pending_jobs=%llu fill_parts_completed=%llu fill_bytes_completed=%llu\n",
            scope, n_kv, device,
            (unsigned long long) s.all_hit_calls, (unsigned long long) s.any_miss_calls,
            (unsigned long long) s.rejected_calls, (unsigned long long) s.output_commits, all_hit_rate,
            (unsigned long long) s.selected_ids, (unsigned long long) s.id_readback_bytes,
            (unsigned long long) s.slot_map_upload_bytes, (unsigned long long) s.input_copy_bytes,
            (unsigned long long) s.output_copy_bytes, (unsigned long long) s.cuda_copy_calls,
            (unsigned long long) s.cpu_copy_calls, (unsigned long long) s.synchronizations,
            (unsigned long long) s.capture_bypasses, (unsigned long long) s.cuda_calls,
            (unsigned long long) s.cpu_calls, (unsigned long long) s.cpu_expert_calls,
            (unsigned long long) s.partitioned_calls, (unsigned long long) s.complete_cpu_fallback_calls,
            (unsigned long long) s.gpu_pending_at_cpu_start, (unsigned long long) s.gpu_completed_during_cpu,
            (unsigned long long) s.full_expert_weight_copy_bytes,
            (unsigned long long) rows, (unsigned long long) s.gpu_hit_draws,
            (unsigned long long) s.cpu_miss_draws, row_hit_rate,
            (unsigned long long) r.cpu_calls, (unsigned long long) r.max_parallel_calls,
            (unsigned long long) r.active_bindings, (unsigned long long) r.pin_resolve_calls,
            (unsigned long long) r.pin_acquisitions, (unsigned long long) r.pin_releases,
            (unsigned long long) r.pending_pin_releases, (unsigned long long) r.pin_release_failures,
            (unsigned long long) r.pending_pin_active, (unsigned long long) r.fills_accepted,
            (unsigned long long) r.fills_completed, (unsigned long long) r.fills_rejected,
            (unsigned long long) r.fills_failed, (unsigned long long) r.fill_outstanding,
            (unsigned long long) r.fill_parts_completed, (unsigned long long) r.fill_bytes_completed);
}

static ggml_cuda_moe_ffn_stats f5a_stats_difference(const ggml_cuda_moe_ffn_stats & now,
        const ggml_cuda_moe_ffn_stats & before) {
    ggml_cuda_moe_ffn_stats d{};
#define F5A_DIFF_STAT(field) d.field = f5a_delta(now.field, before.field)
    F5A_DIFF_STAT(all_hit_calls); F5A_DIFF_STAT(any_miss_calls); F5A_DIFF_STAT(rejected_calls);
    F5A_DIFF_STAT(selected_ids); F5A_DIFF_STAT(id_readback_bytes); F5A_DIFF_STAT(slot_map_upload_bytes);
    F5A_DIFF_STAT(input_copy_bytes); F5A_DIFF_STAT(output_copy_bytes); F5A_DIFF_STAT(cuda_copy_calls);
    F5A_DIFF_STAT(cpu_copy_calls); F5A_DIFF_STAT(synchronizations); F5A_DIFF_STAT(cuda_calls);
    F5A_DIFF_STAT(cpu_calls); F5A_DIFF_STAT(capture_bypasses); F5A_DIFF_STAT(output_commits);
    F5A_DIFF_STAT(full_expert_weight_copy_bytes); F5A_DIFF_STAT(gpu_hit_draws); F5A_DIFF_STAT(cpu_miss_draws);
    F5A_DIFF_STAT(cpu_expert_calls); F5A_DIFF_STAT(gpu_pending_at_cpu_start); F5A_DIFF_STAT(gpu_completed_during_cpu);
    F5A_DIFF_STAT(partitioned_calls); F5A_DIFF_STAT(complete_cpu_fallback_calls);
#undef F5A_DIFF_STAT
    return d;
}

static ggml_cuda_moe_ffn_cpu_resource_stats f5a_resource_difference(
        const ggml_cuda_moe_ffn_cpu_resource_stats & now,
        const ggml_cuda_moe_ffn_cpu_resource_stats & before) {
    ggml_cuda_moe_ffn_cpu_resource_stats d{};
#define F5A_DIFF_RESOURCE(field) d.field = f5a_delta(now.field, before.field)
    F5A_DIFF_RESOURCE(cpu_calls); F5A_DIFF_RESOURCE(pin_resolve_calls); F5A_DIFF_RESOURCE(pin_acquisitions);
    F5A_DIFF_RESOURCE(pin_releases); F5A_DIFF_RESOURCE(pending_pin_releases); F5A_DIFF_RESOURCE(pin_release_failures);
    F5A_DIFF_RESOURCE(fills_accepted); F5A_DIFF_RESOURCE(fills_completed); F5A_DIFF_RESOURCE(fills_rejected);
    F5A_DIFF_RESOURCE(fills_failed); F5A_DIFF_RESOURCE(fill_parts_completed); F5A_DIFF_RESOURCE(fill_bytes_completed);
#undef F5A_DIFF_RESOURCE
    d.max_parallel_calls = now.max_parallel_calls;
    d.active_bindings = now.active_bindings;
    d.pending_pin_active = now.pending_pin_active;
    d.fill_outstanding = now.fill_outstanding;
    return d;
}

static bool report_f5a_checkpoint_deltas(const std::vector<f5a_stats_checkpoint> & checkpoints) {
    if (checkpoints.size() < 2) return false;
    for (size_t i = 1; i < checkpoints.size(); ++i) {
        const auto & before = checkpoints[i - 1];
        const auto & now = checkpoints[i];
        if (before.devices.size() != now.devices.size()) return false;
        const unsigned int from_n_kv = before.n_kv == UINT32_MAX ? 0 : before.n_kv + 1024;
        const unsigned int to_n_kv = now.n_kv + 1024;
        LOG_TEE("f5a_checkpoint_window from_n_kv=%u to_n_kv=%u rows=8\n", from_n_kv, to_n_kv);
        for (size_t d = 0; d < now.devices.size(); ++d) {
            if (before.devices[d].device != now.devices[d].device) return false;
            print_f5a_stats("interval", now.n_kv, now.devices[d].device,
                    f5a_stats_difference(now.devices[d].stats, before.devices[d].stats),
                    f5a_resource_difference(now.devices[d].resource, before.devices[d].resource));
        }
        print_f5a_stats("interval_aggregate", now.n_kv, -1,
                f5a_stats_difference(now.aggregate, before.aggregate),
                f5a_resource_difference(now.aggregate_resource, before.aggregate_resource));
    }
    return true;
}

static bool report_moe_prototype_stats(llama_context * ctx, bool enabled) {
    if (!enabled) {
        LOG_TEE("f4c_prototype_stats mode=blocking state=absent resources=0 bindings=0 leases=0\n");
        return true;
    }

    if (!ctx->moe_ffn_prototype) return false;
    auto & prototype = *ctx->moe_ffn_prototype;
    for (const auto & device : prototype.devices) {
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_stop(device.cuda_backend, device.cpu_resource_id)) {
            LOG_TEE("f4c_prototype_stats error=resource_stop device=%d\n", device.device);
            return false;
        }
    }

    ggml_cuda_moe_ffn_stats aggregate{};
    ggml_cuda_moe_ffn_cpu_resource_stats aggregate_resource{};
    if (!prototype.collect_stats(aggregate) || !prototype.collect_resource_stats(aggregate_resource)) return false;

    for (const auto & device : prototype.devices) {
        ggml_cuda_moe_ffn_stats stats{};
        ggml_cuda_moe_ffn_cpu_resource_stats resource{};
        if (!prototype.collect_device_stats(device.cuda_backend, stats) ||
                !prototype.collect_device_resource_stats(device.cuda_backend, resource)) return false;
        LOG_TEE("f5a_layout device=%d groups=%zu bindings=%zu cpu_graphs=%zu resources=1 legacy_full_expert_copy_bytes=%llu\n",
                device.device, device.groups.size(), device.layer_bindings.size(), device.layer_bindings.size(),
                (unsigned long long) stats.full_expert_weight_copy_bytes);
        print_f5a_stats("total", UINT32_MAX, device.device, stats, resource);
    }

    print_f5a_stats("total_aggregate", 0, -1, aggregate, aggregate_resource);
    return true;
}

#ifdef GGML_USE_CUDA
static bool set_f5a_phase_profile(llama_context * ctx, bool enabled, unsigned int n_kv) {
    if (!ctx || !ctx->moe_ffn_prototype) return false;
    for (const auto & device : ctx->moe_ffn_prototype->devices) {
        if (!ggml_backend_cuda_moe_ffn_set_phase_profile(device.cuda_backend,
                    device.cpu_resource_id, enabled, n_kv)) return false;
    }
    return true;
}

static bool report_f5a_phase_samples(llama_context * ctx) {
    if (!ctx || !ctx->moe_ffn_prototype) return false;
    LOG_TEE("f5a_phase_header kind=device_call_kind_miss_rows_window_ns0_mutex_ns1_ids_sync_ns2_resolve_admission_ns3_miss_input_sync_ns4_gpu_submit_ns5_cpu_compute_ns6_join_commit_ns7_total;fill_ns0_copy_submit_ns1_stream_sync_ns2_publication\n");
    for (const auto & device : ctx->moe_ffn_prototype->devices) {
        std::vector<ggml_cuda_moe_ffn_phase_sample> samples(GGML_CUDA_MOE_FFN_PHASE_SAMPLE_CAPACITY);
        size_t count = 0;
        uint64_t dropped = 0;
        if (!ggml_backend_cuda_moe_ffn_get_phase_samples(device.cuda_backend,
                    device.cpu_resource_id, samples.data(), samples.size(), &count, &dropped, false)) return false;
        LOG_TEE("f5a_phase_samples device=%d count=%zu dropped=%llu\n", device.device, count,
                (unsigned long long) dropped);
        for (size_t i = 0; i < count; ++i) {
            const auto & s = samples[i];
            LOG_TEE("f5a_phase_sample device=%d kind=%d call_kind=%d miss_rows=%d window_n_kv=%u ns0=%llu ns1=%llu ns2=%llu ns3=%llu ns4=%llu ns5=%llu ns6=%llu ns7=%llu\n",
                    s.device, s.kind, s.call_kind, s.miss_rows, s.window_n_kv,
                    (unsigned long long) s.ns[0], (unsigned long long) s.ns[1],
                    (unsigned long long) s.ns[2], (unsigned long long) s.ns[3],
                    (unsigned long long) s.ns[4], (unsigned long long) s.ns[5],
                    (unsigned long long) s.ns[6], (unsigned long long) s.ns[7]);
        }
    }
    return true;
}
#endif

int main(int argc, char ** argv) {

    gpt_params params;
    params.sweep_bench = true;

    if (!gpt_params_parse(argc, argv, params)) {
        print_usage(argc, argv);
        return 1;
    }
    if (params.nrep < 1) params.nrep = 1;
    if (params.sweep_stride < 1) params.sweep_stride = 1;

    const char * prototype_env = std::getenv("IK_LLAMA_MOE_HYBRID_PROTOTYPE");
    const bool use_moe_prototype = prototype_env != nullptr;
    if (use_moe_prototype && std::strcmp(prototype_env, "1") != 0) {
        fprintf(stderr, "%s: error: IK_LLAMA_MOE_HYBRID_PROTOTYPE accepts only the value 1\n", __func__);
        return 2;
    }
#ifndef GGML_USE_CUDA
    if (use_moe_prototype) {
        fprintf(stderr, "%s: error: IK_LLAMA_MOE_HYBRID_PROTOTYPE requires a CUDA build\n", __func__);
        return 2;
    }
#endif
    if (use_moe_prototype && params.split_mode != LLAMA_SPLIT_MODE_LAYER) {
        fprintf(stderr, "%s: error: IK_LLAMA_MOE_HYBRID_PROTOTYPE requires --split-mode layer\n", __func__);
        return 2;
    }

    const char * f5a_checkpoints_env = std::getenv("IK_LLAMA_F5A_COUNTER_CHECKPOINTS");
    const bool f5a_checkpoints = f5a_checkpoints_env != nullptr;
    if (f5a_checkpoints && (std::strcmp(f5a_checkpoints_env, "1") != 0 || !use_moe_prototype)) {
        fprintf(stderr, "%s: error: IK_LLAMA_F5A_COUNTER_CHECKPOINTS requires value 1 and prototype mode\n", __func__);
        return 2;
    }

    const char * f5a_profile_env = std::getenv("IK_LLAMA_F5A_PROFILE_WINDOWS");
    const bool f5a_profile_windows = f5a_profile_env != nullptr;
    if (f5a_profile_windows && std::strcmp(f5a_profile_env, "1") != 0) {
        fprintf(stderr, "%s: error: IK_LLAMA_F5A_PROFILE_WINDOWS accepts only the value 1\n", __func__);
        return 2;
    }
#ifndef GGML_USE_CUDA
    if (f5a_profile_windows) {
        fprintf(stderr, "%s: error: IK_LLAMA_F5A_PROFILE_WINDOWS requires a CUDA build\n", __func__);
        return 2;
    }
#endif
    const char * f5a_phase_env = std::getenv("IK_LLAMA_F5A_PHASE_PROFILE");
    const bool f5a_phase_profile = f5a_phase_env != nullptr;
    if (f5a_phase_profile && (std::strcmp(f5a_phase_env, "1") != 0 || !use_moe_prototype || !f5a_profile_windows)) {
        fprintf(stderr, "%s: error: IK_LLAMA_F5A_PHASE_PROFILE requires value 1, prototype mode and bounded profile windows\n", __func__);
        return 2;
    }
#ifndef GGML_USE_CUDA
    if (f5a_phase_profile) {
        fprintf(stderr, "%s: error: IK_LLAMA_F5A_PHASE_PROFILE requires a CUDA build\n", __func__);
        return 2;
    }
#endif

    const bool use_text = !params.prompt_file.empty() && !params.prompt_is_binary;
    if (use_text && params.prompt.empty()) {
        fprintf(stderr, "%s: error: %s is empty\n", __func__, params.prompt_file.c_str());
        return 1;
    }
    if (!use_text && !params.prompt.empty()) {
        fprintf(stderr, "%s: warning: -p and -bf are ignored, use -f to supply a text\n", __func__);
    }

    if (params.minilog) {
        llama_log_set(llama_selective_log_callback, nullptr);
    }

    // init LLM

    llama_backend_init();
    llama_numa_init(params.numa);

#ifdef GGML_USE_CUDA
    if (use_moe_prototype && ggml_backend_cuda_get_device_count() != 2) {
        fprintf(stderr, "%s: error: IK_LLAMA_MOE_HYBRID_PROTOTYPE requires exactly two visible CUDA devices\n", __func__);
        llama_backend_free();
        return 2;
    }
#endif

    sweep_vram_tracker vram_tracker;
    if (params.sweep_memory) {
        vram_tracker.start();
    }

    // initialize the model

    llama_model_params model_params = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    llama_context_params ctx_params = common_context_params_to_llama(params);

    llama_moe_ffn_prototype_lifecycle prototype_lifecycle{};
    llama_moe_ffn_prototype_config prototype_config{};
    prototype_config.all_devices = true;
    prototype_config.layer = 0; // Valid context-factory anchor; all-device discovery remains cache-driven.
    prototype_config.cpu_threads = 12;
    prototype_config.lifecycle = &prototype_lifecycle;

    llama_context * ctx = use_moe_prototype
        ? llama_init_from_model_with_moe_ffn_prototype(model, ctx_params, prototype_config)
        : llama_init_from_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    if (use_moe_prototype) {
        if (!ctx->moe_ffn_prototype || !ctx->moe_ffn_prototype->valid() ||
                ctx->moe_ffn_prototype->devices.size() != 2 ||
                ctx->moe_ffn_prototype->layer_count() != 43) {
            fprintf(stderr, "%s: error: prototype did not initialize both expected Qwen cache groups and 43 compatible layers\n", __func__);
            llama_free(ctx);
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
        for (const auto & device : ctx->moe_ffn_prototype->devices) {
            if (device.groups.size() != 1 || device.layer_bindings.empty()) {
                fprintf(stderr, "%s: error: prototype device state lacks one group or compatible bindings\n", __func__);
                llama_free(ctx);
                llama_free_model(model);
                llama_backend_free();
                return 1;
            }
        }
        LOG_TEE("f4c_prototype_activation enabled=1 devices=2 groups=2 compatible_layers=43 cpu_threads_per_device=12 seed_experts=0 split=layer\n");
    } else if (ctx->moe_ffn_prototype) {
        fprintf(stderr, "%s: error: ordinary context unexpectedly created prototype state\n", __func__);
        llama_free(ctx);
        llama_free_model(model);
        llama_backend_free();
        return 1;
    } else {
        LOG_TEE("f4c_prototype_activation enabled=0 mode=ordinary blocking-cache-path=1 prototype_resources=0\n");
    }

    const bool use_checkpoint = common_speculative_needs_checkpoint(model);

    const unsigned int n_kv_max = llama_n_ctx(ctx);


    const llama_vocab * vocab = llama_get_vocab(ctx);
    llama_token bos = vocab->token_bos();
    //llama_token eos = llama_token_eos_impl(*vocab);

    const unsigned int n_vocab  = llama_n_vocab(model);

    // decode in batches of ctx_params.n_batch tokens
    auto decode_helper = [](llama_context * ctx, llama_batch & batch, int32_t n_batch) {
        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                LOG_TEE("failed to decode the batch, n_batch = %d, ret = %d\n", n_batch, ret);
                return false;
            }

            llama_synchronize(ctx);
        }

        return true;
    };

    const unsigned int pp = params.n_ubatch;
    const unsigned int tg = params.n_predict > 0 ? params.n_predict : params.n_ubatch / 4;

    std::vector<llama_token> text;
    if (use_text) {
        text = common_tokenize(ctx, params.prompt, true);
        const unsigned int n_text = text.size();
        unsigned int n_end = 0;
        for (unsigned int n_kv = 0, i_loop = 0; n_kv < n_kv_max; n_kv += pp, ++i_loop) {
            n_end = std::max(n_end, n_kv + std::max(pp, i_loop % params.sweep_stride == 0 ? tg : 0u));
        }
        if (n_end > n_text) {
            LOG_TEE("%s: n_text = %u tokens from %s, repeats from position %u\n", __func__, n_text, params.prompt_file.c_str(), n_text);
        } else {
            LOG_TEE("%s: n_text = %u tokens from %s\n", __func__, n_text, params.prompt_file.c_str());
        }
    }
    const std::string text_json = use_text ? ", \"n_text\": " + std::to_string(text.size()) : "";

    if (!params.sweep_bench_output_jsonl) {
        LOG_TEE("\n");
        LOG_TEE("%s: n_kv_max = %d, n_batch = %d, n_ubatch = %d, flash_attn = %d, n_gpu_layers = %d, n_threads = %u, n_threads_batch = %u\n", __func__, n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch);
        LOG_TEE("\n");
        if (params.sweep_memory) {
            LOG_TEE("|%6s | %6s | %6s | %8s | %8s | %8s | %8s | %10s | %10s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s", "RSS HWM", "VRAM delta");
            LOG_TEE("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|-%10s-|-%10s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------", "----------", "----------");
        } else {
            LOG_TEE("|%6s | %6s | %6s | %8s | %8s | %8s | %8s |\n", "PP", "TG", "N_KV", "T_PP s", "S_PP t/s", "T_TG s", "S_TG t/s");
            LOG_TEE("|%6s-|-%6s-|-%6s-|-%8s-|-%8s-|-%8s-|-%8s-|\n", "------", "------", "------", "--------", "--------", "--------", "--------");
        }
    }

    llama_batch batch = llama_batch_init(n_kv_max, 0, 1);

    auto pp_helper = [&](unsigned int n_kv) {
        common_batch_clear(batch);

        for (unsigned int i = 0; i < pp; ++i) {
            common_batch_add(batch, use_text ? text[(n_kv + i) % text.size()] : std::rand() % n_vocab, n_kv + i, { 0 }, false);
        }
        batch.logits[batch.n_tokens - 1] = true;

        return decode_helper(ctx, batch, ctx_params.n_batch);
    };

    // warm up
    if (params.warmup) {
        common_batch_add(batch, bos, 0, { 0 }, false);

        if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }
    if (params.batch_warmup) {
        // clean up KV cache after generation
        llama_kv_cache_seq_rm(ctx, 0, params.n_ubatch, -1);

        // prepare batch of pp size for prompt processing performance measurement
        common_batch_clear(batch);

        for (unsigned int i = 0; i < params.n_ubatch; ++i) {
            common_batch_add(batch, use_text ? text[i % text.size()] : std::rand() % n_vocab, i, { 0 }, false);
        }

        if (!decode_helper(ctx, batch, ctx_params.n_ubatch)) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }
    }

    common_batch_clear(batch);
    llama_kv_cache_clear(ctx);

    llama_reset_timings(ctx);

    int i_loop = 0;
    std::vector<uint8_t> checkpoint_data;
    std::vector<f5a_stats_checkpoint> f5a_checkpoints_data;
    if (f5a_checkpoints) {
        f5a_stats_checkpoint initial{};
        if (!collect_f5a_stats(ctx, UINT32_MAX, initial)) {
            LOG_TEE("f5a_counter_checkpoint error=initial_snapshot\n");
            return 1;
        }
        f5a_checkpoints_data.push_back(std::move(initial));
    }

    for (unsigned int n_kv = 0; n_kv < n_kv_max; n_kv += params.n_ubatch) {
        // clean up KV cache before generation
        //llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);

        const bool measure = i_loop % params.sweep_stride == 0;
        int nrep = measure && i_loop < 1 ? params.nrep : 1;

        size_t checkpoint_size = 0;
        if (use_checkpoint && measure && n_kv > 0) {
            const size_t need = llama_state_seq_get_size(ctx, 0, 0);
            checkpoint_data.resize(need);
            checkpoint_size = llama_state_seq_get_data(ctx, checkpoint_data.data(), need, 0, 0);
            if (checkpoint_size == 0) {
                LOG_TEE("%s: failed to checkpoint sequence at %u\n", __func__, n_kv);
                return 1;
            }
            checkpoint_data.resize(checkpoint_size);
        }

        // first measure token generation performance at this context size
        int64_t t_tg_start = 0;
        int64_t t_tg_end   = 0;
        const bool f5a_profile_this_row = f5a_profile_windows && (n_kv == 0 || n_kv == 24576);

        if (measure) {
            t_tg_start = ggml_time_us();
            //fprintf(stderr, "======================================== tg_start for n_kv = %u\n", n_kv);
            //printf("======================================== tg_start for n_kv = %u\n", n_kv);

            for (int irep = 0; irep < nrep; ++irep) {
                if (use_checkpoint) {
                    if (n_kv == 0) {
                        llama_kv_cache_clear(ctx);
                    }
                } else {
                    llama_kv_cache_seq_rm(ctx, 0, n_kv, -1);
                }

#ifdef GGML_USE_CUDA
                if (f5a_phase_profile && f5a_profile_this_row && irep == 0 &&
                        !set_f5a_phase_profile(ctx, true, n_kv)) {
                    LOG_TEE("f5a_phase_profile error=enable n_kv=%u\n", n_kv);
                    return 1;
                }
                if (f5a_profile_this_row && irep == 0 && cudaProfilerStart() != cudaSuccess) {
                    LOG_TEE("f5a_profile error=start n_kv=%u\n", n_kv);
                    return 1;
                }
#endif

                for (unsigned int i = 0; i < tg; ++i) {
                    common_batch_clear(batch);
                    common_batch_add(batch, use_text ? text[(n_kv + i) % text.size()] : std::rand() % n_vocab, n_kv + i, { 0 }, true);

                    if (!decode_helper(ctx, batch, ctx_params.n_batch)) {
                        LOG_TEE("%s: llama_decode() failed\n", __func__);
                        return 1;
                    }
                }
            }

            //fprintf(stderr, "======================================== tg_end for n_kv = %u\n", n_kv);
            //printf("======================================== tg_end for n_kv = %u\n", n_kv);
            t_tg_end = ggml_time_us();
#ifdef GGML_USE_CUDA
            if (f5a_profile_this_row) {
                const cudaError_t profile_status = cudaProfilerStop();
                if (profile_status != cudaSuccess) {
                    LOG_TEE("f5a_profile error=stop n_kv=%u cuda_error=%s\n", n_kv, cudaGetErrorString(profile_status));
                    return 1;
                }
                if (f5a_phase_profile && !set_f5a_phase_profile(ctx, false, n_kv)) {
                    LOG_TEE("f5a_phase_profile error=disable n_kv=%u\n", n_kv);
                    return 1;
                }
                LOG_TEE("f5a_profile_window_complete n_kv=%u tg_tokens=%u\n", n_kv, tg);
            }
#endif
        } else {
            // keep the token stream aligned with a stride-1 sweep
            for (unsigned int i = 0; i < tg; ++i) {
                (void) std::rand();
            }
        }

        if (use_checkpoint && measure) {
            if (n_kv > 0) {
                const size_t n = llama_state_seq_set_data(ctx, checkpoint_data.data(), checkpoint_data.size(), 0, 0);
                if (n != checkpoint_size) {
                    LOG_TEE("%s: failed to restore sequence (expected %zu bytes, got %zu)\n", __func__, checkpoint_size, n);
                    return 1;
                }
            } else {
                llama_kv_cache_clear(ctx);
            }
        }

        // measure prompt processing performance
        int64_t t_pp_start = 0;
        int64_t t_pp_end   = 0;

        if (measure) {
            t_pp_start = ggml_time_us();

            for (int irep = 0; irep < nrep; ++irep) {
                if (use_checkpoint) {
                    if (n_kv == 0) {
                        llama_kv_cache_clear(ctx);
                    }
                } else {
                    if (!llama_kv_cache_seq_rm(ctx, 0, n_kv, -1)) {
                        LOG_TEE("%s: failed to rewind sequence to %u\n", __func__, n_kv);
                        return 1;
                    }
                }

                if (!pp_helper(n_kv)) {
                    LOG_TEE("%s: llama_decode() failed\n", __func__);
                    return 1;
                }
            }

            t_pp_end = ggml_time_us();
        } else {
            if (!pp_helper(n_kv)) {
                LOG_TEE("%s: llama_decode() failed\n", __func__);
                return 1;
            }
        }

        if (!measure) {
            ++i_loop;
            continue;
        }

        // calculate and print metrics
        const float t_pp = (t_pp_end - t_pp_start) / 1000000.0f / nrep;
        const float t_tg = (t_tg_end - t_tg_start) / 1000000.0f / nrep;

        const float speed_pp = pp / t_pp;
        const float speed_tg = tg / t_tg;

        double rss_hwm_mib    = -1.0;
        double vram_delta_mib = -1.0;
        if (params.sweep_memory) {
            rss_hwm_mib    = get_rss_hwm_mib();
            vram_delta_mib = vram_tracker.sample();
        }

        if(params.sweep_bench_output_jsonl) {
            if (params.sweep_memory) {
                const std::string rss_json  = format_mib(rss_hwm_mib, 3, "null");
                const std::string vram_json = format_mib(vram_delta_mib, 3, "null");
                LOG_TEE(
                    "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                    "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f, \"rss_hwm_mib\": %s, \"vram_delta_mib\": %s%s }\n",
                    n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                    pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg, rss_json.c_str(), vram_json.c_str(), text_json.c_str()
                );
            } else {
                LOG_TEE(
                    "{\"n_kv_max\": %d, \"n_batch\": %d, \"n_ubatch\": %d, \"flash_attn\": %d, \"n_gpu_layers\": %d, \"n_threads\": %u, \"n_threads_batch\": %u, "
                    "\"pp\": %d, \"tg\": %d, \"n_kv\": %d, \"t_pp\": %f, \"speed_pp\": %f, \"t_tg\": %f, \"speed_tg\": %f%s }\n",
                    n_kv_max, params.n_batch, params.n_ubatch, params.flash_attn, params.n_gpu_layers, ctx_params.n_threads, ctx_params.n_threads_batch,
                    pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg, text_json.c_str()
                );
            }
        } else {
            if (params.sweep_memory) {
                const std::string rss  = format_mib(rss_hwm_mib, 1, "n/a");
                const std::string vram = format_mib(vram_delta_mib, 1, "n/a");
                LOG_TEE("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f | %10s | %10s |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg, rss.c_str(), vram.c_str());
            } else {
                LOG_TEE("|%6d | %6d | %6d | %8.3f | %8.2f | %8.3f | %8.2f |\n", pp, tg, n_kv, t_pp, speed_pp, t_tg, speed_tg);
            }
        }

        // Read counters only after the measured TG and PP intervals have ended.
        if (f5a_checkpoints && ((n_kv + pp) % (8 * pp) == 0 || n_kv + pp == n_kv_max)) {
            f5a_stats_checkpoint checkpoint{};
            if (!collect_f5a_stats(ctx, n_kv, checkpoint)) {
                LOG_TEE("f5a_counter_checkpoint error=snapshot n_kv=%u\n", n_kv);
                return 1;
            }
            f5a_checkpoints_data.push_back(std::move(checkpoint));
        }

        ++i_loop;
    }

    if (f5a_checkpoints && !report_f5a_checkpoint_deltas(f5a_checkpoints_data)) {
        LOG_TEE("f5a_counter_checkpoint error=report\n");
        return 1;
    }

    llama_print_timings(ctx);

    llama_batch_free(batch);

    const bool prototype_stats_ok = report_moe_prototype_stats(ctx, use_moe_prototype);
#ifdef GGML_USE_CUDA
    const bool phase_stats_ok = !f5a_phase_profile || report_f5a_phase_samples(ctx);
#else
    const bool phase_stats_ok = true;
#endif

    llama_free(ctx);
    llama_free_model(model);

    LOG_TEE("f4c_prototype_lifecycle_after_context_free mode=%s leases=%d resources=%d bindings=%d cpu_graphs=%d cpu_graph_context_bytes=%zu\n",
            use_moe_prototype ? "hybrid" : "blocking", prototype_lifecycle.live_leases,
            prototype_lifecycle.live_cpu_resources, prototype_lifecycle.live_bindings,
            prototype_lifecycle.live_cpu_graphs, prototype_lifecycle.cpu_graph_context_bytes);
    if (!prototype_stats_ok || !phase_stats_ok || prototype_lifecycle.live_leases != 0 ||
            prototype_lifecycle.live_cpu_resources != 0 || prototype_lifecycle.live_bindings != 0 ||
            prototype_lifecycle.live_cpu_graphs != 0) {
        llama_backend_free();
        return 1;
    }

    llama_backend_free();

    return 0;
}
