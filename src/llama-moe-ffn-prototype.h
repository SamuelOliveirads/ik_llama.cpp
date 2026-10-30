#pragma once

#include "llama-moe-cache.h"
#include "ggml-moe-ffn.h"

#include <cstdint>
#include <vector>

struct llama_context;
struct llama_context_params;
struct llama_model;
struct ggml_context;
struct ggml_tensor;
struct ggml_cgraph;
struct ggml_backend;
struct ggml_backend_buffer;

struct llama_moe_ffn_prototype_lifecycle {
    int live_leases = 0;
    int live_cpu_resources = 0;
    int live_bindings = 0;
    int live_cpu_graphs = 0;
    size_t cpu_graph_context_bytes = 0;
};

struct llama_moe_ffn_prototype_config {
    int32_t layer = -1;
    int32_t device = -1;
    int32_t cpu_threads = 1;
    bool all_devices = false;
    bool test_poison_cpu_priming_inputs = false;
    std::vector<int32_t> seed_experts;
    llama_moe_ffn_prototype_lifecycle * lifecycle = nullptr;
};

struct llama_moe_ffn_prototype_state {
    struct group_state {
        llama_moe_cache::frozen_lease lease;
        llama_moe_cache_frozen_layer snapshot;
        std::vector<ggml_tensor *> stale_slot_maps;
    };

    struct layer_binding {
        llama_moe_cache_frozen_layer snapshot;
        ggml_backend * cuda_backend = nullptr;
        ggml_backend_buffer * cpu_buffer = nullptr;
        ggml_context * cpu_ctx = nullptr;
        ggml_cgraph * cpu_graph = nullptr;
        ggml_tensor * cpu_activations = nullptr;
        ggml_tensor * cpu_ids = nullptr;
        ggml_tensor * cpu_routes = nullptr;
        ggml_tensor * cpu_output = nullptr;
        int32_t binding_id = 0;
        int32_t cpu_resource_id = 0;
        ggml_moe_ffn_layout layout{};
    };

    struct device_state {
        ggml_backend * cuda_backend = nullptr;
        ggml_backend * cpu_backend = nullptr;
        int32_t device = -1;
        int32_t cpu_resource_id = 0;
        int32_t n_embd = 0;
        int32_t n_ff = 0;
        int32_t n_experts_used = 0;
        bool cpu_backend_transferred = false;
        std::vector<group_state> groups;
        std::vector<layer_binding> layer_bindings;
    };

    llama_moe_cache * cache = nullptr;
    llama_moe_ffn_prototype_lifecycle * lifecycle = nullptr;
    std::vector<device_state> devices;

    bool initialize(llama_moe_cache & cache, const llama_model & model,
            const llama_moe_ffn_prototype_config & config);
    void shutdown();
    bool valid() const;
    const layer_binding * find_layer(int32_t il) const;
    const group_state * find_group(int32_t group) const;
    size_t layer_count() const;
    bool collect_stats(ggml_cuda_moe_ffn_stats & stats) const;
    bool collect_device_stats(ggml_backend * backend, ggml_cuda_moe_ffn_stats & stats) const;
    bool collect_resource_stats(ggml_cuda_moe_ffn_cpu_resource_stats & stats) const;
    bool collect_device_resource_stats(ggml_backend * backend, ggml_cuda_moe_ffn_cpu_resource_stats & stats) const;
    ggml_tensor * build_node(ggml_context * ctx, ggml_backend_sched_t sched,
            const ggml_moe_ffn_layout & layout);
};

// Internal-only construction entry point used by the focused context fixture.
struct llama_context * llama_init_from_model_with_moe_ffn_prototype(
        struct llama_model * model, struct llama_context_params params,
        const llama_moe_ffn_prototype_config & config);
