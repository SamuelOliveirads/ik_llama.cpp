#pragma once

#include "ggml.h"

#include <stdint.h>
#include <math.h>

enum {
    GGML_MOE_FFN_MAX_SELECTED = 10,
    GGML_MOE_FFN_PARTS = 3,
    GGML_MOE_FFN_MAX_EMBD = 8192,
    GGML_MOE_FFN_MAX_FF = 32768,
    GGML_MOE_FFN_ROW_ALIGNMENT = 32,
};

enum ggml_moe_ffn_layout_flag {
    GGML_MOE_FFN_LAYOUT_SEPARATE_PARTS = 1u << 0,
    GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS = 1u << 1,
    GGML_MOE_FFN_LAYOUT_NO_HADAMARD = 1u << 2,
    GGML_MOE_FFN_LAYOUT_HOST_RESIDENT = 1u << 3,
};

enum ggml_moe_ffn_graph_flag {
    GGML_MOE_FFN_GRAPH_ONE_TOKEN = 1u << 0,
    GGML_MOE_FFN_GRAPH_NO_LORA = 1u << 1,
    GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS = 1u << 2,
    GGML_MOE_FFN_GRAPH_SILU_LIMIT = 1u << 3,
    GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS = 1u << 4,
    GGML_MOE_FFN_GRAPH_NO_HADAMARD = 1u << 5,
};

// Immutable description of the currently implemented three-part routed FFN.
// Tensor pointers are identities; their metadata is validated by the query.
struct ggml_moe_ffn_layout {
    int32_t n_experts, n_experts_used, n_embd, n_ff, n_slots, n_parts;
    int32_t device, cache_group, cache_layer;
    const void * owner_backend;
    const void * cache_owner;
    const struct ggml_tensor * up, * gate, * down;
    const struct ggml_tensor * bank_up, * bank_gate, * bank_down;
    const struct ggml_tensor * activation, * selected_ids, * route_weights;
    enum ggml_unary_op unary_op;
    float limit;
    uint32_t flags;
    uint32_t graph_flags;
};

// Router-side inputs are deliberately absent from this predicate: the semantic
// op receives completed selected IDs and final route weights. Only values that
// alter expert arithmetic or its representation belong here.
static inline bool ggml_moe_ffn_expert_execution_inputs_supported(
        const struct ggml_tensor * up_bias, const struct ggml_tensor * gate_bias,
        const struct ggml_tensor * down_bias, const struct ggml_tensor * merged_up_gate,
        const struct ggml_tensor * merged_up_gate_bias, const struct ggml_tensor * down_scale) {
    return !up_bias && !gate_bias && !down_bias && !merged_up_gate &&
        !merged_up_gate_bias && !down_scale;
}

static inline bool ggml_moe_ffn_tensor_shape(const struct ggml_tensor * tensor,
        enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    return tensor && tensor->type == type && tensor->ne[0] == ne0 && tensor->ne[1] == ne1 &&
        tensor->ne[2] == ne2 && tensor->ne[3] == 1 && ne0 > 0 && ne1 > 0 && ne2 > 0 &&
        ggml_n_dims(tensor) == (ne2 > 1 ? 3 : ne1 > 1 ? 2 : 1);
}

static inline bool ggml_moe_ffn_tensor_layout(const struct ggml_tensor * tensor,
        enum ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2) {
    if (!ggml_moe_ffn_tensor_shape(tensor, type, ne0, ne1, ne2) ||
        tensor->nb[0] != ggml_type_size(type)) {
        return false;
    }

    const size_t row_bytes = ggml_row_size(type, ne0);
    if (row_bytes == 0 || row_bytes > SIZE_MAX / (size_t) ne1) {
        return false;
    }
    const size_t expert_bytes = row_bytes * (size_t) ne1;
    if (expert_bytes > SIZE_MAX / (size_t) ne2) {
        return false;
    }

    // The executor addresses experts with nb[2], so each slice must be exactly
    // its packed row extent: no inter-row padding, inter-expert padding or aliasing.
    return tensor->nb[1] == row_bytes && tensor->nb[2] == expert_bytes &&
        tensor->nb[3] == expert_bytes * (size_t) ne2 && ggml_is_contiguous(tensor);
}

static inline bool ggml_moe_ffn_bank_layout_matches(const struct ggml_tensor * host,
        const struct ggml_tensor * bank) {
    return host && bank && host->type == bank->type && host->ne[0] == bank->ne[0] &&
        host->ne[1] == bank->ne[1] && host->ne[3] == bank->ne[3] &&
        host->nb[0] == bank->nb[0] && host->nb[1] == bank->nb[1] && host->nb[2] == bank->nb[2];
}

static inline bool ggml_moe_ffn_types_supported(enum ggml_type up, enum ggml_type gate,
        enum ggml_type down) {
    return up == GGML_TYPE_IQ3_S && gate == GGML_TYPE_IQ3_S &&
        (down == GGML_TYPE_IQ4_NL || down == GGML_TYPE_IQ4_XS);
}

static inline bool ggml_moe_ffn_layout_supported(const struct ggml_moe_ffn_layout * l) {
    const uint32_t required = GGML_MOE_FFN_LAYOUT_SEPARATE_PARTS |
        GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS | GGML_MOE_FFN_LAYOUT_NO_HADAMARD |
        GGML_MOE_FFN_LAYOUT_HOST_RESIDENT;
    return l && l->n_experts > 0 && l->n_experts_used >= 1 && l->n_experts_used <= l->n_experts &&
        l->n_experts_used <= GGML_MOE_FFN_MAX_SELECTED && l->n_embd > 0 &&
        l->n_embd <= GGML_MOE_FFN_MAX_EMBD && l->n_embd % GGML_MOE_FFN_ROW_ALIGNMENT == 0 &&
        l->n_ff > 0 && l->n_ff <= GGML_MOE_FFN_MAX_FF &&
        l->n_ff % GGML_MOE_FFN_ROW_ALIGNMENT == 0 && l->n_parts == GGML_MOE_FFN_PARTS &&
        l->unary_op == GGML_UNARY_OP_SILU && isfinite(l->limit) && l->limit >= 0.0f &&
        (l->flags & required) == required &&
        l->up && l->gate && l->down &&
        ggml_moe_ffn_types_supported(l->up->type, l->gate->type, l->down->type) &&
        ggml_moe_ffn_tensor_layout(l->up, l->up->type, l->n_embd, l->n_ff, l->n_experts) &&
        ggml_moe_ffn_tensor_layout(l->gate, l->gate->type, l->n_embd, l->n_ff, l->n_experts) &&
        ggml_moe_ffn_tensor_layout(l->down, l->down->type, l->n_ff, l->n_embd, l->n_experts) &&
        ((l->n_slots == 0 && !l->bank_up && !l->bank_gate && !l->bank_down) ||
         (l->n_slots > 0 && l->device >= 0 && l->cache_group >= 0 && l->cache_layer >= 0 &&
           l->owner_backend && l->cache_owner &&
            ggml_moe_ffn_tensor_layout(l->bank_up, l->up->type, l->n_embd, l->n_ff, l->n_slots) &&
            ggml_moe_ffn_tensor_layout(l->bank_gate, l->gate->type, l->n_embd, l->n_ff, l->n_slots) &&
           ggml_moe_ffn_tensor_layout(l->bank_down, l->down->type, l->n_ff, l->n_embd, l->n_slots) &&
          ggml_moe_ffn_bank_layout_matches(l->up, l->bank_up) &&
          ggml_moe_ffn_bank_layout_matches(l->gate, l->bank_gate) &&
          ggml_moe_ffn_bank_layout_matches(l->down, l->bank_down)));
}

static inline bool ggml_moe_ffn_graph_inputs_supported(const struct ggml_moe_ffn_layout * l) {
    return ggml_moe_ffn_layout_supported(l) && l->n_slots > 0 &&
        ggml_moe_ffn_tensor_shape(l->activation, GGML_TYPE_F32, l->n_embd, 1, 1) &&
        ggml_moe_ffn_tensor_shape(l->selected_ids, GGML_TYPE_I32, l->n_experts_used, 1, 1) &&
        ggml_moe_ffn_tensor_shape(l->route_weights, GGML_TYPE_F32, 1, l->n_experts_used, 1) &&
        ggml_is_contiguous(l->activation) && ggml_is_contiguous(l->selected_ids) &&
        ggml_is_contiguous(l->route_weights);
}

static inline bool ggml_moe_ffn_graph_supported(const struct ggml_moe_ffn_layout * l) {
    const uint32_t required = GGML_MOE_FFN_GRAPH_ONE_TOKEN | GGML_MOE_FFN_GRAPH_NO_LORA |
        GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS | GGML_MOE_FFN_GRAPH_SILU_LIMIT |
        GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS | GGML_MOE_FFN_GRAPH_NO_HADAMARD;
    return l && (l->graph_flags & required) == required && ggml_moe_ffn_graph_inputs_supported(l);
}

struct ggml_moe_ffn_op_params {
    int32_t device;
    int32_t binding_id;
    uint64_t generation;
    int32_t n_embd;
    int32_t n_ff;
    int32_t unary_op;
    float limit;
};
