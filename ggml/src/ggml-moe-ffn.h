#pragma once

#include "ggml.h"

#include <stdint.h>

struct ggml_moe_ffn_op_params {
    int32_t device;
    int32_t binding_id;
    uint64_t generation;
    int32_t n_embd;
    int32_t n_ff;
    int32_t unary_op;
    float limit;
};
