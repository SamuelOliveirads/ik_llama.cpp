#pragma once

#include <cstdint>
struct llama_batch;
struct llama_context;

bool llama_prepare_dsv4_graph_inputs(llama_context & lctx, const llama_batch & batch, bool set_tensors, bool reserve_plan);
void llama_reset_dsv4_state(llama_context * ctx, int32_t seq_id = -1);
bool llama_dsv4_spec_ckpt_save(llama_context * ctx, bool use_gpu);
bool llama_dsv4_spec_ckpt_restore(llama_context * ctx, bool use_gpu);
void llama_dsv4_spec_ckpt_discard(llama_context * ctx);
bool llama_dsv4_spec_ckpt_gpu_active(const llama_context * ctx);
uint64_t llama_dsv4_state_fingerprint(const llama_context * ctx);
