#include "llama-spec-features.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>

#include "llama-model.h"
#include "llama-context.h"

llama_context::dflash_runtime::capture_state::~capture_state() {
    for (ggml_backend_buffer_t buf : gpu_layer_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    if (gpu_ctx != nullptr) {
        ggml_free(gpu_ctx);
    }
}


void llama_reset_dflash_kv_cache_state(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dflash.kv.cache_write_pos = 0;
    ctx->dflash.kv.cache_n_filled = 0;
    ctx->dflash.kv.cache_update_rows = 0;
    ctx->dflash.kv.cache_view_write_pos = 0;
    ctx->dflash.kv.cache_view_n_filled = 0;
    ctx->dflash.kv.cache_applied_window_version = 0;
    ctx->dflash.kv.cache_valid = false;
    ctx->dflash.kv.cache_view_valid = false;
    ctx->dflash.kv.device_input_ready = false;
    ctx->dflash.kv.device_input_rows = 0;
    ctx->dflash.kv.device_input_row_offset = 0;
    std::fill(ctx->dflash.kv.cache_pos.begin(), ctx->dflash.kv.cache_pos.end(), 0);
    std::fill(ctx->dflash.kv.cache_slot_valid.begin(), ctx->dflash.kv.cache_slot_valid.end(), 0);

    for (ggml_backend_buffer_t buf : ctx->dflash.kv.cache_bufs) {
        if (buf != nullptr) {
            ggml_backend_buffer_clear(buf, 0);
        }
    }
}

llama_dflash_kv_cache_transition llama_plan_dflash_kv_cache_transition_for_ctx(
        const struct llama_context * ctx,
        const llama_dflash_window_update & window_update,
        int32_t n_rows) {
    if (ctx == nullptr) {
        llama_dflash_kv_cache_transition plan;
        plan.rebuild_cache = true;
        plan.append_rows = std::clamp(window_update.append_rows, 0, n_rows);
        plan.next_n_filled = n_rows;
        return plan;
    }

    const int32_t cross_ctx = ctx->dflash.visible_cross_ctx > 0
            ? ctx->dflash.visible_cross_ctx
            : std::max<int32_t>(1, (int32_t) ctx->cparams.n_ctx - (int32_t) ctx->model.hparams.dflash_block_size);

    return llama_plan_dflash_kv_cache_transition(
            cross_ctx,
            ctx->dflash.kv.cache_n_filled,
            ctx->dflash.kv.cache_write_pos,
            ctx->dflash.kv.cache_valid,
            ctx->dflash.kv.cache_applied_window_version,
            window_update.version,
            window_update.keep_rows,
            window_update.append_rows,
            window_update.replace,
            n_rows);
}

void llama_set_dflash_visible_cross_ctx(
        struct llama_context * ctx,
        int32_t cross_ctx) {
    if (ctx == nullptr) {
        return;
    }

    ctx->dflash.visible_cross_ctx = std::max<int32_t>(0, cross_ctx);
}

int32_t llama_get_dflash_visible_cross_ctx(
        const struct llama_context * ctx) {
    return ctx != nullptr ? ctx->dflash.visible_cross_ctx : 0;
}

int32_t llama_model_dflash_block_size(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_block_size : 0;
}

int32_t llama_model_dflash_mask_token_id(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_mask_token_id : -1;
}

int32_t llama_model_dflash_n_target_layers(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_n_target_layers : 0;
}

int32_t llama_model_dflash_n_target_features(const struct llama_model * model) {
    return model ? (int32_t) model->hparams.dflash_n_target_features : 0;
}

int32_t llama_model_dflash_target_layer_ids(
        const struct llama_model * model,
        int32_t * layer_ids,
        int32_t capacity) {
    if (model == nullptr || layer_ids == nullptr || capacity <= 0) {
        return 0;
    }

    const int32_t n_layers = std::min<int32_t>((int32_t) model->hparams.dflash_n_target_layers, capacity);
    for (int32_t i = 0; i < n_layers; ++i) {
        layer_ids[i] = (int32_t) model->hparams.dflash_target_layer_ids[i];
    }

    return n_layers;
}

int32_t llama_model_dflash_target_mask_token_id(const struct llama_model * model) {
    if (model == nullptr) {
        return (int32_t) LLAMA_TOKEN_NULL;
    }

    return (int32_t) model->vocab.token_mask();
}

static const ggml_tensor * llama_dflash_output_tensor(
        const struct llama_model * model) {
    if (model == nullptr) {
        return nullptr;
    }

    if (model->output_mtp != nullptr) {
        return model->output_mtp;
    }

    if (model->output != nullptr) {
        return model->output;
    }

    return model->tok_embd;
}

int32_t llama_model_dflash_io_mode(
        const struct llama_model * draft_model,
        const struct llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr || draft_model->arch != LLM_ARCH_DFLASH_DRAFT) {
        return LLAMA_DFLASH_IO_MODE_INVALID;
    }

    const ggml_tensor * draft_output = llama_dflash_output_tensor(draft_model);
    const ggml_tensor * target_output = llama_dflash_output_tensor(target_model);
    if (draft_model->tok_embd == nullptr || draft_output == nullptr || target_model->tok_embd == nullptr || target_output == nullptr) {
        return LLAMA_DFLASH_IO_MODE_INVALID;
    }

    const bool shared_tok = draft_model->tok_embd == target_model->tok_embd;
    const bool shared_output = draft_output == target_output;
    if (shared_tok && shared_output) {
        return LLAMA_DFLASH_IO_MODE_SHARED;
    }

    if (!shared_tok && !shared_output) {
        return LLAMA_DFLASH_IO_MODE_SELF_CONTAINED;
    }

    return LLAMA_DFLASH_IO_MODE_MIXED;
}

bool llama_model_dflash_io_tensors_match(
        const struct llama_model * draft_model,
        int32_t n_embd,
        int32_t n_vocab) {
    const ggml_tensor * output = llama_dflash_output_tensor(draft_model);
    if (draft_model == nullptr || draft_model->tok_embd == nullptr || output == nullptr || n_embd <= 0 || n_vocab <= 0) {
        return false;
    }

    return (int32_t) draft_model->tok_embd->ne[0] == n_embd &&
           (int32_t) draft_model->tok_embd->ne[1] == n_vocab &&
           (int32_t) output->ne[0] == n_embd &&
           (int32_t) output->ne[1] == n_vocab;
}

bool llama_model_share_dflash_io_tensors(
        struct llama_model * draft_model,
        const struct llama_model * target_model) {
    if (draft_model == nullptr || target_model == nullptr) {
        return false;
    }

    if (draft_model->arch != LLM_ARCH_DFLASH_DRAFT) {
        return true;
    }

    if (draft_model->tok_embd == nullptr) {
        draft_model->tok_embd = target_model->tok_embd;
    }

    if (draft_model->output == nullptr) {
        draft_model->output = target_model->output ? target_model->output : target_model->tok_embd;
        if (draft_model->output == nullptr) {
            draft_model->output = draft_model->tok_embd;
        }
    }

    const bool uses_shared_tok = draft_model->tok_embd == target_model->tok_embd;
    const bool uses_shared_output = draft_model->output == target_model->output ||
            draft_model->output == target_model->tok_embd;

    if (draft_model->output_mtp == nullptr) {
        if (target_model->output_mtp != nullptr && uses_shared_tok && uses_shared_output) {
            draft_model->output_mtp = target_model->output_mtp;
        } else if (draft_model->output != nullptr) {
            draft_model->output_mtp = draft_model->output;
        } else {
            draft_model->output_mtp = draft_model->tok_embd;
        }
    }

    const struct ggml_tensor * output = llama_dflash_output_tensor(draft_model);
    return draft_model->tok_embd != nullptr && output != nullptr;
}

static bool llama_set_dflash_target_features_impl(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        bool copy_data,
        const llama_dflash_window_update * window_update) {
    const bool have_full_features = target_features != nullptr && n_floats > 0;
    const bool have_append_features = window_update != nullptr &&
            window_update->append_features != nullptr &&
            window_update->append_floats > 0 &&
            window_update->append_rows > 0;

    const bool have_device_features = ctx != nullptr && window_update != nullptr &&
            window_update->append_rows > 0 && ctx->dflash.kv.device_input_ready &&
            ctx->dflash.kv.device_input_rows == window_update->append_rows;

    if (ctx == nullptr || n_rows <= 0 || (!have_full_features && !have_append_features && !have_device_features)) {
        return false;
    }

    if (have_full_features && copy_data) {
        ctx->dflash.target.features_owned.assign(target_features, target_features + n_floats);
        ctx->dflash.target.features = ctx->dflash.target.features_owned.data();
    } else if (have_full_features) {
        ctx->dflash.target.features_owned.clear();
        ctx->dflash.target.features = target_features;
    } else {
        ctx->dflash.target.features_owned.clear();
        ctx->dflash.target.features = nullptr;
    }
    ctx->dflash.target.features_n_floats = have_full_features ? n_floats : 0;
    ctx->dflash.target.features_n_rows = n_rows;
    if (have_append_features && copy_data) {
        ctx->dflash.target.append_features_owned.assign(
                window_update->append_features,
                window_update->append_features + window_update->append_floats);
        ctx->dflash.target.append_features = ctx->dflash.target.append_features_owned.data();
    } else if (have_append_features) {
        ctx->dflash.target.append_features_owned.clear();
        ctx->dflash.target.append_features = window_update->append_features;
    } else {
        ctx->dflash.target.append_features_owned.clear();
        ctx->dflash.target.append_features = nullptr;
    }
    ctx->dflash.target.append_features_n_floats = have_append_features ? window_update->append_floats : 0;
    ctx->dflash.target.append_features_n_rows = have_append_features ? window_update->append_rows : 0;
        ctx->dflash.target.version = window_update != nullptr && window_update->version > 0
            ? window_update->version
            : ctx->dflash.target.version + 1;
        ctx->dflash.target.keep_rows = window_update != nullptr
            ? std::max<int32_t>(0, std::min(n_rows, window_update->keep_rows))
            : 0;
        ctx->dflash.target.append_rows = window_update != nullptr
            ? std::max<int32_t>(0, std::min(n_rows, window_update->append_rows))
            : n_rows;
        ctx->dflash.target.replace = window_update != nullptr
            ? window_update->replace
            : true;
        if (ctx->dflash.target.keep_rows + ctx->dflash.target.append_rows > n_rows) {
        ctx->dflash.target.keep_rows = std::max<int32_t>(0, n_rows - ctx->dflash.target.append_rows);
        }

            const int32_t cross_ctx = ctx->dflash.visible_cross_ctx > 0
                ? ctx->dflash.visible_cross_ctx
                : std::max<int32_t>(1, (int32_t) ctx->cparams.n_ctx - (int32_t) ctx->model.hparams.dflash_block_size);
            const llama_dflash_window_update cache_window_update = {
                ctx->dflash.target.version,
                ctx->dflash.target.keep_rows,
                ctx->dflash.target.append_rows,
                ctx->dflash.target.replace,
                ctx->dflash.target.append_features,
                ctx->dflash.target.append_features_n_floats,
            };
            const llama_dflash_kv_cache_transition cache_plan = llama_plan_dflash_kv_cache_transition_for_ctx(ctx, cache_window_update, n_rows);

        if (cache_plan.cache_up_to_date) {
            ctx->dflash.kv.cache_view_n_filled = ctx->dflash.kv.cache_n_filled;
            ctx->dflash.kv.cache_view_write_pos = ctx->dflash.kv.cache_write_pos;
            ctx->dflash.kv.cache_view_valid = ctx->dflash.kv.cache_valid;
        } else if (cross_ctx > 0) {
            ctx->dflash.kv.cache_view_n_filled = cache_plan.next_n_filled;
            ctx->dflash.kv.cache_view_write_pos = cache_plan.next_write_pos;
            ctx->dflash.kv.cache_view_valid = cache_plan.next_n_filled > 0;
        }

    if (target_positions != nullptr) {
        if (copy_data) {
            ctx->dflash.target.positions_owned.assign(target_positions, target_positions + n_rows);
            ctx->dflash.target.positions = ctx->dflash.target.positions_owned.data();
        } else {
            ctx->dflash.target.positions_owned.clear();
            ctx->dflash.target.positions = target_positions;
        }
        ctx->dflash.target.positions_n = (size_t) n_rows;
    } else {
        ctx->dflash.target.positions_owned.clear();
        ctx->dflash.target.positions = nullptr;
        ctx->dflash.target.positions_n = 0;
    }

    return true;
}

bool llama_set_dflash_target_features_copy(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        const llama_dflash_window_update * window_update) {
    return llama_set_dflash_target_features_impl(ctx, target_features, n_floats, n_rows, target_positions, true, window_update);
}

bool llama_set_dflash_target_features_view(
        struct llama_context * ctx,
        const float * target_features,
        size_t n_floats,
        int32_t n_rows,
        const llama_pos * target_positions,
        const llama_dflash_window_update * window_update) {
    return llama_set_dflash_target_features_impl(ctx, target_features, n_floats, n_rows, target_positions, false, window_update);
}

static bool llama_dflash_parse_layer_id(const struct ggml_tensor * tensor, int32_t & layer_id) {
    if (tensor == nullptr) {
        return false;
    }

    static constexpr const char * prefix = "l_out-";
    if (std::strncmp(tensor->name, prefix, std::strlen(prefix)) != 0) {
        return false;
    }

    char * end = nullptr;
    const long raw = std::strtol(tensor->name + std::strlen(prefix), &end, 10);
    if (end == tensor->name + std::strlen(prefix) || *end != '\0') {
        return false;
    }

    layer_id = (int32_t) raw;
    if (layer_id >= 1000) {
        layer_id %= 1000;
    }

    return layer_id >= 0;
}

static int32_t llama_dflash_find_layer_index(const struct llama_context * ctx, int32_t layer_id) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return -1;
    }

    const auto & layer_ids = ctx->dflash.capture->layer_ids;
    const auto it = std::find(layer_ids.begin(), layer_ids.end(), layer_id);
    return it == layer_ids.end() ? -1 : (int32_t) std::distance(layer_ids.begin(), it);
}

static ggml_backend_t llama_dflash_backend_for_buffer(
        const llama_context & ctx,
        ggml_backend_buffer_t buffer);

static bool llama_dflash_capture_layout_matches(
        const ggml_tensor * source,
        const ggml_tensor * destination) {
    if (source == nullptr || destination == nullptr || source->type != destination->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (source->ne[i] != destination->ne[i] || source->nb[i] != destination->nb[i]) {
            return false;
        }
    }
    return true;
}

static void llama_dflash_capture_host_fallback(llama_context * ctx) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return;
    }

    auto & capture = *ctx->dflash.capture;
    capture.gpu_features_enabled = false;
    for (size_t layer_idx = 0; layer_idx < capture.layer_ids.size(); ++layer_idx) {
        ggml_tensor * source = layer_idx < capture.gpu_source_tensors.size()
                ? capture.gpu_source_tensors[layer_idx] : nullptr;
        if (source == nullptr || ctx->sched == nullptr || source->ne[0] <= 0 || source->ne[1] <= 0) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(ctx->sched, source);
        if (backend == nullptr) {
            continue;
        }
        const size_t n_floats = (size_t) source->ne[0] * (size_t) source->ne[1];
        capture.layer_rows[layer_idx].resize(n_floats);
        ggml_backend_tensor_get_async(backend, source, capture.layer_rows[layer_idx].data(), 0, n_floats * sizeof(float));
        capture.layer_seen_batch_id[layer_idx] = capture.capture_batch_id;
        capture.row_width = (int32_t) source->ne[0];
        capture.row_count = (int32_t) source->ne[1];
        if (capture.telemetry_enabled) {
            capture.capture_d2h_bytes += n_floats * sizeof(float);
        }
    }
}

static int llama_dflash_capture_eval_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    auto * ctx = static_cast<llama_context *>(user_data);
    if (ctx == nullptr || !ctx->dflash.capture) {
        return false;
    }

    int32_t layer_id = -1;
    if (!llama_dflash_parse_layer_id(tensor, layer_id)) {
        return 0;
    }

    const int32_t layer_idx = llama_dflash_find_layer_index(ctx, layer_id);
    if (layer_idx < 0) {
        return 0;
    }

    //printf("%s -> %d, %d\n", tensor->name, layer_id, layer_idx);

    if (ask) {
        return 2;
    }

    const int32_t row_width = (int32_t) tensor->ne[0];
    const int32_t row_count = row_width > 0 ? (int32_t) (ggml_nelements(tensor) / (int64_t) row_width) : 0;
    if (row_width <= 0 || row_count <= 0) {
        return 0;
    }

    auto & capture = *ctx->dflash.capture;
    if (capture.capture_batch_id == 0) {
        capture.capture_batch_id = 1;
    }
    if (capture.layer_seen_batch_id.size() != capture.layer_ids.size()) {
        capture.layer_seen_batch_id.assign(capture.layer_ids.size(), 0);
    }

    if (capture.gpu_source_tensors.size() == capture.layer_ids.size()) {
        capture.gpu_source_tensors[(size_t) layer_idx] = tensor;
    }
    auto backend = ggml_backend_sched_get_tensor_backend(ctx->sched, tensor);
    capture.layer_backends[(size_t) layer_idx] = backend;
    if (backend == nullptr) {
        capture.device_last_fallback_reason = "capture-backend-unresolved";
        LLAMA_LOG_WARN("DFlash capture callback rejected layer=%d tensor=%s reason=%s; using host recovery\n",
                layer_id, tensor->name, capture.device_last_fallback_reason.c_str());
        llama_dflash_capture_host_fallback(ctx);
        return 2;
    }
    if (capture.gpu_features_enabled) {
        capture.row_width = row_width;
        capture.row_count = row_count;
        capture.layer_seen_batch_id[(size_t) layer_idx] = 0;

        ggml_tensor * destination = capture.gpu_layer_views.size() == capture.layer_ids.size()
                ? capture.gpu_layer_views[(size_t) layer_idx] : nullptr;
        if (destination != nullptr && destination->buffer == nullptr) {
            ggml_backend_view_init(destination);
        }
        ggml_backend_t destination_backend = capture.gpu_layer_backends.size() == capture.layer_ids.size()
                ? capture.gpu_layer_backends[(size_t) layer_idx] : nullptr;
        if (destination_backend == nullptr) {
            destination_backend = llama_dflash_backend_for_buffer(
                    *ctx, destination != nullptr ? destination->buffer : nullptr);
        }
        const bool backend_match = backend != nullptr && destination_backend != nullptr &&
                (backend == destination_backend ||
                 ggml_backend_get_default_buffer_type(backend) == ggml_backend_get_default_buffer_type(destination_backend));
        const bool shape_match = llama_dflash_capture_layout_matches(tensor, destination);
        if (!backend_match || !shape_match) {
            capture.device_last_fallback_reason = !backend_match
                    ? "capture-backend-mismatch" : "capture-layout-mismatch";
            LLAMA_LOG_WARN("DFlash capture callback rejected layer=%d source=%s destination=%s reason=%s\n",
                    layer_id,
                    backend != nullptr ? ggml_backend_name(backend) : "<unknown>",
                    destination_backend != nullptr ? ggml_backend_name(destination_backend) : "<unknown>",
                    capture.device_last_fallback_reason.c_str());
            llama_dflash_capture_host_fallback(ctx);
            return 2;
        }

        ggml_backend_tensor_copy_async(backend, destination_backend, tensor, destination);
        capture.layer_device_produced_batch_id[(size_t) layer_idx] = capture.capture_batch_id;
        capture.layer_device_produced_rows[(size_t) layer_idx] = row_count;
        if (capture.telemetry_enabled) {
            const size_t bytes = ggml_nbytes(tensor);
            capture.device_same_device_bytes += bytes;
        }
        return 2;
    }
    auto & rows = capture.layer_rows[(size_t) layer_idx];
    rows.resize((size_t) row_count * (size_t) row_width);
    if (capture.telemetry_enabled) {
        capture.capture_d2h_bytes += (uint64_t) ggml_nbytes(tensor);
    }
    // Queue the host read on the tensor backend. llama_spec_prepare_dflash_capture()
    // synchronizes the scheduler before materialization consumes these rows.
    ggml_backend_tensor_get_async(backend, tensor, rows.data(), 0, ggml_nbytes(tensor));

    capture.row_width = row_width;
    capture.row_count = row_count;
    capture.layer_seen_batch_id[(size_t) layer_idx] = capture.capture_batch_id;
    return 2;
}

static ggml_backend_buffer_type_t llama_dflash_capture_layer_buft(
        const llama_context & ctx,
        int32_t layer_id) {
    if (layer_id >= 0 && layer_id < (int32_t) ctx.model.layers.size()) {
        const ggml_tensor * wk = ctx.model.layers[layer_id].wk;
        if (wk != nullptr && wk->buffer != nullptr) {
            return ggml_backend_buffer_get_type(wk->buffer);
        }
    }

    if (layer_id >= 0 && layer_id < (int32_t) ctx.model.buft_layer.size() && ctx.model.buft_layer[layer_id].buft != nullptr) {
        return ctx.model.buft_layer[layer_id].buft;
    }

    return llama_default_buffer_type_cpu(true);
}

static bool llama_dflash_init_gpu_capture(
        const llama_context & ctx,
        llama_context::dflash_runtime::capture_state & capture) {
    const char * enabled = std::getenv("IK_DFLASH_DEVICE_FEATURES");
    if (enabled == nullptr || enabled[0] != '1') {
        return true;
    }

    const int32_t capacity = std::max<int32_t>(1, ctx.cparams.n_ubatch);
    ggml_init_params params = {
        /*.mem_size   =*/ (size_t) (std::max(1, (int) capture.layer_ids.size()) + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    capture.gpu_ctx = ggml_init(params);
    if (capture.gpu_ctx == nullptr) {
        LLAMA_LOG_WARN("%s: failed to create GPU capture context; using CPU capture fallback\n", __func__);
        return false;
    }

    capture.gpu_layer_capacity = capacity;
    capture.gpu_layer_tensors.resize(capture.layer_ids.size(), nullptr);
    capture.gpu_layer_bufs.resize(capture.layer_ids.size(), nullptr);
    capture.gpu_layer_backends.resize(capture.layer_ids.size(), nullptr);
    for (size_t i = 0; i < capture.layer_ids.size(); ++i) {
        const int32_t layer_id = capture.layer_ids[i];
        ggml_tensor * tensor = ggml_new_tensor_2d(capture.gpu_ctx, GGML_TYPE_F32, ctx.model.hparams.n_embd, capacity);
        if (tensor == nullptr) {
            LLAMA_LOG_WARN("%s: failed to create GPU capture tensor for layer %d; using CPU capture fallback\n", __func__, layer_id);
            capture.gpu_capture_enabled = false;
            return false;
        }
        ggml_format_name(tensor, "dflash_gpu_capture_%d", layer_id);
        const ggml_backend_buffer_type_t buft = llama_dflash_capture_layer_buft(ctx, layer_id);
        ggml_backend_t capture_backend = nullptr;
        for (ggml_backend_t candidate : ctx.backends) {
            if (candidate != nullptr && ggml_backend_get_default_buffer_type(candidate) == buft) {
                capture_backend = candidate;
                break;
            }
        }
        if (capture_backend == nullptr || ggml_backend_is_cpu(capture_backend)) {
            LLAMA_LOG_WARN("%s: no non-CPU capture backend for layer %d; using CPU capture fallback\n", __func__, layer_id);
            capture.gpu_capture_enabled = false;
            return false;
        }
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
        ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
        if (buffer == nullptr) {
            LLAMA_LOG_WARN("%s: failed to allocate GPU capture buffer for layer %d; using CPU capture fallback\n", __func__, layer_id);
            capture.gpu_capture_enabled = false;
            return false;
        }
        ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
        ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
        ggml_backend_buffer_clear(buffer, 0);
        capture.gpu_layer_tensors[i] = tensor;
        capture.gpu_layer_bufs[i] = buffer;
        capture.gpu_layer_backends[i] = capture_backend;
    }

    capture.gpu_capture_enabled = true;
    capture.gpu_features_enabled = true;
    return true;
}

bool llama_set_dflash_capture_layers(
        struct llama_context * ctx,
        const int32_t * layer_ids,
        int32_t n_layers) {
    if (ctx == nullptr || layer_ids == nullptr || n_layers <= 0) {
        return false;
    }

    auto capture = std::make_unique<llama_context::dflash_runtime::capture_state>();
    const char * telemetry = std::getenv("IK_DFLASH_TELEMETRY");
    capture->telemetry_enabled = telemetry != nullptr && telemetry[0] == '1';
    capture->layer_ids.assign(layer_ids, layer_ids + n_layers);
    capture->layer_rows.resize((size_t) n_layers);
    capture->layer_backends.resize((size_t) n_layers, nullptr);
    capture->gpu_layer_views.resize((size_t) n_layers, nullptr);
    capture->gpu_source_tensors.resize((size_t) n_layers, nullptr);
    capture->layer_seen_batch_id.assign((size_t) n_layers, 0);
    capture->layer_device_produced_batch_id.assign((size_t) n_layers, 0);
    capture->layer_device_produced_rows.assign((size_t) n_layers, 0);
    llama_dflash_init_gpu_capture(*ctx, *capture);
    capture->prev_cb_eval = ctx->cparams.cb_eval;
    capture->prev_cb_eval_user_data = ctx->cparams.cb_eval_user_data;
    ctx->dflash.capture = std::move(capture);
    ctx->dflash.feature_view_buffer.clear();

    ctx->cparams.cb_eval = llama_dflash_capture_eval_callback;
    ctx->cparams.cb_eval_user_data = ctx;
    if (ctx->sched != nullptr) {
        ggml_backend_sched_set_eval_callback(ctx->sched, ctx->cparams.cb_eval, ctx->cparams.cb_eval_user_data);
    }

    return true;
}

struct ggml_tensor * llama_dflash_capture_graph_dst(
        struct llama_context * ctx,
        struct ggml_context * graph_ctx,
        int32_t layer_id,
        int32_t n_rows) {
    if (ctx == nullptr || graph_ctx == nullptr || !ctx->dflash.capture || n_rows <= 0) {
        return nullptr;
    }

    auto & capture = *ctx->dflash.capture;
    if (!capture.gpu_capture_enabled || n_rows > capture.gpu_layer_capacity) {
        return nullptr;
    }

    const int32_t layer_idx = llama_dflash_find_layer_index(ctx, layer_id);
    if (layer_idx < 0 || layer_idx >= (int32_t) capture.gpu_layer_tensors.size()) {
        return nullptr;
    }

    ggml_tensor * tensor = capture.gpu_layer_tensors[(size_t) layer_idx];
    if (tensor == nullptr) {
        return nullptr;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(tensor->buffer);
    ggml_backend_t producer_backend = nullptr;
    for (ggml_backend_t candidate : ctx->backends) {
        if (candidate != nullptr && ggml_backend_get_default_buffer_type(candidate) == buft) {
            producer_backend = candidate;
            break;
        }
    }
    if (producer_backend == nullptr || ctx->sched == nullptr) {
        return nullptr;
    }

    ggml_backend_sched_set_tensor_backend(ctx->sched, tensor, producer_backend);
    ggml_tensor * view = ggml_view_2d(graph_ctx, tensor, tensor->ne[0], n_rows, tensor->nb[1], 0);
    if (view == nullptr) {
        return nullptr;
    }
    ggml_backend_sched_set_tensor_backend(ctx->sched, view, producer_backend);
    ggml_backend_view_init(view);
    if (capture.gpu_layer_views.size() != capture.layer_ids.size()) {
        capture.gpu_layer_views.assign(capture.layer_ids.size(), nullptr);
    }
    capture.gpu_layer_views[(size_t) layer_idx] = view;
    return view;
}

static ggml_backend_t llama_dflash_select_device_input_backend(const llama_context & ctx) {
    if (ctx.model.dflash_fc != nullptr && ctx.model.dflash_fc->buffer != nullptr) {
        const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(ctx.model.dflash_fc->buffer);
        for (ggml_backend_t backend : ctx.backends) {
            if (backend != nullptr && ggml_backend_get_default_buffer_type(backend) == buft) {
                return backend;
            }
        }
    }

    for (ggml_backend_t backend : ctx.backends) {
        if (backend != nullptr && !ggml_backend_is_cpu(backend)) {
            return backend;
        }
    }
    return nullptr;
}

static ggml_backend_t llama_dflash_backend_for_buffer(
        const llama_context & ctx,
        ggml_backend_buffer_t buffer) {
    if (buffer == nullptr) {
        return nullptr;
    }

    const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buffer);
    for (ggml_backend_t backend : ctx.backends) {
        if (backend != nullptr && ggml_backend_get_default_buffer_type(backend) == buft) {
            return backend;
        }
    }

    // A single non-CPU backend is unambiguous even when a scheduler or
    // allocator wraps its buffer type.
    ggml_backend_t only_non_cpu = nullptr;
    int non_cpu_count = 0;
    for (ggml_backend_t backend : ctx.backends) {
        if (backend != nullptr && !ggml_backend_is_cpu(backend)) {
            only_non_cpu = backend;
            ++non_cpu_count;
        }
    }
    return non_cpu_count == 1 ? only_non_cpu : nullptr;
}

static bool llama_dflash_backends_share_device(ggml_backend_t source, ggml_backend_t destination) {
    if (source == nullptr || destination == nullptr) {
        return false;
    }
    if (source == destination) {
        return true;
    }

    const char * source_name = ggml_backend_name(source);
    const char * destination_name = ggml_backend_name(destination);
    const bool source_cuda = source_name != nullptr && std::strncmp(source_name, "CUDA", 4) == 0 &&
            std::strcmp(source_name, "CUDA_Host") != 0;
    const bool destination_cuda = destination_name != nullptr && std::strncmp(destination_name, "CUDA", 4) == 0 &&
            std::strcmp(destination_name, "CUDA_Host") != 0;
    if (source_cuda && destination_cuda && source_name != nullptr && destination_name != nullptr) {
        return std::strcmp(source_name, destination_name) == 0;
    }

    const ggml_backend_buffer_type_t source_buft = ggml_backend_get_default_buffer_type(source);
    const ggml_backend_buffer_type_t destination_buft = ggml_backend_get_default_buffer_type(destination);
    return source_buft != nullptr && source_buft == destination_buft;
}

enum class llama_dflash_copy_route {
    same_device,
    cross_device,
    unsupported,
};

static bool llama_dflash_is_cuda_device_backend(ggml_backend_t backend) {
    const char * name = backend != nullptr ? ggml_backend_name(backend) : nullptr;
    return name != nullptr && std::strncmp(name, "CUDA", 4) == 0 &&
            std::strcmp(name, "CUDA_Host") != 0;
}

static llama_dflash_copy_route llama_dflash_copy_route_for(
        ggml_backend_t source, ggml_backend_t destination) {
    if (llama_dflash_backends_share_device(source, destination)) {
        return llama_dflash_copy_route::same_device;
    }
    if (llama_dflash_is_cuda_device_backend(source) &&
            llama_dflash_is_cuda_device_backend(destination)) {
        return llama_dflash_copy_route::cross_device;
    }
    return llama_dflash_copy_route::unsupported;
}

static void llama_dflash_note_device_copy(
        llama_context::dflash_capture_state & capture,
        llama_dflash_copy_route route,
        size_t bytes) {
    switch (route) {
        case llama_dflash_copy_route::same_device:
            capture.device_same_device_bytes += bytes;
            break;
        case llama_dflash_copy_route::cross_device:
            capture.device_cross_device_bytes += bytes;
            break;
        case llama_dflash_copy_route::unsupported:
            break;
    }
}

bool llama_dflash_prepare_device_transport(struct llama_context * ctx_tgt, struct llama_context * ctx_dft) {
    if (ctx_tgt == nullptr || ctx_dft == nullptr ||
            !ctx_tgt->dflash.capture || !ctx_tgt->dflash.capture->gpu_capture_enabled ||
            ctx_tgt->dflash.capture->gpu_layer_tensors.empty()) {
        return false;
    }

    auto & kv = ctx_dft->dflash.kv;
    auto clear_partial = [&]() {
        for (ggml_backend_buffer_t buf : kv.device_input_bufs) {
            if (buf != nullptr) {
                ggml_backend_buffer_free(buf);
            }
        }
        kv.device_input_bufs.clear();
        for (ggml_backend_buffer_t buf : kv.device_window_bufs) {
            if (buf != nullptr) {
                ggml_backend_buffer_free(buf);
            }
        }
        kv.device_window_bufs.clear();
        kv.device_window_target_features = nullptr;
        kv.device_window_capacity = 0;
        kv.device_window_valid = false;
        if (kv.device_input_ctx != nullptr) {
            ggml_free(kv.device_input_ctx);
            kv.device_input_ctx = nullptr;
        }
        kv.device_input_target_features = nullptr;
        kv.device_input_backend = nullptr;
    };

    if (kv.device_input_target_features != nullptr && kv.device_window_target_features != nullptr &&
            !kv.device_input_bufs.empty() && !kv.device_window_bufs.empty() &&
            kv.device_input_target_features->buffer != nullptr && kv.device_window_target_features->buffer != nullptr &&
            kv.device_input_backend != nullptr) {
        return true;
    }
    if (kv.device_input_target_features != nullptr || kv.device_input_ctx != nullptr || !kv.device_input_bufs.empty()) {
        clear_partial();
    }
    const int64_t layer_width = (int64_t) ctx_tgt->model.hparams.n_embd;
    const int64_t layer_count = (int64_t) ctx_tgt->dflash.capture->layer_ids.size();
    const int64_t window_capacity = ctx_dft->dflash.visible_cross_ctx;
    if (layer_width <= 0 || layer_count <= 0 || layer_width > std::numeric_limits<int64_t>::max() / layer_count ||
            window_capacity <= 0 || window_capacity > std::numeric_limits<int32_t>::max()) {
        clear_partial();
        return false;
    }
    const int64_t feature_width = layer_width * layer_count;
    const int64_t capture_capacity = ctx_tgt->dflash.capture->gpu_layer_capacity;
    if (capture_capacity <= 0 || capture_capacity > std::numeric_limits<int32_t>::max()) {
        clear_partial();
        return false;
    }
    const int64_t capacity = std::max(capture_capacity, window_capacity);
    ggml_init_params params = {
        /*.mem_size   =*/ 3 * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    kv.device_input_ctx = ggml_init(params);
    if (kv.device_input_ctx == nullptr) {
        return false;
    }

    kv.device_input_target_features = ggml_new_tensor_2d(
            kv.device_input_ctx, GGML_TYPE_F32, feature_width, capacity);
    if (kv.device_input_target_features == nullptr) {
        clear_partial();
        return false;
    }

    kv.device_window_target_features = ggml_new_tensor_2d(
            kv.device_input_ctx, GGML_TYPE_F32, feature_width, window_capacity);
    if (kv.device_window_target_features == nullptr) {
        clear_partial();
        return false;
    }
    kv.device_window_capacity = (int32_t) window_capacity;

    const ggml_backend_t device_backend = llama_dflash_select_device_input_backend(*ctx_dft);
    if (device_backend == nullptr) {
        clear_partial();
        return false;
    }
    const ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(device_backend);
    const size_t bytes = ggml_backend_buft_get_alloc_size(buft, kv.device_input_target_features);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, bytes);
    if (buffer == nullptr) {
        LLAMA_LOG_WARN("DFlash device input allocation failed width=%lld capacity=%lld bytes=%zu\n", (long long) feature_width, (long long) capacity, bytes);
        clear_partial();
        return false;
    }
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_tensor_alloc(buffer, kv.device_input_target_features, ggml_backend_buffer_get_base(buffer));
    ggml_backend_buffer_clear(buffer, 0);
    kv.device_input_bufs.push_back(buffer);

    const size_t window_bytes = ggml_backend_buft_get_alloc_size(buft, kv.device_window_target_features);
    ggml_backend_buffer_t window_buffer = ggml_backend_buft_alloc_buffer(buft, window_bytes);
    if (window_buffer == nullptr) {
        LLAMA_LOG_WARN("DFlash device window allocation failed width=%lld capacity=%lld bytes=%zu\n", (long long) feature_width, (long long) window_capacity, window_bytes);
        clear_partial();
        return false;
    }
    ggml_backend_buffer_set_usage(window_buffer, GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    ggml_backend_tensor_alloc(window_buffer, kv.device_window_target_features, ggml_backend_buffer_get_base(window_buffer));
    ggml_backend_buffer_clear(window_buffer, 0);
    kv.device_window_bufs.push_back(window_buffer);
    kv.device_input_backend = device_backend;
    return true;
}

bool llama_dflash_copy_device_input_from_window(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft,
        const std::vector<int32_t> & source_row_indices) {
    if (ctx_dft == nullptr || source_row_indices.empty() ||
            ctx_dft->dflash.kv.device_window_target_features == nullptr ||
            ctx_dft->dflash.kv.device_input_target_features == nullptr ||
            ctx_dft->dflash.kv.device_input_backend == nullptr) {
        return false;
    }

    ggml_tensor * source = ctx_dft->dflash.kv.device_window_target_features;
    ggml_tensor * destination = ctx_dft->dflash.kv.device_input_target_features;
    const int32_t n_rows = (int32_t) source_row_indices.size();
    const int64_t row_width = source->ne[0];
    if (row_width <= 0 || destination->ne[0] != row_width ||
            n_rows > destination->ne[1]) {
        return false;
    }
    for (int32_t row : source_row_indices) {
        if (row < 0 || row >= ctx_dft->dflash.kv.device_window_capacity) {
            return false;
        }
    }

    ggml_init_params view_params = {
        /*.mem_size   =*/ (size_t) (2 * n_rows + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * view_ctx = ggml_init(view_params);
    if (view_ctx == nullptr) {
        return false;
    }

    bool copied = true;
    for (int32_t row_idx = 0; row_idx < n_rows && copied; ++row_idx) {
        ggml_tensor * src_view = ggml_view_2d(
                view_ctx, source, row_width, 1, source->nb[1],
                (size_t) source_row_indices[(size_t) row_idx] * source->nb[1]);
        ggml_tensor * dst_view = ggml_view_2d(
                view_ctx, destination, row_width, 1, destination->nb[1],
                (size_t) row_idx * destination->nb[1]);
        if (src_view == nullptr || dst_view == nullptr) {
            copied = false;
            break;
        }
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            dst_view->nb[i] = src_view->nb[i];
        }
        ggml_backend_tensor_copy_async(
                ctx_dft->dflash.kv.device_input_backend,
                ctx_dft->dflash.kv.device_input_backend,
                src_view, dst_view);
    }

    if (copied) {
        ggml_backend_synchronize(ctx_dft->dflash.kv.device_input_backend);
    }
    ggml_free(view_ctx);
    if (!copied) {
        return false;
    }

    ctx_dft->dflash.kv.device_input_ready = true;
    ctx_dft->dflash.kv.device_input_rows = n_rows;
    ctx_dft->dflash.kv.device_input_row_offset = 0;
    GGML_UNUSED(ctx_tgt);
    return true;
}

bool llama_dflash_rebuild_device_input_from_window(
        struct llama_context * ctx_dft,
        int32_t n_rows,
        int32_t ring_write_pos) {
    if (ctx_dft == nullptr || ctx_dft->dflash.kv.device_window_capacity <= 0 ||
            n_rows <= 0 || n_rows > ctx_dft->dflash.kv.device_window_capacity ||
            ctx_dft->dflash.kv.device_input_target_features == nullptr ||
            n_rows > ctx_dft->dflash.kv.device_input_target_features->ne[1]) {
        return false;
    }
    const int32_t capacity = ctx_dft->dflash.kv.device_window_capacity;
    const int32_t normalized_write_pos = ((ring_write_pos % capacity) + capacity) % capacity;
    const int32_t read_start = (normalized_write_pos - n_rows + capacity) % capacity;
    std::vector<int32_t> source_rows((size_t) n_rows);
    for (int32_t i = 0; i < n_rows; ++i) {
        source_rows[(size_t) i] = (read_start + i) % capacity;
    }
    return llama_dflash_copy_device_input_from_window(nullptr, ctx_dft, source_rows);
}

bool llama_dflash_copy_device_window_rows(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft,
        const std::vector<int32_t> & source_row_indices,
        const std::vector<int32_t> & destination_row_indices) {
    if (ctx_tgt == nullptr || ctx_dft == nullptr || !ctx_tgt->dflash.capture ||
            source_row_indices.empty() || source_row_indices.size() != destination_row_indices.size() ||
            ctx_dft->dflash.kv.device_window_target_features == nullptr ||
            ctx_dft->dflash.kv.device_input_backend == nullptr) {
        return false;
    }

    auto & capture = *ctx_tgt->dflash.capture;
    const int32_t n_layers = (int32_t) capture.layer_ids.size();
    if (!capture.gpu_capture_enabled || n_layers <= 0 ||
            capture.gpu_layer_tensors.size() != (size_t) n_layers ||
            capture.layer_device_produced_batch_id.size() != (size_t) n_layers ||
            capture.layer_device_produced_rows.size() != (size_t) n_layers) {
        return false;
    }

    const int32_t produced_rows = capture.layer_device_produced_rows.front();
    const int32_t layer_width = (int32_t) ctx_tgt->model.hparams.n_embd;
    ggml_tensor * destination = ctx_dft->dflash.kv.device_window_target_features;
    if (capture.capture_batch_id == 0 || produced_rows <= 0 || layer_width <= 0 ||
            destination->ne[0] < (int64_t) layer_width * n_layers ||
            destination->ne[1] < ctx_dft->dflash.kv.device_window_capacity) {
        return false;
    }

    for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        if (capture.layer_device_produced_batch_id[(size_t) layer_idx] != capture.capture_batch_id ||
                capture.layer_device_produced_rows[(size_t) layer_idx] != produced_rows ||
                capture.gpu_layer_tensors[(size_t) layer_idx] == nullptr ||
                capture.gpu_layer_tensors[(size_t) layer_idx]->buffer == nullptr) {
            return false;
        }
    }

    for (size_t i = 0; i < source_row_indices.size(); ++i) {
        if (source_row_indices[i] < 0 || source_row_indices[i] >= produced_rows ||
                destination_row_indices[i] < 0 ||
                destination_row_indices[i] >= ctx_dft->dflash.kv.device_window_capacity) {
            return false;
        }
    }

    ggml_init_params view_params = {
        /*.mem_size   =*/ (size_t) (2 * n_layers * source_row_indices.size() + 1) * ggml_tensor_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * view_ctx = ggml_init(view_params);
    if (view_ctx == nullptr) {
        return false;
    }

    std::vector<ggml_backend_t> source_backends;
    std::vector<llama_dflash_copy_route> copy_routes;
    source_backends.reserve((size_t) n_layers);
    copy_routes.reserve((size_t) n_layers);
    for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        ggml_tensor * source = capture.gpu_layer_tensors[(size_t) layer_idx];
        ggml_backend_t source_backend = capture.gpu_layer_backends.size() == capture.layer_ids.size()
                ? capture.gpu_layer_backends[(size_t) layer_idx] : nullptr;
        if (source_backend == nullptr) {
            source_backend = llama_dflash_backend_for_buffer(*ctx_tgt, source->buffer);
        }
        if (source_backend == nullptr) {
            source_backend = capture.layer_backends[(size_t) layer_idx];
        }
        const llama_dflash_copy_route route = llama_dflash_copy_route_for(
                source_backend, ctx_dft->dflash.kv.device_input_backend);
        if (route == llama_dflash_copy_route::unsupported) {
            capture.device_last_fallback_reason = source_backend == nullptr
                    ? "unknown-source-backend"
                    : "mixed-transport-disabled-or-unsupported-backend";
            LLAMA_LOG_WARN("DFlash device window rejected source=%s destination=%s reason=%s\n",
                    source_backend != nullptr ? ggml_backend_name(source_backend) : "<unknown>",
                    ggml_backend_name(ctx_dft->dflash.kv.device_input_backend),
                    capture.device_last_fallback_reason.c_str());
            ggml_free(view_ctx);
            return false;
        }
        source_backends.push_back(source_backend);
        copy_routes.push_back(route);
    }

    bool copied = true;
    for (int32_t layer_idx = 0; layer_idx < n_layers && copied; ++layer_idx) {
        ggml_tensor * source = capture.gpu_layer_tensors[(size_t) layer_idx];
        ggml_backend_t source_backend = source_backends[(size_t) layer_idx];
        for (size_t i = 0; i < source_row_indices.size(); ++i) {
            ggml_tensor * src_view = ggml_view_2d(
                    view_ctx, source, layer_width, 1, source->nb[1],
                    (size_t) source_row_indices[i] * source->nb[1]);
            ggml_tensor * dst_view = ggml_view_2d(
                    view_ctx, destination, layer_width, 1, destination->nb[1],
                    (size_t) layer_idx * (size_t) layer_width * sizeof(float) +
                    (size_t) destination_row_indices[i] * destination->nb[1]);
            if (src_view == nullptr || dst_view == nullptr) {
                copied = false;
                break;
            }
            ggml_backend_view_init(src_view);
            ggml_backend_view_init(dst_view);
            for (int j = 1; j < GGML_MAX_DIMS; ++j) {
                dst_view->nb[j] = src_view->nb[j];
            }
            ggml_backend_tensor_copy_async(source_backend, ctx_dft->dflash.kv.device_input_backend, src_view, dst_view);
            llama_dflash_note_device_copy(capture, copy_routes[(size_t) layer_idx],
                    (size_t) ctx_tgt->model.hparams.n_embd * sizeof(float));
        }
    }

    if (copied) {
        const int64_t sync_start_us = ggml_time_us();
        for (ggml_backend_t backend : source_backends) {
            if (backend != nullptr) {
                ggml_backend_synchronize(backend);
            }
        }
        ggml_backend_synchronize(ctx_dft->dflash.kv.device_input_backend);
        if (capture.telemetry_enabled) {
            capture.device_sync_count++;
            capture.device_sync_us += (uint64_t) std::max<int64_t>(0, ggml_time_us() - sync_start_us);
        }
    }
    ggml_free(view_ctx);
    if (!copied) {
        return false;
    }

    ctx_dft->dflash.kv.device_window_valid = true;
    return true;
}

bool llama_dflash_upload_device_window(
        struct llama_context * ctx_dft,
        const float * rows,
        int32_t n_rows) {
    if (ctx_dft == nullptr || rows == nullptr ||
            ctx_dft->dflash.kv.device_window_target_features == nullptr ||
            ctx_dft->dflash.kv.device_input_backend == nullptr ||
            n_rows <= 0 || n_rows > ctx_dft->dflash.kv.device_window_capacity) {
        return false;
    }
    const size_t bytes = (size_t) n_rows * (size_t) ctx_dft->dflash.kv.device_window_target_features->ne[0] * sizeof(float);
    ggml_backend_tensor_set_async(ctx_dft->dflash.kv.device_input_backend,
            ctx_dft->dflash.kv.device_window_target_features, rows, 0, bytes);
    ggml_backend_synchronize(ctx_dft->dflash.kv.device_input_backend);
    ctx_dft->dflash.kv.device_window_valid = true;
    return true;
}

bool llama_dflash_read_device_window(
        struct llama_context * ctx_dft,
        float * rows,
        int32_t n_rows) {
    if (ctx_dft == nullptr || rows == nullptr ||
            ctx_dft->dflash.kv.device_window_target_features == nullptr ||
            n_rows <= 0 || n_rows > ctx_dft->dflash.kv.device_window_capacity) {
        return false;
    }
    const size_t bytes = (size_t) n_rows * (size_t) ctx_dft->dflash.kv.device_window_target_features->ne[0] * sizeof(float);
    ggml_backend_tensor_get(ctx_dft->dflash.kv.device_window_target_features, rows, 0, bytes);
    return true;
}
bool llama_dflash_device_window_is_valid(const struct llama_context * ctx) {
    return ctx != nullptr && ctx->dflash.kv.device_window_valid;
}

bool llama_dflash_device_feature_path_enabled(const struct llama_context * ctx) {
    return ctx != nullptr && ctx->dflash.capture && ctx->dflash.capture->gpu_features_enabled;
}

void llama_dflash_disable_gpu_features(struct llama_context * ctx) {
    if (ctx != nullptr && ctx->dflash.capture) {
        ctx->dflash.capture->gpu_features_enabled = false;
    }
}

void llama_dflash_note_device_host_fallback(
        struct llama_context * ctx, const char * reason) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return;
    }
    auto & capture = *ctx->dflash.capture;
    capture.device_host_fallback_count++;
    capture.device_last_fallback_reason = reason != nullptr ? reason : "host-fallback";
}

void llama_dflash_note_device_append(struct llama_context * ctx) {
    if (ctx != nullptr && ctx->dflash.capture && ctx->dflash.capture->telemetry_enabled) {
        ctx->dflash.capture->device_append_count++;
    }
}

void llama_dflash_log_telemetry_summary(
        const struct llama_context * ctx,
        int32_t ring_filled,
        int32_t ring_write_pos,
        int32_t window_rows,
        bool device_only) {
    if (ctx == nullptr || !ctx->dflash.capture || !ctx->dflash.capture->telemetry_enabled) {
        return;
    }

    const auto & capture = *ctx->dflash.capture;
    LLAMA_LOG_INFO(
            "DFlash telemetry gpu_features=%d same_device_bytes=%llu cross_device_bytes=%llu "
            "feature_d2h_bytes=%llu host_fallbacks=%llu device_appends=%llu "
            "sync_count=%llu sync_us=%llu capture_layers=%zu ring_filled=%d ring_write=%d "
            "window_rows=%d device_only=%d fallback_reason=%s\n",
            capture.gpu_features_enabled ? 1 : 0,
            (unsigned long long) capture.device_same_device_bytes,
            (unsigned long long) capture.device_cross_device_bytes,
            (unsigned long long) capture.capture_d2h_bytes,
            (unsigned long long) capture.device_host_fallback_count,
            (unsigned long long) capture.device_append_count,
            (unsigned long long) capture.device_sync_count,
            (unsigned long long) capture.device_sync_us,
            capture.layer_ids.size(),
            ring_filled,
            ring_write_pos,
            window_rows,
            device_only ? 1 : 0,
            capture.device_last_fallback_reason.empty() ? "none" : capture.device_last_fallback_reason.c_str());
}

void llama_dflash_clear_device_append(struct llama_context * ctx) {
    if (ctx != nullptr) {
        ctx->dflash.kv.device_input_ready = false;
        ctx->dflash.kv.device_input_rows = 0;
        ctx->dflash.kv.device_input_row_offset = 0;
    }
}

void llama_clear_dflash_capture(struct llama_context * ctx) {
    if (ctx == nullptr) {
        return;
    }

    ggml_backend_sched_eval_callback prev_cb_eval = nullptr;
    void * prev_cb_eval_user_data = nullptr;
    if (ctx->dflash.capture) {
        prev_cb_eval = ctx->dflash.capture->prev_cb_eval;
        prev_cb_eval_user_data = ctx->dflash.capture->prev_cb_eval_user_data;
    }

    ctx->dflash.capture.reset();
    ctx->dflash.feature_view_buffer.clear();

    if (ctx->cparams.cb_eval == llama_dflash_capture_eval_callback && ctx->cparams.cb_eval_user_data == ctx) {
        ctx->cparams.cb_eval = prev_cb_eval;
        ctx->cparams.cb_eval_user_data = prev_cb_eval_user_data;
        if (ctx->sched != nullptr) {
            ggml_backend_sched_set_eval_callback(ctx->sched, prev_cb_eval, prev_cb_eval_user_data);
        }
    }
}

void llama_begin_dflash_capture_batch(struct llama_context * ctx) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return;
    }

    auto & capture = *ctx->dflash.capture;
    capture.capture_batch_id++;
    capture.row_count = 0;
    capture.row_width = 0;
    std::fill(capture.layer_seen_batch_id.begin(), capture.layer_seen_batch_id.end(), 0);
    std::fill(capture.layer_device_produced_batch_id.begin(), capture.layer_device_produced_batch_id.end(), 0);
    std::fill(capture.layer_device_produced_rows.begin(), capture.layer_device_produced_rows.end(), 0);
    std::fill(capture.gpu_source_tensors.begin(), capture.gpu_source_tensors.end(), nullptr);
}

void llama_finish_dflash_capture_batch(
        struct llama_context * ctx,
        bool is_prompt_warmup) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return;
    }

    GGML_UNUSED(is_prompt_warmup);
    auto & capture = *ctx->dflash.capture;
    // Reset the batch-local reference shape so the next decode only compares layers within
    // the same batch, not against the previous prompt/verify batch.
    capture.row_count = 0;
    capture.row_width = 0;
}

static bool llama_spec_prepare_dflash_capture(
        struct llama_context * ctx,
        int32_t & row_count,
        int32_t & row_width,
        int32_t & n_layers) {
    if (ctx == nullptr || !ctx->dflash.capture) {
        return false;
    }

    auto & capture = *ctx->dflash.capture;
    const int64_t wait_start_us = ggml_time_us();
    llama_synchronize(ctx);
    if (capture.telemetry_enabled) {
        capture.device_sync_count++;
        capture.device_sync_us += (uint64_t) std::max<int64_t>(0, ggml_time_us() - wait_start_us);
    }
    row_count = capture.row_count;
    row_width = capture.row_width;
    n_layers = (int32_t) capture.layer_ids.size();
    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || capture.layer_seen_batch_id.size() != (size_t) n_layers ||
            capture.layer_device_produced_batch_id.size() != (size_t) n_layers || capture.layer_device_produced_rows.size() != (size_t) n_layers) {
        return false;
    }

    if (capture.capture_batch_id == 0) {
        LLAMA_LOG_WARN("%s: DFlash capture batch markers are not initialized (batch_id=0)\n", __func__);
        return false;
    }

    for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        const auto & rows = capture.layer_rows[(size_t) layer_idx];
        const bool cpu_ready = capture.layer_seen_batch_id[(size_t) layer_idx] == capture.capture_batch_id &&
                rows.size() == (size_t) row_count * (size_t) row_width;
        const bool gpu_ready = capture.layer_device_produced_batch_id[(size_t) layer_idx] == capture.capture_batch_id &&
                capture.layer_device_produced_rows[(size_t) layer_idx] == row_count &&
                capture.gpu_layer_tensors.size() == (size_t) n_layers &&
                capture.gpu_layer_tensors[(size_t) layer_idx] != nullptr &&
                capture.gpu_layer_tensors[(size_t) layer_idx]->buffer != nullptr;
        if (!cpu_ready && !gpu_ready) {
            LLAMA_LOG_WARN("%s: DFlash capture rows unavailable for layer %d (rows=%zu expected=%zu cpu=%d gpu=%d)\n",
                    __func__, capture.layer_ids[(size_t) layer_idx], rows.size(),
                    (size_t) row_count * (size_t) row_width, cpu_ready ? 1 : 0, gpu_ready ? 1 : 0);
            return false;
        }
    }

    return true;
}

        static bool llama_spec_materialize_dflash_rows_prepared(
            struct llama_context * ctx,
            int32_t row_count,
            int32_t row_width,
            int32_t n_layers,
            const std::vector<int32_t> & row_indices,
            std::vector<float> & rows_out,
            int32_t & combined_width);

static bool llama_spec_materialize_dflash_rows(
        struct llama_context * ctx,
        const std::vector<int32_t> & row_indices,
        std::vector<float> & rows_out,
        int32_t & combined_width) {
    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    return llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, rows_out, combined_width);
}

static bool llama_spec_materialize_dflash_rows_prepared(
        struct llama_context * ctx,
        int32_t row_count,
        int32_t row_width,
        int32_t n_layers,
        const std::vector<int32_t> & row_indices,
        std::vector<float> & rows_out,
        int32_t & combined_width) {
    rows_out.clear();
    combined_width = 0;
    if (ctx == nullptr || row_indices.empty()) {
        return false;
    }

    if (row_count <= 0 || row_width <= 0 || n_layers <= 0 || ctx->dflash.capture == nullptr) {
        return false;
    }

    combined_width = row_width * n_layers;
    rows_out.resize((size_t) row_indices.size() * (size_t) combined_width);

    auto & capture = *ctx->dflash.capture;
    const auto & layer_rows = capture.layer_rows;
    for (size_t out_row = 0; out_row < row_indices.size(); ++out_row) {
        int32_t row_index = row_indices[out_row];
        if (row_index < 0) {
            row_index += row_count;
        }
        if (row_index < 0 || row_index >= row_count) {
            rows_out.clear();
            combined_width = 0;
            return false;
        }

        float * dst = rows_out.data() + out_row * (size_t) combined_width;
        for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
            const auto & layer = layer_rows[(size_t) layer_idx];
            if (layer.size() == (size_t) row_count * (size_t) row_width) {
                const float * src = layer.data() + (size_t) row_index * (size_t) row_width;
                std::memcpy(dst + (size_t) layer_idx * (size_t) row_width, src, (size_t) row_width * sizeof(float));
            } else {
                const ggml_tensor * source = capture.gpu_layer_tensors[(size_t) layer_idx];
                if (source == nullptr || source->buffer == nullptr) {
                    rows_out.clear();
                    combined_width = 0;
                    return false;
                }
                ggml_backend_tensor_get(source,
                        dst + (size_t) layer_idx * (size_t) row_width,
                        (size_t) row_index * source->nb[1], (size_t) row_width * sizeof(float));
                if (capture.telemetry_enabled) {
                    capture.capture_d2h_bytes += (uint64_t) row_width * sizeof(float);
                }
            }
        }
    }

    return true;
}


static bool llama_spec_prepare_dflash_device_capture(
        struct llama_context * ctx,
        int32_t & row_count,
        int32_t & row_width,
        int32_t & n_layers) {
    if (ctx == nullptr || !ctx->dflash.capture || !ctx->dflash.capture->gpu_features_enabled) {
        return false;
    }

    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    auto & capture = *ctx->dflash.capture;
    for (int32_t layer_idx = 0; layer_idx < n_layers; ++layer_idx) {
        if (capture.layer_device_produced_batch_id[(size_t) layer_idx] != capture.capture_batch_id ||
                capture.layer_device_produced_rows[(size_t) layer_idx] != row_count ||
                capture.gpu_layer_tensors[(size_t) layer_idx] == nullptr ||
                capture.gpu_layer_tensors[(size_t) layer_idx]->buffer == nullptr) {
            return false;
        }
    }
    return true;
}

static bool llama_spec_fill_dflash_device_view(
        struct llama_context * ctx,
        const llama_batch & batch,
        llama_seq_id seq_id,
        bool filter_seq,
        llama_spec_feature_view & view) {
    if (ctx == nullptr || batch.n_tokens <= 0 || batch.pos == nullptr ||
            batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_device_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    const int32_t batch_row_offset = std::max<int32_t>(0, batch.n_tokens - row_count);
    view = {};
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    view.width = row_width * n_layers;
    view.rows.reserve((size_t) batch.n_tokens);
    for (int32_t batch_index = batch_row_offset; batch_index < batch.n_tokens; ++batch_index) {
        if (batch.n_seq_id[batch_index] <= 0 || batch.seq_id[batch_index] == nullptr) {
            view.rows.clear();
            return false;
        }
        const llama_seq_id row_seq_id = batch.seq_id[batch_index][0];
        if (filter_seq) {
            bool has_seq = false;
            for (int32_t j = 0; j < batch.n_seq_id[batch_index]; ++j) {
                if (batch.seq_id[batch_index][j] == seq_id) {
                    has_seq = true;
                    break;
                }
            }
            if (!has_seq) {
                continue;
            }
        }
        view.rows.push_back({ row_seq_id, batch.pos[batch_index], nullptr });
    }
    return !view.rows.empty();
}

bool llama_spec_get_dflash_device_feature_view(
        struct llama_context * ctx,
        const llama_batch & batch,
        llama_spec_feature_view & view) {
    return llama_spec_fill_dflash_device_view(ctx, batch, 0, false, view);
}

bool llama_spec_get_dflash_device_feature_view_for_seq(
        struct llama_context * ctx,
        const llama_batch & batch,
        llama_seq_id seq_id,
        llama_spec_feature_view & view) {
    return llama_spec_fill_dflash_device_view(ctx, batch, seq_id, true, view);
}
bool llama_spec_get_dflash_feature_view(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_spec_feature_view & view) {
    if (ctx == nullptr || batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    const int32_t batch_row_offset = std::max<int32_t>(0, batch.n_tokens - row_count);
    std::vector<int32_t> row_indices;
    std::vector<int32_t> batch_indices;
    row_indices.reserve((size_t) (batch.n_tokens - batch_row_offset));
    batch_indices.reserve((size_t) (batch.n_tokens - batch_row_offset));
    for (int32_t i = batch_row_offset; i < batch.n_tokens; ++i) {
        row_indices.push_back(i - batch_row_offset);
        batch_indices.push_back(i);
    }

    if (row_indices.empty()) {
        return false;
    }

    view = {};
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    if (!llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, ctx->dflash.feature_view_buffer, view.width)) {
        return false;
    }

    view.rows.reserve(batch_indices.size());
    for (int32_t batch_index : batch_indices) {
        if (batch.n_seq_id[batch_index] <= 0 || batch.seq_id[batch_index] == nullptr) {
            view.rows.clear();
            return false;
        }

        view.rows.push_back({
            /* .seq_id = */ batch.seq_id[batch_index][0],
            /* .pos    = */ batch.pos[batch_index],
            /* .data   = */ ctx->dflash.feature_view_buffer.data() + view.rows.size() * (size_t) view.width,
        });
    }

    return true;
}

bool llama_spec_get_dflash_feature_view_for_seq(
        struct llama_context   * ctx,
        const llama_batch      & batch,
        llama_seq_id             seq_id,
        llama_spec_feature_view & view) {
    if (ctx == nullptr || batch.n_tokens <= 0 || batch.pos == nullptr || batch.n_seq_id == nullptr || batch.seq_id == nullptr) {
        return false;
    }

    int32_t row_count = 0;
    int32_t row_width = 0;
    int32_t n_layers = 0;
    if (!llama_spec_prepare_dflash_capture(ctx, row_count, row_width, n_layers)) {
        return false;
    }

    const int32_t batch_row_offset = std::max<int32_t>(0, batch.n_tokens - row_count);
    std::vector<int32_t> row_indices;
    row_indices.reserve((size_t) batch.n_tokens);
    std::vector<int32_t> batch_indices;
    batch_indices.reserve((size_t) batch.n_tokens);
    for (int32_t i = batch_row_offset; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] <= 0 || batch.seq_id[i] == nullptr) {
            return false;
        }

        for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
            if (batch.seq_id[i][j] == seq_id) {
                row_indices.push_back(i - batch_row_offset);
                batch_indices.push_back(i);
                break;
            }
        }
    }

    if (row_indices.empty()) {
        return false;
    }

    view = {};
    view.kind = LLAMA_SPEC_FEATURE_HIDDEN_STATE;
    if (!llama_spec_materialize_dflash_rows_prepared(ctx, row_count, row_width, n_layers, row_indices, ctx->dflash.feature_view_buffer, view.width)) {
        return false;
    }

    view.rows.reserve(row_indices.size());
    for (size_t i = 0; i < batch_indices.size(); ++i) {
        const int32_t batch_index = batch_indices[i];
        view.rows.push_back({
            /* .seq_id = */ seq_id,
            /* .pos    = */ batch.pos[batch_index],
            /* .data   = */ ctx->dflash.feature_view_buffer.data() + i * (size_t) view.width,
        });
    }

    return true;
}

bool llama_spec_copy_dflash_rows_from_output_indices(
        struct llama_context * ctx,
        const std::vector<int32_t> & output_indices,
        std::vector<float> & hidden_rows) {
    int32_t combined_width = 0;
    if (!llama_spec_materialize_dflash_rows(ctx, output_indices, hidden_rows, combined_width)) {
        hidden_rows.clear();
        return false;
    }

    return hidden_rows.size() == (size_t) output_indices.size() * (size_t) combined_width;
}
