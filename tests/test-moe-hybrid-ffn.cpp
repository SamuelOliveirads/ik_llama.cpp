#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-cuda-moe-ffn.h"
#include "ggml-quants.h"
#include "ggml-moe-ffn.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {
constexpr int K = 256;
constexpr int EXPERTS = 14;
constexpr int SLOTS = 12;
constexpr int SELECTED = 10;

struct metrics {
    double max_abs = 0.0;
    double rms = 0.0;
    size_t worst = 0;
};

metrics compare(const std::vector<float> & a, const std::vector<float> & b) {
    metrics result;
    double sum_sq = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double error = std::abs(double(a[i]) - b[i]);
        if (error > result.max_abs) {
            result.max_abs = error;
            result.worst = i;
        }
        sum_sq += error * error;
    }
    result.rms = std::sqrt(sum_sq / std::max<size_t>(1, a.size()));
    return result;
}

bool finite(const std::vector<float> & values) {
    return std::all_of(values.begin(), values.end(), [](float x) { return std::isfinite(x); });
}

bool byte_ranges_overlap(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b);
    return an && bn && ab <= UINTPTR_MAX - an && bb <= UINTPTR_MAX - bn && ab < bb + bn && bb < ab + an;
}

struct fake_cache_owner {
    enum class slot_state : uint8_t { FREE, COPYING, READY };
    int32_t device = -1;
    int32_t group = 0;
    int32_t layer = 0;
    uint64_t owner_generation = 0;
    std::vector<int32_t> slot_of;
    std::vector<int32_t> key_of;
    std::vector<uint64_t> generation;
    std::vector<uint32_t> pins;
    std::vector<slot_state> state;
    std::vector<int32_t> lru_oldest_first;
    uint64_t resolves = 0;
    uint64_t releases = 0;
    bool reject_release = false;
    bool corrupt_resolution = false;
    const ggml_tensor * fill_hosts[GGML_CUDA_MOE_FFN_MAX_FILL_PARTS]{};
    ggml_tensor * fill_banks[GGML_CUDA_MOE_FFN_MAX_FILL_PARTS]{};
    int32_t fill_experts = 0;
    int32_t fill_slots = 0;
    int32_t corrupt_fill = 0;
    bool allow_fill = false;
    uint64_t fill_reservations = 0;
    uint64_t fill_rollbacks = 0;
    bool fill_pending = false;
    ggml_cuda_moe_ffn_fill_job reserved_fill{};
};

void fake_touch(fake_cache_owner & owner, int32_t slot) {
    owner.lru_oldest_first.erase(std::remove(owner.lru_oldest_first.begin(), owner.lru_oldest_first.end(), slot),
            owner.lru_oldest_first.end());
    owner.lru_oldest_first.push_back(slot);
}

bool fake_resolve_and_pin(void * context, int32_t device, int32_t group, int32_t layer,
        uint64_t owner_generation, const int32_t * ids, int32_t n_ids, ggml_cuda_moe_ffn_resolution * result) {
    auto * owner = static_cast<fake_cache_owner *>(context);
    if (!owner || !ids || !result || device != owner->device || group != owner->group ||
            layer != owner->layer || owner_generation != owner->owner_generation ||
            n_ids <= 0 || n_ids > GGML_MOE_FFN_MAX_SELECTED) return false;
    ggml_cuda_moe_ffn_resolution resolved{};
    resolved.n_draws = n_ids;
    for (int32_t i = 0; i < n_ids; ++i) {
        if (ids[i] < 0 || (size_t) ids[i] >= owner->slot_of.size()) return false;
        const int32_t slot = owner->slot_of[ids[i]];
        resolved.draw_slots[i] = slot;
        if (slot < 0) continue;
        if ((size_t) slot >= owner->key_of.size() || owner->key_of[slot] != ids[i] ||
                owner->state[slot] != fake_cache_owner::slot_state::READY || owner->pins[slot]) return false;
        bool duplicate = false;
        for (int32_t j = 0; j < resolved.n_pins; ++j) duplicate |= resolved.pins[j].slot == slot;
        if (!duplicate) {
            auto & pin = resolved.pins[resolved.n_pins++];
            pin.owner = owner;
            pin.device = device;
            pin.group = group;
            pin.layer = layer;
            pin.slot = slot;
            pin.key = ids[i];
            pin.owner_generation = owner_generation;
        }
    }
    for (int32_t i = 0; i < resolved.n_pins; ++i) {
        const int32_t slot = resolved.pins[i].slot;
        auto & generation = owner->generation[slot];
        if (++generation == 0) ++generation;
        ++owner->pins[slot];
        resolved.pins[i].generation = generation;
        fake_touch(*owner, slot);
    }
    *result = resolved;
    if (owner->corrupt_resolution) result->draw_slots[0] = INT32_MAX;
    ++owner->resolves;
    return true;
}

bool fake_release_pins(void * context, const ggml_cuda_moe_ffn_pin_token * pins, int32_t n_pins) {
    auto * owner = static_cast<fake_cache_owner *>(context);
    if (!owner || owner->reject_release || !pins || n_pins <= 0 || n_pins > GGML_MOE_FFN_MAX_SELECTED) return false;
    for (int32_t i = 0; i < n_pins; ++i) {
        const auto & pin = pins[i];
        if (pin.owner != owner || pin.device != owner->device || pin.group != owner->group ||
                pin.layer != owner->layer || pin.owner_generation != owner->owner_generation ||
                pin.slot < 0 || (size_t) pin.slot >= owner->key_of.size() ||
                owner->key_of[pin.slot] != pin.key || owner->generation[pin.slot] != pin.generation ||
                owner->pins[pin.slot] != 1) return false;
        for (int32_t j = 0; j < i; ++j) if (pins[j].slot == pin.slot) return false;
    }
    for (int32_t i = 0; i < n_pins; ++i) --owner->pins[pins[i].slot];
    ++owner->releases;
    return true;
}

bool fake_reserve_fill(void * context, int32_t device, int32_t group, int32_t layer,
        uint64_t owner_generation, int32_t expert, ggml_cuda_moe_ffn_fill_job * job) {
    auto * owner = static_cast<fake_cache_owner *>(context);
    if (!owner || !job || (owner->corrupt_fill == 0 && !owner->allow_fill) || owner->fill_pending ||
            device != owner->device || group != owner->group || layer != owner->layer ||
            owner_generation != owner->owner_generation || expert < 0 || expert >= owner->fill_experts ||
            owner->slot_of[expert] >= 0 ||
            owner->fill_slots <= 0) return false;
    int32_t free_slot = -1;
    for (int32_t slot = 0; slot < owner->fill_slots; ++slot) {
        if (owner->state[slot] == fake_cache_owner::slot_state::FREE &&
                owner->key_of[slot] == -1 && owner->pins[slot] == 0) {
            free_slot = slot;
            break;
        }
    }
    int32_t old_key = -1;
    if (free_slot < 0) for (const int32_t slot : owner->lru_oldest_first) {
        if (owner->state[slot] == fake_cache_owner::slot_state::READY &&
                owner->key_of[slot] >= 0 && owner->pins[slot] == 0) {
            free_slot = slot;
            old_key = owner->key_of[slot];
            break;
        }
    }
    if (free_slot < 0) return false;
    ggml_cuda_moe_ffn_fill_job reserved{};
    reserved.owner = owner;
    reserved.device = device;
    reserved.group = group;
    reserved.layer = layer;
    reserved.slot = free_slot;
    reserved.key = layer * owner->fill_experts + expert;
    reserved.owner_generation = owner_generation;
    reserved.generation = ++owner->generation[reserved.slot];
    if (reserved.generation == 0) reserved.generation = ++owner->generation[reserved.slot];
    reserved.n_parts = GGML_CUDA_MOE_FFN_MAX_FILL_PARTS;
    for (int32_t i = 0; i < reserved.n_parts; ++i) {
        const ggml_tensor * host = owner->fill_hosts[i];
        ggml_tensor * bank = owner->fill_banks[i];
        if (!host || !host->data || !bank || !bank->data || !host->nb[2] || host->nb[2] != bank->nb[2] ||
                (size_t) expert > SIZE_MAX / host->nb[2] || (size_t) reserved.slot > SIZE_MAX / bank->nb[2]) return false;
        const size_t bytes = host->nb[2];
        reserved.parts[i] = {
            (const uint8_t *) host->data + (size_t) expert * host->nb[2],
            (uint8_t *) bank->data + (size_t) reserved.slot * bank->nb[2],
            bytes,
        };
        reserved.total_bytes += bytes;
    }
    switch (owner->corrupt_fill) {
        case 0: break;
        case 1: ++reserved.parts[0].bytes; break;
        case 2: reserved.parts[0].source = (const uint8_t *) reserved.parts[0].source + 1; break;
        case 3: reserved.parts[0].destination = (uint8_t *) reserved.parts[0].destination + 1; break;
    }
    if (old_key >= 0) {
        owner->slot_of[old_key] = -1;
        owner->key_of[free_slot] = reserved.key;
        owner->state[free_slot] = fake_cache_owner::slot_state::COPYING;
        owner->lru_oldest_first.erase(std::remove(owner->lru_oldest_first.begin(),
                owner->lru_oldest_first.end(), free_slot), owner->lru_oldest_first.end());
    } else {
        owner->key_of[free_slot] = reserved.key;
        owner->state[free_slot] = fake_cache_owner::slot_state::COPYING;
        owner->lru_oldest_first.erase(std::remove(owner->lru_oldest_first.begin(),
                owner->lru_oldest_first.end(), free_slot), owner->lru_oldest_first.end());
    }
    owner->reserved_fill = reserved;
    owner->fill_pending = true;
    ++owner->fill_reservations;
    *job = reserved;
    return true;
}

bool fake_finish_fill(void * context, const ggml_cuda_moe_ffn_fill_job * job, bool copied) {
    auto * owner = static_cast<fake_cache_owner *>(context);
    if (!owner || !job || !owner->fill_pending || job->owner != owner ||
            job->device != owner->reserved_fill.device || job->group != owner->reserved_fill.group ||
            job->layer != owner->reserved_fill.layer || job->slot != owner->reserved_fill.slot ||
            job->key != owner->reserved_fill.key || job->generation != owner->reserved_fill.generation ||
            job->owner_generation != owner->reserved_fill.owner_generation) return false;
    owner->fill_pending = false;
    if (copied) {
        owner->slot_of[job->key % owner->fill_experts] = job->slot;
        owner->state[job->slot] = fake_cache_owner::slot_state::READY;
    } else {
        owner->key_of[job->slot] = -1;
        owner->state[job->slot] = fake_cache_owner::slot_state::FREE;
        ++owner->fill_rollbacks;
    }
    fake_touch(*owner, job->slot);
    return true;
}

bool exercise_fake_replacement_contract(const fake_cache_owner & initial) {
    fake_cache_owner owner = initial;
    owner.allow_fill = true;
    owner.corrupt_fill = 0;
    ggml_cuda_moe_ffn_fill_job job{};

    if (!fake_reserve_fill(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, 11, &job) || job.slot != 11 ||
            owner.state[11] != fake_cache_owner::slot_state::COPYING ||
            !fake_finish_fill(&owner, &job, true) || owner.slot_of[11] != 11 ||
            owner.state[11] != fake_cache_owner::slot_state::READY) return false;

    const int32_t id1 = 1;
    ggml_cuda_moe_ffn_resolution pin{};
    if (!fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, &id1, 1, &pin) || !fake_release_pins(&owner, pin.pins, pin.n_pins) ||
            owner.lru_oldest_first.back() != 1) return false;
    for (int32_t id = 0; id < owner.fill_slots; ++id) {
        if (id == id1) continue;
        if (!fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer,
                    owner.owner_generation, &id, 1, &pin) ||
                !fake_release_pins(&owner, pin.pins, pin.n_pins)) return false;
    }
    if (owner.lru_oldest_first.front() != 1) return false;
    owner.pins[1] = 1; // Preserve a pinned LRU head while a distinct miss is admitted.

    fake_cache_owner no_victim = owner;
    no_victim.pins[1] = 0;
    for (size_t s = 0; s < no_victim.pins.size(); ++s) {
        if (no_victim.state[s] == fake_cache_owner::slot_state::READY) no_victim.pins[s] = 1;
    }
    no_victim.state[2] = fake_cache_owner::slot_state::COPYING;
    no_victim.slot_of[2] = -1;
    const auto keys_before = no_victim.key_of;
    const auto slots_before = no_victim.slot_of;
    const auto generations_before = no_victim.generation;
    const auto pins_before = no_victim.pins;
    const auto states_before = no_victim.state;
    const auto lru_before = no_victim.lru_oldest_first;
    if (fake_reserve_fill(&no_victim, no_victim.device, no_victim.group, no_victim.layer,
                no_victim.owner_generation, 12, &job) ||
            no_victim.key_of != keys_before || no_victim.slot_of != slots_before ||
            no_victim.generation != generations_before || no_victim.pins != pins_before ||
            no_victim.state != states_before || no_victim.lru_oldest_first != lru_before) return false;

    const int32_t overwritten = 0;
    const int32_t replacement = 12;
    if (!fake_reserve_fill(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, replacement, &job) || job.slot != overwritten ||
            owner.slot_of[overwritten] != -1 || owner.key_of[overwritten] != replacement ||
            owner.state[overwritten] != fake_cache_owner::slot_state::COPYING) return false;
    ggml_cuda_moe_ffn_resolution copying{};
    if (!fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, &replacement, 1, &copying) ||
            copying.draw_slots[0] != -1 || copying.n_pins != 0 ||
            fake_reserve_fill(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, 13, &job)) return false;
    if (!fake_finish_fill(&owner, &owner.reserved_fill, true) || owner.slot_of[replacement] != overwritten) return false;
    if (!fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, &replacement, 1, &copying) ||
            copying.draw_slots[0] != overwritten || !fake_release_pins(&owner, copying.pins, copying.n_pins)) return false;

    if (!fake_reserve_fill(&owner, owner.device, owner.group, owner.layer,
                owner.owner_generation, 13, &job)) return false;
    const int32_t failed_slot = job.slot;
    const int32_t failed_old_key = initial.key_of[failed_slot];
    if (failed_old_key < 0 || owner.slot_of[failed_old_key] >= 0 ||
            !fake_finish_fill(&owner, &owner.reserved_fill, false) ||
            owner.state[failed_slot] != fake_cache_owner::slot_state::FREE ||
            owner.key_of[failed_slot] != -1 || owner.slot_of[failed_old_key] != -1) return false;
    return true;
}

const ggml_cuda_moe_ffn_cache_callbacks fake_cache_callbacks = {
    fake_resolve_and_pin,
    fake_release_pins,
    fake_reserve_fill,
    fake_finish_fill,
};

bool exercise_fake_pin_contract(fake_cache_owner & owner) {
    const int32_t ids[SELECTED] = {0, 0, 1, 2, 3, 4, 5, 6, 7, 8};
    ggml_cuda_moe_ffn_resolution resolved{};
    if (!fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer, owner.owner_generation, ids, SELECTED, &resolved) ||
            resolved.n_draws != SELECTED || resolved.n_pins != SELECTED - 1) return false;
    const auto exact = resolved.pins[0];
    auto invalid = exact;
    ++invalid.device;
    if (fake_release_pins(&owner, &invalid, 1)) return false;
    invalid = exact; ++invalid.group;
    if (fake_release_pins(&owner, &invalid, 1)) return false;
    invalid = exact; ++invalid.key;
    if (fake_release_pins(&owner, &invalid, 1)) return false;
    invalid = exact; ++invalid.generation;
    if (fake_release_pins(&owner, &invalid, 1)) return false;
    if (!fake_release_pins(&owner, resolved.pins, resolved.n_pins) ||
            fake_release_pins(&owner, resolved.pins, resolved.n_pins)) return false;
    ggml_cuda_moe_ffn_resolution next{};
    if (!fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer, owner.owner_generation, ids, SELECTED, &next) ||
            next.pins[0].generation == exact.generation ||
            fake_release_pins(&owner, resolved.pins, resolved.n_pins) ||
            !fake_release_pins(&owner, next.pins, next.n_pins)) return false;
    if (fake_resolve_and_pin(&owner, owner.device + 1, owner.group, owner.layer, owner.owner_generation, ids, SELECTED, &next) ||
            fake_resolve_and_pin(&owner, owner.device, owner.group + 1, owner.layer, owner.owner_generation, ids, SELECTED, &next) ||
            fake_resolve_and_pin(&owner, owner.device, owner.group, owner.layer, owner.owner_generation + 1, ids, SELECTED, &next)) return false;
    return std::all_of(owner.pins.begin(), owner.pins.end(), [](uint32_t count) { return count == 0; });
}

ggml_context * make_context(size_t bytes, bool no_alloc) {
    ggml_init_params params{bytes, nullptr, no_alloc};
    return ggml_init(params);
}

bool layout_parameterization_check(const ggml_moe_ffn_layout & base) {
    ggml_context * ctx = make_context(64 * 1024, true);
    if (!ctx) return false;
    ggml_moe_ffn_layout layout = base;
    layout.n_experts_used = 6;
    layout.activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_embd, 1);
    layout.selected_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, layout.n_experts_used, 1);
    layout.route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, layout.n_experts_used, 1);
    auto * router_logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_experts, 1);
    auto * selection_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_experts, 1);
    auto * biased_router_logits = ggml_add(ctx, router_logits, selection_bias);
    const bool no_expert_extras = ggml_moe_ffn_expert_execution_inputs_supported(
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    layout.graph_flags = GGML_MOE_FFN_GRAPH_ONE_TOKEN | GGML_MOE_FFN_GRAPH_NO_LORA |
        GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS | GGML_MOE_FFN_GRAPH_SILU_LIMIT |
        GGML_MOE_FFN_GRAPH_NO_HADAMARD;
    if (no_expert_extras)
        layout.graph_flags |= GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS;
    const bool router_bias_upstream_accepted = biased_router_logits && no_expert_extras &&
            ggml_moe_ffn_graph_supported(&layout);
    auto * expert_up_bias = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_ff, layout.n_experts);
    ggml_moe_ffn_layout biased_layout = layout;
    const bool expert_bias_supported = ggml_moe_ffn_expert_execution_inputs_supported(
            expert_up_bias, nullptr, nullptr, nullptr, nullptr, nullptr);
    if (!expert_bias_supported)
        biased_layout.graph_flags &= ~GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS;
    const bool expert_bias_rejected = !expert_bias_supported && !ggml_moe_ffn_graph_supported(&biased_layout);
    auto * expert_down_scale = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, layout.n_experts);
    ggml_moe_ffn_layout scaled_layout = layout;
    const bool expert_scale_supported = ggml_moe_ffn_expert_execution_inputs_supported(
            nullptr, nullptr, nullptr, nullptr, nullptr, expert_down_scale);
    if (!expert_scale_supported)
        scaled_layout.graph_flags &= ~GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS;
    const bool expert_scale_rejected = !expert_scale_supported && !ggml_moe_ffn_graph_supported(&scaled_layout);
    const bool supported = ggml_moe_ffn_graph_supported(&layout);
    layout.limit = 10.0f;
    const bool finite_clamp_supported = ggml_moe_ffn_graph_supported(&layout);
    layout.limit = -1.0f;
    const bool negative_clamp_rejected = !ggml_moe_ffn_layout_supported(&layout);
    layout.limit = INFINITY;
    const bool infinite_clamp_rejected = !ggml_moe_ffn_layout_supported(&layout);
    layout.limit = NAN;
    const bool nan_clamp_rejected = !ggml_moe_ffn_layout_supported(&layout);
    layout.limit = 0.0f;
    layout.graph_flags &= ~GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS;
    const bool final_route_contract_required = !ggml_moe_ffn_graph_supported(&layout);

    auto * xs_down = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_XS, layout.n_ff, layout.n_embd, layout.n_experts);
    auto * xs_bank_down = ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_XS, layout.n_ff, layout.n_embd, layout.n_slots);
    layout = base;
    layout.down = xs_down;
    layout.bank_down = xs_bank_down;
    layout.n_experts_used = 6;
    layout.activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_embd, 1);
    layout.selected_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, layout.n_experts_used, 1);
    layout.route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, layout.n_experts_used, 1);
    layout.graph_flags = GGML_MOE_FFN_GRAPH_ONE_TOKEN | GGML_MOE_FFN_GRAPH_NO_LORA |
        GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS | GGML_MOE_FFN_GRAPH_SILU_LIMIT |
        GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS | GGML_MOE_FFN_GRAPH_NO_HADAMARD;
    const bool iq4_xs_down_supported = xs_down && xs_bank_down && ggml_moe_ffn_graph_supported(&layout);
    auto * q8_down = ggml_new_tensor_3d(ctx, GGML_TYPE_Q8_0, layout.n_ff, layout.n_embd, layout.n_experts);
    layout.down = q8_down;
    const bool unlisted_down_type_rejected = q8_down && !ggml_moe_ffn_layout_supported(&layout);

    auto * bad_host_slice = ggml_view_3d(ctx, const_cast<ggml_tensor *>(base.up),
            base.up->ne[0], base.up->ne[1], base.up->ne[2],
            base.up->nb[1], base.up->nb[2] + ggml_type_size(base.up->type), 0);
    layout = base;
    layout.n_experts_used = 6;
    layout.activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_embd, 1);
    layout.selected_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, layout.n_experts_used, 1);
    layout.route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, layout.n_experts_used, 1);
    layout.graph_flags = GGML_MOE_FFN_GRAPH_ONE_TOKEN | GGML_MOE_FFN_GRAPH_NO_LORA |
        GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS | GGML_MOE_FFN_GRAPH_SILU_LIMIT |
        GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS | GGML_MOE_FFN_GRAPH_NO_HADAMARD;
    layout.up = bad_host_slice;
    const bool padded_expert_slice_rejected = bad_host_slice && !ggml_moe_ffn_graph_supported(&layout);

    auto * bad_bank_row = ggml_view_3d(ctx, const_cast<ggml_tensor *>(base.bank_up),
            base.bank_up->ne[0], base.bank_up->ne[1], base.bank_up->ne[2],
            base.bank_up->nb[1] + ggml_type_size(base.bank_up->type),
            base.bank_up->nb[2], 0);
    layout = base;
    layout.n_experts_used = 6;
    layout.activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, layout.n_embd, 1);
    layout.selected_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, layout.n_experts_used, 1);
    layout.route_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, layout.n_experts_used, 1);
    layout.graph_flags = GGML_MOE_FFN_GRAPH_ONE_TOKEN | GGML_MOE_FFN_GRAPH_NO_LORA |
        GGML_MOE_FFN_GRAPH_FINAL_ROUTE_WEIGHTS | GGML_MOE_FFN_GRAPH_SILU_LIMIT |
        GGML_MOE_FFN_GRAPH_NO_EXPERT_EXECUTION_EXTRAS | GGML_MOE_FFN_GRAPH_NO_HADAMARD;
    layout.bank_up = bad_bank_row;
    const bool padded_bank_row_rejected = bad_bank_row && !ggml_moe_ffn_graph_supported(&layout);

    layout.n_experts_used = layout.n_experts + 1;
    const bool excessive_selection_rejected = !ggml_moe_ffn_layout_supported(&layout);
    ggml_free(ctx);
    return supported && router_bias_upstream_accepted && expert_bias_rejected && expert_scale_rejected &&
        finite_clamp_supported && negative_clamp_rejected &&
        infinite_clamp_rejected && nan_clamp_rejected && final_route_contract_required && iq4_xs_down_supported &&
        unlisted_down_type_rejected && padded_expert_slice_rejected &&
        padded_bank_row_rejected && excessive_selection_rejected;
}

void fill_quantized(ggml_tensor * tensor, int salt) {
    const int64_t row = tensor->ne[0];
    const size_t count = ggml_nelements(tensor);
    std::vector<float> source(count);
    for (size_t i = 0; i < count; ++i) {
        source[i] = 0.08f * std::sin(float(i + salt * 31) * 0.071f) +
                    0.03f * std::cos(float(i * 3 + salt) * 0.037f);
    }
    std::vector<float> importance(row, 1.0f);
    ggml_quantize_chunk(tensor->type, source.data(), tensor->data, 0,
            count / row, row, importance.data(), nullptr);
}

std::vector<float> make_vector(size_t count, int seed) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = 0.4f * std::sin(float(i + seed * 17) * 0.019f);
    }
    return result;
}

void set_tensor(ggml_tensor * tensor, const void * data) {
    ggml_backend_tensor_set(tensor, data, 0, ggml_nbytes(tensor));
}

std::vector<float> get_f32(ggml_tensor * tensor) {
    std::vector<float> result(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, result.data(), 0, result.size() * sizeof(float));
    return result;
}

struct fixture {
    ggml_context * host_ctx = nullptr;
    ggml_context * cpu_ctx = nullptr;
    ggml_context * cuda_ctx = nullptr;
    ggml_backend_t cpu = nullptr;
    ggml_backend_t cuda = nullptr;
    int32_t cpu_resource_id = -1;
    bool owns_cpu_resource = false;
    bool owns_backends = false;
    ggml_backend_buffer_t cpu_buffer = nullptr;
    ggml_backend_buffer_t cuda_buffer = nullptr;
    ggml_tensor * host_up = nullptr;
    ggml_tensor * host_gate = nullptr;
    ggml_tensor * host_down = nullptr;
    ggml_tensor * cpu_activation = nullptr;
    ggml_tensor * cpu_ids = nullptr;
    ggml_tensor * cpu_routes = nullptr;
    ggml_tensor * cpu_output = nullptr;
    ggml_cgraph * cpu_graph = nullptr;
    ggml_tensor * cpu_single_ids = nullptr;
    ggml_tensor * cpu_single_routes = nullptr;
    ggml_tensor * cpu_single_output = nullptr;
    ggml_cgraph * cpu_single_graph = nullptr;
    ggml_tensor * semantic = nullptr;
    ggml_tensor * semantic_consumer = nullptr;
    ggml_tensor * consumer_bias = nullptr;
    ggml_cgraph * semantic_graph = nullptr;
    ggml_cgraph * activation_graph = nullptr;
    ggml_tensor * stock_ids = nullptr;
    ggml_tensor * stock_routes = nullptr;
    ggml_tensor * stock_output = nullptr;
    ggml_cgraph * stock_graph = nullptr;
    ggml_tensor * stock_zero_output = nullptr;
    ggml_cgraph * stock_zero_graph = nullptr;
    float limit = 0.0f;
    metrics ordinary_cpu_cuda_control{};
    bool ordinary_cpu_cuda_control_valid = false;
    ggml_tensor * ids = nullptr;
    ggml_tensor * routes = nullptr;
    ggml_tensor * seed = nullptr;
    ggml_tensor * producer_weight = nullptr;
    ggml_tensor * activation = nullptr;
    std::vector<float> producer_values;
    ggml_cuda_moe_ffn_binding binding{};
    std::vector<int32_t> expert_to_slot{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, -1, -1};
    fake_cache_owner cache_owner;
    int device = -1;
    int32_t binding_id = -1;
    uint64_t generation = UINT64_C(0x100000001);
    fixture * shared_parent = nullptr;
    std::vector<fixture *> shared_bindings;
};

bool create_fixture(fixture & f, int device, fixture * shared = nullptr, float limit = 0.0f) {
    f.device = device;
    f.limit = limit;
    f.shared_parent = shared;
    f.cache_owner.device = device;
    f.cache_owner.owner_generation = f.generation;
    f.cache_owner.group = 3;
    f.cache_owner.layer = shared ? 1 : 0;
    f.cache_owner.slot_of = f.expert_to_slot;
    f.cache_owner.key_of.assign(SLOTS, -1);
    f.cache_owner.generation.assign(SLOTS, 1);
    f.cache_owner.pins.assign(SLOTS, 0);
    f.cache_owner.state.assign(SLOTS, fake_cache_owner::slot_state::FREE);
    f.cache_owner.lru_oldest_first.resize(SLOTS);
    for (int32_t slot = 0; slot < SLOTS; ++slot) f.cache_owner.lru_oldest_first[slot] = slot;
    for (int32_t expert = 0; expert < EXPERTS; ++expert) {
        const int32_t slot = f.expert_to_slot[expert];
        if (slot >= 0) {
            f.cache_owner.key_of[slot] = expert;
            f.cache_owner.state[slot] = fake_cache_owner::slot_state::READY;
        }
    }
    f.host_ctx = make_context(16 * 1024 * 1024, false);
    f.cpu_ctx = make_context(16 * 1024 * 1024, true);
    f.cuda_ctx = make_context(64 * 1024 * 1024, true);
    f.owns_backends = shared == nullptr;
    f.cpu = shared ? shared->cpu : ggml_backend_cpu_init();
    f.cuda = shared ? shared->cuda : ggml_backend_cuda_init(device, nullptr, nullptr);
    if (!f.host_ctx || !f.cpu_ctx || !f.cuda_ctx || !f.cpu || !f.cuda) return false;
    if (!shared) ggml_backend_cpu_set_n_threads(f.cpu, 1);

    f.host_up = ggml_new_tensor_3d(f.host_ctx, GGML_TYPE_IQ3_S, K, K, EXPERTS);
    f.host_gate = ggml_new_tensor_3d(f.host_ctx, GGML_TYPE_IQ3_S, K, K, EXPERTS);
    f.host_down = ggml_new_tensor_3d(f.host_ctx, GGML_TYPE_IQ4_NL, K, K, EXPERTS);
    const int salt = shared ? 17 : 0;
    fill_quantized(f.host_up, 1 + salt);
    fill_quantized(f.host_gate, 2 + salt);
    fill_quantized(f.host_down, 3 + salt);

    f.cpu_activation = ggml_new_tensor_2d(f.cpu_ctx, GGML_TYPE_F32, K, 1);
    f.cpu_ids = ggml_new_tensor_2d(f.cpu_ctx, GGML_TYPE_I32, SELECTED, 1);
    f.cpu_routes = ggml_new_tensor_3d(f.cpu_ctx, GGML_TYPE_F32, 1, SELECTED, 1);
    ggml_tensor * up_gate = ggml_moe_up_gate(f.cpu_ctx, f.host_up, f.host_gate,
            f.cpu_activation, f.cpu_ids, GGML_UNARY_OP_SILU);
    *(float *)(up_gate->op_params + 1) = f.limit;
    ggml_tensor * down = ggml_mul_mat_id(f.cpu_ctx, f.host_down, up_gate, f.cpu_ids);
    f.cpu_output = ggml_mul_multi_add(f.cpu_ctx, down, f.cpu_routes);
    f.cpu_graph = ggml_new_graph_custom(f.cpu_ctx, 32, false);
    ggml_build_forward_expand(f.cpu_graph, f.cpu_output);
    f.cpu_single_ids = ggml_new_tensor_2d(f.cpu_ctx, GGML_TYPE_I32, 1, 1);
    f.cpu_single_routes = ggml_new_tensor_3d(f.cpu_ctx, GGML_TYPE_F32, 1, 1, 1);
    ggml_tensor * single_up_gate = ggml_moe_up_gate(f.cpu_ctx, f.host_up, f.host_gate,
            f.cpu_activation, f.cpu_single_ids, GGML_UNARY_OP_SILU);
    *(float *)(single_up_gate->op_params + 1) = f.limit;
    ggml_tensor * single_down = ggml_mul_mat_id(f.cpu_ctx, f.host_down, single_up_gate, f.cpu_single_ids);
    f.cpu_single_output = ggml_mul_multi_add(f.cpu_ctx, single_down, f.cpu_single_routes);
    f.cpu_single_graph = ggml_new_graph_custom(f.cpu_ctx, 32, false);
    ggml_build_forward_expand(f.cpu_single_graph, f.cpu_single_output);
    f.cpu_buffer = ggml_backend_alloc_ctx_tensors(f.cpu_ctx, f.cpu);
    if (!f.cpu_buffer) return false;
    const std::vector<float> cpu_zeros(K, 0.0f);
    const std::vector<int32_t> cpu_seed_ids{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> cpu_seed_routes(SELECTED, 1.0f / SELECTED);
    const int32_t cpu_single_id = 0;
    const float cpu_single_route = 1.0f;
    ggml_backend_tensor_set(f.cpu_activation, cpu_zeros.data(), 0, K * sizeof(float));
    ggml_backend_tensor_set(f.cpu_ids, cpu_seed_ids.data(), 0, SELECTED * sizeof(int32_t));
    ggml_backend_tensor_set(f.cpu_routes, cpu_seed_routes.data(), 0, SELECTED * sizeof(float));
    ggml_backend_tensor_set(f.cpu_single_ids, &cpu_single_id, 0, sizeof(cpu_single_id));
    ggml_backend_tensor_set(f.cpu_single_routes, &cpu_single_route, 0, sizeof(cpu_single_route));
    if (ggml_backend_graph_compute(f.cpu, f.cpu_graph) != GGML_STATUS_SUCCESS) return false;
    if (ggml_backend_graph_compute(f.cpu, f.cpu_single_graph) != GGML_STATUS_SUCCESS) return false;

    ggml_tensor * cache_up = ggml_new_tensor_3d(f.cuda_ctx, GGML_TYPE_IQ3_S, K, K, SLOTS);
    ggml_tensor * cache_gate = ggml_new_tensor_3d(f.cuda_ctx, GGML_TYPE_IQ3_S, K, K, SLOTS);
    ggml_tensor * cache_down = ggml_new_tensor_3d(f.cuda_ctx, GGML_TYPE_IQ4_NL, K, K, SLOTS);
    f.ids = ggml_new_tensor_2d(f.cuda_ctx, GGML_TYPE_I32, SELECTED, 1);
    f.routes = ggml_new_tensor_3d(f.cuda_ctx, GGML_TYPE_F32, 1, SELECTED, 1);
    f.seed = ggml_new_tensor_2d(f.cuda_ctx, GGML_TYPE_F32, K, 1);
    f.producer_weight = ggml_new_tensor_2d(f.cuda_ctx, GGML_TYPE_F32, K, K);
    f.producer_values.resize(ggml_nelements(f.producer_weight));
    for (size_t i = 0; i < f.producer_values.size(); ++i) {
        const size_t row = i / K;
        const size_t column = i % K;
        f.producer_values[i] = row == column ? 0.5f : 0.00001f * std::sin(float(i) * 0.013f);
    }
    f.activation = ggml_mul_mat(f.cuda_ctx, f.producer_weight, f.seed);
    for (int i = 0; i < 4; ++i) f.activation = ggml_mul_mat(f.cuda_ctx, f.producer_weight, f.activation);
    f.activation_graph = ggml_new_graph_custom(f.cuda_ctx, 16, false);
    ggml_build_forward_expand(f.activation_graph, f.activation);
    f.consumer_bias = ggml_new_tensor_2d(f.cuda_ctx, GGML_TYPE_F32, K, 1);

    f.stock_ids = ggml_new_tensor_2d(f.cuda_ctx, GGML_TYPE_I32, SELECTED, 1);
    f.stock_routes = ggml_new_tensor_3d(f.cuda_ctx, GGML_TYPE_F32, 1, SELECTED, 1);
    ggml_tensor * stock_up_gate = ggml_moe_up_gate(f.cuda_ctx, cache_up, cache_gate,
            f.activation, f.stock_ids, GGML_UNARY_OP_SILU);
    *(float *)(stock_up_gate->op_params + 1) = f.limit;
    ggml_tensor * stock_down = ggml_mul_mat_id(f.cuda_ctx, cache_down, stock_up_gate, f.stock_ids);
    f.stock_output = ggml_mul_multi_add(f.cuda_ctx, stock_down, f.stock_routes);
    f.stock_graph = ggml_new_graph_custom(f.cuda_ctx, 32, false);
    ggml_build_forward_expand(f.stock_graph, f.stock_output);
    if (f.limit > 0.0f) {
        ggml_tensor * zero_up_gate = ggml_moe_up_gate(f.cuda_ctx, cache_up, cache_gate,
                f.activation, f.stock_ids, GGML_UNARY_OP_SILU);
        *(float *)(zero_up_gate->op_params + 1) = 0.0f;
        ggml_tensor * zero_down = ggml_mul_mat_id(f.cuda_ctx, cache_down, zero_up_gate, f.stock_ids);
        f.stock_zero_output = ggml_mul_multi_add(f.cuda_ctx, zero_down, f.stock_routes);
        f.stock_zero_graph = ggml_new_graph_custom(f.cuda_ctx, 32, false);
        ggml_build_forward_expand(f.stock_zero_graph, f.stock_zero_output);
    }

    f.semantic = ggml_moe_ffn(f.cuda_ctx, f.activation, f.ids, f.routes, device, 1, f.generation,
            K, K, GGML_UNARY_OP_SILU, f.limit);
    f.semantic_consumer = ggml_add(f.cuda_ctx, f.semantic, f.consumer_bias);
    f.semantic_graph = ggml_new_graph_custom(f.cuda_ctx, 64, false);
    ggml_build_forward_expand(f.semantic_graph, f.semantic_consumer);
    f.cuda_buffer = ggml_backend_alloc_ctx_tensors(f.cuda_ctx, f.cuda);
    if (!f.cuda_buffer) return false;

    ggml_backend_tensor_set(f.producer_weight, f.producer_values.data(), 0,
            f.producer_values.size() * sizeof(float));
    const std::vector<float> bias(K, 0.125f);
    ggml_backend_tensor_set(f.consumer_bias, bias.data(), 0, bias.size() * sizeof(float));
    const size_t up_expert_bytes = ggml_row_size(GGML_TYPE_IQ3_S, K) * K;
    const size_t down_expert_bytes = ggml_row_size(GGML_TYPE_IQ4_NL, K) * K;
    for (int expert = 0; expert < EXPERTS; ++expert) {
        const int slot = f.expert_to_slot[expert];
        if (slot < 0) continue;
        ggml_backend_tensor_set(cache_up, (const uint8_t *) f.host_up->data + expert * up_expert_bytes,
                slot * up_expert_bytes, up_expert_bytes);
        ggml_backend_tensor_set(cache_gate, (const uint8_t *) f.host_gate->data + expert * up_expert_bytes,
                slot * up_expert_bytes, up_expert_bytes);
        ggml_backend_tensor_set(cache_down, (const uint8_t *) f.host_down->data + expert * down_expert_bytes,
                slot * down_expert_bytes, down_expert_bytes);
    }
    // Keep the authoritative owner map unchanged; this otherwise-free bank
    // slot is only an ordinary-CUDA reference for synthetic miss expert 11.
    const int reference_slot = SLOTS - 1;
    const int reference_expert = 11;
    ggml_backend_tensor_set(cache_up, (const uint8_t *) f.host_up->data + reference_expert * up_expert_bytes,
            reference_slot * up_expert_bytes, up_expert_bytes);
    ggml_backend_tensor_set(cache_gate, (const uint8_t *) f.host_gate->data + reference_expert * up_expert_bytes,
            reference_slot * up_expert_bytes, up_expert_bytes);
    ggml_backend_tensor_set(cache_down, (const uint8_t *) f.host_down->data + reference_expert * down_expert_bytes,
            reference_slot * down_expert_bytes, down_expert_bytes);

    f.binding.generation = f.generation;
    f.binding.layout = { EXPERTS, SELECTED, K, K, SLOTS, GGML_MOE_FFN_PARTS, device, f.cache_owner.group,
        f.cache_owner.layer, f.cuda, &f.cache_owner, f.host_up, f.host_gate, f.host_down,
        cache_up, cache_gate, cache_down, nullptr, nullptr, nullptr, GGML_UNARY_OP_SILU, f.limit,
        GGML_MOE_FFN_LAYOUT_SEPARATE_PARTS | GGML_MOE_FFN_LAYOUT_NO_EXTRA_WEIGHTS |
        GGML_MOE_FFN_LAYOUT_NO_HADAMARD | GGML_MOE_FFN_LAYOUT_HOST_RESIDENT, 0 };
    if (K == 256 && !layout_parameterization_check(f.binding.layout)) return false;
    f.cache_owner.fill_experts = EXPERTS;
    f.cache_owner.fill_slots = SLOTS;
    f.cache_owner.fill_hosts[0] = f.host_up;
    f.cache_owner.fill_hosts[1] = f.host_gate;
    f.cache_owner.fill_hosts[2] = f.host_down;
    f.cache_owner.fill_banks[0] = cache_up;
    f.cache_owner.fill_banks[1] = cache_gate;
    f.cache_owner.fill_banks[2] = cache_down;
    f.binding.cache_callbacks = &fake_cache_callbacks;
    f.binding.cache_context = &f.cache_owner;
    if (shared) {
        f.cpu_resource_id = shared->cpu_resource_id;
    } else {
        ggml_cuda_moe_ffn_cpu_resource resource{f.cpu, K, K, SELECTED, 1};
        f.cpu_resource_id = ggml_backend_cuda_moe_ffn_cpu_resource_register(f.cuda, &resource);
        if (f.cpu_resource_id <= 0) return false;
        f.owns_cpu_resource = true;
    }
    f.binding.cpu_resource_id = f.cpu_resource_id;
    f.binding.cpu_graph = f.cpu_single_graph;
    f.binding.cpu_activations = f.cpu_activation;
    f.binding.cpu_ids = f.cpu_single_ids;
    f.binding.cpu_route_weights = f.cpu_single_routes;
    f.binding.cpu_output = f.cpu_single_output;
    f.binding_id = ggml_backend_cuda_moe_ffn_register(f.cuda, &f.binding);
    if (f.binding_id <= 0) return false;
    if (shared) shared->shared_bindings.push_back(&f);
    ggml_moe_ffn_op_params op_params{device, f.binding_id, f.generation, K, K, GGML_UNARY_OP_SILU, f.limit};
    memcpy(f.semantic->op_params, &op_params, sizeof(op_params));
    return true;
}

void destroy_fixture(fixture & f) {
    if (f.owns_cpu_resource && f.cpu_resource_id > 0 && f.cuda &&
            !ggml_backend_cuda_moe_ffn_cpu_resource_stop(f.cuda, f.cpu_resource_id)) std::abort();
    if (f.owns_cpu_resource) {
        const auto siblings = f.shared_bindings;
        f.shared_bindings.clear();
        for (fixture * sibling : siblings) {
            sibling->shared_parent = nullptr;
            destroy_fixture(*sibling);
        }
    }
    if (!f.owns_cpu_resource && f.binding_id > 0 && f.cuda && f.cpu_resource_id > 0 &&
            !ggml_backend_cuda_moe_ffn_cpu_resource_stop(f.cuda, f.cpu_resource_id)) std::abort();
    if (f.binding_id > 0 && f.cuda && !ggml_backend_cuda_moe_ffn_unregister(f.cuda, f.binding_id)) std::abort();
    if (f.shared_parent) {
        auto & siblings = f.shared_parent->shared_bindings;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), &f), siblings.end());
        f.shared_parent = nullptr;
    }
    if (f.cuda_buffer) ggml_backend_buffer_free(f.cuda_buffer);
    if (f.cpu_buffer) ggml_backend_buffer_free(f.cpu_buffer);
    if (f.cuda_ctx) ggml_free(f.cuda_ctx);
    if (f.cpu_ctx) ggml_free(f.cpu_ctx);
    if (f.host_ctx) ggml_free(f.host_ctx);
    if (f.owns_cpu_resource && f.cpu_resource_id > 0 && f.cuda) {
        ggml_backend_cuda_moe_ffn_cpu_resource_unregister(f.cuda, f.cpu_resource_id);
    }
    if (f.owns_backends) {
        if (f.cuda) ggml_backend_free(f.cuda);
        // The resource owns and frees the CPU backend after successful registration.
        if (!f.owns_cpu_resource && f.cpu) ggml_backend_free(f.cpu);
    }
}

bool query_support_without_allocating_output(fixture & f, int owner_device, int32_t binding_id, uint64_t generation) {
    ggml_context * ctx = make_context(1024 * 1024, true);
    ggml_tensor * activation = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, SELECTED, 1);
    ggml_tensor * routes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, SELECTED, 1);
    ggml_tensor * op = ggml_moe_ffn(ctx, activation, ids, routes, owner_device,
            binding_id, generation, K, K, GGML_UNARY_OP_SILU, f.limit);
    const bool unallocated = op->buffer == nullptr;
    const bool supported = ggml_backend_supports_op(f.cuda, op);
    ggml_free(ctx);
    return unallocated && supported;
}

bool run_case(fixture & f, int iteration, const std::vector<int32_t> & ids,
        const std::vector<float> & routes, const std::vector<float> & input_seed,
        bool expect_hit, metrics & cuda_error, metrics & cpu_error,
        bool expect_complete_cpu_fallback = false) {
    set_tensor(f.ids, ids.data());
    set_tensor(f.routes, routes.data());
    std::vector<float> scaled_seed = input_seed;
    if (f.limit > 0.0f) for (float & value : scaled_seed) value *= 4096.0f;
    set_tensor(f.seed, scaled_seed.data());
    if (ggml_backend_graph_compute(f.cuda, f.activation_graph) != GGML_STATUS_SUCCESS) return false;
    const auto activation = get_f32(f.activation);
    if (!finite(activation)) return false;

    std::memcpy(f.cpu_activation->data, activation.data(), K * sizeof(float));
    std::memcpy(f.cpu_ids->data, ids.data(), SELECTED * sizeof(int32_t));
    std::memcpy(f.cpu_routes->data, routes.data(), SELECTED * sizeof(float));
    if (ggml_backend_graph_compute(f.cpu, f.cpu_graph) != GGML_STATUS_SUCCESS) return false;
    const auto cpu = get_f32(f.cpu_output);
    if (!finite(cpu)) return false;
    const bool has_miss = std::any_of(ids.begin(), ids.end(), [&](int32_t id) {
        return id < 0 || (size_t) id >= f.expert_to_slot.size() || f.expert_to_slot[id] < 0;
    });
    metrics cpu_cuda_control{};
    std::vector<float> ordinary_cuda;
    std::vector<float> mixed_reference;
    if (expect_hit || has_miss) {
        std::vector<int32_t> slots(ids.size(), -1);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (ids[i] < 0 || (size_t) ids[i] >= f.expert_to_slot.size()) return false;
            slots[i] = f.expert_to_slot[ids[i]] >= 0 ? f.expert_to_slot[ids[i]] : SLOTS - 1;
        }
        set_tensor(f.stock_ids, slots.data());
        set_tensor(f.stock_routes, routes.data());
        if (ggml_backend_graph_compute(f.cuda, f.stock_graph) != GGML_STATUS_SUCCESS) return false;
        ordinary_cuda = get_f32(f.stock_output);
        if (!finite(ordinary_cuda)) return false;
        cpu_cuda_control = compare(ordinary_cuda, cpu);
        f.ordinary_cpu_cuda_control = cpu_cuda_control;
        f.ordinary_cpu_cuda_control_valid = true;
        std::printf("ordinary_cpu_cuda_control iteration=%d limit=%g max=%.9g rms=%.9g worst=%zu\n",
                iteration, f.limit, cpu_cuda_control.max_abs, cpu_cuda_control.rms, cpu_cuda_control.worst);
        if (f.limit > 0.0f && expect_hit) {
            if (ggml_backend_graph_compute(f.cuda, f.stock_zero_graph) != GGML_STATUS_SUCCESS) return false;
            const auto zero = get_f32(f.stock_zero_output);
            const metrics threshold_delta = compare(ordinary_cuda, zero);
            if (!finite(zero) || threshold_delta.max_abs <= 1.0e-4) {
                std::fprintf(stderr, "clamp threshold was not exercised limit=%g delta=%g\n",
                        f.limit, threshold_delta.max_abs);
                return false;
            }
            std::printf("clamp_threshold limit=%g ordinary_limit10_vs_zero_max=%.9g rms=%.9g crossed=1\n",
                    f.limit, threshold_delta.max_abs, threshold_delta.rms);
        }
        if (!expect_hit) {
            mixed_reference.assign(K, 0.0f);
            std::vector<float> draw_routes(SELECTED, 0.0f);
            for (int draw = 0; draw < SELECTED; ++draw) {
                std::vector<float> contribution;
                const int32_t expert = ids[draw];
                const int32_t resident_slot = f.expert_to_slot[expert];
                if (resident_slot >= 0) {
                    std::fill(draw_routes.begin(), draw_routes.end(), 0.0f);
                    draw_routes[draw] = routes[draw];
                    set_tensor(f.stock_routes, draw_routes.data());
                    if (ggml_backend_graph_compute(f.cuda, f.stock_graph) != GGML_STATUS_SUCCESS) return false;
                    contribution = get_f32(f.stock_output);
                } else {
                    ggml_backend_tensor_set(f.cpu_single_ids, &expert, 0, sizeof(expert));
                    ggml_backend_tensor_set(f.cpu_single_routes, &routes[draw], 0, sizeof(float));
                    if (ggml_backend_graph_compute(f.cpu, f.cpu_single_graph) != GGML_STATUS_SUCCESS) return false;
                    contribution = get_f32(f.cpu_single_output);
                }
                if (!finite(contribution)) return false;
                for (int k = 0; k < K; ++k) mixed_reference[k] += contribution[k];
            }
        }
    }
    if (ggml_backend_graph_compute(f.cuda, f.semantic_graph) != GGML_STATUS_SUCCESS) return false;
    const auto semantic = get_f32(f.semantic);
    if (!finite(semantic)) return false;
    cpu_error = compare(semantic, cpu);
    const auto & expected = expect_complete_cpu_fallback ? cpu : (expect_hit ? ordinary_cuda : mixed_reference);
    cuda_error = compare(semantic, expected);
    if ((!expect_hit && !has_miss) || cuda_error.max_abs > 1e-5 || cuda_error.rms > 5e-6) {
        std::fprintf(stderr, "clamp_semantic_reference_failure limit=%g ordinary_cpu_cuda_max=%.9g ordinary_cpu_cuda_rms=%.9g semantic_reference_max=%.9g semantic_reference_rms=%.9g semantic_cpu_max=%.9g semantic_cpu_rms=%.9g\n",
                f.limit, cpu_cuda_control.max_abs, cpu_cuda_control.rms, cuda_error.max_abs,
                cuda_error.rms, cpu_error.max_abs, cpu_error.rms);
        return false;
    }

    const auto semantic_consumer = get_f32(f.semantic_consumer);
    std::vector<float> expected_consumer(K);
    for (int i = 0; i < K; ++i) expected_consumer[i] = semantic[i] + 0.125f;
    const metrics consumer_error = compare(semantic_consumer, expected_consumer);
    if (!finite(semantic_consumer) || consumer_error.max_abs > 1e-5) return false;
    std::printf("case=%d branch=%s stock_cuda_max=%.8g stock_cuda_rms=%.8g stock_worst=%zu semantic_cpu_max=%.8g semantic_cpu_rms=%.8g cpu_worst=%zu consumer_delta=%.8g\n",
            iteration, expect_hit ? "all-hit" : "any-miss", cuda_error.max_abs, cuda_error.rms,
            cuda_error.worst, cpu_error.max_abs, cpu_error.rms, cpu_error.worst, consumer_error.max_abs);
    return true;
}

bool run_scheduler_order_case(fixture & f, const std::vector<std::vector<int32_t>> & route_ids,
        const std::vector<std::vector<float>> & route_weights, bool all_hit_only = false) {
    ggml_context * cpu_ctx = make_context(8 * 1024 * 1024, true);
    ggml_context * cuda_ctx = make_context(8 * 1024 * 1024, true);
    if (!cpu_ctx || !cuda_ctx) return false;
    ggml_tensor * seed = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * weight = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_F32, K, K);
    ggml_tensor * ids = ggml_new_tensor_2d(cpu_ctx, GGML_TYPE_I32, SELECTED, 1);
    ggml_tensor * routes = ggml_new_tensor_3d(cpu_ctx, GGML_TYPE_F32, 1, SELECTED, 1);
    std::vector<float> weight_data(size_t(K) * K);
    for (int row = 0; row < K; ++row) for (int col = 0; col < K; ++col) {
        weight_data[size_t(row) * K + col] = row == col ? 0.75f : 0.000003f * std::sin(float(row * K + col));
    }
    ggml_tensor * activation = ggml_mul_mat(cpu_ctx, weight, seed);
    ggml_tensor * semantic = ggml_moe_ffn(cuda_ctx, activation, ids, routes, f.device,
            f.binding_id, f.generation, K, K, GGML_UNARY_OP_SILU, f.limit);
    ggml_set_output(semantic);
    ggml_tensor * bias = ggml_new_tensor_2d(cuda_ctx, GGML_TYPE_F32, K, 1);
    ggml_tensor * consumer = ggml_add(cuda_ctx, semantic, bias);
    ggml_cgraph * graph = ggml_new_graph_custom(cuda_ctx, 32, false);
    ggml_build_forward_expand(graph, consumer);

    ggml_backend_t backends[] = { f.cuda, f.cpu };
    ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false);
    if (!sched) { ggml_free(cuda_ctx); ggml_free(cpu_ctx); return false; }
    ggml_backend_sched_set_tensor_backend(sched, activation, f.cpu);
    ggml_backend_sched_set_tensor_backend(sched, seed, f.cpu);
    ggml_backend_sched_set_tensor_backend(sched, weight, f.cpu);
    ggml_backend_sched_set_tensor_backend(sched, ids, f.cpu);
    ggml_backend_sched_set_tensor_backend(sched, routes, f.cpu);
    ggml_backend_sched_set_tensor_backend(sched, bias, f.cuda);
    ggml_backend_sched_set_tensor_backend(sched, consumer, f.cuda);
    const bool allocated = ggml_backend_sched_alloc_graph(sched, graph);
    const int copies = allocated ? ggml_backend_sched_get_n_copies(sched) : 0;
    const bool ownership = allocated && ggml_backend_sched_get_tensor_backend(sched, activation) == f.cpu &&
        ggml_backend_sched_get_tensor_backend(sched, ids) == f.cpu &&
        ggml_backend_sched_get_tensor_backend(sched, routes) == f.cpu &&
        ggml_backend_sched_get_tensor_backend(sched, semantic) == f.cuda &&
        semantic->src[0] && semantic->src[0]->type == GGML_TYPE_F32 && semantic->src[0]->ne[0] == K &&
        semantic->src[1] && semantic->src[1]->type == GGML_TYPE_I32 && semantic->src[1]->ne[0] == SELECTED &&
        semantic->src[2] && semantic->src[2]->type == GGML_TYPE_F32 && semantic->src[2]->ne[1] == SELECTED &&
        copies >= 1 && ggml_backend_sched_get_n_splits(sched) == 2;
    bool no_weight_nodes = true;
    for (int i = 0; allocated && i < graph->n_nodes + graph->n_leafs; ++i) {
        const ggml_tensor * tensor = i < graph->n_nodes ? graph->nodes[i] : graph->leafs[i - graph->n_nodes];
        no_weight_nodes = no_weight_nodes && tensor != f.binding.layout.up && tensor != f.binding.layout.gate &&
            tensor != f.binding.layout.down && tensor != f.binding.layout.bank_up && tensor != f.binding.layout.bank_gate &&
            tensor != f.binding.layout.bank_down;
    }

    const std::vector<float> bias_data(K, 0.125f);
    std::vector<float> seed_data(K);
    bool okay = ownership && no_weight_nodes;
    for (int step = 0; okay && step < 3; ++step) {
        const int route = all_hit_only ? (step == 1 ? 2 : 0) : step;
        seed_data = make_vector(K, 30 + step);
        ggml_backend_tensor_set(seed, seed_data.data(), 0, K * sizeof(float));
        ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
        ggml_backend_tensor_set(ids, route_ids[route].data(), 0, SELECTED * sizeof(int32_t));
        ggml_backend_tensor_set(routes, route_weights[route].data(), 0, SELECTED * sizeof(float));
        ggml_backend_tensor_set(bias, bias_data.data(), 0, bias_data.size() * sizeof(float));
        const bool compute_ok = ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS;
        okay = compute_ok;
        const auto scheduled_output = get_f32(semantic);
        const auto scheduled_consumer = get_f32(consumer);
        const auto activation_value = get_f32(activation);
        std::memcpy(f.cpu_activation->data, activation_value.data(), K * sizeof(float));
        std::memcpy(f.cpu_ids->data, route_ids[route].data(), SELECTED * sizeof(int32_t));
        std::memcpy(f.cpu_routes->data, route_weights[route].data(), SELECTED * sizeof(float));
        okay = okay && finite(scheduled_output) && finite(scheduled_consumer) &&
            ggml_backend_graph_compute(f.cpu, f.cpu_graph) == GGML_STATUS_SUCCESS;
        const auto expected = get_f32(f.cpu_output);
        std::vector<float> expected_consumer(K);
        for (int i = 0; i < K; ++i) expected_consumer[i] = expected[i] + bias_data[i];
        const metrics scheduled_error = compare(scheduled_output, expected);
        const metrics consumer_error = compare(scheduled_consumer, expected_consumer);
        if (scheduled_error.max_abs > 2e-4 || consumer_error.max_abs > 2e-4) {
            std::vector<int32_t> observed_ids(SELECTED);
            std::vector<float> observed_routes(SELECTED);
            std::vector<float> observed_activation(K);
            ggml_backend_tensor_get(semantic->src[1], observed_ids.data(), 0, SELECTED * sizeof(int32_t));
            ggml_backend_tensor_get(semantic->src[2], observed_routes.data(), 0, SELECTED * sizeof(float));
            ggml_backend_tensor_get(semantic->src[0], observed_activation.data(), 0, K * sizeof(float));
            std::fprintf(stderr, "scheduler mismatch step=%d max=%.8g rms=%.8g y0=%.8g ref0=%.8g consumer0=%.8g ref_consumer0=%.8g activation0=%.8g observed_activation0=%.8g route0=%.8g observed_route0=%.8g ids0=%d observed_id0=%d semantic_alias=%d\n",
                    step, scheduled_error.max_abs, scheduled_error.rms, scheduled_output[0], expected[0], scheduled_consumer[0], expected_consumer[0],
                    activation_value[0], observed_activation[0], route_weights[route][0], observed_routes[0], route_ids[route][0], observed_ids[0],
                    semantic->data == consumer->data);
        }
        okay = okay && scheduled_error.max_abs <= 2e-4 && consumer_error.max_abs <= 2e-4;
        ggml_cuda_moe_ffn_stats op_stats{};
        ggml_cuda_moe_ffn_cpu_resource_stats resource_stats{};
        const bool got_stats = ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &op_stats) &&
            ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &resource_stats);
        const int pending_slots = std::count(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), 1u);
        int route_misses = 0;
        for (const int32_t id : route_ids[route]) route_misses += f.expert_to_slot[id] < 0;
        std::printf("scheduler_step=%d route=%d branch=%s selected=%d misses=%d ids0=%d ids9=%d splits=%d copy_slots=%d status=%d max_error=%.8g rms=%.8g consumer_error=%.8g pins=%d pending_pin=%llu fill_outstanding=%llu fake_fill_pending=%d stats=%d op_data=%p consumer_src0_data=%p consumer_data=%p source_alias=%d\n", step, route,
                route_misses ? "any-miss" : "all-hit", SELECTED, route_misses, route_ids[route][0], route_ids[route][SELECTED - 1],
                ggml_backend_sched_get_n_splits(sched), copies,
                int(compute_ok), scheduled_error.max_abs, scheduled_error.rms, consumer_error.max_abs, pending_slots,
                (unsigned long long) resource_stats.pending_pin_active,
                (unsigned long long) resource_stats.fill_outstanding,
                f.cache_owner.fill_pending, got_stats, semantic->data, consumer->src[0]->data, consumer->data,
                semantic->data == consumer->src[0]->data);
    }
    ggml_backend_sched_free(sched);
    ggml_free(cuda_ctx);
    ggml_free(cpu_ctx);
    return okay;
}
}

int main(int argc, char ** argv) {
    int device = 0;
    float limit = 0.0f;
    std::string mode = "full";
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--device") device = std::stoi(argv[++i]);
        else if (std::string(argv[i]) == "--mode") mode = argv[++i];
        else if (std::string(argv[i]) == "--limit") limit = std::stof(argv[++i]);
    }
    if (!std::isfinite(limit) || (limit != 0.0f && limit != 10.0f)) {
        std::fprintf(stderr, "focused fixture supports only --limit 0 or 10\n");
        return 2;
    }
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        std::fprintf(stderr, "invalid or unavailable CUDA device %d\n", device);
        return 2;
    }
    fixture f;
    if (!create_fixture(f, device, nullptr, limit)) {
        std::fprintf(stderr, "fixture registration/setup failed on device %d\n", device);
        destroy_fixture(f);
        return 1;
    }
    if (!exercise_fake_pin_contract(f.cache_owner)) {
        std::fprintf(stderr, "authoritative fake cache pin identity/dedup contract failed\n");
        destroy_fixture(f);
        return 1;
    }
    if (!exercise_fake_replacement_contract(f.cache_owner)) {
        std::fprintf(stderr, "synthetic pin-safe FREE/COPYING/READY replacement contract failed\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("synthetic_replacement free_first=1 touch_mru=1 pinned_head_skipped=1 copying_skipped=1 no_victim_unchanged=1 overwrite_no_resurrection=1 one_job=1\n");
    const std::vector<std::vector<int32_t>> route_ids = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 11},
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 11},
    };
    std::vector<std::vector<float>> route_weights(4, std::vector<float>(SELECTED));
    for (int call = 0; call < 4; ++call) {
        float total = 0.0f;
        for (int i = 0; i < SELECTED; ++i) {
            route_weights[call][i] = float((i + 1) * (call + 2));
            total += route_weights[call][i];
        }
        for (float & weight : route_weights[call]) weight /= total;
    }
    const auto run_malformed_cases = [&]() {
        ggml_cuda_moe_ffn_cpu_resource_stats before{}, after{};
        if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &before)) return false;
        for (int32_t corruption = 1; corruption <= 3; ++corruption) {
            f.cache_owner.corrupt_fill = corruption;
            metrics invalid_fill_cuda_error, invalid_fill_cpu_error;
            if (!run_case(f, 10 + corruption, route_ids[1], route_weights[1],
                        make_vector(K, 50 + corruption), false, invalid_fill_cuda_error, invalid_fill_cpu_error) ||
                    !ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &after) ||
                    f.cache_owner.fill_pending || f.cache_owner.fill_reservations != uint64_t(corruption) ||
                    f.cache_owner.fill_rollbacks != uint64_t(corruption) || f.cache_owner.key_of[SLOTS - 1] != -1 ||
                    after.fills_accepted != before.fills_accepted || after.fills_completed != before.fills_completed ||
                    after.fill_parts_completed != before.fill_parts_completed || after.fill_bytes_completed != before.fill_bytes_completed ||
                    after.fill_outstanding != 0) return false;
            const auto & bad = f.cache_owner.reserved_fill.parts[0];
            const size_t output_bytes = size_t(K) * sizeof(float);
            std::printf("malformed_range corruption=%d dest=%p bytes=%zu semantic=%p consumer=%p overlap_semantic=%d overlap_consumer=%d expected_bank_slot=%p\n",
                    corruption, bad.destination, bad.bytes, f.semantic->data, f.semantic_consumer->data,
                    byte_ranges_overlap(bad.destination, bad.bytes, f.semantic->data, output_bytes),
                    byte_ranges_overlap(bad.destination, bad.bytes, f.semantic_consumer->data, output_bytes),
                    (uint8_t *) f.binding.layout.bank_up->data + size_t(SLOTS - 1) * f.binding.layout.bank_up->nb[2]);
        }
        f.cache_owner.corrupt_fill = 0;
        std::printf("fill_descriptor_validation malformed_bytes_source_destination=3 exact_rollbacks=3 queue_publications=0 transport_parts=0\n");
        return true;
    };
    if (mode == "scheduler-hit" || mode == "scheduler-miss" || mode == "malformed-scheduler") {
        const bool malformed_ok = mode != "malformed-scheduler" || run_malformed_cases();
        const bool okay = malformed_ok && run_scheduler_order_case(f, route_ids, route_weights, mode == "scheduler-hit");
        destroy_fixture(f);
        return okay ? 0 : 1;
    }
    if (mode == "scheduler-before-malformed") {
        bool prefix_ok = true;
        for (int i = 0; prefix_ok && i < 4; ++i) {
            metrics cuda_error, cpu_error;
            prefix_ok = run_case(f, i, route_ids[i], route_weights[i], make_vector(K, i + 1),
                    i == 0 || i == 2, cuda_error, cpu_error);
        }
        const bool okay = prefix_ok && run_scheduler_order_case(f, route_ids, route_weights);
        destroy_fixture(f);
        return okay ? 0 : 1;
    }
    if (mode != "full") {
        std::fprintf(stderr, "unknown fixture mode: %s\n", mode.c_str());
        destroy_fixture(f);
        return 2;
    }
    double worst_cuda = 0.0;
    double worst_cpu = 0.0;
    for (int i = 0; i < 4; ++i) {
        metrics cuda_error, cpu_error;
        if (!run_case(f, i, route_ids[i], route_weights[i], make_vector(K, i + 1), i == 0 || i == 2, cuda_error, cpu_error)) {
            std::fprintf(stderr, "semantic/reference mismatch in transition %d\n", i);
            destroy_fixture(f);
            return 1;
        }
        worst_cuda = std::max(worst_cuda, cuda_error.max_abs);
        worst_cpu = std::max(worst_cpu, cpu_error.max_abs);
    }

    ggml_cuda_moe_ffn_stats stats{};
    if (!ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &stats) ||
        stats.all_hit_calls != 2 || stats.any_miss_calls != 2 || stats.cpu_calls != 2 ||
        stats.cuda_calls != 4 || stats.gpu_hit_draws != 38 || stats.cpu_miss_draws != 2 ||
        stats.cpu_expert_calls != 2 || stats.partitioned_calls != 2 ||
        stats.complete_cpu_fallback_calls != 0 || stats.capture_bypasses != 4 || stats.output_commits != 4 ||
         stats.selected_ids != 40 || stats.id_readback_bytes != 160 || stats.slot_map_upload_bytes != 152 ||
        stats.input_copy_bytes != 2128 || stats.output_copy_bytes != 6144 ||
        stats.cuda_copy_calls != 20 || stats.cpu_copy_calls != 6 || stats.synchronizations != 6 ||
        stats.full_expert_weight_copy_bytes != 0) {
        std::fprintf(stderr, "execution counter invariant failed: hit=%llu miss=%llu cpu=%llu gpu=%llu ids=%llu idbytes=%llu slotbytes=%llu input=%llu output=%llu copies=%llu cpucopies=%llu sync=%llu bypass=%llu commits=%llu\n",
                (unsigned long long)stats.all_hit_calls, (unsigned long long)stats.any_miss_calls,
                (unsigned long long)stats.cpu_calls, (unsigned long long)stats.cuda_calls,
                (unsigned long long)stats.selected_ids, (unsigned long long)stats.id_readback_bytes,
                (unsigned long long)stats.slot_map_upload_bytes, (unsigned long long)stats.input_copy_bytes,
                (unsigned long long)stats.output_copy_bytes, (unsigned long long)stats.cuda_copy_calls,
                (unsigned long long)stats.cpu_copy_calls, (unsigned long long)stats.synchronizations,
                (unsigned long long)stats.capture_bypasses, (unsigned long long)stats.output_commits);
        destroy_fixture(f);
        return 1;
    }
    const ggml_cuda_moe_ffn_stats transition_stats = stats;

    ggml_cuda_moe_ffn_cpu_resource_stats fill_before{}, fill_after{};
    if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &fill_before)) {
        destroy_fixture(f);
        return 1;
    }
    if (!run_malformed_cases()) {
        std::fprintf(stderr, "malformed fill descriptor case failed\n");
        destroy_fixture(f);
        return 1;
    }

    set_tensor(f.ids, route_ids[1].data());
    set_tensor(f.routes, route_weights[1].data());
    set_tensor(f.seed, make_vector(K, 31).data());
    f.cache_owner.corrupt_resolution = true;
    const bool rejected_before_dispatch = ggml_backend_graph_compute(f.cuda, f.semantic_graph) == GGML_STATUS_FAILED;
    f.cache_owner.corrupt_resolution = false;
    if (!rejected_before_dispatch ||
            !std::all_of(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), [](uint32_t count) { return count == 0; })) {
        std::fprintf(stderr, "pre-dispatch transfer failure leaked current cache pins\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("pre_dispatch_failure=1 immediate_pin_release=1\n");

    ggml_cuda_moe_ffn_stats before_dispatch_failure{};
    if (!ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &before_dispatch_failure) ||
            !ggml_backend_cuda_moe_ffn_test_fail_next_dispatch_after_start(f.cuda, f.binding_id)) {
        std::fprintf(stderr, "could not arm post-dispatch fallback contract test\n");
        destroy_fixture(f);
        return 1;
    }
    metrics dispatched_failure_cuda_error, dispatched_failure_cpu_error;
    ggml_cuda_moe_ffn_stats after_dispatch_failure{};
    if (!run_case(f, 4, route_ids[0], route_weights[0], make_vector(K, 32), true,
                dispatched_failure_cuda_error, dispatched_failure_cpu_error, true) ||
            !ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &after_dispatch_failure) ||
            after_dispatch_failure.complete_cpu_fallback_calls != before_dispatch_failure.complete_cpu_fallback_calls + 1 ||
            std::count(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), 1u) != SELECTED) {
        std::fprintf(stderr, "post-dispatch fallback did not retain the submitted hit-read pins\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("post_dispatch_failure_recovery=complete_cpu_output pending_pin_tokens=%d\n", SELECTED);

    set_tensor(f.ids, route_ids[0].data());
    set_tensor(f.routes, route_weights[0].data());
    set_tensor(f.seed, make_vector(K, 32).data());
    metrics pending_cuda_error, pending_cpu_error;
    if (!run_case(f, 6, route_ids[0], route_weights[0], make_vector(K, 33), true,
                pending_cuda_error, pending_cpu_error)) {
        std::fprintf(stderr, "pending-record setup execution failed\n");
        destroy_fixture(f);
        return 1;
    }
    const uint64_t releases_before_rejection = f.cache_owner.releases;
    f.cache_owner.reject_release = true;
    const bool retained_after_release_rejection =
            ggml_backend_graph_compute(f.cuda, f.semantic_graph) == GGML_STATUS_FAILED &&
            std::count(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), 1u) == SELECTED;
    f.cache_owner.reject_release = false;
    if (!retained_after_release_rejection || f.cache_owner.releases != releases_before_rejection) {
        std::fprintf(stderr, "rejected pending release discarded or changed the active pin record\n");
        destroy_fixture(f);
        return 1;
    }
    metrics retry_cuda_error, retry_cpu_error;
    if (!run_case(f, 5, route_ids[0], route_weights[0], make_vector(K, 33), true,
                retry_cuda_error, retry_cpu_error) || f.cache_owner.releases != releases_before_rejection + 1) {
        std::fprintf(stderr, "successful re-entry did not release the retained record exactly once\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("pending_release_rejection=1 retained_pins=%d retry_released_once=1\n", SELECTED);

    if (!run_scheduler_order_case(f, route_ids, route_weights)) {
        std::fprintf(stderr, "scheduler CPU-input -> semantic CUDA -> CUDA-consumer contract failed\n");
        destroy_fixture(f);
        return 1;
    }

    fixture same_device;
    if (!create_fixture(same_device, device, &f) || same_device.cpu_resource_id != f.cpu_resource_id ||
        same_device.binding_id == f.binding_id) {
        std::fprintf(stderr, "same-device shared CPU resource setup failed\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    ggml_cuda_moe_ffn_cpu_resource_stats shared_stats{};
    if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &shared_stats) ||
        shared_stats.active_bindings != 2) {
        std::fprintf(stderr, "shared CPU resource did not report both layer bindings\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    const uint64_t cpu_calls_before_shared_sequence = shared_stats.cpu_calls;
    metrics shared_hit_error, shared_cpu_error;
    if (!run_case(same_device, 0, route_ids[0], route_weights[0], make_vector(K, 40), true,
                shared_hit_error, shared_cpu_error)) {
        std::fprintf(stderr, "second binding all-hit execution failed\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    if (!std::all_of(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), [](uint32_t count) { return count == 0; })) {
        std::fprintf(stderr, "next binding entry did not release the prior binding's pin record\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    if (std::count(same_device.cache_owner.pins.begin(), same_device.cache_owner.pins.end(), 1u) != SELECTED) {
        std::fprintf(stderr, "binding entry did not install its own pending pins\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    const auto second_binding_output = get_f32(same_device.semantic);
    if (!run_case(f, 0, route_ids[0], route_weights[0], make_vector(K, 40), true,
                shared_hit_error, shared_cpu_error) || get_f32(f.semantic) == second_binding_output ||
        !std::all_of(same_device.cache_owner.pins.begin(), same_device.cache_owner.pins.end(), [](uint32_t count) { return count == 0; }) ||
        !run_case(same_device, 1, route_ids[1], route_weights[1], make_vector(K, 41), false,
                shared_hit_error, shared_cpu_error) ||
        !std::all_of(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), [](uint32_t count) { return count == 0; }) ||
        !ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &shared_stats) ||
        shared_stats.cpu_calls != cpu_calls_before_shared_sequence + 1 ||
        shared_stats.max_parallel_calls != 1 || shared_stats.active_bindings != 2) {
        std::fprintf(stderr, "shared resource did not serialize distinct binding work correctly: before=%llu after=%llu parallel=%llu bindings=%llu\n",
                (unsigned long long) cpu_calls_before_shared_sequence, (unsigned long long) shared_stats.cpu_calls,
                (unsigned long long) shared_stats.max_parallel_calls, (unsigned long long) shared_stats.active_bindings);
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    std::printf("shared_resource=device:%d id:%d active_bindings:%llu cpu_calls:%llu max_parallel:%llu\n",
            device, f.cpu_resource_id, (unsigned long long) shared_stats.active_bindings,
            (unsigned long long) shared_stats.cpu_calls, (unsigned long long) shared_stats.max_parallel_calls);
    const bool parent_rejected_before_stop = !ggml_backend_cuda_moe_ffn_unregister(f.cuda, f.binding_id);
    const bool sibling_rejected_before_stop = !ggml_backend_cuda_moe_ffn_unregister(f.cuda, same_device.binding_id);
    metrics still_live_parent_cuda, still_live_parent_cpu, still_live_sibling_cuda, still_live_sibling_cpu;
    if (!parent_rejected_before_stop || !sibling_rejected_before_stop ||
            !ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &stats) ||
            !run_case(same_device, 0, route_ids[0], route_weights[0], make_vector(K, 42), true,
                    still_live_sibling_cuda, still_live_sibling_cpu) ||
            !run_case(f, 0, route_ids[0], route_weights[0], make_vector(K, 43), true,
                    still_live_parent_cuda, still_live_parent_cpu) ||
            !ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &shared_stats) ||
            shared_stats.active_bindings != 2) {
        std::fprintf(stderr, "binding unregister before shared transport stop changed or invalidated a binding\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    std::printf("shared_transport_unregister_before_stop=rejected both_bindings_usable=1 active_bindings=%llu\n",
            (unsigned long long) shared_stats.active_bindings);
    if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &shared_stats) ||
        shared_stats.active_bindings != 2 ||
        !run_case(f, 0, route_ids[0], route_weights[0], make_vector(K, 42), true,
                shared_hit_error, shared_cpu_error)) {
        std::fprintf(stderr, "shared resource became unusable after rejected unregister\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("shared_resource_after_unregister=device:%d active_bindings:%llu cpu_calls:%llu\n",
            device, (unsigned long long) shared_stats.active_bindings,
            (unsigned long long) shared_stats.cpu_calls);

    auto * op_params = (ggml_moe_ffn_op_params *) f.semantic->op_params;
    const ggml_moe_ffn_op_params valid_params = *op_params;
    const std::vector<float> sentinel(K, 17.0f);
    auto expect_preflight_failure = [&](ggml_moe_ffn_op_params invalid, const char * label) {
        *op_params = invalid;
        ggml_backend_tensor_set(f.semantic, sentinel.data(), 0, sentinel.size() * sizeof(float));
        const bool failed = ggml_backend_graph_compute(f.cuda, f.semantic_graph) == GGML_STATUS_FAILED;
        const auto after = get_f32(f.semantic);
        *op_params = valid_params;
        if (!failed || after != sentinel) {
            std::fprintf(stderr, "%s did not reject before output commit\n", label);
            return false;
        }
        return true;
    };
    auto unknown = valid_params;
    unknown.binding_id += 100000;
    if (!expect_preflight_failure(unknown, "unknown binding")) { destroy_fixture(f); return 1; }
    auto stale = valid_params;
    stale.generation += 1;
    if (!expect_preflight_failure(stale, "stale generation")) { destroy_fixture(f); return 1; }

    auto same_low_bits_wrong_generation = valid_params;
    same_low_bits_wrong_generation.generation += (UINT64_C(1) << 32);
    if ((uint32_t) same_low_bits_wrong_generation.generation != (uint32_t) valid_params.generation ||
            !query_support_without_allocating_output(f, f.device, f.binding_id, f.generation) ||
            query_support_without_allocating_output(f, f.device, f.binding_id,
                    same_low_bits_wrong_generation.generation) ||
            !expect_preflight_failure(same_low_bits_wrong_generation, "generation mismatch above UINT32_MAX")) {
        std::fprintf(stderr, "generation lookup truncated bits above UINT32_MAX\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("generation_width exact=%llu rejected_same_low32=%llu\n",
            (unsigned long long) f.generation,
            (unsigned long long) same_low_bits_wrong_generation.generation);

    std::vector<int32_t> invalid_expert = route_ids[0];
    invalid_expert[2] = EXPERTS;
    const auto valid_routes = route_weights[0];
    const std::vector<float> invalid_sentinel(K, 29.0f);
    ggml_backend_tensor_set(f.ids, invalid_expert.data(), 0, sizeof(int32_t) * SELECTED);
    ggml_backend_tensor_set(f.routes, valid_routes.data(), 0, sizeof(float) * SELECTED);
    ggml_backend_tensor_set(f.seed, make_vector(K, 9).data(), 0, sizeof(float) * K);
    ggml_backend_tensor_set(f.semantic, invalid_sentinel.data(), 0, sizeof(float) * K);
    const bool invalid_rejected = ggml_backend_graph_compute(f.cuda, f.semantic_graph) == GGML_STATUS_FAILED;
    const bool invalid_uncommitted = get_f32(f.semantic) == invalid_sentinel;
    if (!invalid_rejected || !invalid_uncommitted) {
        std::fprintf(stderr, "invalid selected expert was not rejected before output commit\n");
        destroy_fixture(f);
        return 1;
    }

    auto wrong_device = f.binding;
    wrong_device.layout.device = device == 0 ? 1 : 0;
    ggml_tensor fake_down = *f.host_down;
    fake_down.type = GGML_TYPE_F16;
    auto unsupported = f.binding;
    unsupported.layout.down = &fake_down;
    auto non_host = f.binding;
    non_host.layout.up = f.binding.layout.bank_up;
    if (ggml_backend_cuda_moe_ffn_register(f.cuda, &wrong_device) >= 0 ||
        ggml_backend_cuda_moe_ffn_register(f.cuda, &unsupported) >= 0 ||
        ggml_backend_cuda_moe_ffn_register(f.cuda, &non_host) >= 0) {
        std::fprintf(stderr, "invalid device/type/host binding was accepted\n");
        destroy_fixture(f);
        return 1;
    }

    if (ggml_backend_cuda_get_device_count() > 1) {
        const int other_device = device == 0 ? 1 : 0;
        fixture other;
        if (!create_fixture(other, other_device)) {
            std::fprintf(stderr, "second device fixture registration failed\n");
            destroy_fixture(other);
            destroy_fixture(f);
            return 1;
        }
        if (other.binding_id != f.binding_id || other.generation != f.generation ||
            !query_support_without_allocating_output(f, f.device, f.binding_id, f.generation) ||
            !query_support_without_allocating_output(other, other.device, other.binding_id, other.generation) ||
            query_support_without_allocating_output(f, other.device, other.binding_id, other.generation) ||
            query_support_without_allocating_output(f, f.device, f.binding_id + 100000, f.generation) ||
            query_support_without_allocating_output(f, f.device, f.binding_id, f.generation + 1)) {
            std::fprintf(stderr, "owner-aware supports_op or colliding binding IDs failed\n");
            destroy_fixture(other);
            destroy_fixture(f);
            return 1;
        }

        metrics other_cuda_error, other_cpu_error;
        if (!run_case(other, 0, route_ids[0], route_weights[0], make_vector(K, 12), true,
                other_cuda_error, other_cpu_error)) {
            std::fprintf(stderr, "second owner/device pair failed execution\n");
            destroy_fixture(other);
            destroy_fixture(f);
            return 1;
        }

        ggml_context * other_ctx = make_context(2 * 1024 * 1024, true);
        ggml_tensor * other_activation = ggml_new_tensor_2d(other_ctx, GGML_TYPE_F32, K, 1);
        ggml_tensor * other_ids = ggml_new_tensor_2d(other_ctx, GGML_TYPE_I32, SELECTED, 1);
        ggml_tensor * other_routes = ggml_new_tensor_3d(other_ctx, GGML_TYPE_F32, 1, SELECTED, 1);
        ggml_tensor * foreign = ggml_moe_ffn(other_ctx, other_activation, other_ids, other_routes,
                f.device, f.binding_id, f.generation, K, K, GGML_UNARY_OP_SILU, 0.0f);
        ggml_cgraph * other_graph = ggml_new_graph_custom(other_ctx, 8, false);
        ggml_build_forward_expand(other_graph, foreign);
        ggml_backend_buffer_t other_buffer = ggml_backend_alloc_ctx_tensors(other_ctx, other.cuda);
        std::vector<float> other_sentinel(K, 23.0f);
        ggml_backend_tensor_set(foreign, other_sentinel.data(), 0, other_sentinel.size() * sizeof(float));
        const bool other_support_rejected = !ggml_backend_supports_op(other.cuda, foreign);
        const bool foreign_rejected = ggml_backend_graph_compute(other.cuda, other_graph) == GGML_STATUS_FAILED;
        const bool foreign_unchanged = get_f32(foreign) == other_sentinel;
        ggml_backend_buffer_free(other_buffer);
        ggml_free(other_ctx);
        if (!other_support_rejected || !foreign_rejected || !foreign_unchanged) {
            std::fprintf(stderr, "binding ID resolved on the wrong CUDA device\n");
            destroy_fixture(other);
            destroy_fixture(f);
            return 1;
        }
        destroy_fixture(other);
    }

    metrics reuse_cuda_error, reuse_cpu_error;
    if (!run_case(f, 3, route_ids[0], route_weights[0], make_vector(K, 4), true,
                reuse_cuda_error, reuse_cpu_error)) {
        std::fprintf(stderr, "existing binding failed with reused graph/output\n");
        destroy_fixture(f);
        return 1;
    }
    if (!ggml_backend_cuda_moe_ffn_get_stats(f.cuda, f.binding_id, &stats) ||
        stats.output_commits == 0 || stats.full_expert_weight_copy_bytes != 0) {
        std::fprintf(stderr, "binding output/copy counter invariant failed\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("device=%d worst_cuda_ref_max=%.8g worst_cpu_cuda_max=%.8g ids=%llu id_readback=%llu slot_upload=%llu input_copy=%llu output_copy=%llu cuda_copy_calls=%llu cpu_copy_calls=%llu sync=%llu cuda=%llu cpu=%llu capture_bypass=%llu commits=%llu expert_weight_copy_bytes=%llu\n",
            device, worst_cuda, worst_cpu, (unsigned long long) transition_stats.selected_ids,
            (unsigned long long) transition_stats.id_readback_bytes, (unsigned long long) transition_stats.slot_map_upload_bytes,
            (unsigned long long) transition_stats.input_copy_bytes, (unsigned long long) transition_stats.output_copy_bytes,
            (unsigned long long) transition_stats.cuda_copy_calls, (unsigned long long) transition_stats.cpu_copy_calls,
            (unsigned long long) transition_stats.synchronizations,
            (unsigned long long) transition_stats.cuda_calls, (unsigned long long) transition_stats.cpu_calls,
            (unsigned long long) transition_stats.capture_bypasses, (unsigned long long) transition_stats.output_commits,
            (unsigned long long) stats.full_expert_weight_copy_bytes);
    ggml_cuda_moe_ffn_cpu_resource_stats pin_stats{};
    if (!ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &pin_stats) ||
            pin_stats.pin_resolve_calls == 0 || pin_stats.pin_acquisitions < pin_stats.pin_releases ||
            pin_stats.pin_release_failures != 1 || pin_stats.pending_pin_releases == 0 ||
            pin_stats.pending_pin_active != 1) {
        std::fprintf(stderr, "resource-scoped pending pin counters are inconsistent\n");
        destroy_fixture(f);
        return 1;
    }
    std::printf("pin_lifecycle resolves=%llu acquired=%llu released=%llu pending_releases=%llu failures=%llu active_before_teardown=%llu\n",
            (unsigned long long) pin_stats.pin_resolve_calls,
            (unsigned long long) pin_stats.pin_acquisitions,
            (unsigned long long) pin_stats.pin_releases,
            (unsigned long long) pin_stats.pending_pin_releases,
            (unsigned long long) pin_stats.pin_release_failures,
            (unsigned long long) pin_stats.pending_pin_active);
    if (!ggml_backend_cuda_moe_ffn_cpu_resource_stop(f.cuda, f.cpu_resource_id) ||
            !ggml_backend_cuda_moe_ffn_cpu_resource_stop(f.cuda, f.cpu_resource_id) ||
            !ggml_backend_cuda_moe_ffn_unregister(f.cuda, same_device.binding_id) ||
            !ggml_backend_cuda_moe_ffn_unregister(f.cuda, f.binding_id)) {
        std::fprintf(stderr, "shared transport stop did not establish one unregister boundary for all bindings\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    const bool unregister_once = !ggml_backend_cuda_moe_ffn_unregister(f.cuda, f.binding_id) &&
            !ggml_backend_cuda_moe_ffn_unregister(f.cuda, same_device.binding_id);
    if (!unregister_once ||
            !ggml_backend_cuda_moe_ffn_cpu_resource_get_stats(f.cuda, f.cpu_resource_id, &shared_stats) ||
            shared_stats.active_bindings != 0) {
        std::fprintf(stderr, "shared binding unregister was not exactly once after transport stop\n");
        destroy_fixture(same_device);
        destroy_fixture(f);
        return 1;
    }
    same_device.binding_id = -1;
    f.binding_id = -1;
    std::printf("shared_transport_stop_once=1 bindings_unregistered=2 repeated_binding_unregister_rejected=1\n");
    destroy_fixture(same_device);
    destroy_fixture(f);
    if (!std::all_of(f.cache_owner.pins.begin(), f.cache_owner.pins.end(), [](uint32_t count) { return count == 0; })) {
        std::fprintf(stderr, "teardown did not release the final pending pins\n");
        return 1;
    }
    std::printf("teardown_final_pending_release=1 releases=%llu\n", (unsigned long long) f.cache_owner.releases);
    return 0;
}
