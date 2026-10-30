#include "llama.h"

#include <random>

#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-moe-ffn-prototype.h"
#include "ggml-moe-ffn.h"
#include "ggml-cuda-moe-ffn.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

constexpr size_t CACHE_BYTES = 8ull * 1024 * 1024 * 1024;
constexpr int N_SELECTED = 10;
constexpr int N_SCORED = 32;
constexpr int N_PROMPT = 16;

const char * const CORPUS = R"corpus(When the small observatory reopened after the storm, the volunteers found the brass clock still keeping careful time. They swept rainwater from the steps, made tea, and wrote down what the clouds had hidden overnight. A useful system is not merely fast on a quiet day; it should also explain its choices, recover from interruption, and leave enough evidence for the next person to continue the work.

The maintenance script is deliberately plain:
```python
def checked_copy(source, destination):
    if source is None or not source.is_readable():
        raise ValueError("source is unavailable")
    destination.write(source.read_bytes())
    return destination.stat().st_size
```
The operator runs the check before publishing the new file, then records its checksum and the exact command that produced it. Small explicit steps make failures easier to locate than a clever shortcut.

Em uma manhã tranquila, a equipe revisou os resultados e comparou cada etapa com a referência. A pergunta não era se duas máquinas produziam números idênticos, mas se a diferença permanecia compreensível e estável. 研究者はログを確認し、必要なデータだけを保存した。После этого они отправили короткое сообщение: всё готово, проверка завершена, и следующий запуск может начинаться с чистого состояния.

The final note is intentionally ordinary. It says that measurements belong to their inputs, that missing observations must not be silently invented, and that a future maintainer should be able to reproduce the result without guessing. The team closed the notebook, left the original files untouched, and went home before the rain returned.)corpus";

bool depends_on(const ggml_tensor * node, const ggml_tensor * target,
        std::unordered_set<const ggml_tensor *> & seen) {
    if (!node) return false;
    if (node == target) return true;
    if (!seen.insert(node).second) return false;
    if (depends_on(node->view_src, target, seen)) return true;
    for (const ggml_tensor * src : node->src) {
        if (depends_on(src, target, seen)) return true;
    }
    return false;
}

bool node_count_and_contract(llama_context & ctx, const llama_batch & batch, int expected,
        llama_moe_ffn_prototype_state * proto, bool check_contract) {
    ggml_cgraph * graph = llm_build_context::llama_build_graph(ctx, batch, false, 0);
    if (!graph) return false;
    ggml_tensor * semantic = nullptr;
    int count = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i]->op == GGML_OP_MOE_FFN) {
            semantic = graph->nodes[i];
            ++count;
        }
    }
    if (count != expected) {
        std::fprintf(stderr, "semantic graph nodes: got=%d expected=%d tokens=%d\n", count, expected, batch.n_tokens);
        return false;
    }
    if (!check_contract) return semantic == nullptr;
    if (!semantic || !proto) return false;
    std::unordered_set<const ggml_tensor *> closure;
    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * op = graph->nodes[i];
        if (op->op != GGML_OP_MOE_FFN) continue;
        if (!op->src[0] || !op->src[1] || !op->src[2]) return false;
        ggml_moe_ffn_op_params params{};
        std::memcpy(&params, op->op_params, sizeof(params));
        bool assigned_to_owner = false;
        for (const auto & device : proto->devices) for (const auto & layer : device.layer_bindings) {
            if (layer.binding_id == params.binding_id && layer.snapshot.generation == params.generation &&
                    layer.snapshot.device_group == params.device &&
                    ggml_backend_sched_get_tensor_backend(ctx.sched, op) == layer.cuda_backend) assigned_to_owner = true;
        }
        if (!assigned_to_owner) return false;
        for (const auto & device : proto->devices) for (const auto & layer : device.layer_bindings) {
            const ggml_tensor * layer_weights[] = {
                layer.snapshot.host_up, layer.snapshot.host_gate, layer.snapshot.host_down,
                layer.snapshot.cached_up, layer.snapshot.cached_gate, layer.snapshot.cached_down,
            };
            for (const ggml_tensor * src : op->src) {
                std::vector<const ggml_tensor *> stack{src};
                while (!stack.empty()) {
                    const ggml_tensor * current = stack.back();
                    stack.pop_back();
                    if (!current || !closure.insert(current).second) continue;
                    if (std::find(std::begin(layer_weights), std::end(layer_weights), current) != std::end(layer_weights)) {
                        std::fprintf(stderr, "expert tensor leaked into semantic source closure: %s\n", current->name);
                        return false;
                    }
                    if (current->view_src) stack.push_back(current->view_src);
                    for (const ggml_tensor * dep : current->src) if (dep) stack.push_back(dep);
                }
            }
        }
    }

    bool has_consumer = false;
    for (int i = 0; i < graph->n_nodes; ++i) {
        if (graph->nodes[i] == semantic) continue;
        std::unordered_set<const ggml_tensor *> seen;
        if (depends_on(graph->nodes[i], semantic, seen)) has_consumer = true;
    }
    return has_consumer;
}

struct run_result {
    std::vector<std::vector<float>> logits;
    std::vector<llama_token> targets;
    ggml_cuda_moe_ffn_stats stats = {};
    ggml_cuda_moe_ffn_cpu_resource_stats resource_stats = {};
    struct device_account { int32_t device = -1; ggml_cuda_moe_ffn_stats stats = {}; ggml_cuda_moe_ffn_cpu_resource_stats resource = {}; };
    std::vector<device_account> devices;
    struct window { int begin = 0, end = 0; uint64_t gpu = 0, cpu = 0, fills = 0, done = 0, rejected = 0, failed = 0, bytes = 0; };
    std::vector<window> windows;
};

bool logits_finite(llama_context * ctx, llama_model * model, std::vector<float> & out) {
    const float * logits = llama_get_logits(ctx);
    if (!logits) return false;
    const size_t n = (size_t) llama_n_vocab(model);
    out.assign(logits, logits + n);
    return std::all_of(out.begin(), out.end(), [](float value) { return std::isfinite(value); });
}

bool evaluate_sequence(llama_context * ctx, llama_model * model, const std::vector<llama_token> & tokens,
        run_result & result, llama_moe_ffn_prototype_state * proto) {
    if (tokens.size() < N_PROMPT + N_SCORED + 1) return false;
    std::vector<llama_token> prompt(tokens.begin(), tokens.begin() + N_PROMPT);
    llama_batch prompt_batch = llama_batch_get_one(prompt.data(), (int32_t) prompt.size(), 0, 0);
    if (llama_decode(ctx, prompt_batch) != 0) return false;
    ggml_cuda_moe_ffn_stats prev_s{};
    ggml_cuda_moe_ffn_cpu_resource_stats prev_r{};
    bool first_window = true;
    for (int32_t i = 0; i < N_SCORED; ++i) {
        llama_token token = tokens[N_PROMPT + i];
        llama_batch next = llama_batch_get_one(&token, 1, N_PROMPT + i, 0);
        if (i == 0 && !node_count_and_contract(*ctx, next,
                    proto ? (int) proto->layer_count() : 0, proto, proto != nullptr)) return false;
        if (llama_decode(ctx, next) != 0) return false;
        std::vector<float> row;
        if (!logits_finite(ctx, model, row)) return false;
        result.logits.push_back(std::move(row));
        result.targets.push_back(tokens[N_PROMPT + i + 1]);
        if (proto && (i + 1) % 16 == 0) {
            ggml_cuda_moe_ffn_stats s{};
            ggml_cuda_moe_ffn_cpu_resource_stats r{};
            if (!proto->collect_stats(s) || !proto->collect_resource_stats(r)) return false;
            static ggml_cuda_moe_ffn_stats prev_s{};
            static ggml_cuda_moe_ffn_cpu_resource_stats prev_r{};
            auto delta = [](uint64_t a, uint64_t b) { return a - b; };
            run_result::window w;
            w.begin = i - 15; w.end = i + 1;
            w.gpu = delta(s.gpu_hit_draws, first_window ? 0 : prev_s.gpu_hit_draws);
            w.cpu = delta(s.cpu_miss_draws, first_window ? 0 : prev_s.cpu_miss_draws);
            w.fills = delta(r.fills_accepted, first_window ? 0 : prev_r.fills_accepted);
            w.done = delta(r.fills_completed, first_window ? 0 : prev_r.fills_completed);
            w.rejected = delta(r.fills_rejected, first_window ? 0 : prev_r.fills_rejected);
            w.failed = delta(r.fills_failed, first_window ? 0 : prev_r.fills_failed);
            w.bytes = delta(r.fill_bytes_completed, first_window ? 0 : prev_r.fill_bytes_completed);
            result.windows.push_back(w);
            prev_s = s; prev_r = r; first_window = false;
        }
    }
    if (proto && !proto->collect_stats(result.stats)) return false;
    return true;
}

llama_model * load_model(bool dual_device) {
    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = 99;
    params.max_gpu = dual_device ? 2 : 1;
    params.main_gpu = 0;
    params.split_mode = dual_device ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
    static float tensor_split[] = { 1.0f, 1.0f };
    params.tensor_split = dual_device ? tensor_split : nullptr;
    params.ncmoe = 999;
    params.use_mmap = true;
    params.fit = false;
    params.merge_up_gate_exps = false;
    return llama_model_load_from_file(
        "/mnt/ssd_adata/llm_models/qwen38-flash-next-target/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf",
        params);
}

llama_context_params context_params(size_t cache_bytes) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 512;
    params.n_batch = 128;
    params.n_ubatch = 32;
    params.n_threads = 24;
    params.n_threads_batch = 24;
    params.moe_cache_size = cache_bytes;
    return params;
}

bool lifecycle_zero(const llama_moe_ffn_prototype_lifecycle & lifecycle) {
    return lifecycle.live_bindings == 0 && lifecycle.live_cpu_resources == 0 && lifecycle.live_leases == 0 &&
            lifecycle.live_cpu_graphs == 0 && lifecycle.cpu_graph_context_bytes == 0;
}

bool inspect_frozen_group(llama_context & ctx, llama_moe_ffn_prototype_state & proto) {
    if (!ctx.moe_cache || !proto.valid() || proto.devices.empty()) return false;
    std::vector<int32_t> frozen_layers;
    for (const auto & device : proto.devices) for (const auto & group : device.groups) {
        if (group.snapshot.group_layers.size() != group.stale_slot_maps.size()) return false;
        for (size_t i = 0; i < group.snapshot.group_layers.size(); ++i) {
            const int32_t il = group.snapshot.group_layers[i];
            frozen_layers.push_back(il);
            if (ctx.moe_cache->get_slot_map(il, 1, N_SELECTED) != nullptr) return false;
            ggml_tensor * stale = group.stale_slot_maps[i];
            if (!stale || llama_context::sched_copy_moe_cache(device.cuda_backend, stale, nullptr, nullptr,
                        GGML_BACKEND_SCHED_COPY_PHASE_QUERY, &ctx) ||
                    llama_context::sched_copy_moe_cache(device.cuda_backend, stale, nullptr, nullptr,
                        GGML_BACKEND_SCHED_COPY_PHASE_EXECUTE, &ctx)) return false;
        }
    }
    bool unrelated_group_available = false;
    bool has_unfrozen_group = false;
    for (int32_t il = 0; il < (int32_t) ctx.model.layers.size(); ++il) {
        if (ctx.moe_cache->backend(il) &&
                std::find(frozen_layers.begin(), frozen_layers.end(), il) == frozen_layers.end()) {
            has_unfrozen_group = true;
            if (!ctx.moe_cache->get_slot_map(il, 1, N_SELECTED)) return false;
            unrelated_group_available = true;
        }
    }
    return (!has_unfrozen_group || unrelated_group_available) && proto.valid();
}

bool create_context(llama_model * model, bool prototype, size_t cache_bytes, const std::vector<int32_t> & seeds,
        llama_moe_ffn_prototype_lifecycle & lifecycle, llama_context ** out, bool dual_device = false,
        bool poison_priming_inputs = false) {
    llama_context_params params = context_params(cache_bytes);
    if (!prototype) {
        *out = llama_init_from_model(model, params);
        return *out != nullptr;
    }
    llama_moe_ffn_prototype_config config;
    config.layer = 0;
    config.device = 0;
    config.cpu_threads = dual_device ? 12 : 24;
    config.all_devices = dual_device;
    config.test_poison_cpu_priming_inputs = poison_priming_inputs;
    config.seed_experts = seeds;
    config.lifecycle = &lifecycle;
    *out = llama_init_from_model_with_moe_ffn_prototype(model, params, config);
    return *out != nullptr;
}

struct arm_summary { double mean_nll = 0, ppl = 0; };
struct row_delta { double nll = 0, kl = 0, prob_rms = 0, max_logit = 0; int top1 = 0, top10 = 0; };
struct pair_summary { double nll_mean = 0, nll_rms = 0, nll_max_abs = 0, p50 = 0, p90 = 0, p95 = 0;
    double kl = 0, prob_rms = 0, max_logit = 0, top1 = 0, top10 = 0; std::vector<row_delta> rows; };

double row_nll(const std::vector<float> & logits, llama_token target) {
    const double mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    for (float x : logits) sum += std::exp((double) x - mx);
    return mx + std::log(sum) - logits[(size_t) target];
}

std::vector<size_t> top_indices(const std::vector<float> & v, size_t n) {
    std::vector<size_t> ids(v.size());
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + n, ids.end(), [&](size_t a, size_t b) { return v[a] > v[b]; });
    ids.resize(n);
    return ids;
}

bool exact_repeat(const char * label, const run_result & a, const run_result & b) {
    if (a.targets != b.targets || a.logits.size() != b.logits.size()) {
        std::printf("repeat_check arm=%s exact=0 reason=shape_or_targets\n", label);
        return false;
    }
    for (size_t i = 0; i < a.logits.size(); ++i) {
        if (a.logits[i].size() != b.logits[i].size()) {
            std::printf("repeat_check arm=%s exact=0 position=%zu reason=vocab_size\n", label, i);
            return false;
        }
        if (std::memcmp(a.logits[i].data(), b.logits[i].data(), a.logits[i].size() * sizeof(float)) != 0) {
            double max_delta = 0;
            for (size_t j = 0; j < a.logits[i].size(); ++j)
                max_delta = std::max(max_delta, std::abs((double) a.logits[i][j] - b.logits[i][j]));
            std::printf("repeat_check arm=%s exact=0 position=%zu target=%d max_logit_abs=%.12g\n",
                    label, i, a.targets[i], max_delta);
            return false;
        }
    }
    std::printf("repeat_check arm=%s exact=1 rows=%zu\n", label, a.logits.size());
    return true;
}

bool summarize_arm(const run_result & run, arm_summary & out) {
    if (run.logits.size() != N_SCORED || run.targets.size() != N_SCORED) return false;
    for (size_t i = 0; i < run.logits.size(); ++i) {
        if (run.logits[i].empty() || run.targets[i] < 0 || (size_t) run.targets[i] >= run.logits[i].size()) return false;
        const double nll = row_nll(run.logits[i], run.targets[i]);
        if (!std::isfinite(nll)) return false;
        out.mean_nll += nll;
    }
    out.mean_nll /= N_SCORED;
    out.ppl = std::exp(out.mean_nll);
    return std::isfinite(out.mean_nll) && std::isfinite(out.ppl);
}

bool compare_pair(const run_result & ref, const run_result & cand, pair_summary & out) {
    if (ref.logits.size() != N_SCORED || cand.logits.size() != N_SCORED || ref.targets != cand.targets) return false;
    std::vector<double> abs_delta;
    double nll_sq = 0, kl_sum = 0, prob_sq = 0, max_logit = 0, top1 = 0, top10 = 0;
    uint64_t prob_n = 0;
    for (size_t r = 0; r < N_SCORED; ++r) {
        const auto & a = ref.logits[r]; const auto & b = cand.logits[r];
        if (a.size() != b.size() || ref.targets[r] < 0 || (size_t) ref.targets[r] >= a.size()) return false;
        const double nll_delta = row_nll(b, ref.targets[r]) - row_nll(a, ref.targets[r]);
        if (!std::isfinite(nll_delta)) return false;
        out.rows.push_back({nll_delta, 0, 0, 0, 0, 0});
        abs_delta.push_back(std::abs(nll_delta)); out.nll_mean += nll_delta; nll_sq += nll_delta * nll_delta;
        const double ma = *std::max_element(a.begin(), a.end()), mb = *std::max_element(b.begin(), b.end());
        double sa = 0, sb = 0;
        for (size_t i = 0; i < a.size(); ++i) { sa += std::exp((double) a[i] - ma); sb += std::exp((double) b[i] - mb); }
        const double za = ma + std::log(sa), zb = mb + std::log(sb);
        double row_kl = 0, row_prob_sq = 0, row_logit_max = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const double lpa = (double) a[i] - za, lpb = (double) b[i] - zb;
            const double pa = std::exp(lpa), pb = std::exp(lpb), d = pa - pb;
            row_kl += pa * (lpa - lpb); row_prob_sq += d * d;
            row_logit_max = std::max(row_logit_max, std::abs((double) a[i] - b[i]));
        }
        const auto ta = top_indices(a, 10), tb = top_indices(b, 10);
        int overlap = 0; for (size_t id : ta) overlap += std::find(tb.begin(), tb.end(), id) != tb.end();
        const int same_top1 = ta[0] == tb[0];
        out.rows.back().kl = row_kl; out.rows.back().prob_rms = std::sqrt(row_prob_sq / a.size());
        out.rows.back().max_logit = row_logit_max; out.rows.back().top1 = same_top1; out.rows.back().top10 = overlap;
        kl_sum += row_kl; prob_sq += row_prob_sq; prob_n += a.size(); max_logit = std::max(max_logit, row_logit_max);
        top1 += same_top1; top10 += overlap / 10.0;
    }
    std::sort(abs_delta.begin(), abs_delta.end());
    auto pct = [&](double p) { return abs_delta[(size_t) std::ceil(p * abs_delta.size()) - 1]; };
    out.nll_mean /= N_SCORED; out.nll_rms = std::sqrt(nll_sq / N_SCORED); out.nll_max_abs = abs_delta.back();
    out.p50 = pct(.50); out.p90 = pct(.90); out.p95 = pct(.95); out.kl = kl_sum / N_SCORED;
    out.prob_rms = std::sqrt(prob_sq / prob_n); out.max_logit = max_logit; out.top1 = top1 / N_SCORED; out.top10 = top10 / N_SCORED;
    return std::isfinite(out.kl) && std::isfinite(out.prob_rms) && std::isfinite(out.max_logit);
}

bool write_run_artifacts(const std::string & prefix, const run_result & run) {
    if (run.logits.size() != N_SCORED || run.targets.size() != N_SCORED) return false;
    const size_t vocab = run.logits.front().size();
    std::ofstream logits(prefix + ".logits.f32", std::ios::binary);
    std::ofstream targets(prefix + ".targets.i32", std::ios::binary);
    if (!logits || !targets) return false;
    for (const auto & row : run.logits) {
        if (row.size() != vocab) return false;
        logits.write(reinterpret_cast<const char *>(row.data()), (std::streamsize) (row.size() * sizeof(float)));
    }
    targets.write(reinterpret_cast<const char *>(run.targets.data()), (std::streamsize) (run.targets.size() * sizeof(llama_token)));
    return logits.good() && targets.good();
}

int first_differing_position(const run_result & a, const run_result & b) {
    if (a.targets != b.targets || a.logits.size() != b.logits.size()) return -1;
    for (size_t i = 0; i < a.logits.size(); ++i) {
        if (a.logits[i].size() != b.logits[i].size() ||
                std::memcmp(a.logits[i].data(), b.logits[i].data(), a.logits[i].size() * sizeof(float)) != 0) return (int) i;
    }
    return -1;
}

struct metric_sample { const char * name; double value; };

std::vector<metric_sample> off_metrics(const pair_summary & p) {
    return {{"nll_delta_mean", p.nll_mean}, {"nll_delta_rms", p.nll_rms}, {"nll_delta_max_abs", p.nll_max_abs},
        {"nll_abs_p50", p.p50}, {"nll_abs_p90", p.p90}, {"nll_abs_p95", p.p95}, {"directed_kl", p.kl},
        {"probability_rms", p.prob_rms}, {"top1_agreement", p.top1}, {"top10_overlap", p.top10},
        {"max_logit_abs", p.max_logit}};
}

void print_metric_spread(const char * name, const std::vector<double> & values) {
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    double sq = 0;
    for (double value : values) sq += (value - mean) * (value - mean);
    const double sd = values.size() > 1 ? std::sqrt(sq / (values.size() - 1)) : 0;
    const auto mm = std::minmax_element(values.begin(), values.end());
    const double cv = std::abs(mean) > 1e-15 ? sd / std::abs(mean) : std::numeric_limits<double>::quiet_NaN();
    std::printf("spread metric=%s mean=%.12g sample_sd=%.12g min=%.12g max=%.12g cv=%.12g\n",
            name, mean, sd, *mm.first, *mm.second, cv);
}

void write_window_csv(const std::string & prefix, const std::vector<run_result> & hybrids) {
    std::ofstream csv(prefix + ".hybrid-windows.csv");
    csv << "run,begin,end,gpu_hit_rows,cpu_miss_rows,fills_accepted,fills_completed,fills_rejected,fills_failed,transport_bytes\n";
    for (size_t r = 0; r < hybrids.size(); ++r) for (const auto & w : hybrids[r].windows)
        csv << r + 1 << ',' << w.begin << ',' << w.end << ',' << w.gpu << ',' << w.cpu << ',' << w.fills << ','
            << w.done << ',' << w.rejected << ',' << w.failed << ',' << w.bytes << '\n';
}

bool run_ordinary(llama_model * model, const std::vector<llama_token> & tokens, size_t cache_bytes,
        const char * label, run_result & result, bool dual_device = false) {
    llama_moe_ffn_prototype_lifecycle lifecycle; llama_context * ctx = nullptr;
    if (!create_context(model, false, cache_bytes, {}, lifecycle, &ctx, dual_device)) return false;
    const bool cache_path = cache_bytes == 0 ? (!ctx->moe_cache || !ctx->moe_cache->get_slot_map(0, 1, N_SELECTED)) :
            (ctx->moe_cache && ctx->moe_cache->get_slot_map(0, 1, N_SELECTED));
    std::vector<llama_token> prompt(tokens.begin(), tokens.begin() + N_PROMPT);
    const bool prompt_path = node_count_and_contract(*ctx, llama_batch_get_one(prompt.data(), N_PROMPT, 0, 0), 0, nullptr, false);
    const bool ok = cache_path && !ctx->moe_ffn_prototype && prompt_path && evaluate_sequence(ctx, model, tokens, result, nullptr);
    std::printf("arm=%s cache_bytes=%zu semantic_ops=0 prompt_semantic_nodes=0 rows=%zu path_ok=%d\n", label, cache_bytes, result.logits.size(), ok);
    llama_free(ctx);
    return ok && lifecycle_zero(lifecycle);
}

bool run_hybrid(llama_model * model, const std::vector<llama_token> & tokens, const char * label, run_result & result,
        bool dual_device = false) {
    llama_moe_ffn_prototype_lifecycle lifecycle; llama_context * ctx = nullptr;
    if (!create_context(model, true, CACHE_BYTES, {}, lifecycle, &ctx, dual_device)) return false;
    auto & proto = *ctx->moe_ffn_prototype;
    const size_t n_eligible = proto.layer_count();
    const size_t n_groups = std::accumulate(proto.devices.begin(), proto.devices.end(), size_t(0),
            [](size_t n, const auto & d) { return n + d.groups.size(); });
    if (!proto.valid() || !inspect_frozen_group(*ctx, proto) || n_eligible < 2 ||
            proto.devices.size() != (dual_device ? 2u : 1u) ||
            lifecycle.live_leases != (int) n_groups || lifecycle.live_cpu_resources != (int) proto.devices.size() ||
            lifecycle.live_bindings != (int) n_eligible || lifecycle.live_cpu_graphs != (int) n_eligible) {
        llama_free(ctx); return false;
    }
    std::vector<llama_token> prompt(tokens.begin(), tokens.begin() + N_PROMPT);
    bool ok = node_count_and_contract(*ctx, llama_batch_get_one(prompt.data(), N_PROMPT, 0, 0), 0, &proto, false) &&
            evaluate_sequence(ctx, model, tokens, result, &proto);
    ok = ok && proto.valid() && inspect_frozen_group(*ctx, proto);
    for (const auto & device : proto.devices)
        ok = ok && ggml_backend_cuda_moe_ffn_cpu_resource_stop(device.cuda_backend, device.cpu_resource_id);
    ok = ok && proto.collect_stats(result.stats);
    ggml_cuda_moe_ffn_cpu_resource_stats rs{};
    const bool have_rs = proto.collect_resource_stats(rs);
    result.resource_stats = rs;
    for (const auto & device : proto.devices) {
        run_result::device_account account;
        account.device = device.device;
        const bool device_stats = proto.collect_device_stats(device.cuda_backend, account.stats) &&
                proto.collect_device_resource_stats(device.cuda_backend, account.resource);
        const uint64_t device_expected = (uint64_t) device.layer_bindings.size() * N_SCORED;
        const bool device_ok = device_stats && device_expected > 0 &&
                account.stats.all_hit_calls + account.stats.any_miss_calls == device_expected &&
                account.stats.output_commits == device_expected && account.stats.gpu_hit_draws > 0 &&
                account.stats.cpu_miss_draws > 0 &&
                account.stats.gpu_hit_draws + account.stats.cpu_miss_draws == device_expected * N_SELECTED &&
                account.stats.full_expert_weight_copy_bytes == 0 &&
                account.resource.active_bindings == device.layer_bindings.size() &&
                account.resource.fills_accepted > 0 && account.resource.fills_completed == account.resource.fills_accepted &&
                account.resource.fills_failed == 0 && account.resource.fill_outstanding == 0 &&
                account.resource.pending_pin_active == 0;
        std::printf("device_account arm=%s device=%d groups=%zu bindings=%zu semantic_ops=%llu gpu_rows=%llu cpu_rows=%llu fills=%llu completed=%llu rejected=%llu failed=%llu transport_bytes=%llu active_bindings=%llu outstanding=%llu pins=%llu path_ok=%d\n",
                label, account.device, device.groups.size(), device.layer_bindings.size(),
                (unsigned long long)(account.stats.all_hit_calls + account.stats.any_miss_calls),
                (unsigned long long)account.stats.gpu_hit_draws, (unsigned long long)account.stats.cpu_miss_draws,
                (unsigned long long)account.resource.fills_accepted, (unsigned long long)account.resource.fills_completed,
                (unsigned long long)account.resource.fills_rejected, (unsigned long long)account.resource.fills_failed,
                (unsigned long long)account.resource.fill_bytes_completed, (unsigned long long)account.resource.active_bindings,
                (unsigned long long)account.resource.fill_outstanding, (unsigned long long)account.resource.pending_pin_active,
                device_ok);
        ok = ok && device_ok;
        result.devices.push_back(account);
    }
    const uint64_t expected = (uint64_t) n_eligible * N_SCORED;
    ok = ok && expected > 0 && result.logits.size() == (size_t) N_SCORED && result.targets.size() == (size_t) N_SCORED &&
            result.stats.all_hit_calls + result.stats.any_miss_calls == expected &&
            result.stats.output_commits == expected && result.stats.capture_bypasses >= expected &&
            result.stats.gpu_hit_draws > 0 && result.stats.cpu_miss_draws > 0 &&
            result.stats.gpu_hit_draws + result.stats.cpu_miss_draws == expected * N_SELECTED &&
            result.stats.full_expert_weight_copy_bytes == 0 && have_rs && rs.active_bindings == (uint64_t) n_eligible &&
            rs.fills_accepted > 0 && rs.fills_completed == rs.fills_accepted && rs.fills_failed == 0 &&
            rs.fill_outstanding == 0 && rs.pending_pin_active == 0 && proto.valid() && inspect_frozen_group(*ctx, proto);
    std::printf("arm=%s semantic_ops=%llu expected_ops=%llu all_hit=%llu any_miss=%llu gpu_rows=%llu cpu_rows=%llu output_commits=%llu capture_bypasses=%llu accepted=%llu completed=%llu rejected=%llu failed=%llu transport_bytes=%llu full_expert_weight_copy_bytes=%llu outstanding=%llu pins=%llu path_ok=%d\n",
            label, (unsigned long long)(result.stats.all_hit_calls + result.stats.any_miss_calls), (unsigned long long)expected,
            (unsigned long long)result.stats.all_hit_calls, (unsigned long long)result.stats.any_miss_calls,
            (unsigned long long)result.stats.gpu_hit_draws, (unsigned long long)result.stats.cpu_miss_draws,
            (unsigned long long)result.stats.output_commits, (unsigned long long)result.stats.capture_bypasses,
            (unsigned long long)rs.fills_accepted, (unsigned long long)rs.fills_completed, (unsigned long long)rs.fills_rejected,
            (unsigned long long)rs.fills_failed, (unsigned long long)rs.fill_bytes_completed,
            (unsigned long long)result.stats.full_expert_weight_copy_bytes, (unsigned long long)rs.fill_outstanding,
            (unsigned long long)rs.pending_pin_active, ok);
    for (const auto & w : result.windows) std::printf("window arm=%s positions=%d-%d gpu_rows=%llu cpu_rows=%llu fills=%llu completed=%llu rejected=%llu failed=%llu transport_bytes=%llu\n",
            label, w.begin, w.end - 1, (unsigned long long)w.gpu, (unsigned long long)w.cpu, (unsigned long long)w.fills,
            (unsigned long long)w.done, (unsigned long long)w.rejected, (unsigned long long)w.failed, (unsigned long long)w.bytes);
    llama_free(ctx);
    return ok && lifecycle_zero(lifecycle);
}

} // namespace

int main(int argc, char ** argv) {
    const bool init_only = argc == 6 && std::strcmp(argv[5], "--init-only") == 0;
    if ((argc != 5 && !init_only) || std::strcmp(argv[1], "--devices") != 0 ||
            (std::strcmp(argv[2], "1") != 0 && std::strcmp(argv[2], "2") != 0) ||
            std::strcmp(argv[3], "--output-prefix") != 0) {
        std::fprintf(stderr, "usage: %s --devices 1|2 --output-prefix PATH [--init-only]\n", argv[0]);
        return 2;
    }
    const bool dual_device = std::strcmp(argv[2], "2") == 0;
    llama_backend_init();
    llama_model * model = load_model(dual_device);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }
    if (dual_device) {
        std::unordered_set<int32_t> owners;
        for (size_t il = 0; il < model->layers.size() && il < model->default_layer_device.size(); ++il)
            if (model->layers[il].ffn_up_exps && model->layers[il].ffn_gate_exps && model->layers[il].ffn_down_exps)
                owners.insert(model->default_layer_device[il]);
        std::printf("effective_split=%d eligible_layer_owner_count=%zu\n", (int) model->split_mode, owners.size());
        if (model->split_mode != LLAMA_SPLIT_MODE_LAYER || owners.size() != 2) {
            std::fprintf(stderr, "dual-device fixture requires effective LAYER ownership on both devices\n");
            llama_free_model(model); llama_backend_free(); return 1;
        }
    }
    if (init_only) {
        llama_moe_ffn_prototype_lifecycle lifecycle;
        llama_context * ctx = nullptr;
        bool ok = create_context(model, true, CACHE_BYTES, {}, lifecycle, &ctx, dual_device, true);
        size_t checked_bindings = 0;
        if (ok) {
            const auto & proto = *ctx->moe_ffn_prototype;
            ok = proto.valid() && proto.layer_count() > 0;
            for (const auto & device : proto.devices) for (const auto & layer : device.layer_bindings) {
                const float * activations = static_cast<const float *>(layer.cpu_activations->data);
                const int32_t * ids = static_cast<const int32_t *>(layer.cpu_ids->data);
                const float * routes = static_cast<const float *>(layer.cpu_routes->data);
                const float * output = static_cast<const float *>(layer.cpu_output->data);
                ok = ok && activations && ids && routes && output &&
                        std::all_of(activations, activations + layer.n_embd, [](float x) { return x == 0.0f; }) &&
                        ids[0] == 0 && routes[0] == 1.0f &&
                        std::all_of(output, output + layer.n_embd,
                                [](float x) { return std::isfinite(x) && x == 0.0f; });
                ++checked_bindings;
            }
            ok = ok && checked_bindings == proto.layer_count();
            std::printf("context_init_only poisoned_priming_inputs=1 initialized_bindings=%zu valid=%d\n",
                    checked_bindings, ok ? 1 : 0);
            llama_free(ctx);
        }
        const bool clean = lifecycle_zero(lifecycle);
        llama_free_model(model);
        llama_backend_free();
        return ok && clean ? 0 : 1;
    }
    int32_t n_tokens = llama_tokenize(model, CORPUS, (int32_t) std::strlen(CORPUS), nullptr, 0, true, false);
    if (n_tokens >= 0) { llama_free_model(model); llama_backend_free(); return 1; }
    std::vector<llama_token> tokens((size_t) -n_tokens);
    n_tokens = llama_tokenize(model, CORPUS, (int32_t) std::strlen(CORPUS), tokens.data(), (int32_t) tokens.size(), true, false);
    if (n_tokens < N_PROMPT + N_SCORED + 1) {
        std::fprintf(stderr, "tokenized corpus too short: %d\n", n_tokens);
        llama_free_model(model); llama_backend_free(); return 1;
    }
    tokens.resize(N_PROMPT + N_SCORED + 1);
    const std::string prefix = argv[4];
    { std::ofstream corpus(prefix + ".corpus.txt", std::ios::binary); corpus.write(CORPUS, std::strlen(CORPUS)); }
    { std::ofstream ids(prefix + ".token-ids.txt"); for (size_t i = 0; i < tokens.size(); ++i) ids << i << ',' << tokens[i] << '\n'; }
    std::printf("corpus token_count=%zu scored_positions=%d prompt_tokens=%d devices=%d token_ids_file=%s.token-ids.txt\n",
            tokens.size(), N_SCORED, N_PROMPT, dual_device ? 2 : 1, prefix.c_str());

    bool ok = true;
    run_result blocking;
    ok = run_ordinary(model, tokens, CACHE_BYTES, "BLOCKING", blocking, dual_device) &&
            write_run_artifacts(prefix + ".blocking", blocking);
    arm_summary blocking_summary{};
    ok = ok && summarize_arm(blocking, blocking_summary);
    std::printf("numerical_arm=BLOCKING rows=%zu mean_nll=%.12g ppl=%.12g\n",
            blocking.logits.size(), blocking_summary.mean_nll, blocking_summary.ppl);

    std::vector<run_result> hybrids(2);
    std::vector<arm_summary> hybrid_summary(2);
    std::ofstream metrics(prefix + ".comparison.csv");
    metrics << std::setprecision(17)
            << "arm,rows,mean_nll,ppl,nll_delta_mean,nll_delta_rms,nll_delta_max_abs,directed_kl,probability_rms,top1_agreement,top10_overlap,max_logit_abs\n";
    for (size_t i = 0; ok && i < hybrids.size(); ++i) {
        const std::string label = "HYBRID" + std::to_string(i + 1);
        const std::string run_prefix = prefix + ".hybrid-0" + std::to_string(i + 1);
        pair_summary p;
        ok = run_hybrid(model, tokens, label.c_str(), hybrids[i], dual_device) &&
                write_run_artifacts(run_prefix, hybrids[i]) && summarize_arm(hybrids[i], hybrid_summary[i]) &&
                compare_pair(blocking, hybrids[i], p);
        if (!ok) break;
        metrics << label << ',' << hybrids[i].logits.size() << ',' << hybrid_summary[i].mean_nll << ',' << hybrid_summary[i].ppl
                << ',' << p.nll_mean << ',' << p.nll_rms << ',' << p.nll_max_abs << ',' << p.kl << ',' << p.prob_rms
                << ',' << p.top1 << ',' << p.top10 << ',' << p.max_logit << '\n';
        const std::string rows_path = run_prefix + ".aligned.csv";
        std::ofstream rows(rows_path);
        rows << std::setprecision(17) << "position,target,blocking_nll,hybrid_nll,nll_delta,directed_kl,probability_rms,top1_agreement,top10_overlap,max_logit_abs\n";
        for (size_t pos = 0; pos < p.rows.size(); ++pos) {
            const row_delta & d = p.rows[pos];
            rows << pos << ',' << blocking.targets[pos] << ',' << row_nll(blocking.logits[pos], blocking.targets[pos]) << ','
                << row_nll(hybrids[i].logits[pos], hybrids[i].targets[pos]) << ',' << d.nll << ',' << d.kl << ','
                << d.prob_rms << ',' << d.top1 << ',' << d.top10 << ',' << d.max_logit << '\n';
        }
        std::printf("comparison reference=BLOCKING candidate=%s nll_mean=%.12g nll_rms=%.12g nll_max_abs=%.12g kl=%.12g probability_rms=%.12g top1=%.9g top10=%.9g max_logit_abs=%.12g\n",
                label.c_str(), p.nll_mean, p.nll_rms, p.nll_max_abs, p.kl, p.prob_rms, p.top1, p.top10, p.max_logit);
    }
    if (ok) {
        pair_summary p;
        ok = compare_pair(hybrids[0], hybrids[1], p);
        if (ok) std::printf("comparison reference=HYBRID1 candidate=HYBRID2 rows=%zu kl=%.12g probability_rms=%.12g top1=%.9g top10=%.9g max_logit_abs=%.12g first_diff=%d\n",
                p.rows.size(), p.kl, p.prob_rms, p.top1, p.top10, p.max_logit, first_differing_position(hybrids[0], hybrids[1]));
    }
    std::printf("f4b_summary devices=%d blocking_rows=%zu hybrid_contexts=%zu result=%d output_prefix=%s\n",
            dual_device ? 2 : 1, blocking.logits.size(), hybrids.size(), ok, prefix.c_str());
    llama_free_model(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
