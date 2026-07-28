# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## The paper (read this before touching the algorithm)

This repo is the official implementation of **R²-Mem: Reflective Experience for Memory Search**
(Xinyuan Wang, Wenyu Mao, Junkang Wu, Xiang Wang, Xiangnan He — USTC; arXiv:2605.13486v1, 13 May 2026, preprint).

**A local copy is checked in at `paper/R2-Mem-2605.13486v1.pdf` (25 pages). Read it with the Read tool — do not re-download it.**
Upstream, if ever needed: https://arxiv.org/abs/2605.13486

Where things are in the PDF:
- §3 Preliminary — the deep memory search loop (Planning → Searching → Reflection), and the trajectory definition `τ = (s₀…s_T)`, `sᵢ = (qᵢ, actionᵢ, mᵢ)`. This is exactly what `gam/agents/research_agent.py` implements.
- §4.2 Rubric-guided Evaluating — score each step over rubric dimensions, emit `(scoreₖ, reasonₖ, adviceₖ)`. → `exp/judge_model.py`.
- §4.3 self-Reflection-based experience — `Learner_SR(sₖ, rₖ, aₖ) → e_q^t`, `q ∈ {good, bad}`, `t ∈ {P, R}`. → `exp/self_reflection.py`.
- §4.4 Deep Search with experience — two banks 𝒟^P / 𝒟^R, condition construction (planning: `cᵢ = qᵢ`; reflection: `cᵢ = [Q + mᵢ]`), situation abstraction `LLM_σ`, Top-K similarity retrieval over `φ(c + σ)`. → `exp/eval/research_agent_exp_main.py`.
- §5 Experiments — LoCoMo / HotpotQA / NarrativeQA; baselines RAG, A-MEM, Mem0, LightMem, MemoryOS, Memory-R1, GAM. Headline: +22.6% F1, −12.9% tokens, −20.2% iterations.
- Appendices B/C — dataset + baseline details, scalability, ablations, self-evolution.

Key numbers worth remembering: rubrics are 4 dimensions × 0–3 = **0–12 total per step**; the released thresholds are `Low_threshold=5` / `High_threshold=10`, matching the `(K_low, K_high) = (5, 10)` setting in Figure 4. Experience is distilled from **10% of LoCoMo / 20% of HotpotQA and NarrativeQA** trajectories.

## What the repo actually contains

Two layers that must not be conflated:

- **`gam/`** — the *baseline*, an unmodified copy of GAM (General Agentic Memory, VectorSpaceLab). This is the strongest deep-search baseline from the paper, not R²-Mem. Installed as its own package (`general-agentic-memory`) with its own `pyproject.toml`.
- **`exp/`** — the *R²-Mem contribution*. Everything novel lives here.

`eval/` runs the GAM baseline; `exp/eval/` runs R²-Mem. The two directories contain near-duplicate ~800-line driver scripts per dataset (`eval/hotpotqa_test.py` vs `exp/eval/datasets_test/hotpotqa_exp.py`, etc.). When fixing a bug in one, check whether its twin needs the same fix.

### Baseline architecture (`gam/`)

Dual-agent, LLM-backend-agnostic:
- `MemoryAgent.memorize(message)` — decorates raw text into a `Page` (header + content) plus an abstract; abstracts form the `MemoryState` that Planning sees.
- `ResearchAgent.research(request)` — the iterative loop, `max_iters` capped. Per iteration: `_planning` → `_search` (dispatches to `keyword`/`vector`/`page_index` retrievers per the plan, dedupes hits by `page_id`, sorts by score) → `_integrate` (LLM merges hits into the running `Result`) → `_reflection` (an `InfoCheck` call for "enough?", then a `GenerateRequests` call to synthesize the next query). Breaks early when `decision.enough`.
- Every LLM call goes through `AbsGenerator.generate_single(prompt=..., schema=...)` with a JSON schema; token usage is accumulated on `agent.total_tokens`. Two backends: `VLLMGenerator`, `OpenAIGenerator`.
- Retrievers (`BM25Retriever`, `DenseRetriever`, `IndexRetriever`) share a `build(page_store)` / `update(page_store)` / `search(List[str], top_k)` interface returning `List[List[Hit]]`. `ResearchAgent._update_retrievers()` re-indexes when the page count changes.

### R²-Mem additions (`exp/`)

Offline (build the experience bank), then online (use it):

1. `exp/judge_model.py` — the **Evaluator**. Three interchangeable judges over a whole serialized trajectory: `gpt_judger` (gpt-4o, the default in the paper), `deepseek_judger`, and `judger_vllm` (the local model judging itself — this is the *self-evolution* variant of §5.6/RQ5). Returns per-step, per-module rubric scores. Planning dimensions: Info Needs Coverage / Info Needs Non-Redundancy / Tool–Info Alignment / Planning Efficiency. Reflection dimensions: Sufficiency Judgment Accuracy / Minimal Sufficiency Recognition / Follow-up Query Quality / Answer Completeness Awareness.
2. `exp/self_reflection.py` — the **Learner** and the offline driver. Reads GAM traces, calls the Evaluator, keeps only steps scoring `≥ High_threshold` (label `high_quality`) or `≤ Low_threshold` (label `low_quality`) and *skips everything in between*, then calls `Exp_Planning` / `Exp_Reflection` to distill `{thinking, summary, situation, experience}` + `condition`. Finally `exp_bank()` embeds `"({situation}){condition}"` with BGE-M3 and writes a FAISS `IndexFlatIP` plus a parallel metadata JSON — **two banks, `Planning_*` and `Reflection_*`, per model size.**
3. `exp/eval/research_agent_exp_main.py` — `ResearchAgent_exp`, the online agent. Same loop as GAM's `ResearchAgent`, but each iteration first abstracts a situation (`generate_situation_planning` / `generate_situation_reflection`) and calls `retrieve_experience(query, name, return_k=3, recall_k=10)` to pull Top-K experiences from the matching bank, which are injected into the `*_PROMPT_exp` variants in `exp/prompts/research_prompts_exp.py`. Note `research(request, category)` takes an extra `category` argument the baseline does not.
4. `exp/eval/experience_encoder.py` — process-wide singleton BGE-M3 `SentenceTransformer`. Loaded once; do not instantiate encoders elsewhere.

Bank file naming is a hard contract between writer and reader:
`self_reflection.py` writes `{expbank_path}/Planning_{model_size}_situation.index` + `_metadata.json`; `research_agent_exp_main.py` reads `{exp_bank_path}/{name}_{model_size}_situation.index` with `name ∈ {Planning, Reflection}`. Changing `model_size` in one file without the other silently breaks retrieval.

## Configuration is hardcoded, not CLI-driven

There is no config file. The offline stage and the R²-Mem agent are configured by **editing module-level constants**, most of which ship as placeholders (`"your_api_key"`, `"your_bge_model"`, `"..."`). Any fresh checkout must be edited before it runs:

| File | Constants |
|---|---|
| `exp/self_reflection.py` | `GPT_API_key` / `DEEPSEEK_API_key`, `bge_model_path`, `base_url`, `model_name`, `model_size`, `datasets`, `path_locomo`, `expbank_path`, `High_threshold`, `Low_threshold`, `max_samples`, and the hardcoded `convs = [26]` list in `main()` |
| `exp/eval/research_agent_exp_main.py` | `exp_bank_path`, `model_size` |
| `exp/eval/experience_encoder.py` | `_MODEL_PATH` (BGE-M3) |
| `exp/eval/locomo.py` | `results_path` (R²-Mem output), `GAM_path` (baseline output to replay from) |
| `exp/evaluate_locomo.py` | `exp_results_path`, `noexp_results_path` in `main()` |
| `scripts/*.sh`, `exp/eval/exp_scripts/*.sh` | `outputdir` / `base_outputdir` (all default to `your_output_path`), vLLM ports, `--embedding-model-path` |

The dataset drivers in `eval/` and `exp/eval/datasets_test/` are the exception — they are argparse-driven, with separate `--memory-*`, `--research-*`, `--working-*` model triples (key/base-url/model/api-type), so the three roles can point at different backends.

## Running things

Setup (needs local Qwen2.5-\*-Instruct and BAAI/bge-m3 weights on disk):

```bash
pip install -r requirements.txt
cd gam && pip install -e ".[all]" && cd ..
bash scripts/download_data.sh
export PYTHONPATH=$(pwd)          # required for every command below
```

Serve the backbone (the port must match whatever the scripts use — `scripts/eval_hotpotqa.sh` says 8002, everything else says 8001):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model qwen2.5-7B-Instruct --served-model-name qwen2.5-7B-Instruct \
    --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 32768 --port 8001
```

The four stages, in order — **each depends on the previous one's output on disk**:

```bash
# 1. GAM baseline. Produces per-question research_trace_q*.json + qa_results.json,
#    which are the raw material for reflection. Nothing downstream works without this.
bash scripts/eval_locomo.sh          # or eval_hotpotqa.sh / eval_narrativeqa.sh

# 2. Offline: evaluate traces, distill experience, build the FAISS banks.
python exp/self_reflection.py

# 3. Online: R²-Mem with the experience bank.
python exp/eval/locomo.py                       # LoCoMo
bash exp/eval/exp_scripts/exp_hotpotqa.sh       # HotpotQA
bash exp/eval/exp_scripts/exp_narrativeqa.sh    # NarrativeQA

# 4. Score R²-Mem against GAM (F1 + BLEU-1, broken out by question category).
python exp/evaluate_locomo.py
python exp/evaluate_hotpotqa_narrativeqa.py
```

Note the asymmetry in stage 3: LoCoMo goes through `exp/eval/locomo.py`, which **replays** questions read out of the GAM run's `qa_results.json` (it needs `GAM_path` pointed at stage 1's output), while HotpotQA/NarrativeQA run their own full pipeline from the raw dataset.

To run a single sample instead of a sweep, use the drivers' `--start-idx` / `--end-idx` rather than editing scripts.

There is no test suite, no linter config in use, and no CI. `black`/`mypy` settings exist in `gam/pyproject.toml` but are inherited from upstream GAM and are not run over `exp/`.

## Gotchas

- **`exp/self_reflection.py` needs both `PYTHONPATH=$(pwd)` and to be invoked as `python exp/self_reflection.py`.** It does a bare `from judge_model import *` (resolved via the script's own directory) *and* `from exp.prompts... import ...` (resolved via repo root). Running it as `python -m exp.self_reflection` or from inside `exp/` breaks one import or the other.
- **`scripts/download_data.sh` is incomplete.** It calls `python download_data/download_narrativeqa.py`, which does not exist in the repo — NarrativeQA must be fetched manually into `data/narrativeqa`. It also downloads HotpotQA `eval_6400`, but every eval script iterates over `eval_400 eval_1600 eval_3200`. (Paper Table 2 reports these three as the 56K/224K/448K context settings.)
- `data/` ships only `locomo/locomo10.json`; the other datasets are gitignored-by-omission and are the user's responsibility.
- Comments and debug prints across `gam/` and parts of `exp/` are a mix of Chinese and English, and the loop prints token counts on every call. Match the surrounding language when editing a file rather than normalizing it.
- Nearly every LLM call is wrapped in a broad `try/except` that prints the error and returns an empty/default object (e.g. `_planning` returning an all-empty `SearchPlan`). Silent degradation, not crashes, is the failure mode — when results look wrong, grep stdout for `Error in ...` before assuming a logic bug.
- README calls the code "will update later"; treat the README's structure diagram as authoritative for intent but verify against the files, which have moved ahead of it.
