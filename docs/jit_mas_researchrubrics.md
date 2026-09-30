# ResearchRubrics Integration

## Inspected Primary Sources

Inspected on 2026-09-30, without paid model calls or benchmark runs:

| Source | Fixed version | Relevant material |
| --- | --- | --- |
| [Paper](https://arxiv.org/abs/2511.07685) | [2511.07685v1](https://arxiv.org/html/2511.07685v1), 2025-11-10 | Sections 3.3, 3.4, 4.1 and Appendix E: human criteria, signed importance, independent grading and judge prompts |
| [Official code](https://github.com/scaleapi/researchrubrics) | `2dc80e2d4c38ddd80439517c259d93c6954b193f` | Evaluator, prompts, aggregation and failure handling |
| [Official dataset](https://huggingface.co/datasets/ScaleAI/researchrubrics) | `85de3115053d1453ed612caacf4a405edc1ad756` | `processed_data.jsonl`, 101 whole tasks; revision verified through the dataset API and file read |

Relevant immutable source files:

- [evaluate_single_report.py](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/src/evaluate_rubrics/evaluate_single_report.py)
- [evaluate_reports_batch.py](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/src/evaluate_rubrics/evaluate_reports_batch.py)
- [calculate_compliance_score.py](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/src/calculate_metrics/calculate_compliance_score.py)
- [system_prompt.txt](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/src/prompts/system_prompt.txt)
- [user_prompt.txt](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/src/prompts/user_prompt.txt)
- [Code license](https://github.com/scaleapi/researchrubrics/blob/2dc80e2d4c38ddd80439517c259d93c6954b193f/LICENSE)
- [Dataset card at pinned revision](https://huggingface.co/datasets/ScaleAI/researchrubrics/blob/85de3115053d1453ed612caacf4a405edc1ad756/README.md)

The code license is MIT, with the literal copyright holder
`Copyright (c) 2025 [Authors/Institution]`. The Hugging Face card also declares
MIT. The paper is CC BY 4.0. The two copied prompt files and test-only official
metric script retain the upstream MIT notice in
`benchmark/adapter/researchrubrics_prompts/LICENSE`. Prompt text is unchanged
apart from terminal newline normalization. No benchmark task text is vendored;
the two tiny fixtures are newly written synthetic software tests.

## Observed Semantics

The JSONL fields are `prompt`, `sample_id`, `domain`, `conceptual_breadth`,
`logical_nesting`, `exploration`, and `rubrics`. Each rubric contains `criterion`,
`weight`, and `axis`. We preserve the original fields and add a stable rubric ID
derived from sample ID, rubric position and criterion hash. Rubric position
prevents collisions when a task repeats criterion wording across axes.

The paper studies binary and ternary grading. The fixed public evaluator uses
binary `Satisfied` / `Not Satisfied` with scores 1 / 0. Its system prompt judges
only evidence in the submitted document; the user prompt supplies the complete
document, one criterion, category and signed weight. It requests confidence,
reasoning, evidence quotations and missing elements. The criterion examples are
guidance rather than an exhaustive list of acceptable responses.

The fixed code computes:

```text
numerator   = sum(weight * score for every criterion)
denominator = sum(weight for every criterion with weight > 0)
compliance  = numerator / denominator if denominator > 0 else 0.0
```

Negative weights remain in the numerator, so negative task scores are possible.
No clipping is permitted. A denominator of zero returns 0.0 and is additionally
marked `zero_denominator` locally. Task scores, rather than individual correlated
criteria, are the unit for a dataset mean.

The source `_evaluate_single` makes at most three attempts and parses JSON.
`evaluate_rubric` turns a final exception into `Error`, score 0, confidence 0 and
`success=False`. The failed criterion retains its original weight in the
positive denominator. Our adapter follows that scoring rule, returns all rows,
and marks the result incomplete. A failed negative criterion also scores zero
under the official rule; this can obscure a penalty, so incomplete results must
never authorize experience updates or be presented as complete evaluations.

## Reused Mechanism And Local Extension

ResearchRubrics supplies a benchmark and an independent outcome evaluator, not
a team architecture. JIT-MAS uses its fine-grained feedback after final answer
submission to compare predicted requirements with observed success and failure.
It does not expose official rubrics to the planner or use them to prescribe
fixed roles. Predicted importance remains distinct from private official weights.

The local extension keeps evidence quotations, locates exact quotations in the
answer, preserves the entire raw result and reasoning, and records errors and
an evaluator version. A quotation absent from the answer is retained but marked
unverified. These locations establish what the answer says; they do not prove
that a cited external source is true or supports the claim.

`is_pass` is a legacy JIT compatibility diagnostic: complete evaluation, positive
denominator and compliance at least 1. It is not a newly claimed official
benchmark metric. `official_compliance` preserves the published aggregation
semantics, but a configurable judge endpoint/model is not necessarily the model
used in the paper. The adapter's version includes the upstream commit, actual
prompt texts, judge identity/endpoint and failure/token/length policy.

## Code Contract

`benchmark/adapter/researchrubrics.py` provides:

- `split_item(raw) -> (public, private)`: public fields are allowlisted; original
  hidden fields are kept only in the private record.
- `ResearchRubricsAdapter.load_dataset(path)`: JSONL loading, whole-task duplicate
  rejection and public items only. Duplicate detection uses exact IDs and
  whitespace-normalized, case-folded prompts; this is not semantic deduplication.
- `private_record(task_id)`: coordinator/evaluator-only deep copy. It must never
  be included in agent configuration or a mounted execution workspace.
- `evaluate(answer, task_id, private_record=record)`: official score and a full
  `feedback` list, `complete`, `failed_count`, usage and evaluator identity.
- `official_compliance_score(rows)`: the fixed signed aggregation.

For existing runners the required `answer` slot contains an opaque sample ID,
not a reference answer. `format_task` includes only the public question and
explicit public constraints. The private original record never enters the task
string. The adapter is registered in `jit/harness_ops.py` and
`benchmark/registry.py`; `benchmark/config/researchrubrics.yaml` can be loaded
through `scripts.eval.runner.make_adapter`. ResearchRubrics does not implicitly
inherit execution-model judge credentials.

An injected `judge(messages, **kwargs)` uses JIT's model interface and may be a
metered model. The adapter reports per-call usage but does not debit that
model's ledger again. The default real judge reuses `OpenAIServerModel`, sets
its newly configurable `max_attempts=1`, and disables SDK retries so the adapter
owns the three-attempt policy. The existing model defaults remain five attempts.
No model ID, endpoint, credential or pricing table is hardcoded here. Monetary
cost stays `null` without a trustworthy price table.

## Deliberate Differences And Limits

- The public evaluator supports document chunking and evidence synthesis. This
  first adapter evaluates the whole document and refuses over-limit documents
  with an incomplete result instead of silently truncating them. The character
  limit is an engineering bound, not a token-aware context guarantee. Real
  context limits and timeouts still depend on the configured provider.
- The adapter validates verdict/score agreement, finite numbers and expected
  fields. Malformed output uses the same bounded failure protocol. The upstream
  parser is more permissive. Here a JSON parse failure also waits before retry;
  upstream waits on API failures but immediately retries JSON decode failures.
- The upstream dataframe truncates reasoning and its JSONL writer omits some
  evidence/error information. This adapter preserves it for attribution.
- The upstream implementation hardcodes a Gemini context/pricing map. We use
  explicit endpoint/token settings and report unknown cost rather than adopt
  obsolete prices or pretend other models reproduce the paper's judge.
- Splitting Python dictionaries is not a security boundary. Generated Python
  running on the host can potentially read host files, process memory or
  credentials. Put the original dataset outside all agent mounts. Use the
  JIT-MAS execution guard and a genuine sandbox; explicit `unsafe-local` runs
  cannot claim hidden-data security. The legacy generic runner itself supplies
  no such sandbox.
- The downloaded dataset has a single Hugging Face `train` split. This is not
  an authorized evolution/test protocol. JIT-MAS requires its own saved whole-task
  evolution/validation/test manifest, and must never split criteria from one
  task across partitions. Current-task feedback is released only after final
  submission. Frozen held-out runs must not feed preceding test feedback into
  later tasks.

## Preparation And Offline Checks

Prepare the pinned source file in an evaluator-only directory outside execution
mounts. For example, after installing the optional `huggingface_hub` CLI:

```powershell
hf download ScaleAI/researchrubrics processed_data.jsonl --repo-type dataset --revision 85de3115053d1453ed612caacf4a405edc1ad756 --local-dir C:/jit-evaluator-data/researchrubrics
$env:RESEARCHRUBRICS_DATASET = 'C:/jit-evaluator-data/researchrubrics/processed_data.jsonl'
```

The adapter and its tests do not automatically download data or invoke models.
Run the bounded, offline suite:

```powershell
.venv/Scripts/python.exe -m pytest tests/jit_mas/test_researchrubrics.py -q
```

It checks scoring against the executable calculation extracted from the pinned
official script, signed/zero-denominator/failure cases, public/private canaries,
bounded retry handling, full evidence feedback and public runner registration.
No performance result can be inferred from these fixtures. Real model calls,
the paper's model configuration, sandboxed benchmark execution and benchmark
performance remain unexecuted and unverified in this implementation session.
