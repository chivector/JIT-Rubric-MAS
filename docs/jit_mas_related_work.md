# JIT-MAS: Primary Sources and Design Boundaries

Inspected on 2026-09-30. Repository revisions below were resolved through the
GitHub commits API, and source files were read from their commit-pinned raw URLs.
This is a mechanism audit, not a reproduction of any paper's reported scores.
No upstream MAS framework or source implementation was vendored into JIT-MAS.

## Meta-Team

- Paper: [Evolve as a Team, v1](https://arxiv.org/html/2605.29790v1),
  sections 3 and Appendix D.
- Code revision: [`36dc85d9dc2219d292fa180f347479738a84acb2`](https://github.com/zz-haooo/Meta-Team/tree/36dc85d9dc2219d292fa180f347479738a84acb2).
- License audit: the recursive tree at this revision contains no license file;
  fetching root `LICENSE` returns HTTP 404. Public visibility is not permission
  to copy. JIT-MAS uses independently written code and prompts.

### Existing Mechanisms

The paper preserves local execution context and exchanges selected cross-agent
evidence during reflection. It distinguishes agent, interaction, and team changes.
Its candidate pool is persistent, but the active roster can be selected dynamically;
it would be inaccurate to describe Meta-Team as always executing a fixed team.
[Paper, sections 3.2 and D](https://arxiv.org/html/2605.29790v1).

Inspected implementation:

- [`core/reflection_runner.py`](https://github.com/zz-haooo/Meta-Team/blob/36dc85d9dc2219d292fa180f347479738a84acb2/core/reflection_runner.py):
  `finalize_task` transitions to reflection after task validation; phase guards
  bound L1/L2/L3 and agents continue into reflection. `_apply_reflection` validates
  certain prompt edits before updating agent/team files.
- [`core/message_store.py`](https://github.com/zz-haooo/Meta-Team/blob/36dc85d9dc2219d292fa180f347479738a84acb2/core/message_store.py):
  addressed mailboxes, send/read status, communication partners, termination.
- [`tools/reflection.py`](https://github.com/zz-haooo/Meta-Team/blob/36dc85d9dc2219d292fa180f347479738a84acb2/tools/reflection.py):
  conditional behavioral patches, teammate profiles, collaboration notes,
  evidence-bearing team suggestions, and restricted proposal targets. Its prompt
  mechanisms emphasize general practices over task diaries.
- [`evaluation/researchrubrics.py`](https://github.com/zz-haooo/Meta-Team/blob/36dc85d9dc2219d292fa180f347479738a84acb2/evaluation/researchrubrics.py)
  and [`scripts/run_rr_paper_experiment.sh`](https://github.com/zz-haooo/Meta-Team/blob/36dc85d9dc2219d292fa180f347479738a84acb2/scripts/run_rr_paper_experiment.sh):
  per-criterion binary evaluation, signed aggregation, serial evolution, frozen
  holdout runs. The inspected experiment script partitions 20/81 tasks. We do
  not run this script or inherit these experiment settings.

### JIT-MAS Combination and Extension

`RubricAttributor` conducts global outline, independent local analyses, and global
integration. Local analyses receive their complete observable `RunResult` trace;
the global analyzer receives the index, artifacts and local findings. A local
analyzer can request up to eight indexed events in one additional exchange.
Evidence IDs are checked; unsupported findings remain uncertain. The experiences
target task-conditioned rubric prediction, organization, and reusable capability
practices. Each new task gets a new TeamSpec and JIT harness. Since 2026-10-01,
the first reconciled proposal is applied directly through structural/provenance,
base-version and duplicate guards, without a paired quality-promotion gate.
Downstream quality requires separate measurement: neither a valid schema nor an
LLM endorsement establishes useful experience. These are project design choices,
not claims attributed to Meta-Team.

## Co-STORM

- Paper: [Into the Unknown Unknowns, v1](https://arxiv.org/html/2408.15232v1),
  sections 3-4 and prompt appendix D.
- Code revision: [`fb951af7744dab086e34962e9bc6fe878e145f83`](https://github.com/stanford-oval/storm/tree/fb951af7744dab086e34962e9bc6fe878e145f83).
- [License](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/LICENSE):
  MIT, copyright Stanford Open Virtual Assistant Lab, 2024. The current changes
  borrow ideas and independently formulate prompts; no source was copied.

### Existing Mechanisms

Co-STORM combines topic-specific expert perspectives, a moderator, and an
organized knowledge store for interactive information seeking. The moderator
uses relevant but unused retrieved information to introduce missed perspectives;
the knowledge organization supports subsequent report generation.
[Paper, sections 3-4](https://arxiv.org/html/2408.15232v1).

Inspected paths under `knowledge_storm/collaborative_storm/`:

- [`modules/expert_generation.py`](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/collaborative_storm/modules/expert_generation.py):
  `GenerateExpertGeneral` and `GenerateExpertWithFocus` specify perspectives,
  expertise, interests and discussion focus from the topic/background.
- [`engine.py`](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/collaborative_storm/engine.py):
  `DiscourseManager.get_next_turn_policy` coordinates speaker updates,
  moderator intervention and knowledge reorganization.
- [`modules/co_storm_agents.py`](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/collaborative_storm/modules/co_storm_agents.py):
  expert utterances use their role and knowledge summary; `Moderator` identifies
  uncited snippets and ranks relevance versus redundancy.
- [`modules/information_insertion_module.py`](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/collaborative_storm/modules/information_insertion_module.py):
  intent-linked information is placed into a hierarchical knowledge structure,
  with optional node expansion.
- [`modules/article_generation.py`](https://github.com/stanford-oval/storm/blob/fb951af7744dab086e34962e9bc6fe878e145f83/knowledge_storm/collaborative_storm/modules/article_generation.py):
  `WriteSection` grounds generated sections in indexed source snippets and their
  originating question/query. The original prompt requests Wikipedia sections.

### JIT-MAS Combination and Extension

`GlobalAnalyzer.predict` proposes task-specific capabilities; `local_plan` asks
each candidate to identify gaps and contest the draft; `reconcile` produces an
executable, budgeted responsibility/dependency graph. Our prompts adapt the
perspective and evidence principles to operational rubrics. We do not import
DSPy, require a moderator role, fix a speaker count, assume human participation,
or impose Wikipedia style on creative tasks. Rubric relationships and execution
dependencies are separate structures. This combination is not described as a
Co-STORM implementation or a demonstrated performance improvement.

## Engineering Assumptions

- JSON output plus typed validation is a practical interface, not proof of the
  truth of predictions, reflections, or cited evidence.
- Defaults bound planning to one local round and attribution to one local round
  with one optional indexed evidence request. A smaller context is not assumed
  to be inherently better; complete local observable traces are preserved.
- Agents may share a model endpoint while retaining independent message lists.
  No provider-internal reasoning is required or reconstructed.
- Planning confidence is not calibrated probability. Alignment is semantic but
  model-dependent. Evidence-supported attribution remains a hypothesis.
- These software tests exercise controlled fixtures and cannot estimate real
  benchmark accuracy, cross-task quality gains, or scientific novelty.

## ResearchRubrics

- Paper: [ResearchRubrics v1](https://arxiv.org/abs/2511.07685), CC BY 4.0.
- Official code: [`2dc80e2d4c38ddd80439517c259d93c6954b193f`](https://github.com/scaleapi/researchrubrics/tree/2dc80e2d4c38ddd80439517c259d93c6954b193f),
  MIT. The literal copyright holder is `2025 [Authors/Institution]`.
- Dataset: [ScaleAI/researchrubrics](https://huggingface.co/datasets/ScaleAI/researchrubrics/tree/85de3115053d1453ed612caacf4a405edc1ad756),
  pinned revision `85de3115053d1453ed612caacf4a405edc1ad756`; the card declares MIT.

Existing mechanism: ResearchRubrics is an open-ended task benchmark with
independent, per-criterion judging and an official judge prompt. It is not a
multi-agent system. The inspected official implementation sums signed
`weight * score` and divides by the sum of positive weights; a zero denominator
returns zero. A failed criterion remains represented with its original weight.
See the [pinned implementation and field audit](jit_mas_researchrubrics.md).

Project extension: the adapter preserves individual criterion status, raw
judgments and failure details, and marks incomplete evaluation. Hidden criteria
remain in private records until submission. Separate semantic alignment compares
those criteria with frozen predicted requirements; it never changes the official
score. Engineering assumption: scripted judge responses test protocol and
arithmetic fidelity only. They are not benchmark measurements or a substitute
for externally evaluated real submissions.
