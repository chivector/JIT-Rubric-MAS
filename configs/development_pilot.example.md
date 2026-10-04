# Development pilot configuration

`development_pilot.example.json` retains the v19 model parameters and common budgets
and opts into the registered v20 patch and membership-observation settings. All five v20
campaigns are sealed: 30 fixed slots, 25 graded, 2 generation failures
0 recorded evaluation exception types and 3 incomplete evaluations
without an exception type. Initial-MAS graded 7 of 10. The single
full suite passed with 2,467 tests and 60 subtests. The six-source descriptive
normalized failure-zero macro is Direct 0.5119798628, Initial-MAS
0.4685185185 and Native JIT 0.5476313523. This is a small exposed
development inventory with serving-model diagnostic judges for four sources;
it does not establish a clean held-out or uniform benchmark win. Actual
calls, tokens and failures are retained in the experiment record.

v17 introduced `planning_response_format=json_schema` and
`public_refinement_guard=true` relative to v16. v20 retains the v19 model roles,
sampling parameters and common budgets, and opts into `public_revision_mode=patch`
and `public_membership_observations=true`. The template matches the actual v20
configuration except for endpoint, model alias and response-model placeholders.
Each source version needs its own registration and validation. The template
is not a completed experiment registration and contains no credentials or local
data paths.

Copy the file to your own configuration and set all five model roles to your
provider's credential-free HTTPS endpoint and model alias. The template uses
`https://api.example.com/v1` and `YOUR_MODEL_ALIAS`; these are placeholders.
`expected_response_model` is `null` until you bind the exact response model name
reported by your provider. The four generation roles, `meta/global/local/exec`,
use temperature `0.6`; the evaluator role, `judge`, stays at `0.0`. Keep these
settings the same across compared methods and register the final configuration
before generation. The temperature is an experimental choice for the configured
provider, not an official recommendation for every model.

The API key is read from `JIT_BENCHMARK_API_KEY` in the launching process's
environment. For PowerShell, hidden interactive input avoids storing the key
in the configuration or command history:

```powershell
$pilotApiSecret = Read-Host "API key" -AsSecureString
$env:JIT_BENCHMARK_API_KEY = [System.Net.NetworkCredential]::new("", $pilotApiSecret).Password
```

The template retains the used endpoint's `thinking=disabled` and
`reasoning_effort=none` options. Set these consistently to values your provider
supports, and set its actual context window. The public refinement path requires
the provider to support the configured response formats. The template's
`json_schema` mode requests strict JSON schema for public review, ordinary final
prose revisions and structured construction. The optional `json_schema_review`
mode uses strict review and construction but JSON object for ordinary final prose;
`json_object` retains JSON object for both public stages. The template enables `public_positional_construction` and
`public_numeric_construction` for supported constraints stated in the public
task. Both require `json_schema` or `json_schema_review`, and their generic
defaults are off. This template uses `public_numeric_construction_layout=template`:
the complete prose contains unique alphabetic number markers, and a closed mapping
supplies exactly one integer per marker. Local validation checks marker uniqueness,
absence of other digits and the requested lexical conjunction minimum; rendering
inserts the supplied integers without appending prose. The optional `array` and
`named_objects` layouts remain available. Grammar, meaning and other constraints
still require evaluation. Mixed positional and
numeric requirements conservatively retain ordinary revision. No private
checker parameters enter these construction prompts.

The word-slot protocol is `public-word-position-slots-v2`. Target slots must
contain ASCII letters and avoid the entire finite Treebank compound family
`cannot`, `gimme`, `gonna`, `gotta`, `lemme`, `wanna`, including mixed case.
These can be split into multiple words by a general tokenizer. The same positive
finite-trie pattern governs the two array schemas and local validation. An
unsupported keyword makes compilation decline; the original task remains in
force. No runtime NLTK import, resource download, word substitution or extra
sample is added. This convention still does not certify every sentence boundary,
tokenizer, grammatical requirement or task result.

`public_positional_draft_guidance=true` asks intermediate contributors and the
synthesizer to retain compact complete content while leaving the supported exact
sentence/word position to the already configured final construction stage. Its
generic default is off; final public constraints remain in force.

The template also enables `public_positional_draft_projection=true`. The library
default is off, and enabling it requires `public_refinement`,
`public_positional_construction` and `public_positional_draft_guidance`. Only
complete, independent positive positional instructions that the public compiler
supports are deferred in a deep copy of the execution Actor's public task.
Planning, the original task, final refinement and evaluation retain the complete
original constraints. Recognized quoted, negative, conditional or ambiguous scope
causes an atomic conservative decline and keeps the entire original Actor task.
This guard is conservative and does not establish coverage of every quotation or
language form; v14 quotation gaps and any later fixes are recorded separately
in the experiment history. Other
constraints, public rubrics and frozen evidence are retained, and upstream
contributions or rubrics may still reintroduce exact positions. The audit records
the decision; no private evaluator inputs guide it.

The template sets `public_construction_response_format=json_schema` explicitly
for construction revisions. Public review and ordinary revisions retain the
configured mode. Optional JSON object construction transport uses the same strict
local schema and renderer before submission. A requested remote schema does not
prove that the endpoint enforces it. Invalid outputs fail with their cost recorded,
without format fallback, JSON-prefix selection or another sample, except for the
narrow ordinary-revision candidate guard described below.

`public_refinement_guard=true` enables guard v4 in the v20 source. It can retain
the sole initial completed artifact after ordinary revision locally fails
JSON/schema validation, after supported structural body loss, or after a
sufficiently established regression of finite public literal minima. The
original structural rule remains unchanged: at least 1,000 initial non-whitespace
characters and three punctuation/line chunks, with at most 100 revised
non-whitespace characters, one revised chunk and a revision-to-draft ratio at
most 0.1. A separate new branch checks a single strict Markdown ATX heading with
no body, the same initial size/chunk floor and ratio ceiling. It does not
broaden the 100-character ceiling for ordinary prose.

The title-scope probe recognizes only finite independent positive title-only
instructions. A recognized title-only task suppresses structural selection;
unknown only-output scope suppresses the new heading branch, while the prior
short-output heuristic remains in place. This is a textual aid with incomplete
language coverage, not a general instruction classifier.

The literal compiler recognizes finite independent positive English Use/Include
instructions specifying one ASCII alphabetic word and a digit minimum. It
counts the complete decoded artifact, not a bounded diagnostic vocabulary.
An original-case standalone strict count at least the minimum is sufficient
for `PASS`; even the broader overlapping casefolded substring count below the
minimum is sufficient for `FAIL`; intermediate observations are `UNKNOWN`.
The casefolded strict count is diagnostic only. Unicode word units and internal
apostrophes/dashes prevent an ASCII prefix in a compound or combined unit from
establishing the strict pass.

If a frequency-like family or recognized scope is unsupported, the complete
frequency plan is unknown and cannot select a candidate by frequency. Quoted
examples, negation, conditionals, scoped/case-qualified counts and unsupported
minima conservatively decline within the compiler's finite syntax. Frequency
selection requires every compiled initial rule to be `PASS` and at least one
revision rule to be clearly `FAIL`; an ambiguous count is not a failure. These
local counts do not certify other public requirements, meaning or quality.

The v19 implementation supplies public artifact counts and finite printed
decimal comparisons to the existing review call. Counts cover the complete
decoded artifact; Han character counts are not word counts, tokens, implicit
body-only scope or a precise interpretation of an approximate length request.
Finite arithmetic results do not validate units, evidence, membership or the
meaning of an expression that is correctly described as false. Unsupported
numeric notation remains outside these observations. At most 128 relations
are observed. Matches are bounded substrings, not validation of a complete
expression; unsupported spacing can leave a partial numeric relation.

Review issues can use an exact `[TASK_QUOTE]...[/TASK_QUOTE]` anchor from the
original public question or constraints. The local observation verifies only
substring presence. An issue-level exact state means at least one supported
quote matched, not that all its quotations matched. It does not prove that the
issue follows from the quote or
that the quoted text is an applicable positive instruction. Quote-grounding observations for unanchored issues
remain unknown and valid evidence corrections are not discarded merely because
they lack a task quote. Inferred rubrics and upstream PASS labels remain fallible
planning material rather than new hard task requirements.

In full revision mode, validated empty review issues instruct ordinary revision to
copy the draft character for character while preserving escaped JSON newlines.
This instruction does not certify the initial draft; a missed issue can leave a
bad draft unchanged. Typed construction still produces its required fields.
The existing full-rewrite layout check retains an eligible ordinary initial draft
only when the validated review has no issues, it has at least two blank-line paragraphs, the revised artifact is one
nonempty line, and revision equals initial with only CR/LF characters removed.
Finite recognized single-line/paragraph, code, CSV, heading and other layout
requirements conservatively disable this branch or make its scope unknown.
Other whitespace or content changes do not qualify. Its language coverage is
limited, and it does not infer overall writing quality. Review and revision
remain two fixed calls with full candidate and cost records.

Eligibility means a nonempty execution `final_answer`; it does not certify
semantic correctness or satisfaction of all public constraints. A projected
initial draft or active positional/numeric construction is ineligible. Review
failures, provider errors, exhausted budgets and typed-construction failures
continue to fail. Both candidate hashes, raw component failures, selection
reason and all calls/costs are retained; no evaluator, score comparison, extra
sample or retry chooses the answer. The library default is off, and the flag
requires `public_refinement=true`.
Both public review and revision remain the same two model calls; compilation
and candidate checks add no model API, grader call or quality resampling.
The guard can still retain an overlong initial artifact when the task calls
for a correct short revision. The prior short-prose branch can still miss
101 characters or two chunks; the heading branch recognizes only its finite
ATX shape. For the prior 1,000/100 rule, the 0.1 ratio is mathematically implied
and is not an independent guarantee. Public diagnostic lists and positions can
be truncated; they are not official checkers, and absent diagnostics do not
mean a constraint passed or failed. The literal compiler counts the complete
decoded artifact separately, without certifying semantic correctness.

`public_revision_mode=patch` changes only ordinary public revision. Its library
default remains `full`. The existing second component call returns `edits`, each
with a validated review `issue_index`, unique exact original-draft `old_text`
and supported `new_text`. All edits locate the same original draft. Overlapping
occurrences, overlapping source spans, missing matches, edits that depend on
earlier edits and an empty final artifact fail local applicability checks.
When validated review issues are empty, only an empty edit list is allowed,
and the complete original artifact is preserved locally. This can preserve a
missed defect. An issue index and a unique source span do not prove a repair is
correct. A whole article can still be one span; minimal editing remains a prompt
requirement. Only content outside actual edited spans is guaranteed unchanged.

Active positional and numeric construction retain their existing typed schemas.
Without an enabled guard, patch application errors fail. With an enabled guard,
eligible ordinary initial artifacts can be retained after patch application
errors, with the validated raw response, revision hashes, failure phase and cost
recorded. Provider, review, deadline and typed-construction errors remain failures.
Two calls remain fixed, including empty reviews. No score selects the edits.

`public_membership_observations=true` adds original public condition spans and
limited numerical table observations to existing contributor/writer and review
inputs. Its library default is false. Only complete CSV/Markdown tables in the
canonical fixed public pack can supply values. Unique complete-header binding,
declared source units, raw row/cell provenance and the original year are needed;
unsupported qualifiers, unit denominators and compound suffixes yield UNKNOWN.
Date agreement uses the given `source.date` declaration and an unambiguous
requested year; it does not parse arbitrary contradictory cohorts in source prose
or certify historical applicability. Arithmetic is conditional on supplied scope
and transcription. Inferred rubrics and upstream PASS remain fallible claims.
Overall membership remains UNKNOWN and independently_verified false: no semantic
eligibility, aliases, source truth or complete set is certified. UNKNOWN is not FAIL.

These observations do not construct or select a final answer set. They guide
the existing models while preserving the original task and evidence. Finite
limits and truncation are recorded. The tested numerical source sidecar is
68,225 characters before JSON whitespace; it adds real input tokens to the same
common budget. Calls, tokens, component failures and comparisons retain actual
costs. Complete result metadata remains separate for every source version.

`planning_response_format=json_schema` requests strict schemas for predict,
local-plan and reconciliation records; its library default is `json_object`.
Generated planning annotations have compact length limits, while the original
task, evidence and final deliverable remain complete. In iterative planning
with no common call ceiling, generated `max_calls` and `total_max_calls` must
remain `null`; expected-call estimates do not create execution ceilings.
Local schema and graph validation still apply if the provider ignores the
requested schema. If a response ends at `length`, its raw text stays in the
audit; the existing contract-correction request uses the original inputs and
failure metadata instead of replaying the truncated tail. This does not add
a correction attempt or permit a different format/model fallback.

All five role-level `frequency_penalty` values remain `null`; the common provider
configuration sends no role-level repetition penalty. The component option
`public_revision_frequency_penalty=0.25` applies only to the final public revision
and is recorded with its actual call options and hash. It does not set penalties
for planning, harness generation, local execution, public review or judging.
The library default remains `None`; earlier adverse variants are retained, and
this combination must be evaluated with its full registered inventory.
Unsupported parameters fail and retain their cost without fallback or another
sample.

The common per-task envelope is 2,000,000 tokens and 900 active seconds shared by
generation and deferred evaluation, with one candidate and no model-call cap.
Execution and judge responses have an 8,192-token limit; meta/global/local
responses have a 16,000-token limit. Actual consumption and failures are still
reported. `available_tools=[]` grants no external actor tools; frozen public
evidence must be supplied separately when registering a shared-evidence run.

The runner defaults to fixed source EVO selection. `--partition exposed_test`
is a separate, explicit development exposure option; its contaminated tasks
cannot be reported as clean formal TEST results. The runner seals all method
submissions before scoring and does not replace failed or low-scoring slots.
Instruction tasks use the pinned official checker; other sources can use the
same-model diagnostic judge and are not independent-judge evidence.
