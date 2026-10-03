# Development pilot configuration

`development_pilot.example.json` contains the representative v16 validated
development configuration and common budgets. The complete v16 inventory,
including failures and negative results, is recorded in the experiment history.
Initial-MAS did not beat both baselines overall; this is not a recommendation of
a winning configuration. The example matches the used configuration except for
the endpoint, model alias and exact response-model pin placeholders. Source changes
need their own registrations and validation. The template is not a completed
experiment registration and contains no credentials or local data paths.

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
without format fallback, JSON-prefix selection or another sample.

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
