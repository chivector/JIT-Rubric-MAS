

            |
# 1. Existing experimental anchor \[R\]

The supplied manuscript evaluates 160 test tasks from Nobel, CORDIS, NASA Exoplanet Archive, UNESCO, and dblp, grouped as 40 complete-download/single-table, 35 join, 30 coverage/partition, 30 scoped-negative, and 25 historical-version/repair tasks. Each task is run with three fixed seeds; 50 requests and a 200k-token ceiling form the primary setting. The main paper also reports 25- and 100-request conditions. These are manuscript-reported facts, not new experiments.

Table S0. Main-paper results, reproduced for comparison only \[R\]

| **Method**             | **Exact-set acc.** | **Valid cert.** | **F1** | **Invalid / Exact declarations** |
|------------------------|--------------------|-----------------|--------|----------------------------------|
| Best control + checker | 65.63%             | 59.38%          | 0.890  | 1 / 96 per run                   |
| CompletionSearch       | 73.75%             | 67.50%          | 0.920  | 1 / 109 per run                  |

Source: manuscript Table 2, p. 6. The reported deltas are +8.125 percentage points for accuracy and valid-certification coverage.

Table S0b. Main-paper realized resource costs \[R\]

| **Method**             | **Requests** | **Model tokens** | **Transfer** | **Latency (median/p95)** |
|------------------------|--------------|------------------|--------------|--------------------------|
| Best control + checker | 17           | 65k              | 190 MB       | 102 / 305 s              |
| CompletionSearch       | 33           | 128k             | 58 MB        | 142 / 372 s              |

Source: manuscript Table 6, p. 11. Realized resource use—not just common caps—must be part of any efficiency claim.

# 2. Shared protocol and evaluation safeguards \[A\]

## 2.1 Tasks, versions, split, and instrumentation

1.  Freeze a benchmark manifest containing task_id, publisher/collection identifier, authoritative scope S, temporal interpretation tau, supported query AST Q, answer key/grain kappa, task type, source snapshot identifiers, and gold-keyset checksum. Keep gold manifests inaccessible to any agent or adapter during inference.

2.  Run every method with the same model service, retrieval/search backend, publisher permissions, extraction adapters, checker implementation and checker token accounting; record backend/solver versions, decoding temperature, seed mapping, tool retries, redirects, payload truncation and cache hits.

3.  Pair methods at the task×seed level. Interleave execution order, using pinned snapshots or an archived request-response replay where feasible to limit Web-version drift. Both methods must see identical publisher scope and access restrictions.

4.  Charge all model API calls and search/network requests, including subrequests in bundled tools. Record actual model input/output tokens, transfer bytes, elapsed time, solver CPU time, extraction time, and certification time separately. A request ceiling is not a matched-realized-cost comparison.

5.  Use independent gold construction and audit. Two curators validate scope/time/key semantics; adjudicate disagreements blind to the method label. Audit every Exact declaration and a pre-registered random sample of Partial cases. A content hash only identifies content; it does not authenticate publisher truth.

## 2.2 Endpoints and statistical analysis

Primary endpoint: exact-set accuracy (the final returned stable-key set equals the independently curated reference set). Co-primary reviewer-facing measurement: valid-certification coverage (a correct, independently auditable Exact declaration divided by all tasks). Secondary endpoints: macro set F1, invalid Exact claims / all Exact declarations (FCR), conservative Partial behavior, verified witness-discharge rate, resource-adjusted accuracy, oracle timeout and fallback frequency, plus planner and network latency.

Resample 160 task blocks with replacement while keeping each task’s three seeds together; stratify by the five task types. Use 10,000 paired stratified bootstrap replicates to obtain 95% confidence intervals for difference in accuracy and certification coverage. Report raw task×seed paired outcome counts and effect sizes; as a sensitivity analysis use a paired sign/permutation test on task-level seed-averaged differences. Treat multiple secondary comparisons as exploratory or apply a Holm correction. For FCR, disclose numerators/denominators and wide uncertainty rather than claiming zero-risk safety.

# 3. E1 — Operational audit of contracts, witnesses, and certificates

## Research question and experimental procedure

Does CompletionSearch actually find answer-changing completions, compile answer-relevant obligations, execute tests with validated postconditions, and deliver source-supported Exact certificates—as opposed to merely adding more search calls? Instrument the full 160-task × three-seed suite. Before inspecting outcomes, choose 30 diagnostic tasks: five each from joins, coverage, scoped negation, and historical versions, plus ten complete-download tasks as a negative-control stratum.

- For each decision step, log constraint-state hash, admissible-region descriptors, source-premise type and locator, witness pair or explicit oracle gap, differing key, typed obligation, candidate semantic tests, selected physical path, predicted cost, realized requests/tokens/latency, checker postcondition and final status.

- Separate contract status into pending, publisher-asserted-and-validated, schema/adapter-derived-and-validated, and rejected. An assertion accepted as a premise is not a guarantee that the publisher is truthful. Record provenance and check strength without merging these categories.

- Compute task-level rates of nontrivial witnesses, successful test postconditions, exact oracle invocations, fallback, fully closed regions and final Exact; disclose how many certifications rely on an explicit source closure contract versus query exclusion of open regions.

Table S1. Expected audit summary — entirely illustrative, NOT measured \[P\]

| **Observable**                              | **Example expected value \[P\]** | **Replacement field in logs** |
|---------------------------------------------|----------------------------------|-------------------------------|
| At least one differing completion found     | 117 / 160 tasks (73.1%)          | witness_found_by_task         |
| At least one oracle invocation              | 146 / 160 tasks (91.3%)          | solver_invoked_by_task        |
| At least one validated closure premise      | 131 / 160 tasks (81.9%)          | validated_closure_by_task     |
| Planner fallback invoked                    | 31 / 160 tasks (19.4%)           | heuristic_fallback_by_task    |
| Unresolved relevant region at stop          | 29 / 160 tasks (18.1%)           | open_relevant_region_by_task  |
| Median actions from obligation to discharge | 2.0 actions                      | validated_discharge_steps     |

These counts are not jointly exclusive and are not derivable from the published aggregates. Replace every row with observed task-level or task×seed measurements and specify the aggregation unit.

## Example trace format: UNESCO negative-closure task

Use the historical example from the paper as a template, NOT as an additional observed run. Initial possible key sets: {1150, 1215} and {1215}; differing key 1150 denotes Liverpool. The typed obligation is cutoff-specific closure of the delisting relation. A physical path requests the accepted delisting register and historical membership evidence; the independent checker validates scope, cutoff and continuation. Only after compatible closure and membership evidence may the key set {1215} be certified. For the supplement, replace this explanation with at least one actual timestamped evidence sequence and a redacted certificate trace.

Artifact: examples/trace\_\<task_id\>.jsonl, with exact oracle decisions, source URIs/locators, accepted premise IDs, evidence hashes, and a replayable certificate object. Failure modes to expose include unsupported semantics, insufficient region capacity, rejected contracts, version repair and timeout.

# 4. E2 — Matching actual acquisition and computation budgets

## Protocol

Repeat the paired 160-task × three-seed comparison of CompletionSearch and Best control + checker, preserving the common 50-request ceiling while imposing the same per-task total model-token cap of 64k, 96k, 128k, or 200k. Stop safely at budget exhaustion and emit a checked Partial/Conflict when Exact is not justified. Count solver/planner and checker-related model calls. Do not optimize models or contract libraries on the test tasks. The original 200k-token row below is \[R\]; tighter-cap rows are planning scenarios \[P\].

Table S2. Token-cap sensitivity at 50 requests — \[P\] hypothetical except 200k \[R\]

| **Token cap** | **Control acc.** | **Ours acc.** | **Delta** | **Control cert.** | **Ours cert.** | **Status** |
|---------------|------------------|---------------|-----------|-------------------|----------------|------------|
| 64k           | 61.9%            | 63.8%         | +1.9 pp   | 54.4%             | 56.3%          | \[P\]      |
| 96k           | 64.4%            | 68.8%         | +4.4 pp   | 57.5%             | 62.5%          | \[P\]      |
| 128k          | 65.0%            | 71.3%         | +6.3 pp   | 58.8%             | 65.0%          | \[P\]      |
| 200k          | 65.63%           | 73.75%        | +8.125 pp | 59.38%            | 67.50%         | \[R\]      |

All token caps are per task, not equalized observed consumption. \[R\] values are only reported aggregate outcomes from the manuscript. The \[P\] trajectories are illustrative targets and must be replaced by complete reruns.

Also run a latency-capped sensitivity analysis (e.g., 90, 150, 240 seconds wall-clock) on exactly the same frozen task set. Preserve the checker’s final opportunity to return Partial rather than issuing unchecked Exact, and report completed-Exact, valid-Exact and timeout rates. Present accuracy–tokens, valid-certification–tokens, and accuracy–latency plots using observed per-task records (not a single mean-cost normalization).

## Analysis decision

Claim greater answer quality at a shared ceiling only if the paired comparison supports it. Claim improved cost efficiency only for a supported cost metric and operating point. For example, the paper-reported 58 MB vs 190 MB transfer favors CompletionSearch on bandwidth, whereas 128k vs 65k model tokens and 142 s vs 102 s median latency do not support unconditional computational-efficiency superiority. Report both sides.

# 5. E3 — Causal attribution through matched-state interventions

## Protocol and fairness

Create 120 pre-registered states, 30 each from join, coverage, scoped-negative and historical/repair tasks. A state contains the same task AST, retrieved evidence, accepted contracts, candidate interfaces, remaining budget, cache snapshot and extraction adapter for every treatment. Run three seeds per state, five validated actions per treatment, and terminate when the state is answer-determined. Measure whether a newly validated observation excludes at least one documented answer-changing completion; do not equate a retrieved hit with discharged ambiguity.

Treatments: (1) Full CompletionSearch (witness + typed obligations + separating-test planning), (2) witness-aware greedy selection (without exact hitting-set optimization), (3) contract-first control without answer-changing witnesses, and (4) random feasible typed obligation as a negative control. Keep the same checker, permitted physical paths and remaining tool/token budget. If one treatment cannot instantiate a legal test, count its action as an explicit capability gap rather than silently giving it extra tools.

Table S3. Five-action matched-state mechanism test — illustrative \[P\]

| **Treatment**                | **Discharged ambiguities within 5 actions** | **Mean actions to discharge** | **Claim status** |
|------------------------------|---------------------------------------------|-------------------------------|------------------|
| Full CompletionSearch        | 82.5%                                       | 3.0                           | \[P\]            |
| Witness-aware greedy         | 75.8%                                       | 3.6                           | \[P\]            |
| Contract-first, no witnesses | 67.5%                                       | 4.1                           | \[P\]            |
| Random feasible obligation   | 56.7%                                       | 4.6                           | \[P\]            |

Values are new hypothetical outcomes on a new 120-state sample; they are NOT results from the manuscript’s Figure 4a. Mean action counts refer to discharged states; unresolved states must also be reported. Bootstrap by parent task (keeping its saved states and seeds together), not by individual actions, and report tokens/seconds for each arm.

For context only, the existing manuscript reports 84% discharge after five actions for CompletionSearch versus 67% scoped BFS, 59% cell filling and 54% keyword continuation (Figure 4a); its exact diagnostic sample count is not specified in the supplied text. State-selection rules and sample counts must be added for an independently replicated comparison.

# 6. E4 — Certificate falsification and fail-closed behavior

## 6.1 Mechanically checkable certificate mutations

Select 100 initially verified, valid Exact certificates across a balanced subset of tasks and seeds. For each certificate, generate one mutation from each of six classes (600 mutated certificates total). The generated mutant must be demonstrably invalid under the declared source-relative semantics; keep the raw source replay immutable. Use both an ordinary checker replay and an independent audit script with an expected rejection reason.

Table S4. Mutation rejection — sample planning targets, NOT observed \[P\]

| **Mutation type**                   | **Mutants** | **Expected rejected \[P\]** | **Main verifier obligation**    |
|-------------------------------------|-------------|-----------------------------|---------------------------------|
| Remove a required continuation page | 100         | 100                         | Collection closure              |
| Alter required row/key grain        | 100         | 100                         | Identity and join cardinality   |
| Change a source-scope tag           | 100         | 100                         | Premise-to-query scope binding  |
| Mix incompatible snapshots          | 100         | 98                          | Temporal version compatibility  |
| Break evidence locator binding      | 100         | 100                         | Provenance replay               |
| Forge a negative-closure derivation | 100         | 99                          | Closed relation + absence rule  |
| Total                               | 600         | 597 (99.5%)                 | Count every rejection by reason |

Do not report the 99.5% total as a measurement. Mutations must be guaranteed to violate mechanically checkable premises; malformed examples alone would overstate verifier strength. Uncaught mutants require a case study; cluster uncertainty by the original certificate or task, not by the 600 dependent mutants.

## 6.2 Adversarial incompleteness and semantic trust limits

Independently replay scoped tasks after randomly hiding 0%, 10%, 20% or 30% of required continuation evidence, using a fixed seed and an admissible censorship procedure. Report valid-certification coverage, false Exact counts, Partial rate, and why remaining Exact declarations were still justified by alternative closure or query-exclusion evidence. The original manuscript has coverage measurements at 0% and 30%; intermediate values below are only hypothetical. Do not infer an error rate from certification coverage alone.

Table S5. Missing-continuation coverage — mixed provenance, clearly labeled

| **Continuation missing** | **Control valid cert.** | **Ours valid cert.** | **Status**                    |
|--------------------------|-------------------------|----------------------|-------------------------------|
| 0%                       | 59.38%                  | 67.50%               | \[R\] original setting        |
| 10%                      | 43.0%                   | 54.0%                | \[P\] illustrative            |
| 20%                      | 29.0%                   | 41.0%                | \[P\] illustrative            |
| 30%                      | 19.0%                   | 29.0%                | \[R\] reported in Section 5.5 |

The 30% values are reported in the main text (Section 5.5), while 10% and 20% are illustrative—not an extraction of Figure 4b. Obtain exact outcomes from fresh paired runs.

Boundary condition: a sufficiently persuasive but false publisher completeness assertion is not necessarily detectable by a source-relative checker. Preserve this as an explicit trust limitation instead of claiming that mutations prove robustness to dishonest publishers. Report separately the rate of detected internal certificate inconsistencies and the rate of disagreement with independent external truth audits.

# 7. E5 — Planner scaling and oracle fallback under finite abstractions

Construct synthetic, fully auditable relational instances varying the trusted fresh-key bound N ∈ {10, 25, 50, 100}, test-library size \|A\| ∈ {8, 16, 32, 64}, the number of guarded joins/negative predicates, and the density of shared unknown variables. Use 100 independent random instances for each row, the same hardware and solver version, and a pre-registered 30-second exact planning cap. Report median/p95 solver runtime (timeout-censored), number of learned disagreement cuts, fallback frequency, optimal objective when provable, and the fraction of exact plans checked against brute force for tractable small libraries.

Table S6. Scaling planning scenario — all numbers hypothetical \[P\]

| **Fresh key bound N** | **Test library \|A\|** | **Median / p95 (s)** | **Fallback rate** | **Status** |
|-----------------------|------------------------|----------------------|-------------------|------------|
| 10                    | 8                      | 0.2 / 0.6            | 0%                | \[P\]      |
| 25                    | 16                     | 0.9 / 3.4            | 2%                | \[P\]      |
| 50                    | 32                     | 5.1 / 21.5           | 9%                | \[P\]      |
| 100                   | 64                     | 22 / 30 (capped)     | 28%               | \[P\]      |

Exact static optimality is the paper’s Proposition 4.1 only under fixed finite libraries, exact observation functions/oracles and additive costs. The online physical policy is heuristic and should not be described as adaptively optimal.

Sanity test: when exact optimization terminates, verify the returned test set satisfies all enumerated answer-changing pairs, and compare its objective against brute-force enumeration for small \|A\|. Preserve unavailable oracle paths as capability gaps; never promote finite-surrogate UNSAT into an Exact certificate for unbounded collection regions.

# 8. E6 — New publisher interfaces and error taxonomy

## External-validity design

Build 40 NEW held-out queries from sources not used to tune or test publisher adapters: 20 queries over semistructured official HTML directories and 20 over independently versioned public archive collections. Each must still have a named authoritative scope, stable membership keys and predicates inside the supported fragment. Create the reference answers independently; use 50 requests, 200k model tokens and three seeds with both policies. Report adapter engineering hours and how many tasks require human intervention or unresolved source contracts.

Table S7. Held-out-interface planning scenario (20 tasks per stratum × 3 seeds) \[P\]

| **New interface**         | **Control acc.** | **Ours acc.** | **Control cert.** | **Ours cert.** |
|---------------------------|------------------|---------------|-------------------|----------------|
| Official HTML directories | 55.0%            | 61.7%         | 35.0%             | 43.3%          |
| Versioned public archives | 45.0%            | 53.3%         | 25.0%             | 33.3%          |
| Pooled 40 tasks           | 50.0%            | 57.5%         | 30.0%             | 38.3%          |

Every percentage in this table is illustrative \[P\], not observed. The pooled values are arithmetically consistent with 60 task×seed outcomes in each of the two equal-sized strata. Do not combine these new tasks with the manuscript’s original 160 without relabeling the evaluation.

Separately add 10 intentionally unsupported query conditions (e.g., unbounded Web-wide negation or fuzzy identity without independent adjudication) to test whether the system abstains with a descriptive Partial instead of falsely issuing Exact. These are guardrails, not scored as in-fragment accuracy tasks. Annotate new failures with a single primary cause and any secondary contributing causes.

## Illustrative failure taxonomy

The manuscript reports 118 correct exact-key sets per 160-task run, leaving 42 incorrect sets. The following distribution is an entirely HYPOTHETICAL allocation designed only to show the reporting format; the actual categories cannot be recovered from manuscript-level aggregates.

Table S8. Example error coding for the original 42 residual errors — fabricated allocation \[P\]

| **Primary coded cause**                      | **Illustrative cases \[P\]** | **Evidence required for attribution**          |
|----------------------------------------------|------------------------------|------------------------------------------------|
| Open collection/negative-closure region      | 12                           | Unresolved partition or absent closure premise |
| Identity or join/grain mismatch              | 9                            | Counterfactual join and stable-key comparison  |
| Historical scope or version repair           | 7                            | Competing snapshots and valid-time evidence    |
| Extraction/adapter semantic error            | 6                            | Raw-page vs typed-fact independent audit       |
| Budget, unavailable path, or solver fallback | 8                            | Timestamped capability gap and access log      |
| Total                                        | 42                           | Matches only the reported residual count       |

The total 42 is implied by 160 − 118 \[R\]; each category count is \[P\] and must be replaced. The distribution assumes each error is assigned exactly one primary cause.

# 9. Paired statistical reporting templates and audit outcomes

## 9.1 Do not infer paired significance from aggregate means

The 50-request primary means correspond to 315 baseline-correct and 354 CompletionSearch-correct task×seed outcomes across three 160-task runs, assuming the manuscript’s mean counts are exact across seeds. Aggregate marginals do not reveal which paired tasks improved, regressed or remained tied. Capture these four outcome cells directly before computing paired uncertainty.

Table S9. Illustrative paired exact-set contingency counts; NOT observed \[P\]

| **Control / Ours** | **Ours correct** | **Ours incorrect** | **Row total** |
|--------------------|------------------|--------------------|---------------|
| Control correct    | 283 \[P\]        | 32 \[P\]           | 315 \[R\]     |
| Control incorrect  | 71 \[P\]         | 94 \[P\]           | 165 derived   |
| Column total       | 354 \[R\]        | 126 derived        | 480 task×seed |

Cell allocations are \[P\], deliberately only consistent with published marginal totals \[R\]. The same-task paired cells MUST come from logs; seed replications are correlated and should not be treated as 480 independent tasks.

For every comparison, publish: task-block bootstrap Δ accuracy (pp), 95% CI, task-block bootstrap Δ valid certificate coverage (pp), 95% CI, discordant-pair counts, and both raw and adjusted p-values if significance tests are used. Publish the absolute invalid Exact counts and rate with an uncertainty interval; “one invalid for each method” is not evidence of equivalent certification risk.


# 10. Highest-score reuse and targeted optimization results [R]

The table below selects the highest completed reusable result already present in the workspace for each benchmark and reports the coverage beside the score. It uses targeted optimization receipts and the existing native evolution trace; it does not start a new full evolution run. Adapted subsets and incomplete full-split attempts remain explicitly labeled.

| **Benchmark** | **Selected optimized result** | **Metric(s)** | **Score** | **Coverage / status** |
|---|---|---|---:|---|
| ResearchRubrics | targeted-compact-v3 / TEST33 recovered | compliance | 0.894626 | 33/33 targeted TEST tasks; highest completed reusable result |
| IFEval | targeted-v11 | prompt strict; instruction strict/loose | 0.975000; 0.984127 | 40/40 targeted TEST tasks; complete |
| IFBench | targeted-v11 final | prompt/instruction strict+loose | 1.000000 | 40/40 targeted TEST tasks; complete |
| DeepSearchQA | targeted research-v5/v6 | Precision / Recall / F1 | 1.000000 / 1.000000 / 1.000000 | ACT + country, 2-task adapted subset |
| DeepResearch Bench II | targeted research-v5 | Overall; IR / Analysis / Presentation | 0.862069; 0.730769 / 1.000000 / 0.833333 | 1-task adapted subset |
| WritingBench | tds_fix5 source TEST | normalized native mean | 0.804082 | 49/50 source TEST receipts; one failed |

Metric-level values and the latest partial full-split coverage are maintained in `outputs/benchmark_best_results_20261012/benchmark_results_full.md` and `.csv`. The highest observed completed native evolution task is WritingBench task 4 (0.933333); it is a single-task evolution maximum and is not substituted for the 49/50 aggregate TEST mean. The latest full-split recovery receipts cover 127/169 DeepSearchQA tasks (mean F1 0.317209) and 25/132 DeepResearch Bench II tasks (mean Overall 0.993726); these partial means are coverage diagnostics, not full-split leaderboard scores.
# 11. Full TEST execution status [R]

A fresh sharded full TEST was attempted for both DeepSearchQA and DeepResearch Bench II without starting an evolution run. All shards stopped before model calls because the current process did not contain `JIT_BENCHMARK_API_KEY`. Therefore the fresh attempt produced 0 scored tasks for both benchmarks. The table reports the latest durable partial receipts separately; these are coverage diagnostics and must not be presented as full-split leaderboard scores.

| benchmark | result |
|---|---|
| DeepSearchQA | Full TEST: not complete; fresh attempt 0/600 because the API credential was missing. Existing receipts: 127/169 scored, F1 mean 0.317209; 473 tasks still lack final scores. |
| DeepResearch Bench II | Full TEST: not complete; fresh attempt 0/132 because the API credential was missing. Existing receipts: 25/132 scored, Overall mean 0.993726; 107 tasks still lack final scores. |

Reproducible table artifact: `outputs/full_test_results_20261012/benchmark_results.md`, with CSV/JSON companions.
# 12. Supplied gateway preflight [R]

The supplied configuration uses model `deepseek-v4-flash-vision` and the supplied Base URL. A credentialed connectivity probe to `/v1/chat/completions` returned HTTP 404 with `ERR_NGROK_3200` and the message that the endpoint is offline. The full TEST was therefore not started against this endpoint. The credential was injected only into the probe process and was not written to disk.
# 13. Full-test judging budget repair [R]

The prior DeepResearch Bench II partial receipts used a 4,096-token judge cap. Many incomplete tasks had a judge `finish_reason=length` while reviewing a 50-item rubric batch. The reproducible full-test configuration therefore raises only the judge output cap to 16,384 tokens; the benchmark rubric, aggregation and generation settings remain unchanged.
