# ESG repair results

Task `6847465956a0f6376a60543e`: Patagonia / UNIQLO-Fast Retailing / H&M.

| Candidate | Evaluation scope | Score | Generation accounting | Evaluation accounting |
|---|---|---:|---|---|
| Historical | All 28 native criteria | 0.0645161290 | Historical run | Historical run |
| Compact v3 | Incomplete subset; truncated generation | unavailable | 1 call, 18,948 tokens | 2 calls, 18,563 tokens |
| Compact v4 | Incomplete subset; truncated generation | unavailable | 1 call, 22,756 tokens | 3 calls, 38,548 tokens |
| Compact v5 | Same seven criteria | 0.5000000000 | 1 call, 15,175 tokens | 7 calls, 35,246 tokens |
| Arithmetic v6 | Same seven criteria | 1.0000000000 | 0 new calls; reuses v5 | 7 calls, 33,564 tokens |
| Arithmetic v6 | All 28 native criteria | 0.2795698925 | 0 new calls; reuses v5 | 28 calls, 133,236 tokens |
| Evidence v7 | All 28 native criteria | 0.4408602151 | 0 new calls; reuses v5 | 28 calls, 215,642 tokens |

All listed v5-v7 evaluations are complete. Full scores cover this one ESG task, not an RR dataset mean. The unchanged native positive-weight denominator is 93; v7 satisfies 41 weighted points. Negative fault criteria remain untriggered. Seven-criterion scores are diagnostic subsets only.

V5 used 11,326 input plus 3,849 output tokens and finished with `stop`; it has no truncation. V7 is a complete 3,249-word local report. No v6/v7 model generation or network retrieval calls occur. The costs of v5 generation are reused rather than counted as zero overall. Historical outputs, original scores and all intermediate candidates are retained.

V6 freezes subjective v5 component inputs and corrects every category sum, contribution, sensitivity and rank with decimal arithmetic. V7 adds archive facts with exact source offsets and hashes: legal entities/ownership; common FY2024 scopes; detailed FR emissions; material definitions/trends; H&M wage/grievance/forced-labor data; Patagonia wage targets and disclosed litigation; energy trends and partial third-party ratings. Component judgments, weights and ranking remain frozen. These judgments are not claimed as empirically calibrated metrics.

V7 source-additions hash: `ff9889ebe39ef244a9390375f3683d30476b81694518b1b551d21ffebf5f758a`.

V7 answer normalized-text hash: `20b70e18b06fdce8d9f50a1e1d0150de6422f5d88569a53f3f1cb16fcf0ec303`.

Remaining failures concern water metrics for every company, complete official material rules, continuous Patagonia three-year scope-normalized trends, comparable three-year materials/waste data for all companies, country sourcing prohibitions, independent controversies and financial outcomes, all-company grievance accessibility/outcomes, current comparable ratings and rating divergences. The correct frozen composite ranks H&M above Patagonia, while one native criterion mandates the opposite ranking. Source limitations remain explicit.

One judgment varies despite unchanged underlying availability: the three-year climate criterion passed v6's nonconsecutive Patagonia series but failed v7's more explicit two-year scope detail. V7's gain also includes a permissive judgment counting subjective E/S/T scores as normalized measures. Accordingly, preserve individual judge reasons alongside aggregate scores; the numerical gain is a real native evaluation result, not proof of a general scoring method.
