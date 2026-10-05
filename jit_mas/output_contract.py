"""Public-task construction advice, without parsing or rewriting submissions."""


PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT = """
PUBLIC CONSTRAINT CONSTRUCTION: When the public task constrains the output, privately
check feasibility before drafting. Follow explicit instructions and their stated
precedence; do not silently relax a conflict, invent a requirement or guess a hidden
checker convention. Use the task's stated units, scope and counting rules. Different
tasks can define words, sentences, matches or paragraphs differently.
For a public N-th-sentence/M-th-word requirement, allocate N meaningful sentence slots
before writing and assign each preceding sentence a distinct content function. Construct
the target sentence from M indexed word slots. Use only the needed N sentences unless
another explicit deliverable requires more. Do not repeat a whole sentence or paragraph to
reach an ordinal or consume the token allowance; preserve explicitly required literal word
repetitions. Once the requested content and positions are covered, stop the artifact and
close its required JSON envelope and checkpoint fields.
Build the required structure first: reserve literal start/end text, required terms
and paragraph/line/section slots, then allocate words or sentences among those slots
before filling in substantive content. Include titles, labels and fixed text in counts
when the instructed scope includes them; never add them merely to make counting easier.
For exact word counts or sentence-local word positions, privately construct numbered
word slots under the public counting convention, place constrained literal words in
their required slots, then fill the remaining slots with coherent content. Remove
the construction numbering from the artifact unless the user requested it. During
revision, preserve slot counts in already correct spans and recheck positions after
any added or removed word; a correct outline does not establish a correct final text.
Use clear word boundaries and simple sentence punctuation when the requested genre
permits, avoiding count ambiguity from abbreviations or hyphenation. Preserve required
literal wording and case while applying any broader style or capitalization rule.
Check interactions on the final decoded answer: an added heading, changed case or
replaced phrase can alter counts, keyword occurrences, paragraph structure or boundary
text. Repair the smallest necessary span, preserve already satisfied constraints and
recheck every explicit constraint after the edit. A self-check is not independent tool
verification. Keep construction, reasoning and counting notes outside the answer; the
answer contains only the requested artifact. If a conflict remains unresolved, report
it honestly in internal checkpoints and in the answer only when its format permits.
""".strip()


PUBLIC_ELIGIBILITY_SCOPE_PROMPT = """
PUBLIC PREDICATE SCOPE: A hard inclusion or exclusion predicate may come only from
the original public question, its explicit constraints, or an explicitly named
public source field and operator. Do not promote a proxy statistic, administrative
procedure, catalog or distributor presence, retrieval success, evidence coverage,
freshness or live-stock assumption, citation/URL requirement, or an applicant
identity characteristic into a new qualification rule unless the public task
states it. Keep the primary answer set separate from verification notes and
unverified candidates. For every original predicate, distinguish TRUE (the
observed value satisfies its stated scope), CONTRADICTED (the observed value
fails that same stated scope), and UNKNOWN (the available public material does
not establish either). UNKNOWN is not contradiction and cannot by itself exclude
a candidate; CONTRADICTED affects only the candidate and original predicate it
actually covers. Do not present an unverified candidate as a qualifying member,
and do not turn a missing proxy observation into a failure of the original task.
""".strip()
