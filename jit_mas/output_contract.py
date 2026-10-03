"""Public-task construction advice, without parsing or rewriting submissions."""


PUBLIC_CONSTRAINT_CONSTRUCTION_PROMPT = """
PUBLIC CONSTRAINT CONSTRUCTION: When the public task constrains the output, privately
check feasibility before drafting. Follow explicit instructions and their stated
precedence; do not silently relax a conflict, invent a requirement or guess a hidden
checker convention. Use the task's stated units, scope and counting rules. Different
tasks can define words, sentences, matches or paragraphs differently.
Build the required structure first: reserve literal start/end text, required terms
and paragraph/line/section slots, then allocate words or sentences among those slots
before filling in substantive content. Include titles, labels and fixed text in counts
when the instructed scope includes them; never add them merely to make counting easier.
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
