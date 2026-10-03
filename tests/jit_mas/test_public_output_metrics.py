"""Public-only diagnostics preserve the decoded artifact and expose units."""

import pytest

from jit_mas.public_output_metrics import public_output_metrics
from jit_mas.schemas import PublicTask


def test_counts_decoded_multilingual_artifact_and_public_terms_without_metadata():
    task = PublicTask(task_id="PRIVATE_IDENTIFIER_CANARY", question='Put "ocean" in position 3.', constraints=['Finish with "P.S.".'])
    answer = "A blue ocean shines.\n\n\u6d77\u6d0b 42\nP.S. ocean"
    report = public_output_metrics(task, answer)
    assert report["characters"] == len(answer)
    assert report["cjk_characters"] == 2
    assert report["decimal_digit_characters"] == 2
    assert report["blank_line_paragraphs"] == 2
    ocean = next(item for item in report["quoted_terms"] if item["term"] == "ocean")
    assert ocean["literal_substring_count"] == ocean["word_boundary_casefold_count"] == 2
    assert ocean["positions_in_punctuation_or_line_chunks"][0] == {"sentence_chunk": 1, "word_positions": [3]}
    assert "PRIVATE_IDENTIFIER_CANARY" not in str(report)
    assert answer == "A blue ocean shines.\n\n\u6d77\u6d0b 42\nP.S. ocean"


def test_distinguishes_literal_case_and_whole_words_and_labels_tokenization():
    task = PublicTask(task_id="t", question='Use “sea” and ‘don\u2019t’.')
    report = public_output_metrics(task, "SEA sea seashore don't don\u2019t")
    sea = next(item for item in report["quoted_terms"] if item["term"] == "sea")
    assert sea["literal_substring_count"] == 2
    assert sea["word_boundary_casefold_count"] == 2
    assert "may differ" in report["counting_basis"]


def test_quotes_are_deduplicated_bounded_and_do_not_parse_unquoted_guesses():
    task = PublicTask(task_id="t", question='Use "one" and "one". Do not guess keyword two.')
    report = public_output_metrics(task, "one two")
    assert [item["term"] for item in report["quoted_terms"]] == ["one"]
    many = PublicTask(task_id="t", question=" ".join('`term%d`' % n for n in range(50)))
    assert len(public_output_metrics(many, "term0")["quoted_terms"]) == 32


def test_empty_draft_is_measured_without_fabricated_compliance():
    report = public_output_metrics({"question": "Write exactly 30 words.", "constraints": []}, "")
    assert report["characters"] == report["regex_word_count"] == report["blank_line_paragraphs"] == 0
    assert "passed" not in report and "compliant" not in report


def test_v2_preserves_base_fields_and_has_no_candidate_or_compliance_fields():
    task = PublicTask(task_id="generic", question='Write a passage with "ocean".')
    answer = "Ocean winds move.\nWe see 12 birds."
    report = public_output_metrics(task, answer)
    expected_base = {
        "characters": 34,
        "non_whitespace_characters": 28,
        "cjk_characters": 0,
        "whitespace_word_count": 7,
        "regex_word_count": 7,
        "nonempty_lines": 2,
        "blank_line_paragraphs": 1,
        "punctuation_or_line_chunks": 2,
        "decimal_digit_characters": 2,
        "quoted_terms": [{
            "term": "ocean",
            "literal_substring_count": 0,
            "word_boundary_casefold_count": 1,
            "positions_in_punctuation_or_line_chunks": [
                {"sentence_chunk": 1, "word_positions": [1]}],
            "position_list_truncated": False,
        }],
    }
    assert {key: report[key] for key in expected_base} == expected_base
    assert report["schema_version"] == "public-artifact-diagnostics-v2"
    assert report["english_word_frequencies"]["frequencies"]["ocean"] == 1
    assert "candidate_version" not in report and "additional_scope" not in report
    assert "compliance" not in report and "score" not in report


def test_unquoted_english_words_and_joined_forms_have_explicit_counts():
    task = {"question": "Describe a coast and its weather."}
    report = public_output_metrics(
        task, "Sea SEA seaside sea-green sea's sea\u2019s alpha2 \u4e2d\u6587sea.")
    words = report["english_word_frequencies"]["frequencies"]
    assert words == {"sea": 2, "sea-green": 1, "sea's": 1, "sea\u2019s": 1, "seaside": 1}
    assert report["quoted_terms"] == []
    assert report["english_word_frequencies"]["total_word_occurrences"] == 6


def _alphabetic_word(index):
    letters = []
    while True:
        letters.append(chr(ord("a") + index % 26))
        index //= 26
        if not index:
            return "term" + "".join(reversed(letters))


def test_frequency_limit_reports_full_and_omitted_types():
    answer = " ".join(_alphabetic_word(index) for index in range(1200))
    words = public_output_metrics(
        {"question": "Describe the subject."}, answer)["english_word_frequencies"]
    assert words["total_distinct_words"] == 1200
    assert words["returned_distinct_words"] == 1024
    assert len(words["frequencies"]) == 1024
    assert words["omitted_distinct_words"] == 176
    assert words["truncated"] is True


def test_sentence_arrays_keep_two_documented_token_conventions():
    answer = "One two.\nThree-four five!\n\u6700\u540e \u4e00\u6bb5\u3002"
    report = public_output_metrics(
        {"question": "Write a short passage."}, answer)["sentence_chunk_tokens"]
    assert report["total_chunks"] == 3
    first, second, last = report["chunks"]
    assert first["whitespace_tokens"] == ["One", "two."]
    assert first["unicode_word_tokens"] == ["One", "two"]
    assert second["whitespace_tokens"] == ["Three-four", "five!"]
    assert second["unicode_word_tokens"] == ["Three-four", "five"]
    assert last["unicode_word_tokens"] == ["\u6700\u540e", "\u4e00\u6bb5"]
    assert [row["chunk_index_1_based"] for row in report["chunks"]] == [1, 2, 3]
    assert report["returned_token_items"] == 12
    assert report["token_items_truncated"] is False
    assert "not an official checker position" in report["counting_basis"]


def test_global_sentence_token_limit_is_shared_by_both_arrays():
    answer = " ".join(["alpha"] * 5000)
    report = public_output_metrics(
        {"question": "Write a long passage."}, answer)["sentence_chunk_tokens"]
    row = report["chunks"][0]
    assert row["whitespace_word_count"] == row["unicode_word_count"] == 5000
    assert len(row["whitespace_tokens"]) == len(row["unicode_word_tokens"]) == 2048
    assert report["returned_token_items"] == 4096
    assert report["total_token_items_in_returned_chunks"] == 10000
    assert row["whitespace_tokens_truncated"] is row["unicode_word_tokens_truncated"] is True
    assert report["token_items_truncated"] is True
    assert "not model-tokenizer tokens" in report["token_allocation"]


def test_sentence_limit_reports_unreturned_chunks_without_claiming_completion():
    answer = "\n".join(["Node."] * 130)
    report = public_output_metrics(
        {"question": "Describe nodes."}, answer)["sentence_chunk_tokens"]
    assert report["total_chunks"] == 130
    assert report["returned_chunks"] == len(report["chunks"]) == 128
    assert report["chunks_truncated"] is True
    assert report["omitted_chunks"] == 2
    assert report["token_items_truncated"] is False


def test_numeric_tokens_distinguish_literals_from_characters_and_identifiers():
    answer = "-12.50, +3, 1,234.00, .75, \u22128, 2e3, A7, two, 4/5, \u0664\u0662."
    report = public_output_metrics({"question": "Discuss some quantities."}, answer)
    numbers = report["numeric_token_diagnostics"]
    assert [row["token"] for row in numbers["tokens"]] == [
        "-12.50", "+3", "1,234.00", ".75", "\u22128", "2e3", "4", "5"]
    assert numbers["total_numeric_tokens"] == 8
    assert report["decimal_digit_characters"] == 21
    assert "cannot replace this numeric-token count" in numbers["counting_basis"]
    for row in numbers["tokens"]:
        assert answer[row["character_start"]:row["character_end_exclusive"]] == row["token"]


def test_numeric_list_limit_preserves_actual_total():
    answer = " ".join(["3"] * 1030)
    report = public_output_metrics(
        {"question": "Discuss repeated quantities."}, answer)["numeric_token_diagnostics"]
    assert report["total_numeric_tokens"] == 1030
    assert report["returned_numeric_tokens"] == len(report["tokens"]) == 1024
    assert report["omitted_numeric_tokens"] == 6
    assert report["truncated"] is True


def test_fanboys_are_lexical_types_not_a_grammar_or_score_claim():
    answer = "For a day, and another day; AND curiosity, nor fear, but hope or courage, yet humility, so purpose."
    report = public_output_metrics(
        {"question": "Write reflective prose."}, answer)["fanboys_lexical_diagnostics"]
    assert report["lexical_occurrence_counts"] == {
        "for": 1, "and": 2, "nor": 1, "but": 1, "or": 1, "yet": 1, "so": 1}
    assert report["different_lexical_types_present"] == 7
    assert report["total_lexical_occurrences"] == 8
    assert "grammatically" in report["counting_basis"]


def test_empty_artifact_returns_empty_counts_and_nontext_is_rejected():
    report = public_output_metrics({"question": "Write a passage."}, "")
    assert report["english_word_frequencies"]["frequencies"] == {}
    assert report["sentence_chunk_tokens"]["chunks"] == []
    assert report["numeric_token_diagnostics"]["tokens"] == []
    assert report["fanboys_lexical_diagnostics"]["different_lexical_types_present"] == 0
    with pytest.raises(TypeError):
        public_output_metrics({"question": "Write a passage."}, None)
