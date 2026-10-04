from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.evidence import (EvidenceConfig, EvidencePackBuilder, apply_evidence_pack,
                              build_evidence_manifest, canonical_url, evidence_pack_path,
                              load_evidence_pack, load_evidence_tasks, save_evidence_pack,
                              _json, _round_robin_pack, _sha, _tokenizer, public_instruction_question,
                              EVIDENCE_QUESTION_HEADER, EVIDENCE_TASK_CONSTRAINT, EVIDENCE_SCOPE,
                              RENDERER_IDENTITY)
from jit_mas.schemas import PublicTask, digest
from scripts.prepare_benchmark_evidence import _public_attachment_loader, _read_tasks, main


class QueryPlanner:
    def __init__(self, queries=None, *, raw=None, fail=False):
        self.queries = ["public first query", "public second query"] if queries is None else queries
        self.raw = raw
        self.fail = fail
        self.calls = []

    def __call__(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        if self.fail:
            raise RuntimeError("synthetic transport failure")
        return SimpleNamespace(content=self.raw if self.raw is not None else json.dumps({"queries": self.queries}))

    def get_token_counts(self):
        return {"input_token_count": 17, "output_token_count": 11}


def task(task_id="public-task", **kwargs):
    return PublicTask(task_id=task_id, question="Compare the public evidence before 2024.", **kwargs)


def make_pack(public=None, **kwargs):
    public = public or task()
    builder = EvidencePackBuilder(QueryPlanner(["public search"]),
                                  lambda query: [{"url": "https://example.org/source", "title": "Source"}],
                                  lambda url: "Public source evidence with a stable locator.", **kwargs)
    return builder.build(public)


def test_single_query_call_and_bounded_deterministic_tool_order():
    planner = QueryPlanner([f"query {i}" for i in range(4)])
    queries, urls = [], []

    def search(query):
        queries.append(query)
        index = int(query[-1])
        return [{"url": f"https://example.org/{index * 4 + rank}#fragment"} for rank in range(8)]

    def crawl(url):
        urls.append(url)
        return f"Evidence from {url}."

    pack = EvidencePackBuilder(planner, search, crawl).build(task())
    assert queries == [f"query {i}" for i in range(4)]
    assert urls == [f"https://example.org/{index}" for index in (0, 4, 8, 12, 1, 5, 9, 13)]
    assert len(planner.calls) == 1
    assert planner.calls[0][1] == {"max_tokens": 4096, "temperature": 0}
    assert pack["usage"]["model_calls"] == 1
    assert pack["usage"]["tokens"] == 28
    assert pack["usage"]["tool_calls"] == 12
    assert len(pack["archive"]) == 13
    assert len(pack["sources"]) == 8
    assert pack["builder_identity"]["candidate_selection_policy"] == "rank-then-query-round-robin-v1"
    crawls = [row for row in pack["archive"] if row["kind"] == "crawl_page"]
    assert [row["query_index"] for row in crawls] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert all("raw_output" in row for row in pack["archive"])


def test_balanced_candidates_deduplicate_after_ranking_and_do_not_count_public_exclusions():
    urls = []
    rows = {
        "first": [{"url": "https://BLOCKED.org/article?utm_source=first"},
                  {"url": "https://public.org/shared#first"},
                  {"url": "https://public.org/first"}],
        "second": [{"url": "https://public.org/shared?utm_source=second"},
                   {"url": "https://public.org/second"}],
    }

    def crawl(url):
        urls.append(url)
        return "Synthetic public evidence."

    pack = EvidencePackBuilder(QueryPlanner(["first", "second"]), rows.get, crawl,
                               config=EvidenceConfig(max_pages=3)).build(
                                   task(), public_excluded_urls=["https://blocked.org/article"])
    assert urls == ["https://public.org/shared", "https://public.org/second", "https://public.org/first"]
    crawls = [row for row in pack["archive"] if row["kind"] == "crawl_page"]
    assert [(row["query_index"], row["rank"]) for row in crawls] == [(1, 1), (1, 2), (0, 3)]
    assert pack["usage"]["tool_calls"] == 5
    assert pack["usage"]["model_calls"] == 1
    assert "https://blocked.org/article" not in urls


def test_single_query_keeps_original_rank_order_and_page_limit():
    urls = []

    def crawl(url):
        urls.append(url)
        return "Synthetic source text."

    pack = EvidencePackBuilder(QueryPlanner(["single"]),
                               lambda query: [{"url": f"https://public.org/{rank}"} for rank in range(5)],
                               crawl, config=EvidenceConfig(max_pages=3)).build(task())
    assert urls == [f"https://public.org/{rank}" for rank in range(3)]
    assert pack["usage"]["tool_calls"] == 4
    assert pack["usage"]["model_calls"] == 1


def test_previous_builder_identity_still_loads_immutable_v1_pack(tmp_path):
    public = task()
    pack = make_pack(public)
    del pack["builder_identity"]["candidate_selection_policy"]
    del pack["builder_identity"]["renderer_identity"]
    del pack["renderer_identity"]
    encoding, _ = _tokenizer()
    pack["body"], pack["rendered"], pack["token_count"] = _round_robin_pack(
        public.task_id, pack["sources"], encoding, EvidenceConfig(**pack["config"]))
    pack["builder_code_sha256"] = "synthetic-historical-builder"
    pack["pack_sha256"] = digest({key: value for key, value in pack.items() if key != "pack_sha256"})
    path = evidence_pack_path(tmp_path, public.task_id)
    save_evidence_pack(pack, path)
    original = path.read_bytes()
    loaded = load_evidence_pack(path, public)
    assert loaded["rendered"] == pack["rendered"]
    assert loaded["body"] == pack["body"]
    assert path.read_bytes() == original


def test_canonical_dedup_and_public_source_exclusion_and_redirect():
    urls = []

    def crawl(url):
        urls.append(url)
        return {"text": "Do not leak redirected excluded source text", "url": "https://blocked.org/article"}

    pack = EvidencePackBuilder(QueryPlanner(["query"]), lambda q: [
        {"url": "https://BLOCKED.org/article?utm_source=a#x"},
        {"url": "https://allowed.org/p?utm_source=x"},
        {"url": "https://allowed.org/p#same"},
    ], crawl).build(task(), public_excluded_urls=["https://blocked.org/article"])
    assert urls == ["https://allowed.org/p"]
    assert pack["sources"][0]["status"] == "excluded"
    assert "Do not leak" not in pack["rendered"]
    assert "Do not leak" in json.dumps(pack["archive"])
    assert canonical_url("HTTPS://Example.org:443/p?b=2&a=1&utm_source=x#f") == "https://example.org/p?a=1&b=2"
    with pytest.raises(ValueError, match="credential-free"):
        canonical_url("https://user:secret@example.org")


def test_window_token_bound_locators_and_exact_unicode_source_slices():
    public = task(attachments=["public.md"])
    original = {"https://a.org/": "one two three " * 1000,
                "https://b.org/": "\u4e2d\u6587\u8bc1\u636e\u3002" * 1000,
                "public.md": "attachment evidence " * 1000}
    config = EvidenceConfig(max_pack_tokens=650, round_robin_chunk_tokens=16)
    pack = EvidencePackBuilder(QueryPlanner(["query"]),
                               lambda q: [{"url": "https://a.org/"}, {"url": "https://b.org/"}],
                               lambda url: original[url], config=config).build(
                                   public, attachment_loader=lambda locator: original[locator])
    assert pack["token_count"] <= 650
    assert pack["tokenizer"]["encoding"] == "cl100k_base"
    assert pack["tokenizer"]["provider_native_tokenizer"] is False
    for source in pack["body"]["sources"]:
        assert source["segments"]
        for segment in source["segments"]:
            assert segment["text"] == original[source["locator"]][segment["span_start"]:segment["span_end"]]
            assert segment["source_sha256"] == source["source_sha256"] == _sha(original[source["locator"]])
            assert "\ufffd" not in segment["text"]
        assert source["truncated"] is True


def synthetic_window_pack(public, texts, *, token_budget=1100, queries=None):
    return EvidencePackBuilder(QueryPlanner(queries or ["complementary public query"]),
        lambda query: [{"url": url} for url in texts], lambda url: texts[url],
        config=EvidenceConfig(max_pack_tokens=token_budget)).build(public)


def assert_exact_segments(pack, texts):
    assert pack["token_count"] <= pack["config"]["max_pack_tokens"]
    for source in pack["body"]["sources"]:
        original = texts[source["locator"]]
        previous_end = -1
        for segment in source["segments"]:
            assert 0 <= segment["span_start"] < segment["span_end"] <= len(original)
            assert segment["span_start"] > previous_end
            assert segment["text"] == original[segment["span_start"]:segment["span_end"]]
            assert segment["source_sha256"] == source["source_sha256"] == _sha(original)
            assert "\ufffd" not in segment["text"]
            previous_end = segment["span_end"]


def test_relevance_windows_retain_long_tail_claim_negation_and_units(tmp_path):
    public = PublicTask(task_id="synthetic-tail", question="What does the zephyr treatment evidence establish?")
    tail = "The earlier apparent benefit did not establish causation. Zephyr reduced the reading by 12.5 mg/L. The unit applies to concentration, not dose."
    texts = {"https://public.org/report": "Navigation and unrelated introductory material. " * 350 + tail}
    pack = synthetic_window_pack(public, texts, token_budget=900, queries=["zephyr concentration evidence"])
    selected = "".join(segment["text"] for segment in pack["body"]["sources"][0]["segments"])
    assert tail in selected
    assert pack["body"]["sources"][0]["truncated"] is True
    assert_exact_segments(pack, texts)
    path = save_evidence_pack(pack, evidence_pack_path(tmp_path, public.task_id))
    assert load_evidence_pack(path, public) == pack


def test_public_query_can_retain_a_fact_absent_from_question_terms():
    public = PublicTask(task_id="synthetic-query", question="Summarize the result.")
    target = "Eligibility includes only adults. Rivermark enrollment was 37. The count excludes withdrawn candidates."
    texts = {"https://public.org/query": "Unrelated background. " * 400 + target}
    pack = synthetic_window_pack(public, texts, token_budget=750, queries=["rivermark enrollment"])
    assert target in "".join(segment["text"] for segment in pack["body"]["sources"][0]["segments"])
    assert_exact_segments(pack, texts)


def test_relevance_windows_retain_chinese_context_and_unicode_offsets():
    public = PublicTask(task_id="synthetic-cjk", question="银杉浓度变化如何解释？")
    target = "这不是因果证据。银杉浓度下降了三毫克每升。该数值使用相同分母。"
    texts = {"https://public.org/chinese": "页面菜单与无关内容。" * 650 + target}
    pack = synthetic_window_pack(public, texts, token_budget=950, queries=["银杉浓度"])
    assert target in "".join(segment["text"] for segment in pack["body"]["sources"][0]["segments"])
    assert_exact_segments(pack, texts)


def test_relevance_windows_preserve_each_source_and_keep_disjoint_spans_separate():
    public = PublicTask(task_id="synthetic-fair", question="Compare aurora and basalt measurements.")
    texts = {
        "https://public.org/first": ("Background. " * 120 + "Only preliminary evidence is available. Aurora measured 7 kg. The measure is mass. "
                                    + "Separate background. " * 180 + "This is not a replication. Basalt measured 4 kg. The method differed. "
                                    + "More background. " * 130),
        "https://public.org/second": "Unrelated lead. " * 300 + "Context was fixed. Aurora measured 9 kg. This is not a causal estimate.",
    }
    pack = synthetic_window_pack(public, texts, token_budget=1300, queries=["aurora basalt"])
    first, second = pack["body"]["sources"]
    assert "Aurora measured 7 kg." in "".join(segment["text"] for segment in first["segments"])
    assert "Basalt measured 4 kg." in "".join(segment["text"] for segment in first["segments"])
    assert "Aurora measured 9 kg." in "".join(segment["text"] for segment in second["segments"])
    assert len(first["segments"]) >= 2
    assert_exact_segments(pack, texts)


def test_short_evidence_is_complete_and_long_unmatched_fallback_is_deterministic(monkeypatch):
    monkeypatch.setattr("jit_mas.evidence.utc_now", lambda: "2026-10-03T00:00:00Z")
    public = PublicTask(task_id="synthetic-fallback", question="Assess zygomatic adaptation.")
    short = {"https://public.org/short": "  Exact short evidence.\n\nUnits: kg. No effect was established.  "}
    pack = synthetic_window_pack(public, short)
    assert pack["body"]["sources"][0]["segments"][0]["text"] == short["https://public.org/short"]
    assert pack["body"]["sources"][0]["truncated"] is False
    assert_exact_segments(pack, short)
    long = {"https://public.org/long": "Unrelated alphabetic sentence. " * 400}
    first = synthetic_window_pack(public, long, token_budget=720, queries=["zygomatic"])
    second = synthetic_window_pack(public, long, token_budget=720, queries=["zygomatic"])
    assert first["body"] == second["body"]
    assert first["body"]["sources"][0]["segments"]
    assert_exact_segments(first, long)


def test_unknown_or_modified_renderer_identity_is_rejected(tmp_path):
    public = task()
    pack = make_pack(public)
    assert pack["renderer_identity"] == RENDERER_IDENTITY
    pack["renderer_identity"]["version"] = "unsupported-renderer"
    pack["pack_sha256"] = digest({key: value for key, value in pack.items() if key != "pack_sha256"})
    path = save_evidence_pack(pack, tmp_path / "unknown-renderer.json")
    with pytest.raises(ValueError, match="Unknown evidence renderer"):
        load_evidence_pack(path, public)


def test_unknown_candidate_selection_identity_is_rejected(tmp_path):
    public = task()
    pack = make_pack(public)
    pack["builder_identity"]["candidate_selection_policy"] = "unsupported-selection"
    pack["pack_sha256"] = digest({key: value for key, value in pack.items() if key != "pack_sha256"})
    path = save_evidence_pack(pack, tmp_path / "unknown-candidates.json")
    with pytest.raises(ValueError, match="Unknown evidence candidate selection"):
        load_evidence_pack(path, public)


def test_public_instruction_extraction_accepts_only_complete_generated_evidence_suffix():
    original = "Write a research report." + EVIDENCE_QUESTION_HEADER + "This marker belongs to the task."
    public = PublicTask(task_id="synthetic-boundary", question=original)
    body = {"task_id": public.task_id, "scope": EVIDENCE_SCOPE, "sources": []}
    applied = public.model_copy(update={"question": original + EVIDENCE_QUESTION_HEADER + _json(body),
                                       "constraints": [EVIDENCE_TASK_CONSTRAINT]})
    assert public_instruction_question(applied) == original
    assert public_instruction_question(applied.model_dump()) == original
    assert public_instruction_question(public) == original
    for changed in (dict(body) | {"task_id": "wrong"}, dict(body) | {"scope": "wrong"},
                    dict(body) | {"extra": True}, dict(body) | {"sources": "wrong"}):
        invalid = applied.model_copy(update={"question": original + EVIDENCE_QUESTION_HEADER + _json(changed)})
        assert public_instruction_question(invalid) == invalid.question
    for suffix in (json.dumps(body), _json(body) + " trailing", "malformed JSON"):
        invalid = applied.model_copy(update={"question": original + EVIDENCE_QUESTION_HEADER + suffix})
        assert public_instruction_question(invalid) == invalid.question
    missing_constraint = applied.model_copy(update={"constraints": []})
    assert public_instruction_question(missing_constraint) == missing_constraint.question


def test_failures_preserve_archive_and_locators_without_hidden_retry():
    def search(query):
        if query == "first":
            raise TimeoutError()
        return [{"url": "https://example.org/missing"}]

    pack = EvidencePackBuilder(QueryPlanner(["first", "second"]), search,
                               lambda url: "Error reading page: synthetic failure").build(
                                   task(attachments=["unreadable.pdf"]))
    assert pack["status"] == "complete"
    assert {row["locator"] for row in pack["sources"]} == {
        "query:0", "https://example.org/missing", "unreadable.pdf"}
    assert all(row["status"] == "error" for row in pack["sources"])
    assert pack["usage"]["model_calls"] == 1
    assert pack["usage"]["tool_calls"] == 3


@pytest.mark.parametrize("planner", [QueryPlanner(raw="invalid JSON"), QueryPlanner(fail=True),
                                    QueryPlanner(["1", "2", "3", "4", "5"])])
def test_query_failures_are_archived_but_cannot_enter_formal_execution(tmp_path, planner):
    pack = EvidencePackBuilder(planner, lambda q: pytest.fail("unexpected search"),
                               lambda url: pytest.fail("unexpected crawl")).build(task())
    assert pack["status"] == "failed"
    assert len(planner.calls) == 1
    assert pack["usage"]["model_calls"] == 1
    path = save_evidence_pack(pack, tmp_path / "failed.json")
    with pytest.raises(ValueError, match="Query-planning failure"):
        load_evidence_pack(path, task())


def test_immutable_archive_and_task_content_binding(tmp_path):
    public = task(tools=["web_search"], capabilities=["network"], attachments=[])
    pack = make_pack(public)
    path = save_evidence_pack(pack, evidence_pack_path(tmp_path, public.task_id))
    assert save_evidence_pack(pack, path) == path
    assert load_evidence_pack(path, public) == pack
    changed = copy.deepcopy(pack)
    changed["created_at"] = "different"
    changed["pack_sha256"] = digest({k: v for k, v in changed.items() if k != "pack_sha256"})
    with pytest.raises(FileExistsError, match="immutable"):
        save_evidence_pack(changed, path)
    with pytest.raises(ValueError, match="public task"):
        load_evidence_pack(path, public.model_copy(update={"question": "Changed question"}))
    applied = apply_evidence_pack(public, pack)
    assert applied.task_id == public.task_id
    assert applied.tools == applied.capabilities == applied.attachments == []
    assert pack["rendered"] in applied.question
    assert "archive" not in applied.question
    assert public.tools == ["web_search"]


def test_manifest_checks_counts_ids_file_and_source_hashes(tmp_path):
    tasks = {"first": task("first"), "second": task("second")}
    for public in tasks.values():
        save_evidence_pack(make_pack(public), evidence_pack_path(tmp_path, public.task_id))
    manifest = build_evidence_manifest(tasks, tmp_path)
    assert manifest["count"] == 2
    assert set(load_evidence_tasks(tasks, tmp_path, expected_count=2)) == set(tasks)
    with pytest.raises(ValueError, match="count"):
        load_evidence_tasks(tasks, tmp_path, expected_count=3)
    with pytest.raises(ValueError, match="count"):
        load_evidence_tasks({"first": tasks["first"]}, tmp_path)
    path = evidence_pack_path(tmp_path, "first")
    data = json.loads(path.read_text())
    data["sources"][0]["text"] = "tampered"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="file hash"):
        load_evidence_tasks(tasks, tmp_path)
    with pytest.raises(ValueError, match="content hash"):
        load_evidence_pack(path, tasks["first"])


def test_repeated_pack_validation_reuses_token_window_work_but_keeps_copy_semantics(tmp_path, monkeypatch):
    public = task("cache-validation")
    pack = synthetic_window_pack(public, {
        "https://public.org/report": "Background. " * 180 + "The measured result was 12 units."
    }, token_budget=700, queries=["measured result"])
    path = save_evidence_pack(pack, evidence_pack_path(tmp_path, public.task_id))
    import jit_mas.evidence as evidence
    evidence._validated_pack_cache.cache_clear()
    original = evidence._validate_pack
    calls = []

    def counted(value, task_value):
        calls.append(True)
        return original(value, task_value)

    monkeypatch.setattr(evidence, "_validate_pack", counted)
    first = load_evidence_pack(path, public)
    second = load_evidence_pack(path, public)
    assert len(calls) == 1
    assert first == second == pack
    first["sources"][0]["text"] = "caller mutation"
    assert load_evidence_pack(path, public)["sources"][0]["text"] != "caller mutation"


def test_public_task_only_and_explicit_cli_network_opt_in(tmp_path):
    with pytest.raises(TypeError, match="PublicTask only"):
        EvidencePackBuilder(QueryPlanner(), lambda q: [], lambda u: "").build({"answer": "secret"})
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(task().model_dump() | {"reference": "private"}), encoding="utf-8")
    with pytest.raises(ValueError):
        _read_tasks(path)
    with pytest.raises(SystemExit) as exc:
        main(["--tasks", str(path), "--output", str(tmp_path / "pack"), "--all-tasks",
              "--model", "test", "--endpoint", "https://invalid.example/v1", "--key-env", "UNSET_TEST_KEY"])
    assert exc.value.code == 2
    assert not (tmp_path / "pack").exists()


def test_public_attachment_local_containment_and_excluded_redirect(tmp_path):
    root = tmp_path / "public"
    root.mkdir()
    (root / "notes.md").write_text("public attachment", encoding="utf-8")
    loader = _public_attachment_loader(root, lambda url: "unexpected")
    assert loader("notes.md") == "public attachment"
    with pytest.raises(ValueError, match="outside"):
        loader("../private.json")
    public = task(attachments=["https://allowed.org/attachment"])
    pack = EvidencePackBuilder(QueryPlanner(["query"]), lambda q: [], lambda u: "").build(
        public, public_excluded_urls=["https://blocked.org/article"],
        attachment_loader=lambda u: {"text": "blocked document content", "url": "https://blocked.org/article"})
    assert pack["sources"][-1]["status"] == "excluded"
    assert "blocked document content" not in pack["rendered"]


def test_shared_pack_executes_complete_offline_pipeline_without_network(tmp_path):
    from jit_mas.config import MASConfig
    from jit_mas.experience import ExperienceStore
    from scripts.run_jit_mas import make_pipeline

    store = ExperienceStore(tmp_path / "experience.sqlite")
    try:
        pipeline = make_pipeline(MASConfig(backend="scripted"), store, tmp_path / "runs")
        pack_dir = tmp_path / "packs"
        for public in pipeline.tasks.values():
            save_evidence_pack(make_pack(public), evidence_pack_path(pack_dir, public.task_id))
        build_evidence_manifest(pipeline.tasks, pack_dir)
        pipeline.tasks = load_evidence_tasks(pipeline.tasks, pack_dir)
        source = pipeline.run("evolve", limit=1)
        evaluated = pipeline.run("evaluate", limit=1)
        assert len(source) == len(evaluated) == 1
        assert evaluated[0]["evaluation"]["complete"] is True
        assert evaluated[0]["evaluation"]["score"] is not None
        assert evaluated[0]["experience_updates"] == []
        assert (Path(evaluated[0]["run_dir"]) / "submission.json").is_file()
    finally:
        store.close()


@pytest.mark.parametrize("options", [{"max_queries": 5}, {"max_pages": 9}, {"max_pack_tokens": 32769},
                                    {"query_max_tokens": 4097}, {"results_per_query": 6}])
def test_config_cannot_exceed_preregistered_caps(options):
    with pytest.raises(ValueError):
        EvidenceConfig(**options)
