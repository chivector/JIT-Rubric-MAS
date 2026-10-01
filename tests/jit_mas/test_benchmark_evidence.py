from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jit_mas.evidence import (EvidenceConfig, EvidencePackBuilder, apply_evidence_pack,
                              build_evidence_manifest, canonical_url, evidence_pack_path,
                              load_evidence_pack, load_evidence_tasks, save_evidence_pack)
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
    assert urls == [f"https://example.org/{i}" for i in range(8)]
    assert len(planner.calls) == 1
    assert planner.calls[0][1] == {"max_tokens": 4096, "temperature": 0}
    assert pack["usage"]["model_calls"] == 1
    assert pack["usage"]["tokens"] == 28
    assert pack["usage"]["tool_calls"] == 12
    assert len(pack["archive"]) == 13
    assert len(pack["sources"]) == 8
    assert all("raw_output" in row for row in pack["archive"])


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


def test_round_robin_token_bound_locators_and_exact_unicode_prefixes():
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
        assert source["text"]
        assert original[source["locator"]].startswith(source["text"])
        assert source["span_start"] == 0
        assert source["span_end"] == len(source["text"])
        assert source["truncated"] is True
        assert "\ufffd" not in source["text"]


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
