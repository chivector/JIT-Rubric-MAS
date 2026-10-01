"""Verified author instruction checkers from the pinned IFBench source release."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import random
import sys
import threading
import types
import zipfile
from pathlib import Path

from .benchmarks import IFBENCH_REVISION
from .schemas import digest


PINNED_FILES = {
    "evaluation_lib.py": "e681a4b03a6a9dfb540fbd9ef6b963de503a5ffac557ed611c326ee385850cf6",
    "ifbench/__init__.py": "2a806107f115facaa3e7e6046ca97247237b4b113b514bcdd763b2530141f694",
    "ifbench/classic_instructions.py": "d26e642f2e71e4f57218e5e05f098a22e36311485779baac401161d3228c9973",
    "ifbench/instructions.py": "a76a3b300b531b95200e53427c6b1fbb72b671cf6d358796797c0fa4b6c7c44f",
    "ifbench/instructions_registry.py": "5119f2919261c77566f698f2f367883aff56cf89443bac5a56b6e30ddbd5aef9",
    "ifbench/instructions_util.py": "6996a6efaf6e3c92a2e926881eadc8c0262fb8d06c8cb56394ef7322f69e1930",
}
IFEVAL_CHECKER_REVISION = "e49bbfe381c9c0e564b937f1c4e163a2273c65cc"
IFEVAL_FILES = {
    "instruction_following_eval/evaluation_lib.py": "35decc06000718487f44d7deafa6d3f48a8ec0886281edf40162c0265b7d248c",
    "instruction_following_eval/instructions.py": "60e086f5342a03ce8e18b64bbcccf86308f523c08aa826707a562150a52f3edf",
    "instruction_following_eval/instructions_registry.py": "ec92d72c264f6d906978613085db262356174300370a3fffe6fefd5969ce9cfc",
    "instruction_following_eval/instructions_util.py": "a73797261eee5bf447e279d82a2b700b1bdd3cb1193412dbab1270a85832bc6b",
}
_CHECKER_LOCK = threading.RLock()


def _verify_files(root, files):
    root = Path(root)
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Pinned author instruction checker source hash mismatch")


class PinnedInstructionChecker:
    def __init__(self, source_root, benchmark):
        if benchmark not in {"ifeval", "ifbench"}:
            raise ValueError("Instruction checker requires IFEval or IFBench")
        root = Path(source_root).resolve()
        pinned_files = IFEVAL_FILES if benchmark == "ifeval" else PINNED_FILES
        revision = IFEVAL_CHECKER_REVISION if benchmark == "ifeval" else IFBENCH_REVISION
        _verify_files(root, pinned_files)
        package_spec = importlib.util.find_spec("ifbench")
        if package_spec is None:
            raise ValueError("Install the pinned author IFBench package before execution")
        installed_root = Path(package_spec.origin).parent
        installed_files = {Path(relative).name: expected for relative, expected in PINNED_FILES.items()
                           if relative.startswith("ifbench/")}
        _verify_files(installed_root, installed_files)
        resource_hashes = {}
        for relative in ("tokenizers/punkt.zip", "tokenizers/punkt_tab.zip", "corpora/stopwords.zip",
                         "taggers/averaged_perceptron_tagger_eng.zip"):
            path = installed_root / ".nltk_data" / relative
            if not path.is_file():
                raise ValueError("Prepare and freeze author checker NLTK resources before execution")
            resource_hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    extracted = path.parent / member.filename
                    if not extracted.is_file() or extracted.read_bytes() != archive.read(member):
                        raise ValueError("Author checker extracted NLTK resource differs from its frozen archive")
        module_name = "_jit_mas_pinned_" + benchmark + "_evaluation_" + revision
        with _CHECKER_LOCK:
            import nltk

            resource_path = str(installed_root / ".nltk_data")
            if resource_path not in nltk.data.path:
                nltk.data.path.insert(0, resource_path)
            evaluation_path = root / "evaluation_lib.py"
            if benchmark == "ifeval":
                namespace = "instruction_following_eval"
                package_path = root / namespace
                if namespace not in sys.modules:
                    package = types.ModuleType(namespace)
                    package.__path__ = [str(package_path)]
                    sys.modules[namespace] = package
                elif sys.modules[namespace].__path__ != [str(package_path)]:
                    raise ValueError("Google IFEval checker namespace differs from pinned source")
                evaluation_path = package_path / "evaluation_lib.py"
            if module_name not in sys.modules:
                spec = importlib.util.spec_from_file_location(module_name, evaluation_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            self.author = sys.modules[module_name]
        self.benchmark = benchmark
        self.identity = {
            "source": ("https://github.com/google-research/google-research/tree/" + revision
                       + "/instruction_following_eval" if benchmark == "ifeval"
                       else "https://github.com/allenai/IFBench/tree/" + revision),
            "revision": revision,
            "code_sha256": dict(pinned_files),
            "nltk_resources_sha256": resource_hashes,
            "dependencies": {name: importlib.metadata.version(name)
                             for name in ("ifbench", "nltk", "langdetect", "immutabledict", "emoji", "syllapy")},
            "metric": "prompt-level strict" if benchmark == "ifeval" else "prompt-level loose",
            "google_author_registry": benchmark == "ifeval",
            "langdetect_seed": 0,
            "instruction_random_seed": "SHA256 of instruction record",
        }

    def check_record(self, prediction, private_record):
        record = copy.deepcopy(private_record)
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Pinned checker requires the original public prompt")
        example = self.author.InputExample(
            key=0, instruction_id_list=record["instruction_id_list"],
            prompt=prompt, kwargs=record["kwargs"])
        checker = (self.author.test_instruction_following_strict if self.benchmark == "ifeval"
                   else self.author.test_instruction_following_loose)
        with _CHECKER_LOCK:
            import langdetect

            previous_state = random.getstate()
            previous_seed = langdetect.DetectorFactory.seed
            try:
                random.seed(int(digest({"prompt": prompt, "instruction_id_list": example.instruction_id_list,
                                        "kwargs": example.kwargs}), 16))
                langdetect.DetectorFactory.seed = 0
                result = checker(example, {prompt: prediction})
            finally:
                random.setstate(previous_state)
                langdetect.DetectorFactory.seed = previous_seed
        return list(result.follow_instruction_list)
