import json
import shutil
from pathlib import Path

src = Path("outputs/formal_ours_20261010/assets/batch_live_gpt4o_fallback_20261011/formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch.json")
dst = src.with_name("formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_launch_w4.json")
doc = json.loads(src.read_text(encoding="utf-8"))
doc["workers"] = 4
doc["version"] = "jit-compose-independent-batch-launch-v7-targeted-live-gpt4o-fallback-20261011-w4"
doc["asset_role"] = "formal_v7_ours_targeted_batch_live_gpt4o_fallback_20261011_no_secret_w4"
doc["provenance"]["source_launch"] = str(dst)
dst.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
stale = Path("tds_fix9")
if stale.exists() and stale.is_dir() and not any(stale.iterdir()):
    stale.rmdir()
print(dst)
