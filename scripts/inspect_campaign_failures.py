from __future__ import annotations
import json
import sqlite3
from pathlib import Path

p = Path('.runtime/v7r/campaign.sqlite')
db = sqlite3.connect(p)
print(db.execute("select name from sqlite_master where type='table'").fetchall())
for table in ('records', 'claims', 'slots'):
    try:
        print(table, db.execute(f'pragma table_info({table})').fetchall())
    except Exception:
        pass
rows = db.execute('select slot_id, body, status, result from slots where status=?', ('failed',)).fetchall()
print('failed', len(rows))
for slot_id, body, status, result in rows[:50]:
    try: body = json.loads(body)
    except Exception: pass
    try: result = json.loads(result) if result else None
    except Exception: pass
    print(json.dumps({'slot_id': slot_id, 'body': body, 'result': result}, ensure_ascii=True)[:2500])
rows = db.execute('select slot_id, body, status, started_at from slots where status=?', ('started',)).fetchall()
print('started', len(rows))
for slot_id, body, status, started_at in rows[:30]:
    print(slot_id, started_at, body[:200])
