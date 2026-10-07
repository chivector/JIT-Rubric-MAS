from __future__ import annotations
import json, sqlite3, sys
from collections import Counter
root = sys.argv[1] if len(sys.argv)>1 else '.runtime/v7r'
db=sqlite3.connect(root+'/campaign.sqlite')
rows=db.execute('select body,status,result from slots').fetchall()
c=Counter()
for body,status,result in rows:
    b=json.loads(body); c[(b.get('kind'),b.get('source'),status)] += 1
for key,n in sorted(c.items()): print(key,n)
for body,status,result in rows:
    b=json.loads(body)
    if status == 'failed':
        print('FAILED', json.dumps(b, ensure_ascii=True)[:500], str(result)[:1000])
