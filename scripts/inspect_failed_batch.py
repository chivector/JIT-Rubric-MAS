import json, sqlite3
p='.runtime/v7r/campaign.sqlite'; db=sqlite3.connect(p)
for body,status,result in db.execute('select body,status,result from slots where status="failed"'):
 b=json.loads(body)
 if b.get('kind')=='batch_evolution':
  print(json.dumps({'body':b,'result':json.loads(result) if result else None},ensure_ascii=True)[:6000])
