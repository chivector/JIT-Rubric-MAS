import sqlite3
c=sqlite3.connect('tds_fix5/campaign.sqlite')
c.row_factory=sqlite3.Row
for t in ['metadata','slots','checkpoints','selections','candidates','batch_selections']:
    try:
        print('\n',t,c.execute('select count(*) n from '+t).fetchone()['n'])
        for r in c.execute('select * from '+t+' limit 2').fetchall():
            d=dict(r)
            for k,v in list(d.items()):
                if isinstance(v,str) and len(v)>300: d[k]=v[:300]+'...'
            print(d)
    except Exception as e:
        print(e)
c.close()
