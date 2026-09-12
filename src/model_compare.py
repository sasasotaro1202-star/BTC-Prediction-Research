import sqlite3
from db import DB, init_db

def compare():
    init_db()
    with sqlite3.connect(DB) as con:
        rows=con.execute("SELECT horizon,model_version,n,accuracy,logloss,brier FROM model_metrics ORDER BY id DESC LIMIT 20").fetchall()
    for row in rows:
        print(row)

if __name__ == '__main__':
    compare()
