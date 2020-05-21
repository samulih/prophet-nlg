from typing import List, Optional
from uralicNLP import semfi

get_db_conn = semfi.__get_connection

class SemFi:
    def __init__(self, db_filename: Optional[str] = None):
        self.conn = get_db_conn('fin')

    def _strip(self, w: str) -> str:
        return w.split('_')[0].replace('|', '')

    def get_most_common(self, pos: str, n: int = 1) -> List[str]:
        sql = 'SELECT word FROM words WHERE pos=? ORDER BY frequency DESC LIMIT ?'
        self.conn.execute(sql, (pos, n))
        return [self._strip(r[0]) for r in self.conn.fetchall()]

    def get_random(self, pos: str, n: int = 1) -> List[str]:
        sql = 'SELECT word FROM words WHERE pos=? ORDER BY random() LIMIT ?'
        self.conn.execute(sql, (pos, n))
        return [self._strip(r[0]) for r in self.conn.fetchall()]
