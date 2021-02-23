from collections import defaultdict
import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple
from more_itertools import flatten, take
from randomdict import RandomDict
from uralicNLP import semfi

get_db_conn = semfi.__get_connection
_cache = defaultdict(RandomDict)


class SemFiSQL:
    def __init__(self, db_filename: Optional[str] = None):
        self.conn = get_db_conn('fin')

    def get_most_common(self, pos: str, n: int = 1) -> List[str]:
        sql = 'SELECT word FROM words WHERE pos=? ORDER BY frequency DESC LIMIT ?'
        self.conn.execute(sql, (pos, n))
        return [r[0] for r in self.conn.fetchall()]

    def get_random(self, pos: str, n: int = 1) -> List[str]:
        sql = 'SELECT word FROM words WHERE pos=? ORDER BY random() LIMIT ?'
        self.conn.execute(sql, (pos, n))
        return [r[0] for r in self.conn.fetchall()]

    def get_frequencies(self, lemmas_pos: Iterable[Tuple[str, str]]) -> Dict[str, float]:
        word_placeholder = ' OR '.join('(id LIKE ? and pos=?)' for x in lemmas_pos)
        sql = f'SELECT word, frequency FROM words WHERE {word_placeholder}'
        params = [('{word}_%', pos) for word, pos in lemmas_pos]
        self.conn.execute(sql, list(flatten(params)))
        return dict(self.conn.fetchall())


class SemFi:
    def __init__(self, db_filename: Optional[str] = None):
        self.conn = get_db_conn('fin')
        self._preload_cache()

    def _preload_cache(self):
        if not _cache:
            sql = 'SELECT pos, id, word, frequency FROM words WHERE frequency > 1 ORDER BY frequency DESC'
            self.conn.execute(sql)
            for pos, id, word, frequency in self.conn:
                _cache[pos][id] = (word, frequency)

    def get_most_common(self, pos: str, n: int = 1) -> List[str]:
        return [x[0] for x in take(_cache[pos], n)]

    def get_random(self, pos: str, n: int = 1) -> List[str]:
        return [x[0] for x in _cache[pos].random_sample(n)]

    def get_frequencies(self, lemmas_pos: Iterable[Tuple[str, str]]) -> Dict[str, float]:
        return dict(_cache[p][f'{l}_{p}'] for l, p in lemmas_pos if _cache[p].get(f'{l}_{p}'))
