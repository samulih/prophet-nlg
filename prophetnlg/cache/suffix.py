from collections import defaultdict
from pathlib import Path
import sqlite3
from uralicNLP import semfi, uralicApi


def main():
    source_conn = semfi.__get_connection('fin')
    path = ''
    for id_, name, filename in source_conn.execute('PRAGMA database_list'):
        if name == 'main' and filename is not None:
            path = filename
            break
    if not path:
        print('SQLite file path not found.')
        return

    source_path = Path(path)
    target_path = source_path.with_name('cache.db')
    target_conn = sqlite3.connect(target_path.as_posix())

    prefixes = defaultdict(dict)

    query = 'SELECT word, pos from words WHERE frequency > 2 AND pos IN ("N", "V", "A") ORDER BY pos, frequency DESC'
    source_conn.execute(query)
    for word, pos in source_conn:
        word = word.split('|')[-1]
        analysis = uralicApi.analyze(word, 'fin')
        # don't cache unknown words
        if not analysis:
            continue
        for i, c in enumerate(word):
            word_part = word[i:]
            prefixes[pos].setdefault(word_part, word)

    target_conn.execute('DROP TABLE IF EXISTS suffixes')
    target_conn.execute('CREATE TABLE suffixes (suffix TEXT, word TEXT, pos TEXT)')
    for pos in prefixes:
        pos_prefixes = prefixes[pos]
        mass_insert = f'INSERT INTO suffixes VALUES (?, ?, ?)'
        values = [(suffix, word, pos) for suffix, word in pos_prefixes.items()]
        target_conn.executemany(mass_insert, values)
    target_conn.execute('CREATE INDEX idx_suffix_pos ON suffixes (suffix, pos)')
    target_conn.commit()
    target_conn.close()


if __name__ == '__main__':
    main()
