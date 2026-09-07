"""One SQLite dataset per method, plus the weights needed to use or resume it.

Training, evaluation, recovery and the version report share these records.
"""
from __future__ import annotations

from contextlib import contextmanager
import io
import json
import math
from pathlib import Path
import sqlite3
import zlib


METHODS = {
    'T1': 'rollout', 't1_rollout': 'rollout', 'T1_rollout': 'rollout',
    'T2': 'mcts_dpw', 't2_mcts_dpw': 'mcts_dpw', 'T2_mcts_dpw': 'mcts_dpw',
    'T3': 'candidate_ppo', 'candidate': 'candidate_ppo', 't3_candidate': 'candidate_ppo',
    'autoregressive': 'ar_ppo', 't3_autoregressive': 'ar_ppo', 't3_ar': 'ar_ppo',
    'T4': 'exit', 't4_exit': 'exit',
    'T5': 'bridge', 't5_bridge_grouping': 'bridge',
    'T6': 'paired_value', 't6_bce': 'paired_value', 't6_adv': 'paired_value',
    't6_adv_gated': 'paired_value',
    'rule': 'baselines', 'grand': 'baselines', 'singleton': 'baselines',
    'frozen_b1': 'baselines', 'frozen_b3': 'baselines',
}


def method_dir(run, method):
    method = str(method)
    if method.startswith('t6_'):
        method = 'T6'
    if method.startswith('T1_'):
        method = 'T1'
    if method == 'B3_continuous':
        method = 'baselines'
    if method.startswith(('t4_teacher', 't4_student')):
        method = 'T4'
    return Path(run) / METHODS.get(method, method)


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'to_dict'):
        return _clean(value.to_dict())
    if hasattr(value, 'tolist'):
        return _clean(value.tolist())
    if hasattr(value, 'item'):
        return _clean(value.item())
    return value


def _json(value):
    return json.dumps(_clean(value), ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def _stream(name):
    return str(name).removesuffix('.jsonl')


class Store:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'data.sqlite'
        self.connection = sqlite3.connect(self.path, timeout=60)
        self.connection.execute('PRAGMA busy_timeout=60000')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS blobs (key TEXT PRIMARY KEY, codec TEXT NOT NULL, data BLOB NOT NULL);
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY, stream TEXT NOT NULL, value TEXT NOT NULL,
                source TEXT, source_line INTEGER, UNIQUE(source, source_line));
            CREATE INDEX IF NOT EXISTS records_stream ON records(stream, id);
            CREATE TABLE IF NOT EXISTS episodes (
                method TEXT NOT NULL, split TEXT NOT NULL, checkpoint TEXT NOT NULL,
                family TEXT NOT NULL, value TEXT NOT NULL, trace BLOB,
                PRIMARY KEY(method, split, checkpoint, family));
        ''')
        self._depth = 0

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @contextmanager
    def transaction(self):
        self._depth += 1
        try:
            yield self
            if self._depth == 1:
                self.connection.commit()
        except BaseException:
            if self._depth == 1:
                self.connection.rollback()
            raise
        finally:
            self._depth -= 1

    def _commit(self):
        if not self._depth:
            self.connection.commit()

    def rows(self, stream):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT value FROM records WHERE stream=? ORDER BY id', (_stream(stream),))]

    def append(self, stream, row, *, source=None, source_line=None):
        self.connection.execute('INSERT OR IGNORE INTO records(stream,value,source,source_line) VALUES(?,?,?,?)',
                                (_stream(stream), _json(row), source, source_line))
        self._commit()

    def streams(self):
        return [row[0] for row in self.connection.execute('SELECT DISTINCT stream FROM records ORDER BY stream')]

    def get(self, key, default=None):
        row = self.connection.execute('SELECT value FROM metadata WHERE key=?', (str(key),)).fetchone()
        if row is None and str(key).endswith('.json'):
            row = self.connection.execute('SELECT value FROM metadata WHERE key=?', (str(key)[:-5],)).fetchone()
        if row is None and not str(key).endswith('.json'):
            row = self.connection.execute('SELECT value FROM metadata WHERE key=?', (str(key)+'.json',)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key, value):
        self.connection.execute('INSERT OR REPLACE INTO metadata VALUES(?,?)', (str(key), _json(value)))
        self._commit()

    def keys(self, prefix=''):
        return [row[0] for row in self.connection.execute('SELECT key FROM metadata WHERE key LIKE ? ORDER BY key', (prefix+'%',))]

    def blob_keys(self, prefix=''):
        return [row[0] for row in self.connection.execute('SELECT key FROM blobs WHERE key LIKE ? ORDER BY key', (prefix+'%',))]

    def blob(self, key):
        row = self.connection.execute('SELECT codec,data FROM blobs WHERE key=?', (str(key),)).fetchone()
        if not row:
            return None
        return zlib.decompress(row[1]) if row[0] == 'zlib' else bytes(row[1])

    def put_blob(self, key, data):
        data = bytes(data)
        packed = zlib.compress(data, 1)
        codec, payload = ('zlib', packed) if len(packed) < len(data) else ('raw', data)
        self.connection.execute('INSERT OR REPLACE INTO blobs VALUES(?,?,?)', (str(key), codec, payload))
        self._commit()

    def load_torch(self, key):
        import torch
        data = self.blob(key)
        return torch.load(io.BytesIO(data), map_location='cpu', weights_only=False) if data is not None else None

    def save_torch(self, key, payload):
        import torch
        output = io.BytesIO()
        torch.save(payload, output)
        self.put_blob(key, output.getvalue())

    def evaluation_keys(self):
        return [tuple(row) for row in self.connection.execute(
            'SELECT DISTINCT method,split,checkpoint FROM episodes ORDER BY method,split,checkpoint')]

    def episodes(self, method, split, checkpoint):
        return [json.loads(row[0]) for row in self.connection.execute(
            'SELECT value FROM episodes WHERE method=? AND split=? AND checkpoint=? ORDER BY rowid',
            (str(method), str(split), str(checkpoint)))]

    def save_episode(self, method, split, checkpoint, row, compressed_records_bytes=None):
        family = row.get('family_id', row.get('evaluation_family_id'))
        if family is None:
            raise ValueError('An evaluation episode needs its family_id')
        self.connection.execute('''INSERT INTO episodes VALUES(?,?,?,?,?,?)
            ON CONFLICT(method,split,checkpoint,family) DO UPDATE SET
            value=excluded.value,trace=COALESCE(excluded.trace,episodes.trace)''',
            (str(method), str(split), str(checkpoint), str(family), _json(row), compressed_records_bytes))
        self._commit()

    def episode_trace(self, method, split, checkpoint, family):
        row = self.connection.execute('SELECT trace FROM episodes WHERE method=? AND split=? AND checkpoint=? AND family=?',
                                      (method, split, checkpoint, family)).fetchone()
        return bytes(row[0]) if row and row[0] is not None else None
