"""Arrow IPC + AES-256-GCM; authenticated decryption stays in process memory."""
import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b'MEFIPC01'


def key_from_file(path):
    key = Path(path).read_bytes()
    if len(key) != 32:
        raise ValueError('Expected a 32-byte binary data key')
    return key


def encrypt_bytes(value, key, identity):
    nonce = os.urandom(12)
    return MAGIC + nonce + AESGCM(key).encrypt(nonce, value, identity.encode())


def decrypt_bytes(value, key, identity):
    if not value.startswith(MAGIC) or len(value) < 36:
        raise ValueError('Invalid encrypted data format')
    return AESGCM(key).decrypt(value[8:20], value[20:], identity.encode())


def serialize(table):
    sink = pa.BufferOutputStream()
    table = table.replace_schema_metadata(None)
    with pa.ipc.new_file(sink, table.schema, options=pa.ipc.IpcWriteOptions(compression='zstd')) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


class EncryptedDataset:
    def __init__(self, root, family, key_file):
        if family not in ('gem', 'main', 'nasdaq100', 'sp500', 'crypto'):
            raise ValueError('Unknown dataset family')
        self.root = Path(root) / family
        self.key = key_from_file(key_file)
        self.family = family
        self.manifest = json.loads(decrypt_bytes((self.root / 'manifest.enc').read_bytes(), self.key, family + '/manifest'))
        if self.manifest['family'] != family or self.manifest['schema_version'] != 1:
            raise ValueError('Dataset identity mismatch')
        self.records = self.manifest['shards']

    def years(self, view):
        return sorted({r['year'] for r in self.records if r['view'] == view})

    def read_table(self, view, year, kind, columns=None):
        entries = [r for r in self.records if (r['view'], r['year'], r['kind']) == (view, int(year), kind)]
        if not entries:
            raise FileNotFoundError(f'No shard: {self.family}/{view}/{year}/{kind}')
        tables = []
        for entry in entries:
            filename = entry['file']
            if Path(filename).name != filename:
                raise ValueError('Shard path escapes dataset')
            ciphertext = (self.root / filename).read_bytes()
            if hashlib.sha256(ciphertext).hexdigest() != entry['sha256']:
                raise ValueError('Encrypted shard checksum mismatch')
            plaintext = decrypt_bytes(ciphertext, self.key, entry['identity'])
            table = pa.ipc.open_file(pa.BufferReader(plaintext)).read_all()
            if table.num_rows != entry['rows']:
                raise ValueError('Shard row count mismatch')
            tables.append(table.select(columns) if columns is not None else table)
        return pa.concat_tables(tables)

    def read_frame(self, view, year, kind, columns=None):
        return self.read_table(view, year, kind, columns).to_pandas()

    def lineage(self, view):
        entry = self.manifest.get('date_mappings', {}).get(view)
        if entry is None:
            raise ValueError('Required target/source mapping missing from encrypted manifest')
        filename = entry['file']
        if Path(filename).name != filename: raise ValueError('Mapping path escapes dataset')
        ciphertext = (self.root / filename).read_bytes()
        if hashlib.sha256(ciphertext).hexdigest() != entry['sha256']:raise ValueError('Mapping checksum mismatch')
        raw = decrypt_bytes(ciphertext, self.key, self.family + '/' + view + '/date_mapping')
        rows = pa.ipc.open_file(pa.BufferReader(raw)).read_all().to_pylist()
        result = {}
        for row in rows:
            date = str(row['target_date']).replace('-', '')[:8]
            if date in result: raise ValueError('Duplicate target source mapping')
            result[date] = {k:(v.isoformat() if hasattr(v,'isoformat') else v) for k,v in row.items()}
        return result
