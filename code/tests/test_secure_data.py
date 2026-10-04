import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pyarrow as pa
import pytest
from cryptography.exceptions import InvalidTag
from handoff.secure_data import encrypt_bytes, decrypt_bytes, serialize


def test_roundtrip_schema_values_nulls_and_no_pandas_metadata():
    table = pa.table({'x': [1., None, 3.], 'asset': ['a', 'b', 'c']}).replace_schema_metadata({b'pandas': b'private'})
    raw = serialize(table)
    restored = pa.ipc.open_file(pa.BufferReader(raw)).read_all()
    assert restored.equals(table.replace_schema_metadata(None))
    assert restored.schema.metadata is None
    key = os.urandom(32)
    cipher = encrypt_bytes(raw, key, 'crypto/1_str/2020/stock/0')
    assert decrypt_bytes(cipher, key, 'crypto/1_str/2020/stock/0') == raw
    assert encrypt_bytes(raw, key, 'crypto/1_str/2020/stock/0') != cipher
    for bad_key, bad_id, bad_cipher in [(os.urandom(32), 'crypto/1_str/2020/stock/0', cipher), (key, 'crypto/1_memory/2020/stock/0', cipher), (key, 'crypto/1_str/2020/stock/0', cipher[:-1] + bytes([cipher[-1] ^ 1]))]:
        with pytest.raises(InvalidTag):
            decrypt_bytes(bad_cipher, bad_key, bad_id)
    with pytest.raises(ValueError):
        decrypt_bytes(cipher[:15], key, 'crypto/1_str/2020/stock/0')
