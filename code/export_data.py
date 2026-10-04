"""Owner-only export. Never include the private directory in a delivery archive."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets

import pyarrow as pa
import pyarrow.parquet as pq
from handoff.secure_data import encrypt_bytes, decrypt_bytes, serialize

FAMILIES = {
    'gem': ('cn_a/gem/gem0', 'cn_a/gem/gem1'),
    'main': ('cn_a/main_board/main0', 'cn_a/main_board/main1'),
    'nasdaq100': ('us/market_wide/us0/nasdaq100', 'us/market_wide/us1/nasdaq100'),
    'sp500': ('us/market_wide/us0/sp500', 'us/market_wide/us1/sp500'),
    'crypto': ('crypto/fixed15/crypto0', 'crypto/fixed15/crypto1'),
}


def selected_columns(schema, kind):
    if kind == 'market':
        return schema.names
    prefixes = ('signal_normalized', 'signal_raw', 'label_H1__', 'label_H5__', 'tensor__', 'context__')
    return [n for n in schema.names if n in ('feature_date', 'ts_code') or n.startswith(prefixes)]


def write_private(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'wb') as handle:
        handle.write(value)


def export_family(source, output, private, family):
    destination = output / family
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    key_path = private / (family + '.key')
    map_path = private / (family + '_identifiers.json')
    if key_path.exists() or map_path.exists():
        raise FileExistsError('Private export identity already exists; use a new version')
    views = {'evaluation': source / FAMILIES[family][0], '1_str': source / FAMILIES[family][1] / 'str', '1_memory': source / FAMILIES[family][1] / 'memory'}
    files = []
    codes = set()
    for view, root in views.items():
        annual = sorted(root.glob('year=*/stock_daily.parquet'))
        if view == 'evaluation':
            annual = [p for p in annual if int(p.parent.name.split('=')[1]) >= 2022]
        if not annual:
            raise FileNotFoundError(f'Missing annual source for {family}/{view}')
        for stock in annual:
            codes.update(pq.ParquetFile(stock).read(columns=['ts_code']).column('ts_code').to_pylist())
            for kind in ('stock', 'market'):
                path = stock.parent / (kind + '_daily.parquet')
                if not path.is_file():
                    raise FileNotFoundError(path)
                files.append((view, int(stock.parent.name.split('=')[1]), kind, path))
    if None in codes:
        raise ValueError('Null asset identity')
    # Preserve original asset ordering, important for deterministic fixed-pair losses.
    aliases = sorted({'asset_' + secrets.token_hex(16) for _ in codes})
    if len(aliases) != len(codes):
        raise ValueError('Alias collision')
    mapping = dict(zip(sorted(codes), aliases))
    key = secrets.token_bytes(32)
    write_private(key_path, key)
    write_private(map_path, json.dumps(mapping).encode())
    destination.mkdir(parents=True)
    manifest = {'schema_version': 1, 'family': family, 'format': 'Arrow IPC zstd + AES-256-GCM', 'views': list(views), 'shards': [], 'privacy': 'Pseudonymous assets; original dates and model numeric inputs retained; not irreversible anonymization', 'variant1_scope': 'original reordered source-exposed diagnostic; not independent out-of-sample evidence'}
    source_audit = []
    for view, year, kind, path in files:
        parquet = pq.ParquetFile(path)
        cols = selected_columns(parquet.schema_arrow, kind)
        count = 0
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                digest.update(block)
        for part, batch in enumerate(parquet.iter_batches(batch_size=32768, columns=cols)):
            table = pa.Table.from_batches([batch]).replace_schema_metadata(None)
            if kind == 'stock':
                table = table.set_column(table.schema.get_field_index('ts_code'), 'ts_code', pa.array([mapping[c] for c in table.column('ts_code').to_pylist()]))
            identity = f'{family}/{view}/{year}/{kind}/{part}'
            raw = serialize(table)
            encoded = encrypt_bytes(raw, key, identity)
            # Every exported shard is authenticated and compared before publishing its manifest.
            decoded = decrypt_bytes(encoded, key, identity)
            if decoded != raw:
                raise ValueError('Encrypted roundtrip mismatch')
            restored = pa.ipc.open_file(pa.BufferReader(decoded)).read_all()
            if restored.schema != table.schema or restored.num_rows != table.num_rows:
                raise ValueError('Serialized schema/row mismatch')
            filename = f'{len(manifest["shards"]):06d}.enc'
            (destination / filename).write_bytes(encoded)
            manifest['shards'].append({'identity': identity, 'view': view, 'year': year, 'kind': kind, 'file': filename, 'rows': table.num_rows, 'sha256': hashlib.sha256(encoded).hexdigest()})
            count += table.num_rows
        source_audit.append({'source': str(path.resolve()), 'sha256': digest.hexdigest(), 'rows': count, 'exported_columns': cols})
        print(f'{family}/{view}/{year}/{kind}: {count} rows', flush=True)
    manifest_bytes = json.dumps(manifest, ensure_ascii=False).encode()
    (destination / 'manifest.enc').write_bytes(encrypt_bytes(manifest_bytes, key, family + '/manifest'))
    write_private(private / (family + '_source_audit.json'), json.dumps(source_audit).encode())
    summary = {'family': family, 'views': list(views), 'shards': len(manifest['shards']), 'encrypted_bytes': sum(p.stat().st_size for p in destination.iterdir()), 'manifest_sha256': hashlib.sha256((destination / 'manifest.enc').read_bytes()).hexdigest(), 'verification': 'all encrypted shards decrypted and serialized bytes matched'}
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    parser.add_argument('--family', choices=FAMILIES, action='append')
    args = parser.parse_args()
    if args.private.resolve().is_relative_to(args.output.resolve().parent):
        raise ValueError('Private keys must stay outside the delivery directory')
    args.private.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.private, 0o700)
    args.output.mkdir(parents=True, exist_ok=True)
    for family in args.family or FAMILIES:
        print(json.dumps(export_family(args.source, args.output, args.private, family)), flush=True)
        from attach_lineage import attach
        attach(args.source, args.output, args.private, family)


if __name__ == '__main__':
    main()
