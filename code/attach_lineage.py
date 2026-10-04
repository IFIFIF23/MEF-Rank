"""Owner utility: add encrypted target/source date ledgers to a completed export."""
import argparse
import hashlib
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
from export_data import FAMILIES
from handoff.secure_data import encrypt_bytes,decrypt_bytes,serialize,key_from_file


def attach(source,output,private,family):
    root=output/family;key=key_from_file(private/(family+'.key'))
    manifest=json.loads(decrypt_bytes((root/'manifest.enc').read_bytes(),key,family+'/manifest'))
    if manifest.get('date_mappings'):return
    maps={}
    for view,sub in [('1_str','str'),('1_memory','memory')]:
        path=source/FAMILIES[family][1]/sub/'date_mapping.parquet'
        if path.exists():
            table=pq.ParquetFile(path).read().replace_schema_metadata(None)
        elif view == '1_str':
            source_manifest=json.loads((path.parent/'manifest.json').read_text())
            if source_manifest.get('time_axis')!='source_forward' or source_manifest.get('copy_method')!='byte_for_byte_file_copy':raise ValueError('Cannot infer an identity mapping for non-forward STR data')
            dates=sorted({str(d) for f in path.parent.glob('year=*/market_daily.parquet') for d in pq.ParquetFile(f).read(columns=['feature_date']).column('feature_date').to_pylist()})
            table=pa.table({'target_date':dates,'source_date':dates,'mapping_basis':['source_forward_byte_copy_manifest']*len(dates)})
        else:raise FileNotFoundError(path)
        if any('code' in n or 'name' in n for n in table.schema.names):raise ValueError('Unexpected direct identifier in date mapping')
        identity=family+'/'+view+'/date_mapping';data=encrypt_bytes(serialize(table),key,identity)
        name=view+'_mapping.enc';(root/name).write_bytes(data)
        maps[view]={'file':name,'sha256':hashlib.sha256(data).hexdigest(),'rows':table.num_rows}
    manifest['date_mappings']=maps
    manifest['evaluation_policy']='original observations from 2022 for 60-day context; metrics start at the registered 2023 date; no evaluation fitting'
    temp=root/'manifest.tmp';temp.write_bytes(encrypt_bytes(json.dumps(manifest).encode(),key,family+'/manifest'));temp.replace(root/'manifest.enc')
    summary=json.loads((root/'summary.json').read_text())
    summary.update(manifest_sha256=hashlib.sha256((root/'manifest.enc').read_bytes()).hexdigest(),encrypted_bytes=sum(p.stat().st_size for p in root.glob('*.enc')),date_mapping_views=list(maps))
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'{family}: target/source mappings encrypted',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--private',type=Path,required=True)
    args=p.parse_args()
    for family in FAMILIES:attach(args.source,args.output,args.private,family)


if __name__=='__main__':main()
