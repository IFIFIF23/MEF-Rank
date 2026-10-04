#!/usr/bin/env python3
"""Install just the locked upstream source dependencies; no training or data access."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import yaml
BASE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    lock=yaml.safe_load((BASE/'third_party/model_sources.lock.yaml').read_text())['sources']
    hashes=json.loads((BASE/'third_party/source_hashes.json').read_text())
    for sid,files in hashes.items():
        dest=BASE/'third_party/sources'/sid
        valid=all((dest/f).is_file() and hashlib.sha256((dest/f).read_bytes()).hexdigest()==h for f,h in files.items())
        if valid:print(f'{sid}: ready');continue
        if args.check_only:raise SystemExit(f'{sid}: missing or changed; run python setup_sources.py')
        rec=lock[sid]
        with tempfile.TemporaryDirectory(prefix='mef-upstream-') as temp:
            checkout=Path(temp)/'source'
            subprocess.run(['git','clone','--filter=blob:none','--no-checkout',rec['repository'],str(checkout)],check=True)
            subprocess.run(['git','fetch','--depth=1','origin',rec['commit']],cwd=checkout,check=True)
            subprocess.run(['git','checkout','--detach',rec['commit']],cwd=checkout,check=True)
            for relative,expected in files.items():
                src=checkout/relative
                if not src.is_file() or hashlib.sha256(src.read_bytes()).hexdigest()!=expected:raise RuntimeError(f'Locked source differs: {sid}/{relative}')
            for relative in files:
                target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(checkout/relative,target)
        print(f'{sid}: installed at {rec["commit"]}')


if __name__=='__main__':main()
