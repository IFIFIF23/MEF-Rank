#!/usr/bin/env python3
"""Train one shared model and run its Off/On settings on encrypted variant-1 data."""
import argparse
from pathlib import Path
import sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE/'src'))
from mef_unified import boot

MODELS=('STR','iTransformer','TimeMixer','ModernTCN','SOFTS','TimeXer','TimeBridge','TimeFilter','SRSNet','ChannelTokenFormer')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,default='STR')
    parser.add_argument('--dataset',choices=('gem','main','nasdaq100','sp500','crypto'),default='gem')
    parser.add_argument('--mode',choices=('off','on','both'),default='both',help='both/on produce paired outputs with one shared checkpoint')
    parser.add_argument('--data-dir',type=Path,default=BASE.parent/'data_encrypted')
    parser.add_argument('--key-file',type=Path,required=True,help='Binary key obtained separately from the data owner')
    parser.add_argument('--seed',type=int,default=20260921)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--epochs',type=int,help='Explicitly override the main and historical epoch budget')
    parser.add_argument('--output-dir',type=Path,default=BASE/'outputs')
    parser.add_argument('--resume',type=Path,help='Same run directory and original training options; resume completed stages')
    parser.add_argument('--smoke',action='store_true',help='Engineering-only reduced dates/epochs/history; not a formal experiment')
    args=parser.parse_args()
    if args.threads<1:parser.error('--threads must be positive')
    from handoff.pipeline import run_pipeline
    run_pipeline(args)


if __name__=='__main__':main()
