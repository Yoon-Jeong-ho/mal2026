#!/usr/bin/env python3
"""Run the locked zero-shot score-band prompt ablation."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mal2026.decoder_prompt_band_ablation import AblationConfig,aggregate,prepare,run_qwen,run_solar

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,required=True);parser.add_argument("--stage",choices=("prepare","qwen-run","solar-run","aggregate"),required=True);args=parser.parse_args()
    config=AblationConfig.from_json(args.config)
    result={"prepare":lambda:prepare(config,args.config),"qwen-run":lambda:run_qwen(config),"solar-run":lambda:run_solar(config),"aggregate":lambda:aggregate(config)}[args.stage]()
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
if __name__=="__main__":main()
