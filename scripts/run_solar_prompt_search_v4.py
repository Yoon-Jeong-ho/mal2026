#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from mal2026.solar_prompt_search_v4 import SearchConfigV4,aggregate_discovery,preflight,prepare,run_candidate
def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--stage",choices=("prepare","preflight","run","aggregate-discovery"),required=True);p.add_argument("--candidate");p.add_argument("--split",choices=("discovery","confirmation"),default="discovery");a=p.parse_args();c=SearchConfigV4.from_json(a.config)
    if a.stage=="prepare":r=prepare(c,a.config)
    elif a.stage=="preflight":r=preflight(c)
    elif a.stage=="run":
        if a.candidate is None:p.error("--candidate required")
        r=run_candidate(c,a.candidate,a.split)
    else:r=aggregate_discovery(c)
    print(json.dumps(r,ensure_ascii=False,sort_keys=True));return 0
if __name__=="__main__":raise SystemExit(main())
