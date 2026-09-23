"""Summarise pilot quality and throughput metrics across processed matches."""
import json
import sys

from tennis_pipeline.metrics import match_metrics

if __name__ == "__main__":
    out = {vid: match_metrics(vid) for vid in sys.argv[1:]}
    print(json.dumps(out, indent=2))
