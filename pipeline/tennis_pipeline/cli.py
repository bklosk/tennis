"""Command-line entry point.

    uv run python -m tennis_pipeline.cli scenes KCcKkUnjbzA Fl33UXv6jKI
    uv run python -m tennis_pipeline.cli track KCcKkUnjbzA
    uv run python -m tennis_pipeline.cli events KCcKkUnjbzA
"""
import argparse
import json

from .paths import DOWNLOADS


def video_path(video_id: str):
    path = DOWNLOADS / f"{video_id}.mp4"
    if not path.exists():
        path = DOWNLOADS / f"{video_id}.f298.mp4"
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["scenes", "track", "events", "crops", "strokes", "align", "report"])
    parser.add_argument("video_ids", nargs="+")
    parser.add_argument("--limit-segments", type=int)
    parser.add_argument("--ball-backend", choices=["mps", "coreml"], default="mps")
    parser.add_argument("--vlm", action="store_true", help="strokes: also run the Qwen3-VL labeling experiment")
    args = parser.parse_args()

    if args.stage == "scenes":
        from . import scenes

        for vid in args.video_ids:
            scenes.embed_video(vid, video_path(vid))
        print(json.dumps(scenes.train_classifier(args.video_ids)))
        for vid in args.video_ids:
            segs = scenes.predict_segments(vid)
            print(vid, len(segs), "segments", round(segs.duration.sum() / 60, 1), "min kept")
    elif args.stage == "track":
        from .process import track_match

        for vid in args.video_ids:
            print(json.dumps(track_match(vid, video_path(vid), args.limit_segments, args.ball_backend)))
    elif args.stage == "events":
        from .process import events_match

        for vid in args.video_ids:
            hits = events_match(vid, video_path(vid))
            print(vid, len(hits), "hits")
    elif args.stage == "crops":
        from .process import crops_match

        for vid in args.video_ids:
            crops_match(vid, video_path(vid))
    elif args.stage == "strokes":
        from . import strokes

        print(json.dumps(strokes.run(args.video_ids, vlm=args.vlm), indent=2))
    elif args.stage == "align":
        from . import align

        for vid in args.video_ids:
            print(json.dumps(align.run(vid), indent=2))
    elif args.stage == "report":
        from . import report

        report.run(args.video_ids)


if __name__ == "__main__":
    main()
