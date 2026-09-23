"""Command-line entry point.

    uv run python -m tennis_pipeline.cli scenes KCcKkUnjbzA Fl33UXv6jKI
    uv run python -m tennis_pipeline.cli track KCcKkUnjbzA
    uv run python -m tennis_pipeline.cli events KCcKkUnjbzA
    uv run python -m tennis_pipeline.cli ocr OLD_MATCH_ID                 # matches without official data
    uv run python -m tennis_pipeline.cli balllabels KCcKkUnjbzA ... --action sample
    uv run python -m tennis_pipeline.cli balltrain --epochs 30
    uv run python -m tennis_pipeline.cli gold KCcKkUnjbzA ... --target 200
"""
import argparse
import json

from .paths import DOWNLOADS

STAGES = ["scenes", "track", "events", "crops", "strokes", "align", "report", "ocr", "balllabels", "balltrain", "gold"]


def video_path(video_id: str):
    """The match video: the full download, else a main-camera pack (`video.pack_segments`)."""
    for name in (f"{video_id}.mp4", f"{video_id}.f298.mp4", f"{video_id}.pack"):
        if (DOWNLOADS / name).exists():
            return DOWNLOADS / name
    return DOWNLOADS / f"{video_id}.mp4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("video_ids", nargs="*")
    parser.add_argument("--limit-segments", type=int)
    parser.add_argument("--ball-backend", choices=["auto", "torch", "ane"], default="auto",
                        help="track: TrackNet on PyTorch (CUDA/MPS) or the Apple Neural Engine (auto: ANE on a Mac)")
    parser.add_argument("--ball-weights", help="track/balltrain: TrackNet weights (default: fine-tuned if present)")
    parser.add_argument("--no-inline-crops", action="store_true",
                        help="track: skip cropping hitters from in-memory frames (crops stage decodes them)")
    parser.add_argument("--scene-labeler", choices=["auto", "vlm", "court"], default="auto",
                        help="scenes: label sample frames with the MLX VLM or the court-keypoint model")
    parser.add_argument("--scene-keyframes", action="store_true",
                        help="scenes: decode keyframes only (~6x faster, coarser segment boundaries)")
    parser.add_argument("--reuse-classifier", action="store_true",
                        help="scenes: keep an existing scene classifier instead of retraining")
    parser.add_argument("--vlm", action="store_true", help="strokes: also run the Qwen3-VL labeling experiment")
    parser.add_argument("--no-serve-detector", action="store_true", help="events: rule-only serve detection")
    parser.add_argument("--ocr-engine", choices=["rapidocr", "vlm"], default="rapidocr")
    parser.add_argument("--rebuild", action="store_true", help="ocr: re-derive points from cached reads")
    parser.add_argument("--action", choices=["sample", "export", "review", "auto", "summary"], default="summary",
                        help="balllabels step (auto: trajectory-teacher labels, no review needed)")
    parser.add_argument("--n", type=int, default=2000, help="balllabels: total frames to label")
    parser.add_argument("--holdout", nargs="*", default=[], help="balllabels: matches kept entirely for validation")
    parser.add_argument("--labeler", default="reviewer", help="balllabels review: name stored with labels")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--compare", nargs="*", help="balltrain: only score these weights on the val split")
    parser.add_argument("--target", type=int, default=200, help="gold: total gold labels wanted")
    parser.add_argument("--model", help="gold: Qwen3-VL model id (MLX)")
    parser.add_argument("--force", action="store_true", help="gold: write labels even if calibration fails")
    args = parser.parse_args()
    if not args.video_ids and args.stage not in ("balltrain", "balllabels"):
        parser.error(f"{args.stage} needs at least one video id")

    if args.stage == "scenes":
        from . import scenes

        for vid in args.video_ids:
            scenes.embed_video(vid, video_path(vid), keyframes_only=args.scene_keyframes)
        if args.reuse_classifier and scenes.CLASSIFIER_PATH.exists():
            print("reusing", scenes.CLASSIFIER_PATH)
        else:
            print(json.dumps(scenes.train_classifier(args.video_ids, args.scene_labeler)))
        for vid in args.video_ids:
            segs = scenes.predict_segments(vid)
            print(vid, len(segs), "segments", round(segs.duration.sum() / 60, 1), "min kept")
    elif args.stage == "track":
        from .process import track_match

        for vid in args.video_ids:
            print(json.dumps(track_match(vid, video_path(vid), args.limit_segments, args.ball_backend,
                                         args.ball_weights, inline_crops=not args.no_inline_crops)))
    elif args.stage == "events":
        from .process import events_match

        for vid in args.video_ids:
            hits = events_match(vid, video_path(vid), serve_detector=not args.no_serve_detector)
            n_serves = int(hits.is_serve.sum()) if len(hits) else 0
            print(vid, len(hits), "hits,", n_serves, "serves")
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
    elif args.stage == "ocr":
        from . import ocr

        for vid in args.video_ids:
            res = ocr.rebuild(vid) if args.rebuild else ocr.run(vid, video_path(vid), args.ocr_engine)
            print(json.dumps(res, indent=2, default=str))
    elif args.stage == "balllabels":
        from . import ball_labels

        if args.action == "sample":
            if not args.video_ids:
                parser.error("balllabels --action sample needs video ids")
            ball_labels.sample(args.video_ids, n=args.n, holdout=args.holdout)
        elif args.action == "export":
            print(ball_labels.export(video_path, args.video_ids or None), "frames exported")
        elif args.action == "review":
            ball_labels.Reviewer(args.labeler).run()
        elif args.action == "auto":
            from . import ball_teacher

            if not args.video_ids:
                parser.error("balllabels --action auto needs video ids")
            ball_teacher.auto_label(args.video_ids, video_path, holdout=args.holdout)
        print(json.dumps(ball_labels.summary(), indent=2))
    elif args.stage == "balltrain":
        from . import tracknet_train

        if args.compare:
            print(json.dumps(tracknet_train.compare(args.compare), indent=2))
        else:
            rep = tracknet_train.train(args.epochs, args.batch, args.lr, init=args.ball_weights)
            print(json.dumps({k: rep[k] for k in ("weights", "train_frames", "val_frames", "val_pretrained",
                                                  "val_finetuned")}, indent=2))
    elif args.stage == "gold":
        from . import gold

        rep = gold.expand(args.video_ids, video_path, target=args.target, model_id=args.model or gold.MODEL,
                          force=args.force)
        print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    main()
