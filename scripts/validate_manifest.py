#!/usr/bin/env python3
"""Validate structural and classification invariants in the generated manifests."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def read(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    matches = read("data/matches.csv")
    videos = read("data/videos.csv")
    assert len(matches) == 756
    assert len({row["match_id"] for row in matches}) == 756
    assert sum(row["match_played"] == "True" for row in matches) == 754
    assert Counter(row["coverage_status"] for row in matches)["not_played_walkover"] == 2

    match_ids = {row["match_id"] for row in matches}
    assert all(row["match_id"] in match_ids for row in videos)
    video_keys = {(row["platform"], row["video_id"]) for row in videos}
    assert len(video_keys) == len(videos)

    for row in videos:
        if row["match_status"] == "verified_official_full_match":
            assert row["is_official_channel"] == "True"
            assert row["is_explicit_full_match"] == "True"
            assert float(row["duration_seconds"]) >= 1_800
            assert float(row["match_score"]) >= 0.88
            assert float(row["minimum_player_score"]) >= 0.78

    by_match = Counter(row["match_id"] for row in videos)
    verified = Counter(
        row["match_id"]
        for row in videos
        if row["match_status"] == "verified_official_full_match"
    )
    candidate_full = Counter(
        row["match_id"] for row in videos if row["match_status"] == "candidate_full_match"
    )
    for row in matches:
        assert int(row["public_video_count"]) == by_match[row["match_id"]]
        assert int(row["verified_official_full_video_count"]) == verified[row["match_id"]]
        assert int(row["candidate_full_video_count"]) == candidate_full[row["match_id"]]

    print(
        f"validated matches={len(matches)} played=754 videos={len(videos)} "
        f"verified_full_videos={sum(verified.values())}"
    )


if __name__ == "__main__":
    main()
