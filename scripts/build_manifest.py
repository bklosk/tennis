#!/usr/bin/env python3
"""Build canonical match and public-video manifests from flat yt-dlp metadata."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

TOURNAMENTS = {"Australian Open", "US Open"}
ROUNDS = {"QF", "SF", "F"}
OFFICIAL_CHANNEL_IDS = {
    "Australian Open": {
        "UCeTKJSW1NTAkf27nNmjWt5A",  # Australian Open
        "UCgUrLyvzsRqgGSmTBbxL84A",  # Tennis Australia
    },
    "US Open": {
        "UCXbboag48Qlr78zzz6SkzkQ",  # US Open Tennis Championships
        "UC7joGi4V3-r9i5tmmw7dM6g",  # United States Tennis Association
    },
}

# The archived match-result snapshot was published before the 2026 US Open.
# These pairings are recoverable from the official tournament video metadata.
US_OPEN_2026 = [
    ("men", "QF", "Ben Shelton", "Carlos Alcaraz"),
    ("men", "QF", "Frances Tiafoe", "Alex Michelsen"),
    ("men", "QF", "Alexander Zverev", "Botic van de Zandschulp"),
    ("men", "QF", "Karen Khachanov", "Alexander Blockx"),
    ("men", "SF", "Frances Tiafoe", "Ben Shelton"),
    ("men", "SF", "Alexander Zverev", "Karen Khachanov"),
    ("men", "F", "Alexander Zverev", "Ben Shelton"),
    ("women", "QF", "Qinwen Zheng", "Elena Rybakina"),
    ("women", "QF", "Mirra Andreeva", "Coco Gauff"),
    ("women", "QF", "Jessica Pegula", "Emma Navarro"),
    ("women", "QF", "Aryna Sabalenka", "Linda Noskova"),
    ("women", "SF", "Coco Gauff", "Elena Rybakina"),
    ("women", "SF", "Aryna Sabalenka", "Jessica Pegula"),
    ("women", "F", "Aryna Sabalenka", "Elena Rybakina"),
]


def normalize(value: str | None) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def slug(value: str) -> str:
    return normalize(value).replace(" ", "-")


def title_year(title: str) -> int | None:
    match = re.search(r"\b(20(?:0\d|1\d|2[0-6]))\b", title)
    return int(match.group(1)) if match else None


def title_round(title: str) -> str | None:
    title = normalize(title)
    if re.search(r"\bquarter ?finals?\b|\bqf\b|\b1 4 finals?\b", title):
        return "QF"
    if re.search(r"\bsemi ?finals?\b|\bsf\b|\b1 2 finals?\b", title):
        return "SF"
    if re.search(r"\bfinals?\b", title):
        return "F"
    return None


def token_score(name: str, title: str) -> float:
    tokens = normalize(name).split()
    title_tokens = set(normalize(title).split())
    return sum(token in title_tokens for token in tokens) / len(tokens)


def extract_title_sides(title: str) -> tuple[str, str] | None:
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    prefix = re.split(r"(?i)\bfull match\b", ascii_title, maxsplit=1)[0]
    match = re.search(
        r"(?i)([^|]{2,90}?)\s+(?:v(?:s\.?)?|beats)\s+([^|]{2,110})",
        prefix,
    )
    if not match:
        return None
    left = match.group(1).split("|")[-1].strip(" !-")
    right = re.split(
        r"(?i)\s+(?:in (?:an? )?|iconic |classic |at the ).*$",
        match.group(2),
        maxsplit=1,
    )[0].strip(" !-")
    return left, right


def name_similarity(name: str, candidate: str, title: str) -> float:
    return max(
        token_score(name, title),
        SequenceMatcher(None, normalize(name), normalize(candidate)).ratio(),
    )


def pair_score(player1: str, player2: str, title: str) -> tuple[float, float]:
    sides = extract_title_sides(title)
    if not sides:
        scores = (token_score(player1, title), token_score(player2, title))
        return sum(scores) / 2, min(scores)

    a, b = sides
    direct = (
        name_similarity(player1, a, title),
        name_similarity(player2, b, title),
    )
    swapped = (
        name_similarity(player1, b, title),
        name_similarity(player2, a, title),
    )
    selected = direct if sum(direct) >= sum(swapped) else swapped
    return sum(selected) / 2, min(selected)


def make_match(
    tournament: str,
    year: int,
    draw: str,
    round_code: str,
    player1: str,
    player2: str,
    result_source: str,
    result: str = "",
) -> dict[str, object]:
    players = sorted((player1, player2), key=normalize)
    match_id = "-".join(
        [
            slug(tournament),
            str(year),
            draw,
            round_code.lower(),
            slug(players[0]),
            slug(players[1]),
        ]
    )
    return {
        "match_id": match_id,
        "tournament": tournament,
        "year": year,
        "draw": draw,
        "round": round_code,
        "player1": players[0],
        "player2": players[1],
        "result": result,
        "match_played": "W/O" not in result.upper(),
        "result_source": result_source,
    }


def load_matches(records_dir: Path) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    source = "Jeff Sackmann tennis data archive (CC BY-NC-SA 4.0)"
    for tour, draw in (("atp", "men"), ("wta", "women")):
        for path in sorted((records_dir / tour).glob(f"{tour}_matches_20*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    tournament = row.get("tourney_name")
                    if tournament == "Us Open":
                        tournament = "US Open"
                    if tournament not in TOURNAMENTS or row.get("round") not in ROUNDS:
                        continue
                    year = int(row["tourney_date"][:4])
                    if 2000 <= year <= 2026:
                        matches.append(
                            make_match(
                                tournament,
                                year,
                                draw,
                                row["round"],
                                row["winner_name"],
                                row["loser_name"],
                                source,
                                row.get("score") or "",
                            )
                        )

    for draw, round_code, player1, player2 in US_OPEN_2026:
        matches.append(
            make_match(
                "US Open",
                2026,
                draw,
                round_code,
                player1,
                player2,
                "Official US Open public video metadata",
            )
        )

    deduped = {row["match_id"]: row for row in matches}
    result = sorted(
        deduped.values(),
        key=lambda row: (
            row["tournament"],
            row["year"],
            row["draw"],
            {"QF": 0, "SF": 1, "F": 2}[row["round"]],
            row["player1"],
        ),
    )
    if len(result) != 756:
        raise ValueError(f"Expected 756 target matches, found {len(result)}")
    return result


def expand_json_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.update(Path(path) for path in matches)
        else:
            paths.add(Path(pattern))
    return sorted(paths)


def load_discoveries(grouped_paths: dict[str, list[Path]]) -> list[dict[str, object]]:
    videos: dict[tuple[str, str], dict[str, object]] = {}
    for tournament, paths in grouped_paths.items():
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                if path.suffix == ".jsonl":
                    entries = [json.loads(line) for line in handle if line.strip()]
                    documents = [{"title": path.stem, "entries": entries}]
                else:
                    documents = [json.load(handle)]
            for document in documents:
                collection = document.get("title") or path.stem
                for entry in document.get("entries") or []:
                    add_discovery(videos, tournament, collection, entry)

    for record in videos.values():
        record["source_collections"] = " | ".join(sorted(record["source_collections"]))
    return list(videos.values())


def add_discovery(
    videos: dict[tuple[str, str], dict[str, object]],
    tournament: str,
    collection: str,
    entry: dict[str, object],
) -> None:
    video_id = entry.get("id") or entry.get("video_id") or ""
    entry_title = entry.get("title") or ""
    normalized_title = normalize(entry_title)
    entry_tournament = (
        str(entry.get("tournament"))
        if entry.get("tournament") in TOURNAMENTS
        else "Australian Open"
        if "australian open" in normalized_title
        else "US Open"
        if "us open" in normalized_title
        else tournament
    )
    platform = normalize(
        entry.get("platform")
        or entry.get("extractor_key")
        or entry.get("ie_key")
        or "youtube"
    )
    platform = "dailymotion" if "dailymotion" in platform else "youtube"
    if platform == "youtube" and not re.fullmatch(r"[\w-]{11}", str(video_id)):
        return
    if platform == "dailymotion" and not re.fullmatch(r"[\w-]{3,64}", str(video_id)):
        return
    key = (platform, str(video_id))
    record = videos.setdefault(
        key,
        {
            "tournament": entry_tournament,
            "platform": platform,
            "video_id": video_id,
            "watch_url": entry.get("webpage_url")
            or entry.get("url")
            or (
                f"https://www.youtube.com/watch?v={video_id}"
                if platform == "youtube"
                else f"https://www.dailymotion.com/video/{video_id}"
            ),
            "title": entry_title,
            "channel": entry.get("channel")
            or entry.get("uploader")
            or entry.get("uploader_channel")
            or "",
            "channel_id": entry.get("channel_id") or entry.get("uploader_id") or "",
            "duration_seconds": entry.get("duration")
            or entry.get("approx_duration_seconds")
            or "",
            "view_count": entry.get("view_count") or "",
            "availability": entry.get("availability") or "public_search_result",
            "source_collections": set(),
            "research_confidence": "",
            "research_year": "",
            "research_draw": "",
            "research_round": "",
            "research_player1": "",
            "research_player2": "",
        },
    )
    record["source_collections"].add(collection)
    research_confidence = str(entry.get("confidence") or "")
    confidence_rank = {"": 0, "highlight": 1, "candidate": 2, "verified": 3}
    if confidence_rank.get(research_confidence, 0) > confidence_rank.get(
        str(record["research_confidence"]), 0
    ):
        record["research_confidence"] = research_confidence
        record["research_year"] = entry.get("year") or ""
        record["research_draw"] = entry.get("draw") or ""
        record["research_round"] = entry.get("round") or ""
        record["research_player1"] = entry.get("player1") or ""
        record["research_player2"] = entry.get("player2") or ""


def match_videos(
    matches: list[dict[str, object]],
    discoveries: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    match_groups: dict[tuple[str, int, str], list[dict[str, object]]] = defaultdict(list)
    for match in matches:
        if match["match_played"]:
            match_groups[(match["tournament"], match["year"], match["round"])].append(match)

    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    matched: list[dict[str, object]] = []
    unmatched: list[dict[str, object]] = []

    for video in discoveries:
        title = str(video["title"])
        research_round = normalize(str(video.get("research_round") or ""))
        round_code = (
            "QF"
            if research_round in {"qf", "quarterfinal", "quarter final"}
            else "SF"
            if research_round in {"sf", "semifinal", "semi final"}
            else "F"
            if research_round in {"f", "final"}
            else title_round(title)
        )
        year = (
            int(video["research_year"])
            if video.get("research_year")
            else title_year(title)
        )
        if year is None or round_code is None:
            continue

        pool = match_groups.get((video["tournament"], year, round_code), [])
        research_draw = str(video.get("research_draw") or "")
        research_title = " vs ".join(
            str(video.get(key) or "") for key in ("research_player1", "research_player2")
        )
        score_title = research_title if research_title != " vs " else title
        scored = sorted(
            (
                (
                    *pair_score(str(match["player1"]), str(match["player2"]), score_title),
                    match,
                )
                for match in pool
                if not research_draw or match["draw"] == research_draw
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        best_score, minimum_score, best_match = scored[0] if scored else (0.0, 0.0, None)

        normalized_title = normalize(title)
        conflicting_event = any(
            event in normalized_title
            for event in (
                "wimbledon",
                "roland garros",
                "french open",
                "miami open",
                "indian wells",
            )
        )
        disqualifier = any(
            marker in normalized_title
            for marker in ("highlight", "condensed", "watch along", "radio commentary")
        )
        explicit_full = "full match" in title.lower() and not disqualifier
        duration = int(video["duration_seconds"] or 0)
        official = (
            video["channel_id"] in OFFICIAL_CHANNEL_IDS[video["tournament"]]
            or normalize(str(video["channel"]))
            in {
                "australian open",
                "tennis australia",
                "us open tennis championships",
                "united states tennis association usta",
            }
        )
        research_complete = video.get("research_confidence") in {"verified", "candidate"}

        base = {
            **video,
            "detected_year": year,
            "detected_round": round_code,
            "is_official_channel": official,
            "is_explicit_full_match": explicit_full,
            "match_score": f"{best_score:.3f}",
            "minimum_player_score": f"{minimum_score:.3f}",
            "last_checked_utc": checked_at,
        }

        if (
            best_match
            and not conflicting_event
            and best_score >= 0.72
            and minimum_score >= 0.55
        ):
            if (
                official
                and explicit_full
                and duration >= 1_800
                and best_score >= 0.88
                and minimum_score >= 0.78
            ):
                status = "verified_official_full_match"
            elif (explicit_full or research_complete) and duration >= 1_200:
                status = "candidate_full_match"
            else:
                status = "candidate_match_video"
            matched.append({**best_match, **base, "match_status": status})
        else:
            unmatched.append(
                {
                    **base,
                    "best_match_id": best_match["match_id"] if best_match else "",
                    "match_status": "unmatched_target_round_candidate",
                }
            )

    matched.sort(
        key=lambda row: (
            row["tournament"],
            row["year"],
            row["draw"],
            row["round"],
            row["match_id"],
            row["video_id"],
        )
    )
    unmatched.sort(key=lambda row: (row["tournament"], row["detected_year"], row["title"]))
    return matched, unmatched


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--australian-open", nargs="+", required=True)
    parser.add_argument("--us-open", nargs="+", required=True)
    parser.add_argument("--manual", nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    matches = load_matches(args.records_dir)
    discoveries = load_discoveries(
        {
            "Australian Open": expand_json_paths(args.australian_open),
            "US Open": expand_json_paths(args.us_open),
            "manual": expand_json_paths(args.manual),
        }
    )
    videos, unmatched = match_videos(matches, discoveries)

    verified_counts: dict[str, int] = defaultdict(int)
    candidate_full_counts: dict[str, int] = defaultdict(int)
    candidate_counts: dict[str, int] = defaultdict(int)
    for video in videos:
        candidate_counts[video["match_id"]] += 1
        if video["match_status"] == "verified_official_full_match":
            verified_counts[video["match_id"]] += 1
        elif video["match_status"] == "candidate_full_match":
            candidate_full_counts[video["match_id"]] += 1
    match_rows = []
    for match in matches:
        match_rows.append(
            {
                **match,
                "public_video_count": candidate_counts[match["match_id"]],
                "verified_official_full_video_count": verified_counts[match["match_id"]],
                "candidate_full_video_count": candidate_full_counts[match["match_id"]],
                "coverage_status": (
                    "not_played_walkover"
                    if not match["match_played"]
                    else "verified_official_full_match"
                    if verified_counts[match["match_id"]]
                    else "public_full_match_candidate"
                    if candidate_full_counts[match["match_id"]]
                    else "candidate_only"
                    if candidate_counts[match["match_id"]]
                    else "not_found"
                ),
            }
        )

    write_csv(
        args.output_dir / "matches.csv",
        match_rows,
        [
            "match_id",
            "tournament",
            "year",
            "draw",
            "round",
            "player1",
            "player2",
            "result",
            "match_played",
            "result_source",
            "public_video_count",
            "verified_official_full_video_count",
            "candidate_full_video_count",
            "coverage_status",
        ],
    )
    write_csv(
        args.output_dir / "videos.csv",
        videos,
        [
            "match_id",
            "tournament",
            "year",
            "draw",
            "round",
            "player1",
            "player2",
            "platform",
            "video_id",
            "watch_url",
            "title",
            "channel",
            "channel_id",
            "duration_seconds",
            "view_count",
            "availability",
            "source_collections",
            "research_confidence",
            "is_official_channel",
            "is_explicit_full_match",
            "match_score",
            "minimum_player_score",
            "match_status",
            "last_checked_utc",
        ],
    )

    coverage_rows = []
    for tournament in sorted(TOURNAMENTS):
        for year in range(2000, 2027):
            for draw in ("men", "women"):
                group = [
                    row
                    for row in match_rows
                    if row["tournament"] == tournament
                    and row["year"] == year
                    and row["draw"] == draw
                ]
                coverage_rows.append(
                    {
                        "tournament": tournament,
                        "year": year,
                        "draw": draw,
                        "target_matches": len(group),
                        "played_matches": sum(row["match_played"] for row in group),
                        "walkovers": sum(not row["match_played"] for row in group),
                        "verified_official_full_matches": sum(
                            row["coverage_status"] == "verified_official_full_match"
                            for row in group
                        ),
                        "public_full_match_candidates": sum(
                            row["coverage_status"] == "public_full_match_candidate"
                            for row in group
                        ),
                        "other_video_candidates": sum(
                            row["coverage_status"] == "candidate_only" for row in group
                        ),
                        "not_found": sum(row["coverage_status"] == "not_found" for row in group),
                    }
                )
    write_csv(
        args.output_dir / "coverage.csv",
        coverage_rows,
        [
            "tournament",
            "year",
            "draw",
            "target_matches",
            "played_matches",
            "walkovers",
            "verified_official_full_matches",
            "public_full_match_candidates",
            "other_video_candidates",
            "not_found",
        ],
    )
    write_csv(
        args.output_dir / "unmatched_candidates.csv",
        unmatched,
        [
            "tournament",
            "platform",
            "video_id",
            "watch_url",
            "title",
            "channel",
            "channel_id",
            "duration_seconds",
            "detected_year",
            "detected_round",
            "best_match_id",
            "match_score",
            "minimum_player_score",
            "match_status",
            "source_collections",
            "research_confidence",
            "last_checked_utc",
        ],
    )

    covered = {
        row["match_id"]
        for row in videos
        if row["match_status"] == "verified_official_full_match"
    }
    print(f"targets={len(matches)} discoveries={len(discoveries)} matched_videos={len(videos)}")
    for tournament in sorted(TOURNAMENTS):
        target_count = sum(
            row["tournament"] == tournament and row["match_played"] for row in matches
        )
        covered_count = sum(
            row["tournament"] == tournament and row["match_id"] in covered for row in matches
        )
        print(f"{tournament}: {covered_count}/{target_count} verified full-match coverage")


if __name__ == "__main__":
    main()
