# Grand Slam public-video index

This repository indexes publicly visible match videos for Australian Open and US Open
men's and women's singles quarterfinals, semifinals, and finals from 2000 through 2026.
It stores metadata and watch-page URLs only; it does not contain or download video.

## Scope

There are 756 target bracket slots:

- 27 seasons
- 2 tournaments
- 2 singles draws
- 7 matches per draw (4 quarterfinals, 2 semifinals, 1 final)

Two slots were walkovers with no match to film, so the playable-video denominator is 754.
They remain in `matches.csv` with `coverage_status=not_played_walkover`.

The canonical player pairings come from Jeff Sackmann's public tennis result data. The
archival mirror is CC BY-NC-SA 4.0 and must be attributed. The 2026 US Open pairings
are filled from official US Open public video metadata because the result-data snapshot
predates that tournament.

## Files

- `data/matches.csv` — one row for each of the 756 target matches, with coverage counts.
- `data/videos.csv` — videos that automatically match a target's year, round, and players.
- `data/coverage.csv` — coverage totals by tournament, season, and draw.
- `data/unmatched_candidates.csv` — target-round videos that could not be safely matched;
  this intentionally retains doubles, wheelchair, junior, mistitled, and ambiguous videos
  for manual review.
- `data/ao_research_discoveries.jsonl` — reviewed Australian Open discovery catalog used
  to distinguish plausible complete matches from highlight-only fallbacks.
- `data/sources.csv` — discovery and result-data provenance.

`match_id` is the stable join key between `matches.csv` and `videos.csv`. Multiple videos
can map to the same match.

### Match statuses

- `verified_official_full_match` — official tournament channel, title explicitly says
  "Full Match", duration is at least 30 minutes, and both players match the canonical draw.
- `candidate_full_match` — title and duration look like a full match, but the source or
  player match needs review.
- `candidate_match_video` — relevant public video such as a condensed match, classic
  replay, or other non-full-match candidate.

These are discovery labels, not statements about copyright or permission to download.

## Refreshing the index

Requirements: Python 3.11+ and a current `yt-dlp`. The scripts default to running the
latest isolated release with `uvx --from yt-dlp yt-dlp`.

```bash
python3 scripts/fetch_match_records.py
python3 scripts/discover_official.py

python3 scripts/build_manifest.py \
  --records-dir .cache/match_records \
  --australian-open '.cache/discovery/australian-open/*.json' \
  --us-open '.cache/discovery/us-open/*.json' \
  --manual data/manual_discoveries.json data/ao_research_discoveries.jsonl

# Optional, slower: query each match not covered by an official full replay.
python3 scripts/search_missing.py

python3 scripts/build_manifest.py \
  --records-dir .cache/match_records \
  --australian-open '.cache/discovery/australian-open/*.json' \
                    '.cache/global_search/australian-open/*.json' \
  --us-open '.cache/discovery/us-open/*.json' \
            '.cache/global_search/us-open/*.json' \
  --manual data/manual_discoveries.json data/ao_research_discoveries.jsonl
```

The global search uses only two workers by default to avoid aggressive request rates.
Search results from unofficial channels remain candidates even when their titles say
"Full Match."

## Verification priorities

Before using a candidate:

1. Open the watch page and confirm it remains public.
2. Verify that it contains the full match rather than highlights or a watch-along.
3. Confirm player names, draw, round, and year.
4. Record the applicable license or written permission separately.
5. If downloading is authorized, preserve the platform ID and `.info.json` metadata.

Public availability does not grant permission to download, train on, or redistribute a
copyrighted broadcast. Do not bypass DRM, authentication, geographic restrictions, or
other access controls.

## Current snapshot

Discovery run: 2026-09-22.

- 3,568 unique public search/playlist results inspected.
- 1,099 videos matched to a canonical target match.
- 396 verified official full-video records covering 370 unique matches.
- Australian Open: 192 of 377 played matches have a verified official full video;
  267 have either a verified or reviewed candidate complete video, and 354 have at
  least one matched public video.
- US Open: 178 of 377 played matches have a verified official full video;
  303 have at least one matched public video candidate.
- Across both tournaments, 78 additional matches have a reviewed or non-official
  full-match candidate but no verified official full video.

The Australian Open currently has slightly more verified official full-match coverage
than the US Open in this scope. This repository does not yet measure Roland-Garros or
Wimbledon, so it does not establish a four-Slam ranking.
