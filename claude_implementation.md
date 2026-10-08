# US Open Late-Round Charting Pipeline: Implementation Guide

Written for an autonomous coding agent. Facts in this guide were checked on 2026-10-08.

## 0. How to use this guide

This guide covers four things: what to build, what is already known, where the traps are, and how success will be judged. It leaves implementation choices to you. That includes programming language, libraries, storage formats, model architectures, and code layout. The one fixed choice is that the user has picked RF-DETR and OpenAI's Decisions API as components. RF-DETR ships as a Python package (`pip install rfdetr`, Python 3.10+), so at least that part will be Python. Everything else is your call.

The guide uses three markers. "Default" means a reasonable starting point; replace it if you can show something better on the evaluation in §11. "Verify" means the fact was true when this was written but may have changed, so check before you depend on it. When this guide and the data disagree, the data wins. Note the discrepancy in your decisions log and in the final report.

Two principles override everything else:

1. **Measure before trusting.** Evaluate every automated field against human-charted labels before using it on uncharted matches. Every field ships with a calibrated confidence.
2. **Never let machine output pass as human charting.** Provenance travels with every value, all the way to the exported files.

## 1. Objective

### 1.1 Target set

The targets are all US Open men's and women's singles quarterfinals, semifinals, and finals from 2001 through 2025. That is 7 matches per draw per year, 350 matches in total (175 men's, 175 women's).

One target was never played. The 2025 women's QF (Sabalenka d. Vondroušová) was a walkover, which leaves 349 matches with play.

Six matches ended in retirement and are partial:

| Year | Event | Match |
|---|---|---|
| 2011 | Men's QF | Djokovic d. Tipsarevic |
| 2014 | Women's SF | Wozniacki d. Peng |
| 2016 | Men's QF | Djokovic d. Tsonga |
| 2018 | Men's SF | del Potro d. Nadal |
| 2021 | Men's QF | Auger-Aliassime d. Alcaraz |
| 2024 | Men's QF | Tiafoe d. Dimitrov |

### 1.2 What to produce

For every playable target match, produce a point-by-point and shot-by-shot record. It should approximate the Match Charting Project (MCP) notation. It should also include two things MCP does not record:

- timestamps into the source video
- both players' court positions at every contact, including serve contact

Every field carries three pieces of metadata:

- a probability distribution, or at least a confidence
- a source: human MCP charting, the official point-by-point feed, or a model
- the model and version that produced it, when the source is a model

Where human or official data exists, keep it as the primary value. Store the model's prediction alongside it for evaluation, and never overwrite.

### 1.3 Priorities

Build in tiers. Deliver each tier for every match before going deeper into the next.

| Tier | Contents |
|---|---|
| 1. Point skeleton | For every point: server and returner, which end each is at, deuce or ad side, serve number, score state before the point, point winner, serve direction, video timestamps |
| 2. Rally structure | Every contact's time and hitter, rally length, both players' positions at each contact, stroke side, stroke type |
| 3. Rest of MCP without the bounce | Shot direction (1/2/3), return depth (7/8/9), approach and net-position markers, point outcome type (winner, forced error, unforced error) |
| 4. Bounce-dependent | Error type (net, wide, deep), depth on rally shots, serve-fault type |

Tier 1 gets the most care, because on its own it supports serve-strategy analysis, which is a likely downstream use. Tier 4 may legitimately ship as "unknown" for many shots. MCP has unknown codes for exactly this purpose.

### 1.4 Definition of done

The project is done when all of the following exist:

- a versioned dataset covering every playable match, with honest unknowns
- an evaluation report (§11) with results per field and per era
- a pipeline that can be re-run end to end from raw inputs
- the human-in-the-loop tools described in §12

## 2. What already exists (checked 2026-10-08)

Much of the target set is already charted, partly or fully. Reproducing the numbers in this section is milestone M0. If your numbers differ, find out why before going further.

### 2.1 Match Charting Project

**Files.** The repository is github.com/JeffSackmann/tennis_MatchChartingProject, and it is still public.

- Match lists: `charting-m-matches.csv` and `charting-w-matches.csv`
- Points, split by decade: `charting-{m,w}-points-to-2009.csv`, `charting-{m,w}-points-2010s.csv`, and `charting-{m,w}-points-2020s.csv`

**Parsing quirks.**

- Read the files as latin-1, not UTF-8.
- Text fields contain stray trailing spaces. Both "US Open" and "US Open " occur, as do "R64" and "R64 ". Strip before comparing.
- Match IDs look like `20260517-M-Rome_Masters-F-Casper_Ruud-Jannik_Sinner`. Dates are YYYYMMDD.

**Match-file columns:** match_id, Player 1, Player 2, Pl 1 hand, Pl 2 hand, Date, Tournament, Round, Time, Court, Surface, Umpire, Best of, Final TB?, Charted by.

**Point-file columns:** match_id, Pt, Set1, Set2, Gm1, Gm2, Pts, Gm#, TbSet, Svr, 1st, 2nd, Notes, PtWinner.

- `1st` holds the first serve and, if it went in, the whole rally.
- `2nd` holds the second serve and the rally after a first-serve fault. About 37% of points have one.
- `Pts` appears to be in server-first order: it shows "15-0" after the server wins the first point. Verify this.
- `Notes` is free text. It sometimes contains numbers that look like serve speeds.

**Size and coverage.** The full repository holds 11,785 matches and about 1.88 million points. Its README still says "over 5,000," which is stale. For the target set:

- 179 of the 350 matches are charted: 120 of 175 men's and 59 of 175 women's.
- Those 179 matches contain about 36,000 points.
- Counting serves, that is roughly 178,000 shots, or about 4.9 shots per point.

**License.** CC BY-NC-SA 4.0. Attribution is required, use must be non-commercial, and derived datasets must carry the same license. The repository README says the author takes violations seriously, so you should too.

### 2.2 Slam point-by-point (official feed)

**Where to get it.** Jeff Sackmann's tennis_slam_pointbypoint repository was removed from GitHub around June 2026. So were tennis_atp, tennis_wta, and tennis_pointbypoint; all now return 404. A third-party archival mirror exists at github.com/Aneeshers/tennis-sackmann-archive, also on Hugging Face as Aneeshers/tennis-sackmann-archive.

- Its `slam_pointbypoint/` folder is a snapshot of upstream commit 6febb77 (October 2024) and covers 2011–2024.
- Its `atp/` and `wta/` folders are June 2026 snapshots that run through 2026.
- It carries the same CC BY-NC-SA license.

Pin a specific commit of the mirror and record it. As a provenance check, the mirror agrees with MCP charting on overlapping matches far better than chance (§2.4). That is good evidence the data is genuine. Still, tell the user you are relying on a third-party copy.

**Origin.** The upstream README, kept in the mirror, says the data was scraped from the IBM SlamTracker feature on the slam websites. It is generally available only for courts with Hawk-Eye installed.

**Files and structure.** Files are named `{year}-usopen-matches.csv` and `{year}-usopen-points.csv`.

- The matches file's `round` column is empty. Derive the round from `match_num`, a four-digit code:
  - first digit is the event: 1 = men's singles, 2 = women's singles
  - second digit is the round: 1 = R128 through 7 = F, so 5 = QF, 6 = SF, 7 = F
  - last two digits are the match index
- Points files can contain a `PointNumber` 0 placeholder row at the start of a match. Drop it.

All 196 late-round matches from 2011–2024 are present except one: the 2011 women's QF Kerber–Pennetta (match_num 2503).

**Field availability.** This varies by year. For the late-round matches:

| Field | Years populated (US Open QF/SF/F) | Notes |
|---|---|---|
| ElapsedTime, PointServer, PointWinner, Speed_KMH, P1/P2 Ace, DoubleFault, Winner, UnfErr, NetPoint, History | 2011–2024 | Speed_KMH of 0 means missing |
| ServeNumber | 2014–2024 | The data dictionary also defines ServeIndicator (1/2); check which years use which |
| WinnerShotType, WinnerType | 2014–2024 | Winner shot type is F or B |
| P1/P2 DistanceRun | 2015–2024 | Appears to be meters per point; zeros are common, so treat zero as possibly missing |
| RallyCount | 2016–2024 | 2011 has a different column, `Rally`; 2012–2015 have no rally length at all |
| ServeWidth, ServeDepth, ReturnDepth | 2016–2024 | Width and depth on about 92–98% of points; return depth on about 76–86% (no return exists on aces, etc.) |
| Serve_Direction, ServingTo | 2011 only | Undocumented; see §2.4 |
| P1/P2 ForcedError | 2011 only | Empty in later years |

**Code definitions,** from the data dictionary:

| Field | Codes |
|---|---|
| ServeWidth | W wide, BW body/wide, B body, BC body/center, C center ("down the T") |
| ServeDepth | CTL close to line, NCTL not close to line |
| ReturnDepth | D deep, ND not deep |

`ElapsedTime` is h:mm:ss since the start of the match. It is a strong anchor for aligning points to video (§6.8).

### 2.3 Tour results and players (mirror `atp/` and `wta/`)

**Target list.** The files `atp_matches_YYYY.csv` and `wta_matches_YYYY.csv` give the authoritative target list.

- `tourney_name` is spelled both "US Open" and "Us Open".
- `round` is QF, SF, or F.
- Filtering 2001–2025 yields exactly 350 rows.
- The `score` string marks retirements (RET) and walkovers (W/O).

**Match-level serve statistics.** These files also hold official per-match stats: aces, double faults, serve points, first serves in, first- and second-serve points won, service games, and break points saved and faced. Coverage for the target set:

| Era | Men's | Women's |
|---|---|---|
| 2001–2005 | 100% | about 60% |
| 2006–2010 | 100% | about 91% |
| 2011–2015 | 100% | about 80% |
| 2016–2025 | 100% | essentially 100% |

Use these stats as aggregate constraints and as validation when you reconstruct a match that has no point-level truth.

**Handedness.** `winner_hand` and `loser_hand` hold R, L, or U, and the players files have it too. MCP match files also record handedness.

**Joining sources.** Join MCP, the feed, and tour records on year, gender, round, and normalized surnames. A join on ASCII-folded last names matched all 97 MCP target matches from 2011–2024 to their feed records.

### 2.4 Coverage and cross-source agreement

Combining sources, 234 of the 349 playable matches already have serve direction on nearly every point. MCP supplies it for all charted matches, and ServeWidth supplies it for 2016–2024. Here are the gaps:

| Era | Playable matches | MCP charted | Serve direction available | Gap: full charting | Gap: serve direction |
|---|---|---|---|---|---|
| 2001–2010 | 140 | 70 | 70 | 70 | 70 |
| 2011–2015 | 70 | 28 | 28 (plus up to 6 via 2011 Serve_Direction) | 42 | 42 (36 if the 2011 field is used) |
| 2016–2024 | 126 | 69 | 124 | 57 | 2 |
| 2025 | 13 | 12 | 12 | 1 | 1 |
| Total | 349 | 179 | 234 | 170 | 115 |

- The full-charting gap is two-thirds women's matches: 115 of 170 (the men's gap is 55).
- The serve-direction gap is almost entirely 2001–2015, which is also the oldest and lowest-quality footage. Plan for the hardest video to be where the work is.
- About 72 matches have no point-level truth at all: neither MCP nor feed data. These are the 70 uncharted 2001–2010 matches, the 2011 women's QF Kerber–Pennetta, and the 2025 women's QF Osaka–Muchová. For those, the scoreboard is the only source of point outcomes (§6.9).

**The 2011 Serve_Direction field.** It is undocumented. Cross-tabulating it against MCP for the seven 2011 target matches present in both sources, with points aligned by index, suggests:

| Code | Meaning |
|---|---|
| 1 | wide |
| 2 | body |
| 3 | T |
| 0 | unknown |

Raw agreement is roughly three-quarters. Confirm with a proper alignment before you use it.

**ServeWidth versus MCP.** A comparison over 67 overlapping 2016–2024 matches whose point counts match between sources (about 13,000 points, aligned by index) found:

- When ServeWidth says W or C, MCP agrees 96% of the time.
- BW is mostly MCP "wide," by about 4:1.
- B is mostly "body."
- BC splits between "body" and "T," by about 1.3:1.

The best deterministic 3-bin mapping agrees about 84–85%:

| ServeWidth | MCP 3-bin |
|---|---|
| W, BW | wide |
| B, BC | body |
| C | T |

Index alignment adds some noise, so treat this as a slight underestimate. Even so, take roughly 85% as the practical ceiling for 3-bin serve direction measured against MCP. Human charters and the tracking feed disagree on borderline serves, so a model cannot be meaningfully "better" than that against either one.

## 3. Scope, constraints and hygiene

### 3.1 Footage

The user supplies the video. Do not download, scrape, or otherwise acquire broadcast footage yourself. Instead:

1. Build a manifest from the files you are given (§6.1).
2. Match each file to a target match.
3. Report which matches lack usable footage, so the user can decide what to do.

Do not redistribute footage or frames. The only time frames leave the machine is as inputs to the API calls in §9.

### 3.2 Licensing

**Data.** MCP and the point-by-point data are both CC BY-NC-SA 4.0. Treat any dataset you derive from them as carrying the same terms, and that includes model outputs trained on them. The terms are:

- attribution to Jeff Sackmann / Tennis Abstract
- non-commercial use only
- share-alike

Label every machine-generated value as machine-generated. Nothing you produce should be formatted or named in a way that could be mistaken for human MCP charting, or submitted as such.

**Models.** RF-DETR's licensing is split by model:

| License | Models |
|---|---|
| Apache 2.0 | Nano, Small, Medium, and Large detection models; the segmentation models; the keypoint preview model |
| PML 1.0 (requires the `rfdetr_plus` extension) | Atto, Femto, Pico, XL, and 2XL |

Stay on the Apache-licensed models unless the user approves otherwise.

### 3.3 Secrets and spend

API keys come from the environment and are never logged. All paid API usage goes through a budget guard with a user-set limit. Run a dry-run cost estimate before any bulk run (§9).

### 3.4 Compute

Assume a single T4-class GPU, possibly on a free or preemptible tier. The user has had trouble finding GPU capacity.

Everything must be resumable per (match, stage). After a preemption, completed work is never redone.

Avoid processing entire broadcasts at full frame rate and full resolution. Three facts make this possible:

- Most of a broadcast is not live play.
- Players move slowly relative to the frame rate.
- Only short windows around contacts need dense analysis.

Measure throughput on the vertical slice (milestone M2) and extrapolate before committing to a design. A rough expectation for all 349 matches is tens of T4-hours, not hundreds.

## 4. Domain knowledge the pipeline depends on

### 4.1 MCP notation

The authoritative definitions are in the MCP charting instructions, linked from tennisabstract.com/charting/meta.html. Read them. The summary below combines those conventions with what actually appears in the data, based on character frequencies across all 1.88 million points.

Build a real tokenizer and grammar, not regexes scattered through the code. The parser must handle at least 99.9% of the `1st`/`2nd` strings in the whole repository. Log and inspect the unparsed remainder; the data contains a few dozen junk characters from typos.

**Point structure.** Each point string has this shape:

1. zero or more `c` characters (lets)
2. a serve token
3. rally shot tokens
4. an outcome

**Serve tokens.** A serve token is a direction digit:

| Digit | Serve direction |
|---|---|
| `4` | wide |
| `5` | body |
| `6` | down the T |
| `0` | unknown |

What follows the serve digit:

- `+` right after the digit marks serve-and-volley.
- A fault is the digit followed by an error-type letter: `n` net, `w` wide, `d` deep, `x` wide and deep, `g` foot fault (`g` also appears alone), `e` unknown, `!` shank.
- An ace is the digit followed by `*`.
- An unreturnable serve (a service winner) is the digit followed by `#`.

After a first-serve fault the point continues in the `2nd` column. A fault there is a double fault.

Patterns to confirm in the data:

| Pattern | Approximate count across all MCP strings |
|---|---|
| ace, `[0456]*` | 129,000 |
| unreturnable, `[0456]#` | 35,000 |
| bare faults | 770,000 |

**Rally shot tokens.** Each rally shot is written in this order:

1. a shot letter
2. optional modifiers
3. a direction digit
4. optionally a depth digit
5. optionally an error-type letter and an outcome symbol

Shot letters:

| Code | Meaning | Code | Meaning |
|---|---|---|---|
| `f` | forehand groundstroke | `b` | backhand groundstroke |
| `r` | forehand slice (incl. defensive chips) | `s` | backhand slice |
| `v` | forehand volley | `z` | backhand volley |
| `o` | overhead/smash | `p` | backhand overhead |
| `u` | forehand drop shot | `y` | backhand drop shot |
| `l` | forehand lob | `m` | backhand lob |
| `h` | forehand half-volley | `i` | backhand half-volley |
| `j` | forehand swinging volley | `k` | backhand swinging volley |
| `t` | trick shot (e.g., tweener) | `q` | unknown shot |

**Direction digits:**

| Digit | Meaning |
|---|---|
| `1` | toward a right-hander's forehand side |
| `2` | middle |
| `3` | toward a right-hander's backhand side |
| `0` | unknown |

The definition is fixed relative to a right-handed receiver, regardless of the actual receiver's handedness. §4.3 explains what that means in court coordinates.

**Depth digits** `7`, `8`, and `9` run from shallow to deep. They are required on returns, but some charters add them to every shot, so expect them anywhere.

**Modifiers** seen in the data:

| Symbol | Meaning |
|---|---|
| `+` | approach shot |
| `-` | shot hit at the net |
| `=` | shot hit at the baseline, used for normally-net shots such as overheads hit from the back |
| `;` | net cord |
| `^` | seen mostly after volleys and occasionally drop shots; check the instructions for its exact meaning |

**Outcomes:**

| Symbol | Meaning |
|---|---|
| `*` | winner |
| `#` | forced error |
| `@` | unforced error |

The error-type letter (`n`, `w`, `d`, `x`, `!`, `e`) comes before the outcome symbol. For example, `f2d#` is a forehand to the middle that went deep, forced.

**Points without a rally** use single-character codes:

| Code | Meaning |
|---|---|
| `S` | point awarded to the server |
| `R` | point awarded to the returner |
| `P`, `Q` | penalty points |
| `V` | time violation |

`C` also appears at the end of some strings around successful challenges and replayed points. Verify all of these against the instructions.

**Two conventions that trip people up:**

1. If a player touches the ball but fails to return it, the shot is credited to that player with an error marker. If they never touch it, the previous shot is the winner. So the outcome symbol depends on two things together: who hit last, and whether the opponent made contact.
2. The string never names players. The hitter of each token is implied by alternation, starting from the server.

### 4.2 Scoring and rules (build and test a state machine)

Implement the scoring rules as a deterministic state machine with tests. Server identity, deuce/ad side, which player is at which end, and the legality of every score you read all come from it.

**Scoring.**

- Games use standard scoring with deuce and advantage.
- Sets go to six games with a two-game margin, with a tiebreak at 6–6.
- Tiebreaks are first to 7 points, won by two, in every set except the final set from 2022 onward.
- From 2022, the final set (fifth for men, third for women) uses a 10-point tiebreak at 6–6.
- Before 2022, the US Open's final set used the ordinary 7-point tiebreak.
- Men play best of five; women play best of three.

**Serving order.**

- Serve alternates every game.
- Within a game, the first point is served from the deuce court, and the side alternates each point.
- In a tiebreak, the player due to serve serves the first point from the deuce court. After that, players alternate every two points, each serving first from the ad court and then from the deuce court.
- The player who received first in a tiebreak serves the first game of the next set.

**Changing ends.**

- Players change ends after the first game of each set and after every two games thereafter, that is, whenever the number of games played in the set is odd.
- At the end of a set, they change ends only if the set's total number of games was odd. Otherwise they change after the first game of the next set.
- Within a tiebreak, they change ends after every six points.
- The tiebreak counts as one game for the set-level rule, so after a 7–6 set (13 games) they change.

Given which end each player started at, these rules fully determine who is near and who is far on every point. That is how you attribute detections to named players. Determine the starting ends from the first point you see, and cross-check with appearance.

### 4.3 Court geometry and coordinates

Use ITF dimensions, in meters:

| Measurement | Value |
|---|---|
| Court length | 23.77 (baselines 11.885 from the net) |
| Singles width | 8.23 (sidelines ±4.115 from the center line) |
| Doubles width | 10.97 (±5.485) |
| Service lines | 6.40 from the net |
| Net height | 0.914 at the center, 1.07 at the posts |

A convenient frame puts the origin on the ground at the center of the net, with x across the court and y along it. Pick sign conventions, document them, and use them everywhere.

**Landmarks for registration.** Use only points on the ground plane:

- the four doubles corners
- the four singles corners
- the four points where the service lines meet the singles sidelines
- the two "T" points where the center service line meets the service lines
- the two center marks on the baselines

Net posts and anything else above the ground are not on the plane. Don't use them for the homography.

**Directions depend on perspective.** MCP direction 1 means toward the receiver's right side as they face the net (a right-hander's forehand). Direction 3 means toward their left. In the image:

- For a receiver at the near end (back to the camera), their right is image right.
- For a receiver at the far end (facing the camera), their right is image left.

Serve geometry follows the same logic. The deuce court is the receiver's right-hand service box. So a wide serve to the deuce court goes to the receiver's right, and a T serve goes to the left of that box. In the ad court it is reversed.

Encode this once, test it, and derive every direction label through it.

### 4.4 Broadcast realities, 2001–2025

The footage will vary much more than a modern dataset would suggest, and the gap matches sit in the worst of it.

**Video and audio formats.**

- Early footage is standard definition, often 4:3 and interlaced: 480i at 29.97 frames/s, which is 59.94 fields/s.
- Later footage is 720p, or 1080i/p.
- Internet rips of any era may have variable frame rate, letterboxing or pillarboxing, non-square pixels, and re-encoding artifacts. Some run at 25 fps if they came from a European feed.
- Audio may be compressed, level-normalized, out of sync with the video by a constant offset, or replaced in places.

**The court and the people on it.**

- The courts went from green to blue inside the lines in the mid-2000s, and the surface supplier changed around 2020.
- Arthur Ashe Stadium has had a roof since 2016. Lighting differs between roof-open, roof-closed, and night sessions, and afternoon matches throw hard shadows across the lines.
- Line judges stand and sit around the court until the switch to electronic line calling around 2020–2021. That changes both what your person detector sees and which audio cues exist for out calls.
- Hawk-Eye challenges began at the 2006 US Open. From then on, challenge graphics in broadcasts show where specific balls landed.

**Broadcast grammar.**

- US network coverage moved from CBS to ESPN in the mid-2010s, and some footage may come from international feeds with their own graphics. Score-bug design and position vary by year and by feed, so detect the bug per video.
- Between points the director cuts to close-ups, crowd shots, graphics, and replays. Replays can be slow-motion versions of the main camera, and they must not be counted as new points.
- Live play is almost always shown from a single high camera behind one baseline. It may pan or zoom slightly during points.
- Commentators and crowds mostly go quiet during rallies. That is what makes audio hit detection viable.

### 4.5 Useful priors (re-estimate from aligned data)

| Quantity | Starting value |
|---|---|
| Shots per point | about 4.9 in the charted target matches |
| Points with a first-serve fault | roughly 37% in MCP overall |
| First-serve speeds | roughly 150–220 km/h (Speed_KMH from 2011 gives per-match distributions) |
| Serve contact to return contact, fast serves | well under a second |
| Time between contacts in baseline rallies | roughly one to one-and-a-half seconds |
| Time between contacts in net exchanges | shorter than in baseline rallies |

Use these to initialize your hit-sequence model. Then replace them with distributions estimated from aligned charted points.

## 5. Architecture

Organize the work as a directed graph of stages. The unit of work is one match, or one video asset within a match.

Each stage follows the same rules:

- It reads declared inputs.
- It writes versioned artifacts keyed by match, stage, and a hash of its configuration and code version.
- It is idempotent. Re-running with the same key does nothing. Changing a stage invalidates only its downstream artifacts.

Store compact intermediates rather than decoded frames: detections, tracks, keypoints in contact windows, audio features, registrations, events, and predictions. Keep a manifest that records, for every match and stage, the status, artifact paths, timings, and failure reasons. That makes preemptions and partial failures cheap.

The data model needs the entities below. Names and storage are your choice.

| Entity | Contents |
|---|---|
| Match | A row of the target list: players, handedness, best-of, final score, retirement/walkover flags, and links to MCP, feed, and tour records |
| VideoAsset | A video file, its technical metadata, and its offset within the match |
| Segment | A contiguous camera shot and its view class |
| Registration | The mapping from a segment's frames to court coordinates, with a quality score |
| Track | A person over time: per-frame court positions and an assigned role (near player, far player, other) |
| HitEvent | A contact: time, hitter, evidence, confidence |
| VideoPoint | A point detected in the video: its serve attempts and contacts |
| TruthPoint | A point from MCP or the feed |
| Alignment | The mapping from VideoPoints to TruthPoints |
| FieldPrediction | One value for one field of one shot or point: distribution, source, model version, timestamp |
| Final point and shot records | Truth and predictions merged by the provenance rules in §10 |

## 6. Stages

### 6.1 Ingest and normalize

**Probe every file** for:

- container and codec
- resolution, and sample and display aspect ratio
- nominal and average frame rate (a mismatch means variable frame rate)
- field order and interlacing (metadata is unreliable for rips, so also measure it, e.g., with an interlace detector)
- audio streams
- duration

**Normalize.**

- Decode by presentation timestamp, and carry real time in seconds everywhere. Never use frame indices as time.
- For interlaced sources, default to a field-rate ("bob") deinterlace. It turns 29.97 interlaced frames into 59.94 progressive ones, doubling the effective temporal resolution, which materially helps contact timing.
- Detect and crop black bars.
- Resample non-square pixels to square before running detectors.
- Extract audio as mono at a rate that preserves transients. 32 kHz or higher is safe.

**Handle incomplete or split footage.**

- When a match is split across several files, give each file an offset on a match timeline.
- Flag matches whose footage is incomplete.
- Flag highlight reels and condensed versions, which have far fewer points than the truth sequence. They are not usable as full-match sources.

**Output** the manifest and a per-match coverage report for the user.

### 6.2 Segmentation and view classification

**Split** each video into camera segments with a cut detector that handles three kinds of transition: hard cuts, dissolves, and the logo wipes broadcasters use around replays.

**Classify** each segment as one of these:

- main live view (the high, behind-the-baseline wide shot)
- other live angle
- close-up
- crowd
- graphics
- replay
- other (studio, commercial)

The main view is distinctive. It has a large fraction of court-colored pixels, long straight white lines consistent with a court model, and it registers successfully to the match's reference frame (§6.3). The default classifier is a small model on features like these, smoothed over time.

To get training labels:

1. Ask the Decisions API predicate questions on one frame per second for a few matches (§9).
2. Have the user confirm a sample in the point browser.
3. Train the cheap classifier, so you aren't paying the API for every frame forever.

**Replays** need special handling, because a slow-motion replay of the main camera looks like live play. Detect them using:

- slow-motion signatures: repeated or blended frames, and implausibly slow player motion
- broadcaster replay transitions
- content matching against the point that just ended: the same player trajectories at a different speed

Exclude replays from point assembly. Output a segment table with times, class, and confidence.

### 6.3 Court registration

**Reference frames.** By default, don't train a court detector. Use a human click per match instead:

1. Extract a clean main-view reference frame for the match.
2. Have a human click at least four, ideally eight or more, named landmarks from §4.3 in a small tool you build (§12).
3. Fit a homography robustly and overlay the projected court model.
4. Let the human accept or adjust.

This takes about a minute per match. It is more reliable across 25 years of court colors, shadows, and camera positions than a model would be.

Some matches will need a second reference when the main camera is repositioned: after a rain delay, between day and night, or when the roof closes. Detect this as a sustained registration failure and ask for another click.

**Per-segment registration.** For every main-view segment, register to the reference automatically. A good default:

1. Build a court-line mask: white, thin, high local contrast, excluding player detections and the score bug.
2. Estimate a homography to the reference with intensity- or edge-based alignment, initialized from the previous segment.
3. Fall back to feature matching restricted to the court region.

Re-estimate within a segment when global motion indicates a pan or zoom, and interpolate between estimates.

**Quality.** Score every registration by how well the projected model lines overlap the detected lines. Flag segments below threshold rather than silently using a bad mapping. Mild lens distortion causes small errors near the frame edges; only correct for it if the evaluation shows it matters.

### 6.4 Player detection, tracking and identity

**Detection.** Run an RF-DETR COCO-pretrained detector on main-view segments at a reduced rate. Around 10 frames per second is a reasonable default; raise it around serves and contacts. Keep the person and tennis racket classes.

**Selecting the two players.** Project each person's ground point into court coordinates. In each half, keep the person with sustained presence and movement in the extended playing area: well behind the baseline and somewhat outside the doubles sidelines. Exclude:

- the chair umpire: elevated, static, beside a net post
- ball kids: small, crouched near the net posts and back corners, moving at the wrong times
- line judges in older footage: static at the lines
- spectators

Make these decisions per track, not per frame.

**Tracking.** Use ByteTrack (available in Roboflow's supervision library, which RF-DETR's examples already use) or an equivalent. Repair fragmented tracks using position continuity first and appearance second.

**Identity.** Assign named identities from the scoring state machine (§4.2). Once you know which end each player started at, the rules tell you who is near on every point. Outfit color is a useful cross-check, but not the primary signal, because players change shirts.

**Ground point.** Prefer the ankles from the keypoint model (§6.5) when they are confident. Otherwise use the bottom center of the box. Handle boxes truncated by the frame edge. Feet leave the ground on serves and overheads, so use a robust estimate over a short window. Interpolate to the full frame rate and smooth with a motion model to get positions and velocities.

**When filtering struggles,** particularly with small far players in SD footage, fine-tune RF-DETR on a few hundred frames labeled "player" versus "other person," drawn across eras and resolutions. RF-DETR's README notes that fine-tuned models should read class names from the detections' data rather than from the COCO class list.

### 6.5 Pose and racket

**Keypoints.** Run RF-DETR's keypoint model densely on each player in windows around candidate contacts and serves. The model is `RFDETRKeypointPreview`: pretrained on the 17 COCO person keypoints, 576 px input, about 9.7 ms per image on a T4 with TensorRT. For far players, crop generously around the box and upscale before inference.

The model is labeled preview, so validate it on a small hand-checked set per era, covering near and far players in both SD and HD. If it is weak on small or blurry players, try another top-down pose estimator on the same crops and keep whichever wins.

**Rackets.** Also keep racket boxes in those windows:

- Racket position relative to the torso indicates stroke side.
- Its path over time, high-to-low versus low-to-high, carries slice information.

**Output** per-player keypoint and racket time series with confidences.

### 6.6 Audio features

**Candidate transients.** Compute an onset-strength signal that emphasizes the frequency band where racket impacts dominate speech and crowd noise. Tune the band; start with a high-pass or band-pass filter that suppresses most speech energy. Peak-pick candidates with a per-video adaptive threshold.

**Classify transients** as racket contact, bounce, shoe squeak, net contact, or other. You don't have to label these by hand:

1. Run a first alignment pass (§7).
2. Treat transients near aligned contact times as positives and all others as negatives.
3. Train a small classifier, then iterate.

**Sync.** Estimate each video's audio-to-video offset by cross-correlating confident audio contacts with visual swing peaks, and correct for it.

**Optional: call detection.** Keyword-spot calls of "out" and "fault": line judges in older footage, and the electronic line-calling voice later. These are evidence for errors and serve faults that no image model can provide.

### 6.7 Serve and hit detection

**Serves first.** A serve has a distinctive sequence:

1. the server near the center mark behind the baseline
2. a ball toss, with the tossing arm high
3. the racket above the head at contact
4. an audio impact
5. a returner reaction

The first serve attempt opens a point. After a fault, the second serve follows from the same side, seconds to tens of seconds later.

**Rally contacts as a sequence problem.** Don't treat contacts as independent detections. Within a point, contacts alternate strictly between server and returner, starting with the serve. So:

1. Combine per-time evidence for "the near player hit now" and "the far player hit now": audio contact probability, wrist and racket speed peaks, and pose at the peak.
2. Add priors on the intervals between contacts.
3. Decode the best alternating sequence with dynamic programming or an HMM.

When the number of shots is known, constrain the decoder to it during training-data construction. MCP gives the shot count for charted matches, and RallyCount gives it for 2016–2024; check RallyCount's counting convention against MCP on overlapping matches. When the count is not known, let the decoder choose.

Bounce sounds, where audible, add structure: a groundstroke has one bounce between contacts, and a volley has none.

**Output** for each contact: time (sub-frame if you can), hitter, serve flag and number, evidence breakdown, and confidence.

### 6.8 Point assembly and alignment to known sequences

**Assemble VideoPoints** from main-view segments: serve attempts, contacts, and an end time when the ball is dead. Signs of a dead ball are no contact after the expected interval, players decelerating, and the director cutting away.

**Align VideoPoints to TruthPoints** wherever a truth sequence exists: MCP for charted matches, and the feed for 2011–2024. A global sequence alignment works well (Needleman–Wunsch or DTW style). Build the match cost from:

- server identity under the rules
- number of serve attempts
- rally length
- ElapsedTime, for the feed

MCP has no timestamps, but the sequence of (server, serve attempts, rally length) is distinctive enough to align on by itself. For 2011–2024 feed matches, on-screen serve-speed graphics read by OCR and matched to Speed_KMH make an optional extra anchor.

**Mapping ElapsedTime to video time.** Use a robust piecewise-linear fit. Broadcast time is live, but rips cut commercials and recordings start late. Use set breaks and long changeover gaps as anchors.

**Allow gaps** on both sides: points the broadcast missed, and spurious VideoPoints from misclassified replays.

**Output** the alignment with per-point confidence, and flag low-confidence stretches for review. This alignment is the keystone of the project; §7 explains why.

### 6.9 Scoreboard reading (only where no truth sequence exists)

About 72 matches have neither MCP nor feed data. For these, the score bug is how you learn who won each point.

**Locate the bug** per video, as a stable overlay region whose content changes at point boundaries. Don't hardcode positions.

**Read it** with local OCR, or with a vision model through the Responses API using structured outputs. The Decisions API returns choices and probabilities, not text. Score states are a finite set, though, so you could try a choice formulation.

**Correct with the state machine.** Run every reading through it. It admits only legal successor scores, which corrects most OCR errors.

**Intermittent bugs.** Some older broadcasts show the bug only intermittently. Then infer per-point winners in a constrained decoder that combines:

- the states you do observe
- per-point outcome evidence: who hit last, whether the opponent made contact, out calls
- consistency with the final score in the tour file

**Validate** each reconstructed match against the tour file's official serve statistics where they exist (§2.3).

### 6.10 Per-shot features

For every contact, compute the features any later model might want:

- hitter and opponent court positions and velocities at contact
- the opponent's position at their next contact
- distance each player covered since the previous contact
- time since the previous contact, and time until the next
- serve side, serve number, and score context
- both players' handedness
- pose descriptors normalized to body scale: wrist positions relative to the torso and shoulders, arm extension, trunk rotation, contact height relative to the body
- racket box trajectory
- audio contact strength
- era, source resolution, and near/far

For serves, also record:

- the server's stance position
- the returner's position at serve contact and at return contact

These serve positions are also valuable outputs in their own right for serve-strategy work.

### 6.11 Field models

Field-by-field recipes are in §8. For every model:

- train on the training split of aligned charted matches (§7)
- select on the validation split
- report on the test split (§11)

### 6.12 Export

See §10.

## 7. Turning the charted matches into training data

Wherever the user has footage for the 179 charted matches, those matches are labeled data from exactly this domain. The alignment in §6.8 is what unlocks them.

**Pairing tokens with contacts.** Once each VideoPoint is matched to an MCP point, match the MCP shot tokens to detected contacts in order. Use a small alignment with insertion and deletion costs, driven by hitter alternation and the known shot count. Each accepted pair gives a contact time and position with MCP's labels attached: stroke letter, direction, depth, modifiers, and outcome. That supervises nearly every Tier 2 and Tier 3 field.

**Accept conservatively at first.** Start with high-confidence pairs only: points whose detected contact count equals MCP's and whose decoder confidence is high. Report acceptance rates by era and near/far, so you can see where detection is failing.

**Iterate.**

1. Train the audio and visual contact detectors on the accepted labels.
2. Re-run detection and alignment.
3. Accept more pairs.
4. Repeat until acceptance stops improving.

Keep the iteration honest: points from test matches must never feed back into training (§11).

**Feed-only matches.** Matches with feed data but no MCP charting make up most of the 2011–2024 gap. They provide weaker labels for the fields the feed carries, and checks on the rest:

- server and point winner
- serve number
- rally count (2016+)
- serve width, serve depth, and return depth (2016+)
- winner and error flags
- distance run (2015+), which validates tracking (§11)

## 8. Field-by-field recipes

### Server, returner, ends, and side

These come from the state machine once alignment or scoreboard reading has placed each point in the match. Check them against pose: the server is the player tossing at the baseline center. Treat any disagreement as an alignment bug, not a modeling problem.

### Serve number and faults

Count serve attempts in each point. The fault type (net, wide, deep, wide-and-deep, foot fault) is Tier 4. Evidence for it:

- out and fault calls in the audio
- visible net disturbance
- the returner's reaction
- the Decisions API on frames after the serve

Otherwise leave it as unknown (`e`).

### Serve direction (4, 5, 6)

This is Tier 1's hardest field and its most valuable one.

**Evidence without the ball,** strongest first:

- where the returner makes contact relative to the target service box: distance from the center service line versus the sideline
- the returner's displacement from their ready position
- the returner's stroke side, combined with their handedness
- the server's stance

For aces and unreturned serves, use the direction and extent of the returner's lunge, and expect lower confidence.

**Labels:**

| Source | Coverage | Notes |
|---|---|---|
| MCP | all eras | three bins |
| ServeWidth | 2016–2024 | five bins; train a five-bin head, or map as in §2.4 |
| 2011 Serve_Direction | 2011 | once decoded (§2.4) |

**Target.** Approach the roughly 85% human-versus-feed ceiling against MCP. Separately, report wide-versus-T accuracy on unambiguous W and C labels, where much higher accuracy is achievable.

### Rally length

Rally length is the number of contacts in the point. MCP counts the serve. Check the feed's RallyCount convention against MCP on overlapping matches before using it as a label.

### Stroke side (forehand or backhand)

This is mostly a pose problem. Interpret all of the following with the hitter's handedness:

- the racket-hand wrist's position relative to the body's midline and the shoulders at contact
- which side the racket box is on
- the two-wrists-together signature of two-handed backhands

MCP letters map to side directly:

| Side | Letters |
|---|---|
| Forehand | `f`, `r`, `v`, `o`, `u`, `l`, `h`, `j` |
| Backhand | `b`, `s`, `z`, `p`, `y`, `m`, `i`, `k` |
| Unknown | `t`, `q` |

### Stroke family

The families are groundstroke, slice, volley, half-volley, swinging volley, overhead, drop shot, lob, and trick. They need temporal context. The default is a sequence model over keypoints, racket trajectory, and position features in the contact window.

Context features help a lot:

| Family | Context signature |
|---|---|
| Volley | near the net, compact swing, no bounce before contact |
| Overhead | racket high, arm extended above the head |
| Lob | the opponent retreats afterward |
| Drop shot | the opponent sprints forward afterward |

Slice versus topspin from racket path is the hardest distinction at low resolution.

The classes are heavily imbalanced: `f` and `b` dominate. Report macro-F1 and per-class recall, not just accuracy.

### Position modifiers

These come from positions:

| Modifier | Definition |
|---|---|
| Approach shot (`+`) | the hitter moves forward into the court toward the net after the shot |
| At the net (`-`) | contact within some distance of the net |
| At the baseline (`=`) | contact near or behind the baseline, for shot types normally hit at the net |

Learn the thresholds from MCP labels, and expect charters to be inconsistent.

### Shot direction (1, 2, 3)

This comes from where the receiver next makes contact. Map that position through the receiver's perspective (§4.3), then threshold it into thirds, with thresholds learned from MCP labels.

A point's final shot has no next contact. For it, use the receiver's movement or the Decisions API on frames after contact. Otherwise leave it as unknown (`0`).

### Return depth (7, 8, 9)

Proxy it with how deep the server is standing at the third shot and how they moved after the return, learned from MCP labels. ReturnDepth (D or ND, 2016–2024) is a second, coarser label.

### Outcome type

This follows from the point's structure:

| Situation | Outcome |
|---|---|
| The last player to make contact won the point, and the opponent never touched the ball | winner |
| The opponent touched the ball but didn't return it | the opponent's shot, with an error marker |

Whether the opponent touched the ball uses the same contact evidence as §6.7.

Forced versus unforced is a judgment call, even for humans. Use features of the situation:

- the time the player had
- the distance they covered
- stretch in the pose
- contact depth
- incoming shot type

Add the Decisions API, and calibrate the result against MCP.

### Error type

Net, wide, deep, wide-and-deep, and shank are all Tier 4. Evidence:

- the net's movement
- out calls in the audio
- line judges' signals in older footage
- players' reactions
- the Decisions API on frames after contact

Ship unknown (`e`) when confidence is low.

### Positions at contact

These are a direct output of §6.4 and §6.7.

## 9. Using OpenAI's Decisions API

### 9.1 What it is

As of 2026-10-08. Verify everything in this subsection against the current docs before building.

**Status.**

- Public beta since 2026-10-06, at `POST /v1/decisions`.
- `gpt-6-luna` is the only model.
- OpenAI expects general availability within weeks.

**Inputs and question types.** It takes text, images, or both, plus a list of named questions. It returns typed answers rather than generated text. There are three question types:

| Type | Returns |
|---|---|
| predicate | the probability that a statement is true |
| choice | one option from a fixed list, with probabilities over the options |
| score | a probability-weighted rating over ordered levels |

**Rules.**

- Each question has a unique name, which is echoed in the response.
- Questions in one request share the same input.
- A decision that depends on another decision's answer needs its own request.
- Images must be sent inline as base64 data URLs. Hosted image URLs and file IDs are not supported.

**Billing.** This endpoint charges $0.10 per million input tokens. Output, cache reads, and cache writes are free. Long-context and regional-processing multipliers still apply.

**Thresholds and limits.** The docs tell developers to set routing and review thresholds from their own labeled examples. No rate limits or input-size limits were documented when this was written, so measure them yourself.

**SDKs and docs.** The docs list minimum SDK versions (for example, Python SDK 3.26.0). The guide is at developers.openai.com/api/docs/guides/decisions.

**Request shape.** The sketch below is illustrative only. Field names are unverified; follow the API reference.

```json
{
  "model": "gpt-6-luna",
  "input": [{ "content": [
    { "type": "input_text", "text": "Contact sheet for the highlighted FAR player (right-handed): 9 frames from -300 ms to +300 ms around racket contact, plus a full-frame context image." },
    { "type": "input_image", "image_url": "data:image/jpeg;base64,..." }
  ]}],
  "questions": [
    { "type": "choice", "name": "stroke_side",
      "instructions": "On which side of the body did the highlighted player hit this shot?",
      "choices": [
        { "value": "forehand", "description": "Racket-hand side." },
        { "value": "backhand", "description": "Non-racket-hand side, one or two hands." },
        { "value": "unclear", "description": "Cannot be determined from these frames." }
      ] },
    { "type": "predicate", "name": "at_net",
      "instructions": "The highlighted player is inside the service line at contact." }
  ]
}
```

### 9.2 Where it fits

The API is a classifier over images. Use it for bounded judgments that are hard to engineer:

- view classification ("main live court view?", "replay?") to bootstrap §6.2
- stroke side and stroke family on contact sheets
- forced versus unforced on the final contact of error-ended points
- error type, on frames after the final contact
- possibly serve direction on unreturned serves

Don't use it for:

- geometry (it returns no coordinates)
- reading text (use structured outputs in the Responses API)
- anything you can compute deterministically

### 9.3 Building inputs

**Contact sheets.** There is no video input, so each question gets a contact sheet:

- a grid of frames sampled around the event; for a contact, something like nine frames spanning a few hundred milliseconds either side
- cropped consistently around the relevant player, with margin
- plus a small full-frame context image with that player boxed

In the text, say which player is highlighted, which end they are at, and their handedness.

**Your own features in the text?** Whether to include features such as position is an empirical question. They can help, but they can also make the API echo your other models. Test both ways.

**Cost control.** Input tokens are the whole cost, so keep images as small as accuracy allows. Measure token usage per request from the responses rather than guessing.

**Question design.** Because dependent questions need separate requests, phrase each question so it stands alone given the image. For example, ask for stroke side and stroke family as two independent choices, not as a hierarchy. Put all questions about the same contact sheet in one request.

### 9.4 Operations

**Caching.** Cache every response, keyed by a hash of the exact input bytes, the question specification, and the model. Store the raw responses.

**Rate handling.** Run with bounded concurrency and exponential backoff, and adapt to whatever rate limits you observe.

**Budget.** Before any bulk run:

1. Estimate the cost: measured tokens per request times the number of requests.
2. Check that estimate against the user's budget.
3. If the budget would be exceeded, abort cleanly.

At the published rate, labeling every contact in the uncharted matches should cost on the order of tens of dollars, even at several thousand tokens per request. Evaluating on the charted matches costs about the same. Measure anyway.

**Abstraction.** Keep the integration behind a provider-agnostic interface: "ask these questions about these images." The endpoint is a beta and may change. If it breaks or underperforms, the same interface can be backed by the Responses API with structured outputs, or by a local model.

### 9.5 Trust

Treat every API answer as an uncalibrated feature until proven otherwise. For each question:

1. Measure accuracy, confusion, and calibration on held-out charted matches, broken down by era, gender, and near/far.
2. Calibrate the probabilities with temperature scaling or isotonic regression, per era if needed.
3. Keep the API in the final system only where it adds measurable lift over the local models in the fusion of §10.

A reasonable expectation is that local pose-based models win on stroke side, and that the API earns its place on forced versus unforced and on error type. Your evaluation decides.

## 10. Fusion, calibration, abstention and export

### 10.1 Fusion and abstention

**Fusion.** For each field, combine the available predictions: local models, API answers, and rule-based proxies. Use a simple stacked model fit on the validation split, then calibrate it so its probabilities mean what they say.

**Abstention.** Choose per-field thresholds from coverage-versus-accuracy curves on validation, targeting a precision the user agrees to. The default is to ship a value only when its calibrated probability is high enough that expected accuracy meets the field's target in §11.

Map abstentions to MCP's own unknown codes so downstream analysis can treat them as missing:

| Field | Unknown code |
|---|---|
| Shot | `q` |
| Direction or serve | `0` |
| Error type | `e` |

### 10.2 Provenance rules

In the merged dataset, sources rank in this order:

1. human MCP charting
2. the official feed
3. models

Never overwrite a human or feed value with a model value. Store the model's value alongside it for evaluation. Every value records its source, model version, and confidence.

### 10.3 Exports

Export at least three things.

**1. Canonical tables.** Point and shot tables with all fields, distributions, timestamps, positions, and provenance.

**2. MCP-format point files.** Use the same columns as the MCP point files: match_id, Pt, Set1, Set2, Gm1, Gm2, Pts, Gm#, TbSet, Svr, 1st, 2nd, Notes, PtWinner.

- Generate them from the canonical tables with your serializer.
- Use match IDs that make machine origin obvious, e.g., a suffix.
- Include a sidecar confidence table.
- Round-trip test the serializer against the MCP parser.

**3. A per-serve table for serve-strategy analysis,** with these columns:

- match, set, game, point, and score state
- server, returner, and both players' handedness
- side and serve number
- serve direction, with its distribution and source
- server and returner positions at serve contact
- returner contact position
- point outcome

## 11. Evaluation

### 11.1 Splits

Split by match, never by point. Stratify by gender and by era: 2001–05, 2006–10, 2011–15, 2016–20, 2021–25.

- **Test set:** fixed, around 20% of the charted matches with footage. Never use it for training, threshold selection, or the iteration in §7.
- **Validation set:** separate, used for tuning.
- **Era-transfer experiment:** train on 2011–2025 charted matches and test on the 2001–2010 charted matches. This mirrors where most of the uncharted footage lives.

Report everything broken down by era, gender, near/far, and source resolution.

### 11.2 Gold set

Some things have no existing labels: exact contact times, exact player positions, and point boundaries in the video. Build a small hand-annotated gold set with the tools in §12:

- a few hundred contacts
- a few hundred position annotations
- spread across eras and near/far

Ask the user to do or check the annotation.

### 11.3 Metrics and checks

| Area | Metric |
|---|---|
| Alignment | share of VideoPoints matched to the correct truth point |
| Contact detection | rally length exact-match and within-one rates; contact time error on the gold set |
| Positions | error on the gold set; correlation between your per-point distance covered and the feed's DistanceRun (2015–2024) |
| Categorical fields | accuracy, macro-F1, confusion matrices, calibration error, coverage-accuracy curves |
| Reconstructed matches without point truth | agreement with the tour files' aces, double faults, serve points, first-serve percentage, and first- and second-serve points won |
| Serve direction | comparison against the human-versus-feed ceiling in §2.4 |

### 11.4 Provisional targets

These are starting guesses. Revise them after measuring baselines and ceilings. When a field can't reach a useful level, tell the user rather than shipping it quietly.

| Field | Metric | Provisional target |
|---|---|---|
| Point alignment | correct-match rate (charted matches, complete footage) | ≥ 98% |
| Server, side, ends | accuracy | ≥ 99.5% |
| Serve number | accuracy | ≥ 98% |
| Serve direction (3-bin) | agreement with MCP | ≥ 80% (ceiling about 85%) |
| Serve direction, W vs T | accuracy on unambiguous W/C labels | ≥ 95% |
| Rally length | exact / within one | ≥ 85% / ≥ 97% |
| Contact time | error vs gold | ≤ 2 frames at 30 fps for ≥ 90% of contacts |
| Player position | median error vs gold, near / far | ≤ 0.5 m / ≤ 0.75 m |
| Stroke side | accuracy, near / far | ≥ 97% / ≥ 93% |
| Stroke family | macro-F1 | ≥ 0.75 |
| Shot direction | accuracy vs MCP | ≥ 75% |
| Return depth | accuracy vs MCP | ≥ 70% |
| Winner vs error | accuracy | ≥ 95% |
| Forced vs unforced | accuracy | ≥ 75% |
| Calibration | expected calibration error, every shipped field | ≤ 0.05 |

## 12. Tools to build for the human in the loop

Build these early. They pay for themselves within the first milestone.

**Court clicker.** Shows a reference frame, asks for named landmarks, fits and overlays the court model, and saves the result.

**Point browser.** Plays any point with overlays:

- registered court lines
- tracks and keypoints
- contact markers
- predicted codes next to truth codes

It should jump straight to disagreements and low-confidence predictions. Most debugging will happen here.

**Gold-set annotator.** For contact times, point boundaries, and player positions.

**Review queue.** Ranks low-confidence predictions in uncharted matches by how much a human answer would help. It records corrections and feeds them back as labels.

**Metrics report generator.** Regenerates every table in §11 from the current artifacts.

## 13. Milestones

| Milestone | Deliverable | Exit criteria |
|---|---|---|
| M0 Data foundation (no video) | Target list with flags; MCP parser and serializer; feed and tour loaders; joins; coverage and gap lists; 2011 field decoding; ServeWidth mapping | Reproduces the §2 numbers or explains the differences; parser handles ≥ 99.9% of MCP strings; round-trip tests pass; state machine tests pass on MCP and feed sequences |
| M1 Footage manifest | Manifest, technical metadata, per-match coverage report | Every user-supplied file mapped to a target match or flagged; the user has seen the coverage report |
| M2 Vertical slice | One charted 2016–2024 match through every stage, crude but complete | Overlay video looks right; alignment and rally-length metrics computed; throughput measured and full-run compute extrapolated |
| M3 Tier 1 at scale | Point skeleton plus serve direction for every match with footage | Tier 1 targets met on test, or the gap explained; serve table exported |
| M4 Tier 2 | Contacts, rally length, positions, stroke side and family; Decisions API comparison | Targets met or explained; API lift measured per question |
| M5 Tier 3 and fusion | Directions, depth, modifiers, outcomes; calibrated fusion and abstention | Targets met or explained; calibration verified |
| M6 Production run | All playable matches processed; QA sample reviewed by the user | Error rates on the QA sample consistent with test-set estimates |
| M7 Report | Final datasets, evaluation report, known limitations, re-run instructions | User sign-off |

## 14. Decision rights

**Decide on your own:**

- language and libraries
- storage
- model sizes and architectures
- sampling rates and features
- thresholds, justified by validation data
- code structure
- anything else not listed below

When two reasonable options exist, pick one, record why in a decisions log, and move on.

**Stop and ask the user before:**

- acquiring any footage yourself (don't)
- any manual task you estimate at more than about an hour of their time (per-match court clicking is expected and fine)
- spending more than the API budget they've set, or starting the first bulk run if no budget is set
- using PML-licensed models, or any data source other than those in §2
- expanding scope (round of 16, other slams)
- shipping a field that misses its target by a wide margin
- acting on data that contradicts this guide in a way that changes the plan

## 15. Known traps

| Trap | Symptom | Prevention |
|---|---|---|
| Replays counted as points | More VideoPoints than truth points; duplicated rallies | Replay detection in §6.2; content matching against the previous point; gaps allowed in alignment |
| Wrong ends after changeovers | Near/far identity flips mid-set; server mismatches | State machine with tiebreak end changes every six points and the set-boundary parity rule; tests on real sequences |
| 2022+ final-set tiebreak | State machine rejects legal scores in final sets | 10-point final-set tiebreak from 2022 |
| Frame index used as time | Drift between audio, video, and truth timestamps | Presentation timestamps in seconds everywhere |
| Interlaced footage treated as progressive | Combing artifacts, poor detections, coarse timing | Measure interlacing; field-rate deinterlace |
| Constant audio offset | Audio contacts consistently early or late versus swings | Per-video offset estimated by cross-correlation |
| Officials or ball kids in player tracks | Position jumps; impossible distance run | Track-level role filtering by zone and behavior |
| Homography drift during pans | Players slowly "slide" on the court model | Re-registration on global motion; registration quality scores |
| MCP direction semantics | Directions systematically mirrored for far-end receivers | Single tested perspective mapping (§4.3) |
| Left-handers | Forehand/backhand swapped for lefties | Handedness from tour and MCP data in every stroke-side feature |
| Index alignment across sources | Silent off-by-one label noise | Proper sequence alignment; check point counts and server sequences |
| Sentinel zeros | Fake slow serves, fake zero movement | Speed_KMH 0 and DistanceRun 0 treated as missing |
| Hardcoded score-bug location | OCR garbage on some years or feeds | Per-video bug localization |
| Encoding and whitespace | Missing matches in joins | latin-1 for MCP; strip all text fields |
| Serve-and-volley vs approach | `+` after a serve digit misparsed | Grammar distinguishes serve `+` from rally `+` |
| Leakage | Test metrics far above production QA | Split by match; test matches excluded from §7 iterations |

## 16. Optional extensions (only after M5)

- **Ball tracking.** A temporal tracker such as WASB-SBDT (reported to outperform TrackNet-style baselines) could fill Tier 4 bounce-dependent fields for the shots where proxies are weak.
- **Hawk-Eye challenge graphics** (2006 onward), parsed as sparse ground truth for bounce locations.
- **Serve-speed graphic OCR,** as an alignment anchor and an extra field for pre-2011 matches.
- **Underarm-serve flag** from pose: no high toss, contact below the shoulder. These are rare; check how MCP records underarm serves before building it.
- **Round of 16 or other slams.** The user must approve this expansion.

## 17. References

- MCP repository: github.com/JeffSackmann/tennis_MatchChartingProject
- MCP homepage, with links to the charting instructions: tennisabstract.com/charting/meta.html
- Archival mirror of the slam point-by-point and tour data: github.com/Aneeshers/tennis-sackmann-archive (also huggingface.co/datasets/Aneeshers/tennis-sackmann-archive)
- RF-DETR: github.com/roboflow/rf-detr; docs at rfdetr.roboflow.com; paper arXiv:2511.09554
- supervision (tracking utilities): github.com/roboflow/supervision
- OpenAI Decisions API guide: developers.openai.com/api/docs/guides/decisions
- WASB-SBDT: github.com/nttcom/WASB-SBDT
- ITF Rules of Tennis: court dimensions, scoring, order of service, change of ends
