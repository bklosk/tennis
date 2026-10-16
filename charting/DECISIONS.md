# Decisions log

Choices made while building the pipeline (2026-10-08/09), with the evidence behind each. Where
this departs from `claude_implementation.md`, the departure is noted.

## Scope and data

* **Goal narrowed to the user's three asks**: player coordinates at every contact, rally length,
  stroke type, tested against MCP. Tiers 3–4 of the guide (shot direction, depth, outcome type,
  error type) and the 349-match production run were not attempted.
* **Footage**: none was on the machine (the earlier `downloads/` folder was gone). The guide says
  not to acquire footage; I asked, and the user chose to re-download the official US Open
  full-match uploads with yt-dlp. 14 MCP-charted matches: 4 dev, 10 held-out test, spread over
  2001–2025 and both genders.
* **Downloads are never resumed** (`--no-continue`) and every file is decode-verified. Resuming
  after a restart spliced two different encodes of the same format into corrupt files; orphaned
  bash retry loops also wrote the same file concurrently. The downloader is now one Python process.
* **Pre-2011 videos at 720p**: those uploads are SD sources; 1080p adds bytes, not detail.

## Vision

* **Automatic court registration instead of a human click per match** (guide default). The user
  is not available during a background run. The pretrained 14-keypoint court network registers
  main views to ~1 px; keypoints must be blob centroids (the network's peaks are saturated
  plateaus, so argmax sits on the top row and shifts the court ~20 px). fp16 breaks this network
  on real frames, so it runs in fp32.
* **"Live view" is geometric, not "the dominant camera"**. The 2024 broadcast shows live points
  from a high centred camera, a low centred one and low cameras offset to one side; requiring the
  dominant layout dropped 10 consecutive points. Any registered view looking down the court from
  behind the near baseline is accepted.
* **RF-DETR Small for people, two views per frame**: the full frame for the near player, an
  upscaled crop of the far half (from the homography) for the far player. YOLO11 missed the far
  player in full frames; RF-DETR (fp16 + TorchScript, identical detections, ~30% faster) did not.
  RF-DETR uses COCO category ids (person = 1) although `class_names` is 0-based.
* **Pose from YOLO11m-pose on upscaled player crops**, not the RF-DETR keypoint preview (an
  XLarge model, too slow on the Mac GPU for dense use).
* **Software decoding**: ffmpeg's software H.264 decoder is ~10x faster than VideoToolbox here.

## Contacts and rally length

* **Audio onsets are candidates, not contacts.** Pre-serve ball bounces are as loud as hits
  (z up to ~28), and one hit brings several onsets (grunt, squeak, bounce).
* **Decisions API labels every candidate**: both players' crops at −0.1/0/+0.1 s; per player
  serve / hit / bounce / none. ~670 tokens per request, ~$0.25 per match including a second pass
  for soft onsets (volleys, touch shots) inside rally context.
* **Decoder**: serves need an API serve label, the server behind the baseline and no likely
  contact in the previous 2.5 s (rules out mid-rally overheads); rallies are the best alternating
  sequence under an interval prior. Parameters were grid-searched on dev only.
* **Double-fault rule**: an unreturned second serve is scored as a double fault (0 contacts, MCP
  convention), because in the charted targets double faults are 10.2% of second-serve points and
  unreturned second serves 0.9%.
* **Tried and dropped** (all measured leave-one-match-out on dev): a learned rally-end model
  (no gain), a second API question design with one player per request and explicit
  "missed swing / ball handling" classes (gains on error points were a selection artifact; worse
  overall), bridging single missed contacts (no gain), racket-box features for slice (no gain).
* **Kept**: a learned contact classifier (API answers + audio + pose + positions + timing, trained
  on MCP-count-constrained labels) blended at weight 0.3 with the API scores: +1.6 points exact,
  +1.5 within-one on dev. Dev matches are scored with the model trained without them.

## Strokes

* **The API is not used for forehand/backhand**: asked directly it scored 56% (it reads the far
  player, who faces the camera, as mirrored); asked for the racket's image side, 54%. Its answers
  stay as features but pose carries the signal.
* **Pose features are mirrored by court end and handedness**, plus an orientation-free projection
  of the racket wrist onto the shoulder axis. Gradient-boosted trees, calibrated, 92.6% on dev
  leave-one-match-out.
* **Families**: half-volleys and swinging volleys are scored as volleys; MCP unknown (`q`) and
  trick (`t`) shots are excluded.

## Evaluation

* **Positions are checked through charted facts** (MCP has no coordinates): serve side, serve
  direction vs returner contact position, shot direction vs receiver position, net play vs
  distance from the net. A hand-labelled position gold set (guide §11.2) was not built.
* **The official feed's RallyCount is not a human ceiling for MCP**: aligned point by point it is
  always 0 or 1 below MCP's count (one less on ~65–71% of points), i.e. a different convention.
* **Handedness at test time comes through the MCP alignment.** Handedness is public (tour files);
  an uncharted match would take it from there, with near/far identity from the scoring state
  machine.

## Video archive (2026-10-09)

* **Scope**: the guide's target set, US Open men's and women's QF/SF/F 2001–2025. The video
  manifest has a full-match upload for 164 of the 350 targets (161 from official channels, 3
  from other uploaders); the rest have highlights only or nothing. The Australian Open rows in
  `data/matches.csv` are not included.
* **One upload per match**: official channel first, then the longest upload.
* **Streamed, not staged**: ~550 GB does not fit on the 209 GB of free disk, so each video is
  downloaded, verified, uploaded, size-checked and deleted before the next one needs the space.
* **Private objects**: uploads set `ACL=private` explicitly (checked: an unauthenticated GET
  returns 403). The Space holds other projects' data; everything here lives under
  `tennis/usopen-video/`.

## Analytics (2026-10-13)

* **Images, not a web page**: the user asked for the analytics as images. `scripts/analytics.py`
  writes PNGs and CSV tables to `reports/analytics/` with a README that embeds them; colors are the
  validated light palette from the dataviz method (blue/orange categorical, one-hue blue density,
  blue-red around gray for win rates).
* **Rally time for matches that are not decoded** (`uso/rallytime.py`): a gradient-boosted classifier
  on player speed, position and audio-onset density, trained on decoded points. Leave one match out
  on the 14 decoded matches: 89% of samples right, precision 0.83, recall 0.82, predicted rally share
  0.33 vs 0.33 true; median depth from predicted rally time is within 0.06 m of the true value per
  match on average (all tracked time: 0.15 m). Every match uses the classifier, decoded ones
  out-of-fold, so position charts compare like with like.
* **Returns are split from rally shots** in the forehand-share chart: a returner's stance is set by
  the serve box (a deuce-court returner 2 m to the forehand side mostly hits a backhand off a T
  serve), which MCP's own labels confirm.
* **Known measurement issue, not fixed here**: depth reads ~0.8 m deeper at the near end than the far
  end behind the baseline (servers agree at the baseline), and the near player leaves the frame past
  ~4.5 m. Players change ends, so group comparisons are fair; absolute depths are ±0.4 m.

## Decoding the batch matches (2026-10-13, pending a budget decision)

* **Match table**: `data/batch_matches.csv` lists the 44 tracked-only matches; 32 have an MCP record
  (decoded and aligned like the evaluation set through `truth.match_row`), 12 do not (decoded with
  `uso.uncharted`, hitters by court end; two have a left-hander, so their forehand/backhand is not used).
  Handedness comes from MCP's match files. MCP's "Best of" column has slips (a 2023 men's QF listed as
  best of 3), so best-of follows the draw.
* **Cost**: measured on the 14 evaluation matches (1.505 onset requests per z >= 8 candidate at
  $0.000067, 0.099 rally contacts per candidate at $0.000177 for the two stroke questions), the 44
  matches come to ~$35.8. The run needs the cap in `uso/decisions.py` raised above the user's $15; it
  stays at $14 until the user confirms the new cap.
* **Ledger tally**: `DecisionClient.spent()` used to re-read the whole ledger on every request under a
  lock (0.13 s at 112k lines, growing linearly), which would have throttled a 500k-request run to a
  few requests per second. It now keeps running totals and reads only appended lines (safe across
  processes; a half-written line waits).
* **Runner**: `scripts/decode_batch.py` fetches each video from the Space (sha256 against the archive
  manifest), decodes it in its own process (two at a time), uploads the new outputs next to the vision
  outputs in the Space and deletes the video. A match that hits the cap stops the batch.

## Jev onset labels (2026-10-15)

* **Benchmark first** (`reports/jev/README.md`): TypeSafe Jev 1.14 got the same contact sheets and
  questions as gpt-6-luna on the held-out match 8gP591_wx_k. Its onset answers separate contacts
  from other sounds better (AUC 0.889 vs 0.840 on ~5,000 MCP-derived labels). Its stroke answers are
  better than luna's but far below pose (forehand/backhand 57–71% vs 97%). Box for players lost to
  RF-DETR, and Box for the ball was unusable. Only the onset labels moved to Jev.
* **Serve over-firing**: with luna's decoder settings, Jev's answers opened 122 points for 93 charted
  ones. Jev labels the near player's pre-serve routine (bounce, toss, between points) "serve" far
  more often: 498 onsets at p ≥ 0.4 against luna's 200. Fixed with a serve threshold per court end
  (`rally.pick_serves_api`: near 0.8, far 0.6), tuned on dev. A serve-specific question was not
  needed.
* **Objective now penalises invented points**: `tune_decoder.py` scores exact × coverage ×
  precision (aligned / detected points). Alignment hides extra points on a charted match, but an
  uncharted match has no alignment. luna's parameters predate this term and were not retuned.
* **Each labeler keeps its own artifacts** (`uso/labeler.py`): label file (`onset_api.parquet` /
  `onset_jev.parquet`), decoder parameters (`decoder_params[_jev].json`) and contact models
  (`hitmodel[_jev]*.pkl`). Every function defaults to luna, so the published evaluation reproduces
  unchanged. `decode_batch.py` and `uso.uncharted` default to Jev. Exported rows say
  `source=model:uso-0.2-jev`.
* **Jev's contact model**: retrained on Jev's answers (`fit_hitmodel.py --labeler jev`). It blends at
  weight 0.7, bias −1.0 (dev leave-one-out; luna uses 0.3 / −0.5).
* **Result**:
  * Dev (decoder settings in-sample, contact models leave-one-out): Jev 65.0% exact / 88.0% within
    one / 95.4% point precision, against luna's 66.3% / 86.7% / 89.4%.
  * Held-out 8gP591_wx_k: Jev 60.7% / 84.5%, 89 points found for 93, against luna's 55.2% / 87.4%,
    95 points.
  * Rally length is a wash; Jev invents fewer points and costs less.
* **No stroke questions with Jev**: the fused stroke model is no better than pose alone on the 10
  test matches (side 91.3% vs 91.6%, family macro-F1 0.372 vs 0.361), so Jev-charted matches use
  `stroke_model_pose.pkl` (`fit_stroke_pose.py`).
* **Cost**: Jev reads ~1,163 tokens per onset sheet at $0.042/M, about $0.24 per video-hour. luna
  was ~$0.37 per video-hour with the stroke questions.
* **Jev spend so far**: $1.21 of the user's $1.50. The cap is in `uso/jev.py`.
