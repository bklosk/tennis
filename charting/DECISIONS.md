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
