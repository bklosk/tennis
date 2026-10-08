Mostly yes, but the two tools solve different halves of the problem, and neither one helps with the ball, which is what TrackNet was supposed to handle.

**You may not need to chart half of these matches.** I checked the Match Charting Project match lists. 179 of the 350 US Open QF/SF/F matches from 2001–2025 are already charted: 120 men's and 59 women's, roughly 36,000 points and 178,000 shots. The slam point-by-point files cover all but one late-round match from 2011–2024. From 2016 on, they also include serve width (five bins from wide to T), serve depth, and return depth. Combining the two sources, 234 of the 350 matches already have serve direction on nearly every point.

One caveat about that data. Sackmann took his ATP, WTA, and point-by-point repos down around June, and only the MCP repo is still up. My numbers come from a third-party archival mirror of the 2011–2024 slam files that keeps his CC BY-NC-SA license, so check its provenance before relying on it.

That leaves a real gap of 171 matches for full shot-by-shot data, about two-thirds of them women's. For serve direction the gap is 116 matches, nearly all from 2001–2015, which is also your oldest and worst footage. If this is for the serve-direction project, those 116 matches are the entire job.

The 179 charted matches are also free labels from exactly your setting. If you can find full broadcasts of them (worth checking before anything else), align each MCP point to the video and each shot to a detected hit. That gives you training data and a test set for every field you want to automate.

**RF-DETR is the right tool for the players.** The COCO-pretrained models already detect person and tennis racket. The repo now also includes a keypoint model (preview) pretrained on COCO person keypoints, so you get wrists and ankles without any training.

- Track the two players with ByteTrack from Roboflow's supervision library.
- Project the ankle points to court coordinates with a homography.
- Filter out ball kids and the umpire by where they stand.
- For the homography, skip the court-detection model. Click the court corners once per match, then register each point's main-camera frame to that reference. It takes about a minute per match and will hold up better across 25 years of broadcast styles than a trained model.

RF-DETR-S runs at about 3.5 ms per frame on a T4 with TensorRT. If you process only live-ball segments at around 10 fps, the whole set is on the order of tens of GPU-hours.

**What RF-DETR can't do is the ball.** It looks at one frame at a time, resized to a few hundred pixels, where a broadcast tennis ball is a couple of blurry pixels. That's exactly why TrackNet stacks consecutive frames. So build the pipeline so it doesn't need the ball:

- **Hits:** detect them from audio, since racket impacts are sharp and crowds mostly go quiet during rallies. Combine that with wrist-speed spikes from the keypoints.
- **Who hit:** shots alternate between players, so the sequence tells you.
- **Shot direction:** use where the opponent is when they next make contact.
- **Serve direction:** use where the returner makes contact.
- **Return depth:** use how deep the server is standing for the third shot.
- **Winner vs. error:** use who hit last and who won the point.

Fit each of these stand-ins against the MCP labels rather than hand-tuning thresholds. For 2011–2024 you also don't need to read the scoreboard, because the point-by-point files list the server and point winner for every point.

**The Decisions API fits the judgment calls.** It went to public beta on October 6, runs only on gpt-6-luna, and answers your predefined questions with a probability, a pick from a fixed list with confidence scores, or a rating on an ordered scale. That suits stroke type, forced vs. unforced, error type, and "main camera or replay?"

It has limits that matter for video:

- Images must be sent inline as base64, and there's no video input. You'd send a grid of about nine frames around each hit, cropped to the player hitting.
- A question that depends on an earlier answer needs a separate request.
- It costs $0.10 per million input tokens with no output charge, so labeling every uncharted shot costs something like tens of dollars.

Accuracy is the open question. I'd expect it to struggle with the far player in old standard-definition footage and with slice vs. topspin. OpenAI's docs say to set thresholds using your own labeled examples, and you have 178,000. Run it on the charted matches and measure accuracy by era and by near vs. far player. Keep it only where it beats a small classifier trained on the keypoint sequences with the same labels. My guess is the classifier wins on forehand/backhand, and the API is more useful for forced vs. unforced.

If you later need actual bounce locations, add a multi-frame ball tracker like WASB-SBDT just for those fields.

I'd build the MCP-to-video aligner first, since everything else trains and tests against it. Want me to write it?
