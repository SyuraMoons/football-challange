# Ball possession from automatic tracking — technical document

More than a Hack 2027, Phase 1. Team **wo ai ni**. Match: FC Barcelona (H) vs Athletic Club (A).

## 1. Approach

A hidden Markov model with three labels — `H`, `A`, `D` — decoded with Viterbi over each half of
the tactical tracking. No video, no neural network, no labelled training data.

Why this rather than a per-frame classifier: the ranking metric (`spellF1`) scores possessions as
objects, and the provided baseline shows that per-frame decisions fragment into hundreds of
spurious possessions. A sequence model with an explicit price for changing state yields
possession-shaped output directly, and lets "no evidence" mean "nothing changes", which is exactly
the challenge's rule that a loose ball stays with the last team.

### 1.1 Alive vs dead

Per frame we compute, from the ATD only:

- **activity** — mean speed of all tracked players, smoothed over one second;
- **ball visible** — whether the tracker reports a ball;
- **ball out** — ball more than 0.5 m outside the pitch lines;
- **ball pace** — the fastest (smoothed) ball speed within ±1.5 s; a placed or hand-carried ball
  stays slow, a ball in play does not.

Three emission components model these: *alive*, *dead with the ball in view* (placed, carried,
out of the pitch) and *dead with the ball lost by the tracker*. Each is a product of a Gaussian
(activity), two Bernoullis (visible, out) and a log-normal (pace). Their parameters are **fitted
without labels** by EM (forward–backward) on the graded frames of the match itself, starting from a
common-sense initial guess (play is faster; the ball is usually seen and moving). Fitted values:

| | alive | dead, ball seen | dead, ball unseen |
|---|---|---|---|
| activity mean (m/s) | 2.75 | 1.36 | 1.02 |
| P(ball visible) | 0.91 | 0.98 | 0.02 |
| P(ball out \| visible) | 0.005 | 0.078 | 0.234 |
| share of frames | 0.38 | 0.25 | 0.37 |

Because consecutive frames are strongly correlated, alive/dead log-likelihoods are down-weighted
(×0.15) before decoding.

### 1.2 Home vs away

For every frame with a ball we take the distance from the ball to the nearest home player and to
the nearest away player. The evidence is signed and bounded: it is full when one team has a player
within 1.5 m and the other team is at least 1 m further away, fades linearly to zero when the
nearest player is 3.5 m away, and is zero when both teams are equally close (a challenge) or the
ball is not tracked. A ball in flight or a contested ball therefore says nothing.

**A ball going past is not a ball controlled.** A pass that travels within reach of an opponent
would otherwise count as control by that opponent. We detect *touches* as changes of the ball's
velocity (more than 4 m/s between the 8 frames before and the 8 frames after, with a player within
3 m). Proximity evidence is dropped on frames where the ball moves faster than 7 m/s — faster than
a dribble — and no touch was detected within half a second. On the full match this removes 73
possessions (377 → 304).

### 1.3 Decoding

States: `H`, `A`, dead-seen, dead-unseen (the two dead states are merged into `D` afterwards).
Changing possession costs 9 log-units — about 0.6 s of clear, uncontested control by the other team,
or less control spread over two touches. Entering or leaving dead ball costs 9; the tracker losing or
finding a dead ball costs 2.

Hard evidence added to the scores:

- **Ball tracked out of the pitch.** If the ball is outside the lines for ≥ 3 frames and reaches
  1 m outside, the ball is dead from the frame it crosses the line until at least 3 s after it was
  last seen outside.
- **Kick-offs.** Smart Tagging marks each kick-off with a ~5 s window bracketing the kick. The ball
  is forced dead in the first second of the window and alive in the last; before the kick-off that
  starts a half, the ball is dead. If the tracker has not found the ball by the end of the window,
  play is held alive until it does, for at most 10 s.

Post-processing, all deterministic:

- **No stoppage shorter than 3 s** unless the ball was seen out; shorter ones are returned to the
  preceding possession.
- **Change of possession is placed where control is gained.** Viterbi is indifferent to where a
  switch sits in a neutral stretch, so each `H↔A` switch is moved to the first frame of new-team
  control (or new-team touch) after the last frame of old-team control.
- **Restart from a ball at rest.** If, within 4 s of a decoded restart and before being struck
  (> 8 m/s), the ball comes to rest (< 1 m/s), the restart is postponed to when it leaves that rest
  (> 2.5 m/s). This handles balls rolled to the spot of a free kick or goal kick.

We never output `X` on graded frames; ungraded frames are `X`.

## 2. Assets used and dropped

| asset | used? | how |
|---|---|---|
| tactical ATD (players, ball, team of each slot) | yes | everything above |
| Smart Tagging | partly | kick-off windows only. Other set-piece tags start 2–12 s before the restart with no fixed offset, and phase `Team` tags are phases, not possessions, so they were dropped |
| tactical homography | no | tracking is already in pitch coordinates |
| tactical video, picture | no | inspected only. The long stretches with no ball in the ATD (132 gaps of 5 s or more hold 99.7 % of the missing frames) are stoppages in the frames we sampled — corners being set up, kick-offs, injuries — so a ball detector to fill them was not built |
| tactical video, audio | tried, not in the submission | see §2.1 |
| broadcast video / ATD / homography | no | not synchronised to the reference timeline |

Track identities are not used: the model only needs "nearest player of each team".

### 2.1 Things tried and dropped

- **Referee whistle from the audio** (`solution/whistle.py`, `run.py --whistle`). The whistle is
  clearly audible on the tactical video's audio track (20–30 dB above the crowd at 3–4.5 kHz at
  both kick-offs and both half-ends). A narrow-band detector finds 144 incidents. After a
  detection in live play, player activity falls by a median 0.76 m/s against 0.06 m/s after a random
  live frame, so the signal is real; but only about 6 in 10 detections are followed by a clear
  drop, the rest being crowd whistling. Used as soft evidence (it cancels the cost of entering dead
  ball) it changed 105 frames of 158,857. That is too little to justify making the submission
  depend on the audio decoder's exact output, so it is off by default and **not used for the
  submitted file**.
- **Touches as direct evidence of possession.** Adding each detected touch as evidence for the
  touching team created more possessions (377 → 425), because deflections and tackles are touches
  but not controlled gains. Touches are used only to gate proximity, as in §1.2.
- **Smart Tagging phases as a consistency check.** The automatic phase tags (Build Up, Progression,
  …) carry a team. We used the share of frames in those phases that our labels give to the tagged
  team as a ground-truth-free check when comparing variants: 0.467 own / 0.180 opponent before the
  pass-by rule, 0.485 / 0.157 after. They are not an input to the model.

## 3. Training data

None. No labelled dataset from this or any other match, no pre-trained model, no external data.
The only fitted quantities are the alive/dead emission parameters of §1.1, estimated by
unsupervised EM on the provided match. No hand annotation or hand correction of any kind was made,
and no rule or constant refers to a specific frame of the match.

**Use of the public 5-minute ground truth.** It was used to check the pipeline (alignment, label
meaning) and to diagnose classes of failure while developing: it exposed a bug in switch placement
and showed that stoppages with the ball in view were being missed, which led to the
dead-ball-in-view component, the ball-out rule and the 3 s minimum stoppage. Each later change
(pass-by rule, touch evidence, whistle, kick-off hold) was also scored on it as a check that
nothing broke; the touch-evidence variant scored lower there as well as on the tag check, and was
dropped. We ran no parameter sweeps against it and the numeric constants (costs, radii, speeds) were set from football reasoning
before scoring, not selected by score. It was never used as training data.

## 4. Possession rules implemented

- **Loose ball / ball in flight / ball untracked while play continues:** stays with the last team.
- **Challenges:** when both teams are equally close to the ball the evidence is zero; a single touch
  or deflection by the opponent (a few frames) does not pay the switch cost, so possession does not
  change. Sustained control, or two successive controls by the same team, does.
- **Dead ball:** ball out of the pitch, or a stretch of low collective activity with the ball slow,
  at rest or lost. Ownership of a dead ball is not reported (`D`).
- **Restarts:** play resumes when the HMM leaves the dead state — typically when the ball reappears
  moving, or leaves its resting position. The first possession after a stoppage goes to the team
  that first shows control.

## 5. Results on the public window (pipeline check only)

| | macroF1 | spellF1 |
|---|---|---|
| `baseline_smooth50` (published floor) | 0.621 | 0.326 |
| this method | 0.874 | 0.600 |

17 possessions is a very small sample (bootstrap interval for spellF1: 0.43–0.78); we make no claim
about the full-match score. Full-match output: 303 possessions, 131 stoppages, labels H 29 % / A
17 % / D 54 % of graded frames.

## 6. Limitations

- **Ball lost during live play.** When the tracker loses the ball for several seconds while play
  continues (crosses, crowded boxes) and activity drops, the stretch can be decoded as dead, and
  changes of possession inside it are invisible.
- **Fouls.** The stoppage starts at the whistle, which the tracking cannot hear; our dead-ball
  onset there lags by roughly a second, until activity falls. The whistle detector of §2.1 is not
  yet precise enough to fix this.
- **Very short possessions** (under ~1 s) are deliberately not reported.
- **Aerial balls** are projected onto the pitch by the tracker, so "nearest player" can be wrong
  under a high ball, and a high ball near the far touchline can look out.
- **Dead-ball share** (54 % of graded frames) is on the high side; some slow build-up may be
  absorbed into stoppages.
- **The tracked ball is smoothed** over about half a second, so touches are located to within a
  few frames only, and light touches during a dribble are not detected at all.

With more time: a whistle detector that separates the referee from the crowd, passes modelled
explicitly (ball leaves player A, arrives at player B), and validation on labelled possession data
from other matches.

## 7. Reproduction

See `solution/README.md`. One command, ~20 s, deterministic:
`python solution/run.py --out wo_ai_ni.csv`.
