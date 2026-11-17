# Ball possession from automatic tracking — technical document

More than a Hack 2027, Phase 1. Team **wo ai ni**. Match: FC Barcelona (H) vs Athletic Club (A).

## 1. Approach

A hidden Markov model with three labels — `H`, `A`, `D` — decoded with Viterbi over each half of
the tactical tracking. No video, no neural network, no labelled data from this match. Two distance
constants were checked and chosen on labelled data from seven other matches (§3).

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
within 0.75 m and the other team is at least 1 m further away, fades linearly to zero when the
nearest player is 3.0 m away, and is zero when both teams are equally close (a challenge) or the
ball is not tracked. A ball in flight or a contested ball therefore says nothing.

**Goalkeepers are put back on their own team.** The tracker keeps one slot per goalkeeper, but
for about a tenth of the match it places that slot on the goalkeeper at the other end, who plays
for the other team; every back-pass and goal kick in those stretches was then credited to the
wrong side. A goalkeeper slot is recognised from the tracking itself (a track that spends more
than 80 % of its time within penalty-area depth of a goal line), and on each frame it is given to
the team defending the half of the pitch it stands in. Which team defends which end in each half
is read from the mean position of each team's outfield tracks. This changes 1,699 graded frames
and removes 19 possessions (303 → 284).

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
- **Except when it comes back over a touchline.** A throw-in is taken by hand from outside the
  line, so a ball that went out over a touchline and is tracked back inside the pitch is the
  throw-in being taken, and the 3 s hold ends on that frame. The same exception covers a high ball
  near the far touchline, which the tracker projects outside the pitch although play goes on:
  it is back inside within a second, and the resulting sub-3 s "stoppage" is then discarded by
  the rule below. Behind a goal line the hold stays, because there the ball is fetched and placed
  for a goal kick or corner and its return says nothing. In the first version the hold applied
  everywhere, and 21 of the 56 ball-out events ended with play resuming on the exact frame the
  hold expired — the model wanted to restart earlier and was not allowed to. This changes 1,281
  graded frames in 19 places.
- **Kick-offs.** Smart Tagging marks each kick-off with a ~5 s window bracketing the kick. The ball
  is forced dead in the first second of the window and alive in the last; before the kick-off that
  starts a half, the ball is dead. If the tracker has not found the ball by the end of the window,
  play is held alive until it does, for at most 10 s.

Post-processing, all deterministic:

- **No stoppage shorter than 3 s**; shorter ones are returned to the preceding possession.
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

Track identities are not used, apart from recognising the two goalkeeper slots (§1.2): the model
only needs "nearest player of each team".

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
- **Moving the start of stoppages.** The public window shows one stoppage that we start about 9 s
  early, where the tracker lost the ball in open play, and two whistle stoppages that we start about
  1 s late. Before changing anything we checked on the full match, without ground truth, whether
  either is a general error. The 29 stoppages that begin with the ball lost well inside the pitch
  have the same activity profile as the 14 stoppages whose start is certain (ball tracked over a
  line after live play): median 2.0 m/s at our start and 1.4 m/s one second later, against 2.3 and
  1.7. Where the whistle detector fires within 4 s of a decoded start, the whistle is a median 0.1 s
  before our start when the ball is in view (18 cases) and 0.6 s after it when the ball is lost
  (16 cases), with a spread of about ±1.5 s in both. Neither check supports a systematic shift, so
  the decoder was left as it is.
- **Smart Tagging phases as a consistency check.** The automatic phase tags (Build Up, Progression,
  …) carry a team. We used the share of frames in those phases that our labels give to the tagged
  team as a ground-truth-free check when comparing variants: 0.467 own / 0.180 opponent before the
  pass-by rule, 0.485 / 0.157 after. The exact tag selection behind those figures was not recorded,
  so `tools/gtfree.py` now fixes it (Build Up and Progression tags with a team): 0.748 / 0.108 for
  the previous submission, 0.738 / 0.101 for this one. The tags are not an input to the model.
- **Trusting a missing ball less in open play.** When the tracker loses the ball for a few seconds
  during play, the stretch can be decoded as a stoppage (§6). On the seven labelled DFL matches of
  §3, with the ball hidden the way the challenge tracker loses it, weakening the "ball missing"
  evidence lowered the score on 6 or 7 of 7 matches (−0.004 to −0.018 spellF1), and a decoder with
  separate "play on, ball lost" states lowered it on 7 of 7 (−0.016). Not used.
- **A lower weight on the alive/dead evidence** (0.07 instead of 0.15). The DFL selection of §3 picked
  it together with the two radii. On this match it relabels 4,285 graded frames from stoppage to
  play; on those frames the ball is in view 95 % of the time and the players move at a median
  1.32 m/s, the profile of the dead-ball-in-view component (1.34 m/s) and not of play (2.67 m/s).
  Stoppages with the ball in view are the one case the DFL simulation cannot represent (§3), so the
  alive/dead weight was left at 0.15. See §3 for how this was found.

## 3. Training data

No labelled data from this match and no pre-trained model. The fitted quantities are the alive/dead
emission parameters of §1.1, estimated by unsupervised EM on the provided match. No hand annotation
or hand correction of any kind was made, and no rule or constant refers to a specific frame of the
match.

**Labelled data from other matches (validation and two constants).** We used the public DFL
Bundesliga dataset (Bassek et al. 2025, *Scientific Data*, figshare doi:10.6084/m9.figshare.28196177,
CC BY 4.0): seven matches with 25 Hz tracking and an official per-frame ball status and ball
possession. The decoder was run on them unchanged and scored with the organisers' metric
(`evaluation/possession_metrics.py`). Three differences from the challenge data were bridged with
figures measured on the challenge tracking alone, never its labels:

- *Ball visibility.* The DFL ball is tracked on every frame; the challenge ball is missing on 41 % of
  graded frames, in long gaps that sit mostly on stoppages. The ball was removed from the DFL
  tracking in that pattern (two random draws per match).
- *Ball smoothness.* The challenge ball is smoother (frame-to-frame ball acceleration about half that
  of the DFL ball at the median, a third at the 90th percentile; about 17 detectable touches per
  minute against 28); a ±8-frame moving average on the DFL ball matches it. Scores were also computed on the raw DFL ball.
- *Possession definition.* The DFL possession flag changes team on every touch (about 8 possessions
  per minute of play, against about 4 in the public window), while the challenge counts controlled
  possession only. A flip in open play shorter than 1.5, 2 or 3 s was given back to the team that had
  the ball, and scores are reported for all three references.

On this test the previous constants scored spellF1 0.561 (smoothed ball, 2 s reference). A
single-constant sweep showed most constants at or near their best (pass speed, touch threshold,
switch and dead-ball costs, minimum stoppage) and flagged the two proximity radii, the alive/dead
weight and the switch cost. A joint grid over those four (108 settings, both ball models) was then
scored, with a selection rule written down before the results came in (`tools/choose.py`): best
mean gain over all six conditions and seven matches, accepted only if it beats the previous
constants on at least 6 of 7 matches and, chosen with one match held out, on at least 6 of 7
held-out matches. It picked control radius 0.75 m (was 1.5), fade radius 3.0 m (was 3.5) and
alive/dead weight 0.07 (was 0.15), switch cost unchanged: up on 7 of 7 matches, held-out gain
+0.007 and positive on 6 of 7.

**What we shipped, and how it was decided.** After that decision we scored the resulting file on
the public window as a pipeline check; it fell from 0.600 to 0.516 spellF1, almost all of it
stoppage frames labelled as play. That drop is what made us look further, and we say so plainly.
The cause was found without ground truth: the change of alive/dead weight alone relabels 4,285 graded
frames of the full match from stoppage to play, and those frames have the ball in view and the
activity of a dead ball (§2.1). The DFL simulation hides the ball during most stoppages, so it barely
contains stoppages with the ball in view; with the simulation matched more closely to the challenge
tracking (ball in view during about a third of stoppages, more losses in open play;
`tools/recheck_dfl.py`) the lower weight still looks harmless there, which confirms the test is blind
to this case rather than evidence for the change. We therefore kept the alive/dead weight at 0.15
and shipped only the two radii. On DFL the radii alone gain +0.006 / +0.016 spellF1 at the 2 s
reference (smoothed / raw ball), up on 6 of 7 matches in both, and +0.009 (6 of 7) under the closer
simulation; at the 1.5 s reference the gain is about zero. The radii-only file was not scored on the
public window before this decision; its score is in §5.

**Use of the public 5-minute ground truth.** It was used to check the pipeline (alignment, label
meaning) and to diagnose classes of failure while developing: it exposed a bug in switch placement
and showed that stoppages with the ball in view were being missed, which led to the
dead-ball-in-view component, the ball-out rule and the 3 s minimum stoppage. Each later change
(pass-by rule, touch evidence, whistle, kick-off hold) was also scored on it as a check that
nothing broke; the touch-evidence variant scored lower there as well as on the tag check, and was
dropped. The goalkeeper correction came from watching a rendering of the output on the video
(a goalkeeper on the ball labelled as the other team); the public window was then used to confirm
that the labels disagreed there, and the fix itself uses the tracking only. The touchline
exception of §1.3 came from a check that needs no ground truth (restarts landing exactly where the
3 s hold expired) and from watching those moments on the video; a first form of it, applied to
goal lines too, produced a spurious possession inside a goal-kick stoppage of the public window,
which is why it is limited to touchlines. The public-window scores are identical before and after.

We did score a small number of variants on it: pass speed 5 / 7 / 10 m/s, whistle weight 0 / 9 / 18,
touch threshold 3 and 6 m/s either side of the shipped 4, and touches as direct evidence. Pass speed
and whistle weight gave identical possession scores at every value, so the window did not choose
them. A touch threshold of 3 scored the same as the shipped value and 6 scored lower. The pass-by
rule and the decision not to use touches as evidence agree with the window, but both rest on the
ground-truth-free checks of §1.2 and §2.1 (possession counts on the full match and the tag check).
The later changes (goalkeeper correction, ball-out rule) were kept although the window's possession
score at IoU 0.5 fell from 0.688 to 0.643 with them. The remaining constants (costs, speeds) were
set from football reasoning before scoring; the radii were later chosen on the DFL matches as
described above. The window was never used as training data.

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
| previous submission (radii 1.5 / 3.5 m) | 0.898 | 0.600 |
| DFL selection with alive/dead weight 0.07 (not shipped, §3) | 0.836 | 0.516 |
| **this method** (radii 0.75 / 3.0 m) | 0.874 | 0.586 |

17 possessions is a very small sample (bootstrap interval for spellF1: 0.37–0.75); the difference
between the previous and the shipped radii is well inside it, and we make no claim about the
full-match score. Full-match output: 250 possessions, 129 stoppages, labels H 30 % / A 16 % / D 54 %
of graded frames.

## 6. Limitations

- **Ball lost during live play.** When the tracker loses the ball for several seconds while play
  continues (crosses, crowded boxes) and activity drops, the stretch can be decoded as dead, and
  changes of possession inside it are invisible.
- **Fouls.** The stoppage starts at the whistle, which the tracking cannot hear; our dead-ball
  onset there lags by roughly a second, until activity falls. The whistle detector of §2.1 is not
  yet precise enough to fix this.
- **Very short possessions** are mostly missed: a change of possession needs about 0.6 s of clear
  control, and only 2 % of our possessions last under a second. The public window contains
  possessions of 0.8 s and 1.3 s, so the reference counts some that we do not.
- **Aerial balls** are projected onto the pitch by the tracker, so "nearest player" can be wrong
  under a high ball, and a high ball near the far touchline can look out.
- **Dead-ball share** (54 % of graded frames) is on the high side. Part of it is real — the
  graded second half begins almost two minutes before its kick-off, and the stoppages before the
  four after-goal kick-offs that Smart Tagging marks, plus one other stop of two and a half
  minutes, add up to about twelve minutes — but some slow build-up may be absorbed into
  stoppages.
- **High balls near a goal line** can still be read as out; only the touchline case is handled.
- **The tracked ball is smoothed** over about half a second, so touches are located to within a
  few frames only, and light touches during a dribble are not detected at all.

With more time: a whistle detector that separates the referee from the crowd, passes modelled
explicitly (ball leaves player A, arrives at player B), and an external validation set whose
stoppages look like the challenge's (ball in view while it is placed), so the alive/dead side can
be tuned as well as the team side.
- **Validation data.** The DFL matches label possession on every touch and track the ball
  everywhere; our bridges for both (§3) are approximations, and the 1.5 / 2 / 3 s references
  disagree on small changes.

## 7. Reproduction

See `solution/README.md`. One command, ~20 s, deterministic:
`python solution/run.py --out wo_ai_ni.csv`.
