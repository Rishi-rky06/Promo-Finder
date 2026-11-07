# Promo Finder results: synthetic_show

**Judgments: HEURISTIC BASELINE (no AI model).** Keyword/structure rules only; every candidate is low-confidence. Timestamps, durations and checks are exact.

Rankings are editorial recommendations, not predictions of social performance (no engagement data).

## Run

- Status: **complete**; 2 episodes found, 2 mined OK
- Skipped input files: 1
- Valid candidates: 2 (montages: 0); rejected by validation: 0; near-duplicates removed: 0
- Montage stage: skipped (heuristic provider or disabled); cold-viewer AI review: skipped (heuristic provider or disabled)
- Weights: hook 0.25, cliffhanger 0.25, standalone 0.2, emotion_curiosity 0.15, visual 0.08, spoiler_safety 0.07
- Automated checks on the 2 shortlisted promos: duration in range 2/2, hook within 3s 2/2, cut within 4s of the cliffhanger 2/2, continuous 2/2, boundary snapped to finish a line 0/2

## Post-first shortlist

| # | ID | Segments | Dur | Score | Conf | Sub-scores | Review flags |
|---|---|---|---|---|---|---|---|
| 1 | E01-C1 | E1 00:03.20-00:38.00 | 34.8s | 62.3 | low | Hook 3 Cliff 5 Standalone 3 Emotion 3 Visual 2 SpoilerSafe 4 | low confidence |
| 2 | E02-C1 | E2 00:07.50-00:36.00 | 28.5s | 46.8 | low | Hook 2 Cliff 5 Standalone 3 Emotion 2 Visual 1 SpoilerSafe 2 | low confidence; high spoiler risk |

### #1 E01-C1: Ep 1 rule-based pick @ L003

- Segment 1: Episode 1 **00:03.20 - 00:38.00** (34.8s; L003..L017)
- Total: **34.8s**; score **62.3** (base 62.3); confidence **low** (heuristic fallback: keyword/structure rules, no model judgment)
- Hook (+0.0s): Opens on: "You buried the wrong sister."  
  Screenplay: `L003 [00:03.20-00:05.00] DIA VICTOR (Low, Menacing): "You buried the wrong sister."`
- Cliffhanger (clip ends 0.0s after it): Cuts on: Ominous low drone swells to a cliffhanger sting.  
  Screenplay: `L017 [00:35.00-00:38.00] SFX Ominous low drone swells to a cliffhanger sting.`
- Why it works: Heuristic: opening-signal 2.6, ending-signal 3.5, intense delivery on 38% of lines. No story understanding.
- What a stranger sees: Not assessed (heuristic has no story understanding).
- Evidence: `L003 [00:03.20-00:05.00] DIA VICTOR (Low, Menacing): "You buried the wrong sister."`; `L017 [00:35.00-00:38.00] SFX Ominous low drone swells to a cliffhanger sting.`
- Sub-scores: hook 3 (rule-based estimate (see heuristic.py)); cliffhanger 5 (rule-based estimate (see heuristic.py)); standalone 3 (rule-based estimate (see heuristic.py)); emotion_curiosity 3 (rule-based estimate (see heuristic.py)); visual 2 (rule-based estimate (see heuristic.py)); spoiler_safety 4 (rule-based estimate (see heuristic.py))
- Spoiler risk: low (estimated from episode position (1/2) only)
- Context needed: not assessed by heuristic
- Weaknesses: keyword/structure heuristic: cannot judge story clarity or whether the ending is truly unresolved
- **Needs human review:** low confidence

### #2 E02-C1: Ep 2 rule-based pick @ L003

- Segment 1: Episode 2 **00:07.50 - 00:36.00** (28.5s; L003..L013)
- Total: **28.5s**; score **46.8** (base 46.8); confidence **low** (heuristic fallback: keyword/structure rules, no model judgment)
- Hook (+0.0s): Opens on: "Hello?"  
  Screenplay: `L003 [00:07.50-00:09.00] DIA MAYA (Whispering): "Hello?"`
- Cliffhanger (clip ends 0.0s after it): Cuts on: Thunderclap and a sudden cliffhanger cut.  
  Screenplay: `L013 [00:34.00-00:36.00] SFX Thunderclap and a sudden cliffhanger cut.`
- Why it works: Heuristic: opening-signal 1.0, ending-signal 3.5, intense delivery on 29% of lines. No story understanding.
- What a stranger sees: Not assessed (heuristic has no story understanding).
- Evidence: `L003 [00:07.50-00:09.00] DIA MAYA (Whispering): "Hello?"`; `L013 [00:34.00-00:36.00] SFX Thunderclap and a sudden cliffhanger cut.`
- Sub-scores: hook 2 (rule-based estimate (see heuristic.py)); cliffhanger 5 (rule-based estimate (see heuristic.py)); standalone 3 (rule-based estimate (see heuristic.py)); emotion_curiosity 2 (rule-based estimate (see heuristic.py)); visual 1 (rule-based estimate (see heuristic.py)); spoiler_safety 2 (rule-based estimate (see heuristic.py))
- Spoiler risk: high (estimated from episode position (2/2) only)
- Context needed: not assessed by heuristic
- Weaknesses: keyword/structure heuristic: cannot judge story clarity or whether the ending is truly unresolved
- **Needs human review:** low confidence, high spoiler risk

## Alternatives (valid, ranked lower)

| Overall | ID | Segments | Dur | Score | Conf | Why lower |
|---|---|---|---|---|---|---|

## Episodes

| Ep | Status | Valid | Rejected | Note |
|---|---|---|---|---|
| 1 | ok | E01-C1 | - | heuristic baseline: 12 windows scored |
| 2 | ok | E02-C1 | - | heuristic baseline: 4 windows scored |
