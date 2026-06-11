# Creativity Simulator Roadmap: Centaur as Synthetic Participant

Goal: the most convincing, scaled-up test of whether LLMs — especially
Centaur-70B / Minitaur-8B (Psych-101 cognitive foundation models) — can serve
as synthetic participants for creativity research: psychometric test
development, intervention testing, and experiment simulation.

Framing note: Psych-101 contains **no creativity or open-ended production
tasks** (its 160 experiments are decision-making, learning, memory — discrete
choices wrapped in << >>). Creativity simulation is therefore a genuine
out-of-distribution generalization test for Centaur, which makes the result
informative in either direction. Published critiques (Orr et al. 2025,
arXiv:2510.03311; Bowers' demonstrations of non-human behavior outside the
training distribution) predict failure; the Centaur paper's held-out-experiment
generalization results predict partial success. Nobody has yet tested Centaur
as a creativity participant — that is the gap this project fills.

## Phase 1 — Centaur on the existing ITCT battery (ready to run)

Add `centaur_70b` and `centaur_8b` to the existing human / CRPO / Llama-base /
Gemini comparison on the 5 task types (AUT, Design, Story, SCTT-Hypothesis,
SCTT-ResearchQuestion; 16 items, 245 synthetic participants each).

1. Generate: `generate_centaur_responses.py` (see docs/RUNNING_CENTAUR.md).
   The 70B runs with no GPU via the featherless provider; the 8B needs one
   24 GB GPU with vLLM.
2. Score with CAP (originality + effectiveness) and DSI, refresh
   `task_csvs_for_cap/*_cap_scored.csv`.
3. Evaluate on three pre-registered criteria, all scripted:
   - **Distribution matching** (`analyze_creativity_scores.py`): KS and
     Cohen's d vs human per task/metric. Current best: CRPO (4/6 metrics).
   - **Covariance structure** (`analyze_covariance.py`): does the model
     reproduce the *human* pattern of modest cross-task correlations
     (human r ~ .2-.5)? Existing chat models are over-coherent
     (r ~ .5-.9; profile-r vs human: gemini .62, cpo .43, llama .24).
     Hypothesis: Centaur, trained on real trial-level human variability,
     should produce more human-like (lower, differentiated) correlations.
   - **Item-difficulty profiles**: correlation of per-item mean scores with
     human item means (current: cpo .87, llama .71, gemini .53).

Decision gate: if Centaur beats the CRPO baseline on >=2 of 3 criteria, it
becomes the engine for Phase 2; otherwise CRPO remains the simulator and
Centaur is reported as a comparison condition.

## Phase 2 — Replicating known correlational/covariance patterns at scale

Re-simulate open trial-level datasets and test whether synthetic data
reproduces published covariance patterns (full survey with verified links:
docs/OPEN_DATASETS_AND_EFFECTS.md). Strongest candidates:

| Target | Data | Test |
|---|---|---|
| OCSAI federated AUT corpus (27,217 responses, 2,039 participants, 9 datasets) | github.com/massivetexts/ocsai | distribution + item-profile match at scale; new item bank avoids replicating only our own lab's items |
| Olson et al. 2021 DAT (~8,900 humans, 98 countries) | osf.io/kbeq6 + github.com/jayolson/divergent-association-task | trivially auto-scored; Wang et al. (Nature Human Behaviour 2026) already published an LLM-vs-human DAT comparison to benchmark Centaur against |
| SemDis validation studies (Beaty & Johnson 2021) | osf.io/gz4fc (+ data: osf.io/3zwxc) | AUT + word association: human-rating vs semantic-distance correlation structure |
| Silvia et al. 2008 DT + Big Five | osf.io/8vrck | DT originality x personality covariance (requires persona/individual-difference conditioning of participants) |
| DSI narrative corpus (>3,500 stories, 27 prompts) | osf.io/ath2s | story-task generalization beyond our 4 themes |
| Multilingual AUT repository (Patterson et al., 28 datasets, ~107k responses) | osf.io/5cy9n | serial-order validity check + cross-item and cross-language generalization |
| Saretzki et al. 2025 German mega-AUT (48.5k rated responses, 2,320 participants, 15 objects) | OSF (link in paper, DOI 10.3390/jintelligence13060064) | cross-language distribution + item-profile matching |
| Hofelich Mohr et al. 2016 AUT + Big Five + response-box manipulation | DRUM, DOI 10.13020/D6K012 | verbatim responses + an experimental design artifact humans show but a simulator plausibly shouldn't — a discriminant-validity probe |

Driving-correlation replication for a *different batch* of items (per the
"don't just replicate the same thing" concern): run the same battery on the
held-out 5th items (SOCK, clean-water design, shoe-path-walk story, island
SCTT) plus OCSAI's 13-item validated AUT bank.

## Phase 3 — Replicating robust experimental effects in silico

Manipulations are pure prompt changes, so each effect is a 2-condition run of
the generator with N=245 per arm:

1. **"Be creative" instruction effect** (top target): explicit vs standard
   instructions -> higher originality (d ~ .5-.8), lower fluency
   (Harrington 1975; Nusbaum, Silvia & Beaty 2014; Said-Metwaly et al. 2020
   meta-analysis, 165 effect sizes). Cleanest, most replicated, fully
   text-native. Open materials: Forthmann et al. 2016 (osf.io/a9qnc).
2. **Serial order effect**: originality rises across response position within
   a session (Christensen et al. 1957; Beaty & Silvia 2012; replicated across
   11+ languages). Requires fluency-style generation (multiple responses per
   item) — natural for Centaur's session-transcript format. Bonus tests: the
   Gf moderation (flatter slope for high-ability personas) and the
   fluency-originality confound / equal-odds rule (summed originality should
   correlate r ~ .5-.7 with fluency; ratio scores should decorrelate).
3. **Production blocking in brainstorming** (Diehl & Stroebe 1987; Mullen et
   al. 1991 meta-analysis, r ~ .56, the most robust effect in this set):
   nominal groups beat interacting groups on unique ideas. Multi-agent
   simulation with enforced turn-taking and shared transcript vs independent
   generation + pooling; second-order prediction: removing the turn
   constraint eliminates/reverses the deficit.
4. **Example fixation/conformity** (Smith, Ward & Schumacher 1993; best
   text-native variant: George & Wiley 2020 verbal AUT with example uses):
   examples pull ideas toward example features; "avoid these" instructions
   raise originality above no-example control.
5. **Episodic specificity induction** (Madore, Addis & Schacter 2015;
   d ~ .5-.77, but a 2025 registered replication failed) and **incubation**
   (Sio & Ormerod 2009, d ~ .29, mechanism contested): exploratory only —
   a negative result is itself informative about what simulators can't do.

Scoring throughout: CAP/OCSAI for originality, DSI for narratives, plus human
spot-rating of a stratified sample to validate AI scoring on synthetic text.

## Why this design is convincing

- Three falsifiable criteria per model (distributions, covariance, item
  profiles), not just mean-matching.
- Both replication *of our own lab's* patterns on new data batches and
  *independent labs'* open datasets.
- Experimental-effect replication separates "sounds human" from "responds to
  manipulations like a human" — the property that actually matters for
  simulating interventions.
- Centaur vs CRPO vs base models contrasts three hypotheses: trained on human
  *behavioral transcripts* vs aligned to human *creativity preferences* vs
  generic instruction tuning.
