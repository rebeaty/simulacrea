# Open Datasets and Robust Experimental Effects for LLM Simulation of Creativity Research

**Purpose.** Reference document for testing whether LLMs — especially Centaur (Llama-3.1-70B fine-tuned on Psych-101 human experiment transcripts; Binz et al., arXiv:2410.20268) — can simulate human participants in creativity research (psychometric test development, intervention/experiment simulation). Compiled for the Cognitive Neuroscience of Creativity Lab (R. Beaty, Penn State).

**Verification notes.** All URLs below were checked via web search/fetch on 2026-06-11. OSF pages (osf.io) return HTTP 403 to automated fetchers in this environment, so OSF links are verified indirectly (via paper data-availability statements, search-engine indexing of the OSF page title, or secondary citations) and flagged accordingly. GitHub repos were verified by direct fetch. Items that could not be confirmed are marked **UNVERIFIED**.

**Three simulation test types referenced below:**
- **T1 Correlation replication** — synthetic participants reproduce known human correlational structure (e.g., DAT–AUT correlation, semantic distance–human rating correlation, fluency–originality relations).
- **T2 Distribution matching** — synthetic response pools match human pools on response-level distributions (originality scores, semantic distance, frequency/commonness of ideas, response diversity).
- **T3 Experimental-effect replication** — synthetic participants reproduce a known causal effect under manipulated instructions/stimuli/conditions.

---

## 1. Open datasets with trial-level human creativity responses

### 1.1 OCSAI / Open-Scoring training corpus (~27k human-rated AUT responses) — **strongest verified resource**

- **Citation:** Organisciak, P., Acar, S., Dumas, D., & Berthiaume, K. (2023). Beyond semantic distance: Automated scoring of divergent thinking greatly improves with large language models. *Thinking Skills and Creativity*, 49, 101356. https://doi.org/10.1016/j.tsc.2023.101356
- **URLs (verified by direct fetch):**
  - Library + training data: https://github.com/massivetexts/ocsai — `data/ocsai1/` contains the fine-tuning JSONL splits (e.g., `finetune-gt_main2_prepared_train.jsonl`, `..._val.jsonl`, `..._test.jsonl`, plus `gt_alltests2` variants). Confirmed present.
  - Study materials/notebooks: https://github.com/massivetexts/llm_aut_study (notebooks + results; data-prep notebook `notebooks/ocsai2-dataprep/cleanDatasets.ipynb` in the ocsai repo documents all constituent datasets and citations).
  - Earlier scoring library: https://github.com/massivetexts/open-scoring
  - Hosted scoring API/UI: https://openscoring.du.edu/ocsai
- **N:** ~27,000 responses aggregated from **nine past studies** (per paper abstract; per-study Ns documented in the cleanDatasets notebook). Participant counts vary by constituent study.
- **Tasks:** Alternate Uses Task (AUT; many prompts: brick, knife, box, rope, etc.); OCSAI-2 training extends to consequences, instances, and sentence-completion tasks, and to multiple languages.
- **Variables:** response text, prompt, ground-truth human originality ratings (rescaled 1–5), study source. Fine-tuned-model benchmark: r ≈ .81 with human raters.
- **Access/license:** MIT license on repo; data freely downloadable, no registration. Verified file counts: `finetune-gt_main2_prepared_train.jsonl` = 16,081 rows (val 1,010, test 3,030; ~20k total in the main AUT split); `finetune-gt_alltests2_no-testdata_prepared_train.jsonl` = 30,246 rows. Row format: `{"prompt": "AUT Prompt:box\nResponse:Paper weight\nScore:\n", "completion": "17"}`.
- **Constituent-study direct download URLs** (extracted from `cleanDatasets.ipynb`, usable even though osf.io page browsing is blocked to robots): Dumas et al. 2020 — osf.io/download/u3yv4; Silvia et al. 2009 — osf.io/download/qdrv8; Hass 2017 — osf.io/ng598; Hass 2018 — osf.io/download/p2b9c; Silvia et al. 2008 — OSF project 4ketx (zip endpoint); Beaty & Johnson 2021 SemDis bundle (incl. Beaty et al. 2018 and **Beaty & Silvia 2012 serial-order data**) — OSF project gz4fc; Hofelich Mohr et al. 2016 — UMN conservancy zip; MOTES + pilot (Acar et al. 2023); Patterson et al. 2023 multilingual — files.osf.io/v1/resources/**5cy9n**/providers/github/processed-data/?zip=; TransDis (Chinese) — osf.io/download/3fk8y and /mcwtu; DiStefano et al. metaphors — osf.io/download/mr5a3.
- **Enables:** T1 (item-level human-rating structure), **T2 (the single best human response pool for distribution matching — compare synthetic vs. human response pools per prompt on OCSAI/SemDis score distributions and response overlap)**, and serves as the scoring backbone for every T3 test.

### 1.2 SemDis validation datasets (Beaty & Johnson 2021)

- **Citation:** Beaty, R. E., & Johnson, D. R. (2021). Automating creativity assessment with SemDis: An open platform for computing semantic distance. *Behavior Research Methods*, 53, 757–780. https://doi.org/10.3758/s13428-020-01453-w
- **URL:** https://osf.io/gz4fc/ (OSF project "Automating Creativity Assessment with SemDis"; indexed by search engines under that title — **page contents not directly fetchable here; file-level inventory UNVERIFIED**). Open-access paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC8062332/
- **N:** Study 1 reanalyzed AUT responses from Beaty et al. (2018, PNAS; N = 163–171 participants, several thousand responses); additional studies use word-association/creative-phrase data with human novelty ratings. Exact OSF file Ns UNVERIFIED.
- **Tasks:** AUT (multiple objects); word association/sentence-level novelty tasks.
- **Variables:** response text, human creativity ratings (1–5 subjective scoring), SemDis scores from five semantic models + latent semantic-distance factor.
- **Access:** OSF, free; includes tutorial and example data per paper. SemDis web app: semdis.wlu.psu.edu.
- **Enables:** T1 (replicate human-rating ↔ semantic-distance correlations with synthetic responses), T2.
- **Note for this lab:** these are in-house datasets (Beaty lab), so trial-level access is trivially available even where public file inventories are unverified here.

### 1.3 Divergent Association Task (DAT) — Olson et al. 2021 PNAS (largest N)

- **Citation:** Olson, J. A., Nahas, J., Chmoulevitch, D., Cropper, S. J., & Webb, M. E. (2021). Naming unrelated words predicts creativity. *PNAS*, 118(25), e2022340118. https://doi.org/10.1073/pnas.2022340118
- **URLs:**
  - Open data (N ≈ 8,900 participants, 98 countries): https://osf.io/kbeq6/ (per the paper's data statement; **OSF page not directly fetchable here**)
  - Data + algorithm code: https://osf.io/vjazn/ ; algorithm: https://osf.io/bm5fd/
  - Scoring code (verified by direct fetch): https://github.com/jayolson/divergent-association-task
- **N:** 8,914 participants (Study 1); ~9k total; one 10-word DAT trial each, plus subsets with AUT and Bridge-the-Associative-Gap criterion tasks.
- **Variables:** the 10 nouns produced, GloVe-based mean pairwise semantic distance (DAT score), demographics; criterion-task scores in subsamples.
- **Access:** free OSF download; scoring fully reproducible from GitHub.
- **Enables:** T2 at scale (does a population of synthetic participants match the human DAT score distribution, including its variance — LLMs are notoriously low-variance?), T1 (DAT–AUT correlation, r ≈ .3–.5 in the paper's subsamples).

### 1.4 DSI narrative creativity datasets (Johnson et al. 2023)

- **Citation:** Johnson, D. R., Kaufman, J. C., Baker, B. S., Patterson, J. D., Barbot, B., Green, A. E., van Hell, J., Kennedy, E., Sullivan, G. F., Taylor, C. L., Ward, T., & Beaty, R. E. (2023). Divergent semantic integration (DSI): Extracting creativity from narratives with distributional semantic modeling. *Behavior Research Methods*, 55, 3726–3759. https://doi.org/10.3758/s13428-022-01986-2
- **URL:** https://osf.io/ath2s/ (code, tutorial, and web app per paper; **file-level inventory UNVERIFIED** — OSF not fetchable here). Open-access paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10615993/
- **N:** 9 studies, 27 narrative prompts, **3,500+ short narratives** with human creativity ratings.
- **Variables:** story text, human creativity ratings, BERT-based DSI scores (DSI explains up to 72% of rating variance).
- **Enables:** T1/T2 in the *narrative* domain — important task diversity beyond AUT; synthetic stories scored with DSI vs. human distribution; in-house lab dataset.

### 1.5 MuCE — Multitask/large-scale creativity ratings corpus (Beaty lab co-authored)

- **Citation:** Ismayilzada, M., Laverghetta Jr., A., Luchini, S. A., Patel, R., Bosselut, A., van der Plas, L., & Beaty, R. E. (2025). Creative Preference Optimization. *Findings of EMNLP 2025*. arXiv:2505.14442. https://aclanthology.org/2025.findings-emnlp.509/
- **URL:** https://github.com/mismayil/creative-preference-optimization (verified to exist). Dataset: **200,000+ human responses and ratings from 30+ psychological creativity assessments**; responses rated by ≥2 raters with rater quality control.
- **Hugging Face (VERIFIED via dataset viewer):** https://hf.co/datasets/CNCL-Penn-State/MuCE — public `full_agreement` config with **23.3K rows** (train 16.9K / val 2.2K / test 2.0K / heldout_item 1.9K / heldout_task 187; columns: ID, Dataset, Set, Language, Prompt, RatingLabel, Response, TaskType, factor-scored human rating, FullPrompt), MIT license. Companion sets **MuCE-SFT** (7.2K rows) and **MuCE-Pref** (10K-100K preference pairs) are also public.
- **Enables:** T1/T2 across many tasks at once — the broadest single resource, and the lab has internal access to the full corpus. **Caveat:** CRPO was trained on MuCE, so MuCE cannot serve as an independent validation set for CRPO (it can for Centaur); the `heldout_item`/`heldout_task` splits mitigate this.

### 1.6 Silvia et al. 2008 — classic AUT dataset with subjective scoring

- **Citation:** Silvia, P. J., Winterstein, B. P., Willse, J. T., Barona, C. M., Cram, J. T., Hess, K. I., Martinez, J. L., & Richard, C. A. (2008). Assessing creativity with divergent thinking tasks: Exploring the reliability and validity of new subjective scoring methods. *Psychology of Aesthetics, Creativity, and the Arts*, 2(2), 68–85.
- **URL:** https://osf.io/8vrck/ — OSF project explicitly titled "Assessing Creativity With Divergent Thinking Tasks (Silvia et al., 2008...)" with materials, de-identified data, and creativity responses, posted for reuse (verified via search index; **file inventory UNVERIFIED**).
- **N:** ~226 participants; multiple AUT prompts (brick, knife) + instances + consequences tasks; rater-level ratings (top-2 and average scoring).
- **Enables:** T1 (latent-variable structure of DT scoring; fluency–originality relations), T2. Silvia's broader OSF profile (osf.io/zs2qj UNVERIFIED handle — search "Paul Silvia OSF") hosts data for many subsequent papers, incl. Nusbaum, Silvia & Beaty (2014) instruction-effect data (**UNVERIFIED — check directly; Silvia is a frequent collaborator**).

### 1.7 Cambridge AUT dataset (small, fully open)

- **Citation:** Sun, L., Gu, H., Myers, R., & Yuan, Z. (2024). A new dataset and method for creativity assessment using the alternate uses task. *Communications in Computer and Information Science* (IC2023). https://doi.org/10.1007/978-981-97-0065-3_9
- **URL (verified by direct fetch):** https://github.com/ghydsgaaa/Cambridge-AUT-dataset — Excel files `Bowl AUT dataset.xlsx`, `Paperclip AUT dataset.xlsx`; each response rated 0–4 for originality by three independent raters.
- **N:** not stated on repo front page (**N UNVERIFIED**; modest scale).
- **Enables:** T2 on held-out prompts not in OCSAI training data (useful to avoid contamination concerns).

### 1.8 Multilingual semantic distance datasets (Patterson et al. 2023)

- **Citation:** Patterson, J. D., Merseal, H. M., Johnson, D. R., Agnoli, S., Baas, M., et al. (2023). Multilingual semantic distance: Automatic verbal creativity assessment in many languages. *Psychology of Aesthetics, Creativity, and the Arts*. Preprint: https://hal.science/hal-04274046
- **Content:** AUT responses with human ratings across ~12 languages from an international consortium (~107,672 responses per the preprint). **OSF project 5cy9n** — indirectly verified: the OCSAI-2 build notebook programmatically downloads `files.osf.io/v1/resources/5cy9n/providers/github/processed-data/?zip=`. In-house lab project, so trial-level data is internally available. Replicates the serial order effect cross-linguistically as a validity check.
- **Enables:** T1/T2 cross-linguistically — a distinctive test of whether synthetic participants reproduce human creativity structure outside English.

### 1.8b German mega-AUT corpus (Saretzki et al. 2025) — largest single-language rated AUT set

- **Citation:** Saretzki, J., Forthmann, B., Benedek, M., et al. (2025). Scoring German Alternate Uses items applying large language models. *Journal of Intelligence*, 13(6), 64. https://doi.org/10.3390/jintelligence13060064 (verified via PMC full text, PMC12194149)
- **N:** **49,391 responses (48,507 analyzed) from 2,320 participants, 15 AUT objects, 8 studies, 5 German-speaking labs** (Graz/Benedek, Münster/Forthmann, others). Every response human-rated by >=2 raters (ICCs .79-.90); CLAUS/OCSAI/GPT-4 machine scores included.
- **Access:** paper states all data/materials/scripts are in an OSF repository (candidate IDs from search: osf.io/8fzqt "ADTA" project, data at osf.io/q5j8g — **UNVERIFIED**, confirm via the paper's data statement).
- **Enables:** T1/T2 at scale in German — strong cross-language out-of-distribution test.

### 1.8c Hofelich Mohr et al. 2016 — AUT + Big Five + design manipulation

- **Citation:** Hofelich Mohr, A., Sell, A., & Lindsay, T. (2016). Thinking inside the box: Visual design of the response box affects creative divergent thinking in an online survey. *Social Science Computer Review*, 34(3), 347-359. Data: DRUM, **DOI 10.13020/D6K012** (conservancy.umn.edu/handle/11299/172116; also fetched programmatically by the OCSAI build notebook).
- **Content:** MTurk AUT (brick or paperclip, 2 min) with manipulated response-box segmentation/size, plus **Big Five Inventory** and demographics; verbatim responses.
- **Enables:** T3 discriminant-validity probe — humans show a survey-design artifact (box layout shifts fluency/originality); whether a simulator reproduces a *task-pragmatics* artifact is informative either way — plus T1 (personality covariance).

### 1.9 CAP — Creativity Assessment Platform

- **Citation:** Patterson, J. D., Pronchick, J., & Beaty, R. E. (2025). CAP: The creativity assessment platform for online testing and automated scoring. *Behavior Research Methods*. https://doi.org/10.3758/s13428-025-02761-9 (PMC12361297)
- **Status:** verified to exist as a platform paper (lab's own). **Public trial-level datasets released with CAP: UNVERIFIED** (PMC fetch blocked here). As the lab's own infrastructure, CAP normative data is the natural in-house complement to the public sets above.

### 1.10 Beaty et al. 2018 PNAS behavioral data

- **Citation:** Beaty, R. E., Kenett, Y. N., Christensen, A. P., et al. (2018). Robust prediction of individual creative ability from brain functional connectivity. *PNAS*, 115(5), 1087–1092.
- **Status:** N = 163 in-scanner AUT participants + three external validation samples. The AUT response text + human ratings were reanalyzed in Beaty & Johnson (2021) and should live in/alongside the SemDis OSF project (1.2). **Standalone public posting of the behavioral trial-level data: UNVERIFIED** (paper's PNAS data statement predates routine OSF deposits). In-house data.

### 1.11 Serial-order trial-level data

- **Beaty & Silvia (2012).** Why do ideas get more creative across time? An executive interpretation of the serial order effect in divergent thinking. *Psychology of Aesthetics, Creativity, and the Arts*, 6(4), 309–319. N = 133, single 10-min AUT, every response rated and time-stamped by serial position. **Public posting UNVERIFIED** (2012-era paper; likely available from Silvia/Beaty directly — in-house).
- **Patterson et al. multi-lab serial-order project:** could not be located as a published registered report or public dataset under that description. **UNVERIFIED — confirm internally with J. Patterson.** (Closest verified Patterson assets are the multilingual SemDis consortium data, 1.8.)

### 1.12 Other / honorable mentions

- **Salesforce TTCW** (Torrance Tests for Creative Writing): https://hf.co/datasets/Salesforce/ttcw_creativity_eval — 48 stories (12 New Yorker + 36 LLM) with expert annotations on 14 Torrance-derived dimensions; BSD-3-Clause; verified on Hugging Face. Small but useful for creative-writing evaluation design, not participant simulation.
- **Forthmann datasets:** Forthmann posts data for many papers (e.g., be-creative × object-frequency, fluency-confound simulations) on OSF; no single canonical corpus URL verified here. **UNVERIFIED — search OSF for "Forthmann" per-paper.**
- **Hugging Face:** beyond **MuCE (public, see 1.5)**, no substantive trial-level human AUT/divergent-thinking corpora found (searches for "creativity", "divergent thinking", "alternate uses", "ocsai" returned only LLM benchmarks and the OCSAI-D drawing models).
- **Published LLM-vs-human comparisons to benchmark against:** Wang, Huang, Shen & Uzzi (2026, *Nature Human Behaviour* 10:531-540) — DAT, 9,198 humans vs 215,542 LLM observations incl. persona prompting; Bellemare-Pepin et al. (2024/2025, *Sci Reports*) — ~100k-human DAT database + creative writing; Hubert, Awa & Zabelina (2024, *Sci Reports*) — N=151 humans, AUT/Consequences/DAT vs GPT-4; Haase & Hanel (2023) — human AUT vs 6 chatbots (data: osf.io/vmk3c, UNVERIFIED). **None test Centaur — that is this project's gap.**
- **AuDrA drawing dataset** (Patterson et al., 2023, BRM; osf.io/kqn9v) — verified to exist but drawing-based, not usable for text-only simulation.

### Dataset summary table

| Dataset | N (participants / responses) | Tasks | Scores available | Access (verified?) | Best test |
|---|---|---|---|---|---|
| OCSAI training corpus | ~27k responses, 9 studies | AUT (+ consequences/instances in v2) | Human originality (1–5), OCSAI | GitHub, MIT — **verified** | T1, T2 |
| SemDis validation (Beaty & Johnson 2021) | ~171 Ps + extra studies | AUT, word association | Human ratings, SemDis (5 models) | OSF gz4fc — indirect | T1, T2 |
| DAT (Olson 2021) | 8,914 Ps | DAT (+AUT subsets) | DAT score, criterion tasks | OSF kbeq6/vjazn — indirect; GitHub scoring **verified** | T2, T1 |
| DSI narratives (Johnson 2023) | 3,500+ stories, 9 studies | Short stories (27 prompts) | Human creativity ratings, DSI | OSF ath2s — indirect | T1, T2 |
| MuCE (Ismayilzada 2025) | 200k+ responses, 30+ tasks | Many (AUT, stories, etc.) | Multi-rater ratings | GitHub subset — **verified exists**; full release pending | T1, T2 |
| Silvia 2008 | ~226 Ps | AUT, instances, consequences | Rater-level subjective scores | OSF 8vrck — indirect | T1 |
| Cambridge AUT | UNVERIFIED N | AUT (bowl, paperclip) | 3-rater 0–4 originality | GitHub — **verified** | T2 |
| Multilingual SemDis (Patterson 2023) | multi-site, ~12 languages | AUT | Human ratings, mSemDis | UNVERIFIED URL; in-house | T1, T2 |
| Beaty & Silvia 2012 serial order | 133 Ps | 10-min AUT, time-stamped | Ratings by serial position | UNVERIFIED public; in-house | T3 |

---

## 2. Robust experimental effects feasible for text-only in-silico replication

### 2.1 Serial order effect — **top T3 target**

- **Canonical citations:** Christensen, P. R., Guilford, J. P., & Wilson, R. C. (1957). Relations of creative responses to working time and instructions. *JEP*, 53(2), 82–88. Beaty, R. E., & Silvia, P. J. (2012). *PACA*, 6(4), 309–319.
- **Effect:** later responses in a DT task are more original/creative than earlier ones; idea production rate declines over time. Beaty & Silvia (2012): essentially linear increase in rated creativity across serial position, slope moderated by fluid intelligence (more intelligent Ps start higher/flatter).
- **Replication status:** very robust — replicated for ~70 years across adults, children (e.g., *J. Intelligence* 2021, 5–6-year-olds), neuroimaging (Wang et al. 2017, *Neuropsychologia*; EEG 2019), and supported indirectly by the time-on-task meta-analysis (Paek et al. 2021). One of the most reliable phenomena in the field.
- **Typical size:** consistent positive position→originality slope; effects moderate (standardized slopes ~.2–.4 across studies; large at the aggregate level).
- **Materials:** trivially available (any AUT prompt + 10-min generation).
- **In-silico design:** sample synthetic participants (temperature/persona variation); elicit sequential idea streams for a fixed object ("keep listing uses one at a time"); score each response with OCSAI/SemDis; fit mixed-effects model of originality on serial position; compare slope, fluency decay, and intercept heterogeneity with Beaty & Silvia (2012) human data. Also test the executive-moderation signature (does a stronger model = flatter slope, paralleling intelligence?). Key risk: LLM output order may be governed by likelihood rather than memory-retrieval dynamics — a *diagnostic* failure mode, which is scientifically interesting either way.

### 2.2 "Be creative" instruction effect — **top T3 target**

- **Canonical citations:** Nusbaum, E. C., Silvia, P. J., & Beaty, R. E. (2014). Ready, set, create: What instructing people to "be creative" reveals about the meaning and mechanisms of divergent thinking. *PACA*, 8(4), 423–432. Forthmann, B., et al. (2016). The be-creative effect in divergent thinking: The interplay of instruction and object frequency. *Intelligence*, 57, 25–32. Meta-analysis: Wei, X., Shen, W., Long, H., & Lu, F. (2023). The power of the "be creative" instruction: A meta-analytical evaluation. (ScienceDirect PII S0023969023000760).
- **Effect size (meta-analytic):** creativity d = .69; originality d = .79; fluency d = .06 (ns) — i.e., a clean dissociation: quality up, quantity unchanged/down.
- **Replication status:** robust across dozens of studies since Harrington (1975); moderated by object frequency and scoring method.
- **Materials:** instructions only — fully open (Nusbaum et al. instructions reproduced in paper).
- **In-silico design:** between-"subjects" manipulation of instruction text ("list as many uses" vs. "be creative — clever, unusual uses"); score with OCSAI; test for the d≈.7 originality gain with null fluency effect, and the instruction × object-frequency interaction (Forthmann 2016). Cheapest, cleanest first replication; also a manipulation-sensitivity check that distinguishes participant simulation from instruction-following confounds (the dissociation pattern — not just "better when asked" — is the target).

### 2.3 Example fixation / conformity — **high-value T3 target**

- **Canonical citations:** Smith, S. M., Ward, T. B., & Schumacher, J. S. (1993). Constraining effects of examples in a creative generation task. *Memory & Cognition*, 21(6), 837–845. Related: Jansson & Smith (1991) design fixation; Marsh, Landau & Hicks (1996); Kohn & Smith (2011) collaborative fixation.
- **Effect:** exposure to examples (novel creatures/toys with specific features) increases inclusion of those features in participants' own creations, even with explicit instructions to diverge; conformity persists under "avoid the examples" instructions.
- **Replication status:** robust; replicated and extended repeatedly across 30 years (idea generation, design, brainstorming); medium-to-large conformity effects on feature inclusion proportions.
- **Materials:** stimuli (example creatures with antennae/four legs/tail, etc.) are fully described in the paper; text descriptions suffice for a text-only analog (original used drawings — text adaptation is faithful for LLMs). **Open availability of original image files: UNVERIFIED**, but unnecessary.
- **In-silico design:** condition A: generate imaginary animals (described in text); condition B: same after seeing three example descriptions sharing critical features; DV = proportion of synthetic creations containing critical features; replicate the conformity increase and its persistence under avoid-instructions. Especially interesting because in-context fixation is a known LLM property — quantitative comparison with human conformity rates is novel.

### 2.4 Episodic specificity induction (ESI)

- **Canonical citations:** Madore, K. P., Addis, D. R., & Schacter, D. L. (2015). Creativity and memory: Effects of an episodic-specificity induction on divergent thinking. *Psychological Science*, 26(9), 1461–1468 (2 experiments, internal replication). Extensions: Madore, Jing & Schacter (2016, *Memory & Cognition* — older adults); Madore, Thakral, Beaty, Addis & Schacter (2019, *Cerebral Cortex* — fMRI, with Beaty).
- **Effect:** brief interview training in recollecting episodic detail selectively boosts AUT fluency/flexibility (number of appropriate uses) but not an object-association control task; selective to episodic-retrieval-dependent performance; medium effects (η²p ~ .1–.2 within the lab's studies).
- **Replication status:** consistently replicated **within the Schacter lab network** (incl. with Beaty: Madore et al. 2015 d = 0.52 and 0.77; 2016; 2019 fMRI); but a **2025 registered replication failed** (PsyArXiv, DOI 10.31234/osf.io/bvedh_v1, Spanish sample, two experiments incl. a direct Madore 2015 replication — no ESI effect on AUT). Weaker external robustness than 2.1–2.3.
- **Materials:** induction protocol (video + Cognitive-Interview-style probes) described in papers; **public stimulus posting UNVERIFIED**.
- **In-silico design:** present a short narrative "video" description; ESI condition: probe the synthetic participant to recall the scene in fine episodic detail; control: general-impressions probes or math task; then AUT. Predicts selective fluency (not originality-rating) boost. Feasible but the construct mapping (does an LLM "have" episodic retrieval to induce?) is the weakest of the top candidates — which itself is a publishable theoretical point.

### 2.5 Incubation effect

- **Canonical citation:** Sio, U. N., & Ormerod, T. C. (2009). Does incubation enhance problem solving? A meta-analytic review. *Psychological Bulletin*, 135(1), 94–120.
- **Effect size:** 117 studies; mean d ≈ 0.29 (low–medium); **largest for divergent thinking tasks**; bigger with undemanding interpolated tasks and longer preparation.
- **Replication status:** meta-analytically supported but heterogeneous; individual studies often underpowered.
- **Materials:** standard DT prompts; openly reconstructible.
- **In-silico feasibility: poor-to-moderate.** "Time away" has no native analog in a stateless model; the closest analog (interpolated distractor content in context, or re-prompting in a fresh session with earlier ideas re-presented) changes the mechanism being tested. Recommended only as an exploratory boundary-condition study, not a headline replication.

### 2.6 Production blocking in group brainstorming

- **Canonical citations:** Diehl, M., & Stroebe, W. (1987). Productivity loss in brainstorming groups: Toward the solution of a riddle. *JPSP*, 53(3), 497–509. Meta-analysis: Mullen, B., Johnson, C., & Salas, E. (1991). Productivity loss in brainstorming groups: A meta-analytic integration. *Basic and Applied Social Psychology*, 13(1), 3–23.
- **Effect:** nominal groups (individuals working alone, pooled) produce substantially more ideas than interacting groups; production blocking (turn-taking) is the dominant cause. Large effect (Mullen et al.: strong productivity loss, larger in bigger groups).
- **Replication status:** extremely robust; decades of replications.
- **Materials:** open (standard brainstorming problems, e.g., "thumbs problem").
- **In-silico design:** multi-agent simulation — N solo LLM "participants" (pooled, deduplicated) vs. N agents in a shared transcript with enforced turn-taking; DV = non-redundant idea count + originality. Feasible and a genuinely novel multi-agent result, but higher engineering cost than 2.1–2.3.

### 2.7 Cue/object-frequency and cue semantic-distance effects

- **Citations:** Forthmann, B., Gerwig, A., Holling, H., Çelik, P., Storme, M., & Lubart, T. (2016). *Intelligence*, 57, 25–32 (object frequency × be-creative instruction); related work on associative cue properties (e.g., Heinen & Johnson 2018, *PACA*; Beaty lab semantic-network studies).
- **Effect:** less frequent/common cue objects yield more original responses; cue properties interact with instruction. Moderate, replicated within the Forthmann program; fewer independent replications than 2.1–2.2.
- **In-silico design:** trivially feasible — vary cue frequency/semantic neighborhood density across prompts, score originality, compare human and synthetic cue-level profiles (item-difficulty correlation across prompts is a strong psychometric test: do prompts rank-order the same way for humans and LLMs?).

### 2.8 Fluency–originality confound (statistical/psychometric effect)

- **Citations:** Forthmann, B., Szardenings, C., & Holling, H. (2020). Understanding the confounding effect of fluency in divergent thinking scores. *PACA*, 14(1), 94–112; classic equal-odds debate (Simonton).
- **Effect:** summed originality scores are strongly contaminated by fluency (r often > .6); average scoring and "be creative"+top-scoring reduce it. Robust — it is partly arithmetic, so it *must* replicate if synthetic data are human-like in structure.
- **In-silico design:** free-fluency generation across synthetic participants; verify the human-typical fluency–summed-originality correlation and its reduction under average/max scoring. Good "sanity-check" target and a strong distribution-matching test of fluency variance (LLMs may produce unnaturally uniform fluency).

### 2.9 Time-on-task effect

- **Citation:** Paek, S. H., Abdulla Alabbasi, A. M., Acar, S., & Runco, M. A. (2021). Is more time better for divergent thinking? A meta-analysis of the time-on-task effect on divergent thinking. *Thinking Skills and Creativity*, 41, 100894.
- **Effect:** more time → more/better ideas with an inverted-J pattern; larger for long-vs-short than timed-vs-untimed comparisons. Meta-analytically supported; overlaps mechanistically with the serial order effect.
- **In-silico note:** clock time has no LLM analog; operationalize as response-budget (number of ideas requested / token budget). Best treated as part of the serial-order study rather than standalone.

---

## 3. Ranked top-6 replication shortlist

**1. "Be creative" instruction effect (experimental, T3).** The best first target. Meta-analytic d ≈ .69–.79 on creativity/originality with a null fluency effect (Wei et al. 2023) makes the prediction *pattern-specific*: a simulator must show quality gains without quantity gains, plus the instruction × object-frequency interaction (Forthmann 2016) as a second-order signature. Materials are pure text and reproducible from Nusbaum, Silvia & Beaty (2014) — a lab-internal lineage, so original instructions and (likely) raw data are at hand. It directly tests whether Centaur responds to instructions the way *participants* do rather than the way *assistants* do, which is the core question of the project. Cost: a few thousand generations; scoring fully automated via OCSAI.

**2. Serial order effect (experimental/dynamic, T3).** Arguably the most robust phenomenon in divergent thinking (since Christensen et al. 1957), with in-house trial-level human data (Beaty & Silvia 2012) for quantitative slope comparison, and convergent meta-analytic support (time-on-task). It probes response *dynamics* — memory search and executive retrieval over a session — which Psych-101-style trial-sequence fine-tuning (Centaur) is specifically supposed to capture, making it a sharp Centaur-vs-base-Llama contrast. A failure (flat or reversed slope) is as informative as a success. Fully text-native; one prompt, sequential elicitation, automated scoring by position.

**3. OCSAI 27k corpus distribution + item-structure replication (correlational, T1/T2).** The largest verified open human response pool (GitHub, MIT). Three nested tests: (a) per-prompt originality-score distribution matching (mean, variance, skew — LLM variance collapse is the expected failure); (b) item-level "difficulty" correlation — do prompts that elicit more original ideas from humans do so for synthetic participants (rank correlation across ~dozens of prompts)?; (c) response-content overlap (what fraction of synthetic ideas appear in the human pool, and at what frequency rank — frequency-rank correlation tests whether the model reproduces the human idea-popularity distribution, the foundation of originality scoring itself). This is the psychometric backbone for every other study and requires no new human data.

**4. DAT distributional replication (correlational, T2/T1, N ≈ 8,914).** The biggest-N open dataset in the field (OSF, with verified open scoring code). Simulate thousands of synthetic participants with persona/temperature variation, score with the exact published GloVe pipeline, and compare full score distributions and the DAT–AUT correlation against the human benchmark. Large N gives precise human targets; the single-trial task is maximally cheap; and the known result that LLMs score high-but-narrow on DAT makes the *variance and tail structure* the scientifically interesting quantity for synthetic-participant claims (can prompting reproduce human heterogeneity, not just the human mean?).

**5. Example fixation/conformity (Smith, Ward & Schumacher 1993; experimental, T3).** Robust over 30 years, medium-large effects, and uniquely well-suited to LLMs because in-context influence is mechanistically native to transformers — the open question is whether the *magnitude and instruction-resistance* of conformity matches human rates (including the classic finding that "avoid the examples" instructions fail to eliminate it). Materials are reconstructible from the paper in text form; DV (critical-feature inclusion proportion) is objectively codable, sidestepping rating-model circularity that affects originality-scored studies. Adds a creative *generation* paradigm beyond AUT, broadening task diversity of the test battery.

**6. DSI narrative creativity replication (correlational, T1/T2).** Diversifies the battery into long-form production: 3,500+ human short stories across 27 prompts with human ratings and an open, validated automated metric (DSI explains up to 72% of rating variance; tutorial/code at osf.io/ath2s; in-house dataset). Tests: prompt-level DSI distribution matching; human-vs-synthetic DSI gap by prompt; and reproduction of the demographic-invariance properties reported in the paper. Narrative tasks resist the "list short clever answers" strategy that can let LLMs game the AUT, so this is the strongest guard against concluding human-likeness from a single response format.

**Runners-up:** production blocking (excellent effect, higher multi-agent engineering cost); fluency–originality confound (run it as a free by-product of targets 2–3); ESI (theoretically rich but lab-network-limited replication base and awkward construct mapping); incubation (poor in-silico construct validity — exploratory only).

---

## 4. Cross-cutting design notes

- **Scoring circularity:** when both synthetic responses and the comparison metric come from LLMs (OCSAI), include SemDis and held-out human ratings (from the open corpora above) as convergent scorers; report results under all three.
- **Contamination:** OCSAI training data, DAT data, and most papers here predate Llama-3.1/Centaur training cutoffs and are public — treat verbatim response overlap as a measured quantity (test 3c), not an assumption.
- **Variance engineering:** human-likeness claims hinge on between-participant heterogeneity. Pre-register the participant-sampling scheme (temperature, personas, Psych-101-style transcript framing for Centaur) before computing distributional comparisons.
- **Centaur specifics:** Centaur (arXiv:2410.20268; *Nature* 2025) was tuned on choice-format experiment transcripts; **Psych-101 contains no creativity or open-ended production tasks** (its 160 experiments are decision-making, learning, memory), so creativity simulation is out-of-distribution on both task content and response format. Run all studies on Centaur vs. base Llama-3.1-70B vs. an instruction-tuned control to attribute any human-likeness gains to Psych-101 tuning. Anticipate the published critiques when framing: Orr et al. (2025, arXiv:2510.03311, "prediction is not explanation") and Bowers et al.'s demonstrations of non-human behavior outside the training distribution (256-digit "memory spans", 1-ms RTs; see *Science* news coverage) — a rigorous creativity OOD test speaks directly to that debate, whichever way it comes out.

*Compiled 2026-06-11. All "indirect" OSF verifications should be re-checked in a browser (OSF blocks automated fetchers); items marked UNVERIFIED need confirmation before being cited in a manuscript.*
