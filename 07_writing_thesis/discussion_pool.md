# Discussion pool

Everything collected for the Discussion chapter, in one place. Merged 2026-08-10 from:

- `~/.claude/projects/C--Users-JLU-MBB-Desktop-Meditation-files/memory/thesis-discussion-predictions-findings.md`
- `~/.claude/projects/C--Users-JLU-MBB-Desktop-Meditation-files/memory/thesis-discussion-methods-choices.md`
- `~/.claude/projects/C--Users-JLU-MBB-Desktop-Meditation-files/memory/int-self-disorders-measure-caveat.md` (writing constraint)
- `C:\Users\JLU-MBB\Desktop\Meditation\files\discussion-drafts.md`

Nothing was dropped in the merge; quoted drafts are verbatim. Hard numbers (estimates, CIs, p/q)
live in `CLAUDE.md` under the results sections — this file holds the *argument*, not the tables.

**Add new remarks under §10 (Open remarks) or straight into the section they belong to.**

---

## 1. Study findings, as they stand

Meditation vs placebo = main analysis; DMT–harmine vs placebo = secondary/exploratory.

- ACW prolongation (longer INT) in **visual, interoceptive, and exteroceptive** regions after
  meditation; **minimal** effect in the **mental and auditory** domains.
- Reduced top-down modulation from the **mental self to visual cortex** post-meditation.
- DMT–harmine **attenuates the exteroceptive ACW increase** (secondary/exploratory, low n).
- Read as aligning with TRoM (Cooper et al., 2022): the self-processing pyramid **inverts toward
  extero-/interoception**.

---

## 2. The two §1.4 predictions, paired with results

§1.4 derives two predictions from TRoM + the three-layer self, to be paired with results as
"expected X, found Y".

**Prediction 1 — timescale change.** Intero/exteroceptive timescales lengthen; the mental layer's
own ACW stays comparatively unchanged — a selective, directional shift, not a uniform one.
→ **matches** the finding (prolongation in intero/extero/visual, mental minimal).

**Prediction 2 — coupling change.** Reduced mental-self dominance → weaker coupling of
mental-layer regions to sensory regions.
→ **matches** the finding (reduced mental→visual top-down modulation).

**Division of labour:** implication 1 = *where integration increases* (the ACW change);
implication 2 = *the mental self stepping back* (the coupling change). The mental self "receding"
lives in implication 2 — which is why implication 1 correctly predicts the mental ACW itself
staying still.

> **Watch.** Intro implication 1 was originally mis-stated (it predicted long-ACW transmodal
> regions change / sensory stable — the reverse of the data). It was corrected to the inversion
> logic (intero/extero lengthen, mental unchanged) so intro and results agree. **Do not let the
> old version resurface.**

---

## 3. The mechanism that joins the two findings

Source: Çatal's 2026 thesis §1.4 (`C:\Users\JLU-MBB\Downloads\Catal_Yasir_2026_thesis-18-32.pdf`),
which assembles the dynamic-INT literature. The argument turns the timescale change and the
coupling change from two separate results into **one mechanism with two visible consequences**.

The chain:

1. A region's timescale depends on its **recurrent connections** — internal loops that reverberate
   incoming activity, so denser recurrence = longer timescale (Chaudhuri et al., 2015; Wang, 2020).
2. This is the same architecture separating self from non-self regions, where stronger recurrence
   and higher basal excitation accompany longer ACW (Keskin et al., 2025 — the "internal echo").
3. Recurrent strength is **not fixed** — Zeraati et al. (2023) showed a slight increase in
   recurrent connectivity produces a large timescale change, and argued the modulation is
   **top-down from higher-order regions**.
4. Timescales are labile in practice — lengthening under attentional demand and in working-memory
   delay (Gao et al., 2020), varying across a session and around behavioural events
   (Manea et al., 2024).

**Applied to TRoM:** if the mental self ordinarily constrains the sensory layers from above, and
top-down input sets their recurrent gain, then the **weakened mental→sensory coupling is the cause
of the timescale change**, not a separate finding.

**Honest limit:** the FC here is *undirected*, so direction cannot be established. Flag transfer
entropy as the follow-up (this also matches limitation 3 in §1.10).

### Draft paragraph
Overlap-clean vs Keskin/Northoff/Cooper/Çatal/Chaudhuri; full text also at `files/mechanism_para.txt`.

> The two findings need not be read as separate effects. A region's timescale depends in part on
> its recurrent connections, the internal loops through which activity arriving in a region is
> reverberated rather than simply passed on, so that denser recurrent connectivity sustains
> activity for longer and lengthens the timescale (Chaudhuri et al., 2015; Wang, 2020). This is the
> same architecture that distinguishes self from non-self regions, where stronger recurrent
> connections and higher basal excitation accompany the longer autocorrelation windows of self
> regions (Keskin et al., 2025). Crucially, the strength of that recurrent activity is not fixed.
> Modelling spiking data from monkeys performing a spatial attention task, Zeraati et al. (2023)
> found that a slight increase in recurrent connectivity produces a large change in a region's
> timescale, and argued that this modulation is imposed from above, by top-down input from
> higher-order regions. Timescales are correspondingly labile in practice, lengthening under
> attentional demand and during working-memory delay (Gao et al., 2020) and varying across a
> recording session and around single behavioural events (Manea et al., 2024).
>
> Placed against the topographic reorganisation model, this offers a mechanism rather than a
> coincidence. If the mental self ordinarily constrains the sensory layers from above, and if such
> top-down input is what sets their recurrent gain, then the weakened coupling of the mental layer
> to the sensory regions is not a second finding alongside the timescale change but its cause: the
> mental self relaxes its hold, the regions beneath it are left to reverberate more freely, and
> their timescales lengthen accordingly. What the present data cannot establish is the direction of
> that influence, since the connectivity reported here is undirected. Whether the coupling change
> drives the timescale change, as this account requires, is a question for directed measures such
> as transfer entropy.

---

## 4. Comparison with Egger et al. (2025) — the published analysis of the SAME scans

Egger analysed this dataset (40 practitioners, 3-day retreat, DMT–harmine vs placebo, fMRI 2 days
pre/post) with within- and between-network connectivity, global connectivity, and cortical
gradients. Findings: meditation alone ↑ segregation between resting-state networks; DMT–harmine
↑ visual-network connectivity (internally and toward attentional/salience networks); **no prolonged
cortical gradient disruption**, read as a return to ordinary topographic organisation.

The argument (result-dependent, so it cannot go in the intro):

1. Egger's null is about the **spatial** hierarchy normalising, yet the ACW changed. Hence
   **spatial normalisation does not entail temporal normalisation** — a region can resume its
   ordinary place in the spatial hierarchy while the timescale of its activity remains altered.
2. Egger parcellated by canonical resting-state networks, not by the self's layers, so only a
   layer-resolved INT analysis can test the TRoM redistribution.

The intro (§1.9) keeps only the short novelty statement (these scans were analysed spatially; INT
adds the temporal dimension plus layer-based parcellation). The substantive comparison belongs here.

---

## 5. Other points from Çatal's chapter

- **ACW-AUC precedent** (justifies the operationalisation): Watanabe et al. 2019; Raut et al. 2020b;
  Manea et al. 2022; Wu & Gollo 2025. Other options are ACW-50 (Honey 2012), ACW-0
  (Golesorkhi 2021), 1/e (Cusinato 2023), exponential-fit τ (Murray 2014) — the last two were
  dropped here as defective.
- **Hierarchical ordering normally survives INT change** (Manea et al., 2024). So a *flattening* of
  the hierarchy (H2) is a departure from what the dynamic-INT literature reports — worth stating as
  non-trivial.
- **Finite-data bias underestimates timescales** with classical ACF methods (Zeraati et al., 2022);
  runs here are 234 timepoints, so concede this as a limitation.
- **Session drift:** Manea found INTs *decreased* across a recording session — precedent for a
  session effect, and also a confound to name.

---

## 6. Methodological choices to defend and criticise

### 6.1 FD threshold — the argument, to be made with a threshold-comparison figure

The thesis censors at **FD > 0.3 mm** (from Goldberg et al. 2024, supervisor-confirmed) and a figure
will compare thresholds. The position: **0.5 mm could equally be defended**, because it is doubtful
that frames in the 0.3–0.5 band are pure artifact rather than usable data.

**Sharpen the framing before writing it.** Every frame contains signal, so the question is not
whether those frames carry signal but **whether the motion artifact in them is large enough to
justify the cost of removing them.** For this thesis that cost is specific and directional — each
censored frame becomes a Lomb–Scargle reconstruction, and interpolated frames are smoother than real
ones, which **inflates the autocorrelation and therefore the ACW**. Stricter censoring thus trades
motion bias for interpolation bias, in a direction that matters for a timescale measure and not for
an FC study. That reframing is stronger than "the threshold is arbitrary".

**Empirical support already in hand:** mean FD ≈ 0.215 mm in both sessions and ~20% of frames
censored in both. With the mean sitting just below the threshold, a large share of censored frames
lie just above it — exactly the 0.3–0.5 band in question.

**Points to concede:** Goldberg is the INT-specific paper and recommends 0.3; INT may be more
motion-sensitive than FC; and lenient censoring is risky in a two-arm design, since residual motion
differing by arm could manufacture a drug effect. (Motion is balanced here — same mean FD and
censoring both sessions — which weakens that objection.)

**No consensus, and worse than usually stated:** Power 2012 proposed 0.5, Power 2014 moved to 0.2,
the field spans 0.2–0.5. Egger et al. (2025) used **0.5 on this very dataset**, so two analyses of
the same scans differ. The sharper criticism: **FD thresholds are not portable across TR.** FD sums
displacement per frame, so a longer TR accumulates more movement per frame and the same number is
effectively stricter. Power's 0.5 came from TR = 2.5 s; this study is TR = 1.8 s, so "stricter than
Power" is not well defined.

**The decisive test, not yet run:** re-run the analysis censoring at 0.5 and report both. If the
effect survives, the threshold is moot. Robustness checks (c) low-censoring subset and
(d) percent-censored correlations approach this but do not replace it. Note this requires
re-denoising on the server, since censoring happens inside the denoising pipeline.

### 6.2 Why Goldberg as the primary pipeline

**The stance to take.** There is no settled methodology for denoising ahead of an intrinsic-timescale
analysis — the in-depth work on what is best for ACW has simply not been done. Goldberg et al. (2024)
is the state of the art that exists, so this thesis follows it, and says so plainly rather than
claiming an independent optimisation. The second pipeline (identical, censoring and interpolation
off) is run for exactly that reason: with no consensus to appeal to, robustness across the one choice
that is genuinely contested has to be demonstrated rather than assumed.

**The positive case:** nearly all denoising benchmarks (Ciric, Parkes, CONN defaults) optimise for
functional connectivity. FC and ACW respond differently to filtering — band-passing is largely
benign for FC but *determines* the ACF. Goldberg et al. (2024), "Strategies for motion- and
respiration-robust estimation of fMRI intrinsic neural timescales", is the only candidate validated
on the quantity actually reported here.

**Why not the alternatives:** `detrend`/Keskin fails empirically (~1042 ROIs with non-positive AUC,
concentrated in somatomotor — a collapsed autocorrelation is no timescale, not a short one); Egger's
CONN pipeline is FC-tuned and scrubs *without* interpolating, breaking the even sampling the ACF
presupposes; GSR was removed project-wide.

**Its problems on this data, to state openly:**

1. The 0.01–0.1 Hz band-pass **inflates ACW by construction** (ACF and power spectrum are transform
   pairs). Absolute values are therefore not comparable across studies using different filters —
   only within-study contrasts are interpretable.
2. Interpolated frames add further autocorrelation (see robustness check (e)).
3. Mild circularity: the frames motion corrupted are reconstructed, then motion is tested as a
   confound.
4. **Only half the protocol was adopted** — the respiratory band-stop was deliberately omitted after
   finding no respiratory peak in `trans_y`. Defensible and empirically grounded, but it is a
   deviation from Goldberg and should be named as one.

### 6.3 Open point — the somatomotor exclusion

The somatomotor exclusion from the FDR family is justified in `CLAUDE.md` by broadband AUC
degeneration — but that is a `detrend` problem, and the analysed pipeline drops zero ROIs. Either
find a reason that holds under the analysed pipeline, or report the six-region correction as a
sensitivity analysis.

**Possible resolution, from the supervisor meeting (§11.4).** The region family named there — the
self regions, visual, and auditory as a control — does not include motor. If that specification was
genuinely a priori, it justifies the five-region family on design grounds and the degeneration
argument is not needed at all. Check honestly whether motor was off the list *before* the results
were seen; if it was not, keep the six-region correction as a sensitivity analysis rather than
back-fitting the justification.

### 6.4 The censoring bias runs *against* the finding

The strongest thing to say about censoring is not that it is harmless, but that its bias points the
wrong way for these results.

- Goldberg et al. report that censoring **short** runs causes a systematic **shortening** of intrinsic
  neural timescales.
- The finding here is a **lengthening**.
- So the censoring bias worked against the effect rather than for it: the positive results survived
  **despite** a downward bias, which makes the bias **conservative with respect to this finding**.

**Scope of the argument — do not overreach.** It defends the lengthening results. It does not rescue
the null drug effect; a conservative bias cannot be invoked to explain away a null.

**Where Goldberg does not transfer.** They validated on ≈58 min of data; runs here are 7.2 min (234
timepoints after dummy removal). Their demonstration that the pipeline tolerates censoring up to 50%
was therefore made on long data. At 7 min the same censoring fraction leaves far less from which to
estimate the ACF, so the confidence intervals here are much wider than theirs, and the 50% tolerance
should not be read as transferring unchanged.

**Verified from this dataset (n=35, 70 runs).** After censoring, a run retains a median of **5.94 min**
(mean 5.72, range **3.63–7.20**) out of 7.2 min. The worst retained run keeps 3.6 min — i.e. the
~50%-censoring regime that Goldberg demonstrated only on long data.

**The matching answer.** Because that wide-CI risk is real, the arms were matched on both motion and
the amount of data lost, so no result can be an artifact of unequal data scarcity:

|                                | placebo          | DMT-harmine      |
|--------------------------------|------------------|------------------|
| mean FD, retained frames (mm)  | 0.169 (SD 0.023) | 0.168 (SD 0.022) |
| frames censored (%)            | 19.8 (SD 13.5)   | 21.4 (SD 15.1)   |
| retained minutes (median)      | 5.98             | 5.84             |

The formal test agrees (`99_QC/01_motion_qc/results/censoring_balance_check.csv`, LMM on percent
censored, n=40): arm +3.21, p = 0.58; session −0.58, p = 0.84; session×arm DiD +1.08, p = 0.80.
Nothing about data loss differs by arm, by session, or by their interaction.

*(The ≈0.215 mm mean FD quoted in §6.1 is over all frames; the 0.168–0.169 here is over retained
frames only — different quantities, don't cross-quote them.)*

**Companion limitation:** short scans **and** a small sample. Shorter recordings carry higher variance,
which makes genuine group differences harder to separate from noise — state this alongside the
matching, not instead of it.

### 6.5 Denoising as an open limitation, stated constructively

There is as yet no proper way to denoise for autocorrelation-based measures, and that unresolved
question propagates into how firmly these results can be held. It belongs in the discussion as an open
methodological problem the field needs to address — not as a pessimistic hedge that undercuts the
work. The register: the data were noisy, the denoising question is unsettled, and the results are
reported with that in view. Neither a claim that the analysis is perfect, nor a retraction of it.

Pair it with §6.4: the limitation is real, and the one direction of bias that *can* be pinned down runs
against the finding rather than towards it.

---

## 7. Material cut from the introduction, parked for the Discussion

### 7.1 Why resting state, and what should change (cut from §1.5, 2026-08-04)

Original wording:

> This matters for the present work in two ways. First, the resting state is where that relation is
> established, so it is the right place to measure it. Second, if meditation and psychedelics alter
> the felt boundary of self and world, that change should appear in both the temporal organisation
> of resting activity and the relation of self-related to sensory regions.

**Why it was cut:** it reads as a conclusion in the middle of a theoretical section, and it announces
the study's rationale before the hypotheses have been stated. In the Discussion it can carry more
weight, because by then the results are in and the two claims can be evaluated rather than promised.

**What to develop in the Discussion:**

1. **Resting state as the right measurement site.** The argument is that rest–stimulus interaction
   and temporo-spatial alignment both make the resting state the place where the self–world relation
   is set, rather than a neutral baseline. In the Discussion this becomes a claim about what the
   resting-state findings mean — not just that INT changed, but that a change measured at rest is a
   change in how input will be met.
2. **The two predicted loci of change.** That an altered felt boundary should show up in (i) the
   temporal organisation of resting activity and (ii) the coupling of self-related to sensory
   regions. This maps onto the temporal and connectivity halves of the study, so the Discussion can
   report whether both moved, only one, or neither — and what each outcome would mean for the
   framework.

**Caveat to carry over:** the felt-boundary-to-neural-change inference is the study's own, not
something the cited sources establish. Keep it framed as an interpretation.

---

## 8. Ready-made prose

**Discussion opener** — recaps prior work → names the gap; Northoff register, overlap-clean; past
tense, switch to present if the discussion uses it:

> Earlier work had established, on the one hand, that meditation reorganises the spatial pattern of
> the brain's connectivity and topography and, on the other, that it alters the temporal, scale-free
> structure of its dynamics. What had remained open, however, was how meditation redistributes
> intrinsic timescales across the regions carrying the self's interoceptive, exteroceptive, and
> mental layers, which is the question the present work set out to address.

Keep the "self's interoceptive, exteroceptive, and mental layers" ordering — "…layers of the self"
is a 7-word verbatim run against Northoff's corpus.

(The mechanism paragraph is in §3.)

---

## 9. Writing constraints that apply to this chapter

- **Measure fidelity — INT/ACW vs adjacent temporal constructs.** When citing empirical support for
  "disrupted timescales in disorders of the self," the measure must actually be INT/ACW. Bipolar
  papers (Martino/Huang [488]; Northoff et al. 2018 [487]) measure neuronal **variability**
  (fractional SD), and Çatal early-psychosis [326] measures **sample entropy + peak frequency** —
  none computed an ACW. Bipolar was therefore dropped from the strictly-INT paragraph, which rests
  on schizophrenia ACW ([490], [493]) and autism INT ([494]). Sample entropy was admitted only after
  explicitly broadening the framing from "intrinsic neural timescales" to "intrinsic temporal
  dynamics." *Before citing a study as INT evidence: ACW/PLE = yes; fSD/variability, entropy, phase
  coherence = flag it and either broaden the framing or cite as convergent, not primary.*
- **Reporting convention — results that hold across every control.** When a result reproduces in both
  denoising pipelines and every robustness check, say so directly: the checks establish that
  **censoring is not the reason for this result**. That is the payoff of having run them.
- **Reporting convention — results that disagree across analyses.** When something is significant in
  one analysis but only trend-level in the other (e.g. interoception under one denoising pipeline
  versus the other), state the dependency in so many words: **"this result depends on this analysis."**
  The validity claim is scoped to the analysis that produced it. Do not promote a trend to a finding by
  reading the two together, and do not bury the disagreement.
- **Drug name:** always "DMT-harmine" for the verum compound.
- **Overlap check:** run drafts against the Northoff corpus before they go in; the ordering fixes
  noted above exist to clear it.
- **One pipeline in the writing.** The thesis reports a single denoising pipeline; never call it
  "maximal" in prose.
- **Figures carry no caption text on the image** — captions go in LaTeX.

---

## 10. Open remarks

*(new remarks go here; move them into the section above once they have a home)*

- **2026-08-10, filed.** Remarks on censoring, the denoising limitation, and reporting conventions went
  into §6.2 (why follow Goldberg at all), §6.4 (bias direction, short-data caveat, arm matching),
  §6.5 (denoising as an open limitation) and §9 (the two reporting conventions).
- **2026-08-10, filed.** The first supervisor meeting on these results → §11 (how Northoff reads them)
  and §12 (future directions), plus the a-priori region-family point in §6.3.

---

## 11. Northoff's reading of these results (first supervisor meeting)

How the supervisor interprets the findings — the frames to write the Discussion in. The *numbers*
quoted in that meeting predate the current n=35 mixed-model analysis, so the interpretation is what
carries over, not the significance levels; where the two diverge it is flagged below.

### 11.1 How he reads the ACW result

- **A two-step reading:** first, visual input processing changes; second, integration into the
  **exteroceptive** self increases — so the **body–environment relationship** is what changes.
- This is the **pyramid inverting toward extero- and interoception** (Cooper et al., 2022 / TRoM) —
  his own framing, and he called the results "very convincing" and "making perfect sense".
- Region by region as he read it: ACW prolongation in **visual, interoceptive, exteroceptive**;
  minimal in **mental and auditory**. **Cognition** showed a larger mean difference but did not
  survive FDR. **Auditory unchanged** — a sensory-specificity control, i.e. a null with a job.
- **Visual is the most robust**, holding across every pipeline variant.

### 11.2 The arm comparison, as he reads it

- **Shared** between meditation and DMT-harmine: visual cortex and interoceptive self processing.
- **Differing:** exteroceptive processing — the DMT-harmine arm attenuates the ACW increase seen in
  the meditators.
- **Phenomenological reading:** both arms relate similarly to the **body** (interoception), but
  differently to the **environment** (exteroception).
- The drug effect here is a **drug × meditation interaction**, not the drug alone.
- **Why the arm contrast is weak by design** — his rationale, and the one to use in the limitations:
  DMT has a **short half-life**, so the day-4 scan captures only trait/residual effects, never acute
  drug effects. Any group difference is therefore a DMT × meditation interaction, **further diluted
  because both arms shared the day-1 meditation.** This is the argument for reporting the drug
  contrast as secondary/exploratory *by design* rather than as a test that failed.
- **Calibrate against the current numbers:** as the analysis now stands the drug DiD is null and
  exteroception is a trend that does not survive FDR. His rationale for *why* the contrast is weak
  stands; his "clear attenuation" should be written down to the trend it is.

### 11.3 The brain as a "global topographic thing"

- He expects a **global** effect, and frames the brain that way. The whole-cortex analysis bore this
  out (session +0.141 s, p = 0.008; placebo +0.214 s, p = 0.004) — his prediction, confirmed.
- **The ratio rationale:** every region carries both its own local intrinsic activity and a global
  component; a **global-to-local ratio** captures how much of the global is represented locally. His
  analogy: global climate change manifests differently region by region. (Not yet built — §12.)

### 11.4 The reporting structure he recommended

- **Primary:** whole-group ACW before vs. after meditation, with **a-priori region selection** — the
  self regions, visual, and auditory as control — justified by the meditation induction technique.
- **Secondary/exploratory:** the drug contrast, given the low subject count.
- **The thesis already does this.** Worth stating plainly: the time-effect-primary framing is
  supervisor-sanctioned, not a retreat from a drug effect that failed to appear.
- The a-priori region list omits motor — see §6.3, where that may resolve an open point.

### 11.5 On method

- He **praised the denoising work** as methodologically sensitive and well executed, and accepted
  **pipeline sensitivity as a valid and important observation**. Useful for §6.5: the limitation is
  acknowledged at supervisor level, not a private worry.
- **Parcels preferred over spheres** — more anatomically grounded, less arbitrary in size.
- **Interpolated as the main result, non-interpolated as supplementary.** In the meeting's shorthand,
  "Parcel Maximal" vs "Parcel GLM", where "GLM" is what the pipeline now calls `maximal_nocensor`.
  That is the current design.
- He offered the **counterpoint that less data manipulation is also defensible** — he is not
  committed to interpolation, so §6.4 should argue the choice rather than assume it.
- Pending at the time: a second opinion from **Philip** in the lab on parcels vs. spheres.

### 11.6 Connectivity and entropy, as discussed

- **Post-meditation: less top-down modulation from the mental self to visual cortex** — consistent
  with meditation phenomenology, and the finding that carries his interpretation of the mental self
  stepping back.
- **Its status now:** same direction under both ROI definitions — spheres −0.189 (q = 0.011),
  parcels −0.146 (q = 0.32). FDR-significant in spheres, trend-level in parcels. A concordant trend,
  not a failure to replicate; report it under the §9 convention as depending on the analysis.
- **Within-region visual FC** rises and survives FDR in all four cells (spheres/parcels ×
  censored/uncensored) — the most robust connectivity result, and the one the meeting did not dwell on.
- **Sample entropy → supplementary**, as a specificity check on the ACW findings. This is where the
  AUC-vs-SampEn dissociation earns its place.

---

## 12. Future directions

Carried from the meeting; these are the "what next" for the closing section.

1. **Global-to-local ratios** — global→exteroceptive and global→interoceptive (§11.3). The global ACW
   half is **built and positive**; the ratios are **not built**. His route: all subjects first, then
   split by arm.
2. **Transfer entropy, with Sohail (Japan)** — directionality of information flow between visual
   cortex and the self regions (cognition→visual vs. visual→cognition), including the temporal
   duration and degree of transfer. This gives §3's undirected-FC limit a named, concrete follow-up
   rather than a generic one.
3. **Visual imagery in the meditation protocol** — retrieve the protocol description and find evidence
   of a visual-imagery component to justify the visual finding narratively. He noted it may not be
   explicit and may need reading between the lines. **Still open, and the most consequential of the
   three:** visual is the most robust result in the thesis and currently has no narrative anchor.
4. *(Admin, not thesis text)* Send the unresponsive author's paper and email to the supervisor, who
   will follow up directly.
