# Developer / Presenter README — FDS Final Project

> **Audience:** the three group members. This document is the *internal* deep-dive: every modeling choice, every line of math, every defensible answer to a question the professor might ask in the 30-minute in-person discussion (May 4-8). Read it cover-to-cover before the discussion.

---

## 0. Files in this directory

| File                                | Role                                                                           |
|:------------------------------------|:-------------------------------------------------------------------------------|
| `FDS_Final_Project.ipynb`           | The submission notebook (all 10 sections, executed, with outputs). **Submit this.** |
| `build_notebook.py`                 | Generator script that produced the notebook from cell text + code. Reproducible. |
| `README.md`                         | Polished, professor-facing README. **Submit this too.**                         |
| `DEVELOPER_README.md` (this file)   | Internal deep-dive for the group. Do **not** submit.                            |

The raw data sits at `../dataverse_files/CCES22_Common_OUTPUT_vv_topost.csv` (192 MB). It is *not* checked into the repo because of size; the notebook documents the path and the Harvard Dataverse DOI.

---

## 1. Reproducing every result from scratch

```bash
# from .../FDS/project
uv run --quiet --with nbformat python3 build_notebook.py        # rebuild .ipynb shell
uv run --quiet \
  --with "pymc>=5" --with arviz --with pandas --with matplotlib \
  --with seaborn --with scipy --with numpy --with networkx \
  --with jupyter --with ipykernel --with nbconvert --with nbformat \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 FDS_Final_Project.ipynb
```

End-to-end execution time on an Apple-Silicon laptop: **~85 seconds** (most of which is environment resolution; PyMC sampling itself is ~15 s for both models combined). All randomness is seeded (`random_seed=20260501` everywhere) so you should get bitwise-identical numerics on rerun.

To export PDF for Gradescope, the recommended path is:

```bash
uv run --quiet --with jupyter --with nbconvert --with playwright \
  python -m playwright install chromium
uv run --quiet --with jupyter --with nbconvert --with playwright \
  jupyter nbconvert --to webpdf --allow-chromium-download FDS_Final_Project.ipynb
```

(Vanilla `--to pdf` requires a full LaTeX install; `webpdf` only needs Chromium.)

**Important:** trim markdown wherever needed to keep the resulting PDF ≤ 20 pages at 12 pt (per `final_project_details.fa26.v2.pdf`). The notebook is currently structured to fit, but you may want to collapse output cells before exporting.

---

## 2. Mapping the 6-step "core requirements" to notebook sections

The professor's grading rubric (`final_project_details.fa26.v2.pdf`) explicitly checks for:

| Rubric question                                              | Notebook section            |
|:-------------------------------------------------------------|:----------------------------|
| Question / estimand identified                               | §1.1                        |
| Description of data                                          | §1.2                        |
| Causal model with DAG                                        | §2.1, §2.2 + DAG cell       |
| Variables clearly labeled (NOT T, Y, Z!)                     | §2.1 ($E$, $V$, $P$)        |
| Statistical model in math notation, full                     | §3 LaTeX block              |
| Outcome distribution justified                               | §3.2                        |
| Priors justified by domain knowledge AND prior pred. sim.    | §3.1 + prior-predictive cell |
| Confound handled statistically                               | §3.3 + interaction encoding |
| Validated on **simulated** data with `ref_val`               | §4 (full pipeline)          |
| Computational model code clear                               | §6.1                        |
| Diagnostics: trace, $\widehat R$, ESS                        | §6.2                        |
| PPC plot includes obs / post mean / HDI of mean / HDI of pred | §7 cell-level + marginal    |
| Discussion answers the question                              | §8.1                        |
| Confound explicitly discussed                                | §8.2                        |
| Future-work plan grounded in current results                 | §9                          |
| Group contributions table                                    | §10                         |

**Index-based encoding (NOT dummy):** the announcement on Brightspace explicitly called this out as required ("Index-based categorical encoding is the approach that I want to see used in the statistical models for the final project"). Our model uses `alpha[E_data, P_data]` — pure index encoding into a `(3, 3)` parameter matrix.

---

## 3. The model in one paragraph (memorize this)

> "We model whether a respondent voted for the **Democratic (incumbent-party) House candidate** as a Bernoulli trial whose log-odds is a 3×3 matrix of parameters indexed by the respondent's economic perception ($E$ ∈ Better/Same/Worse) and party identification ($P$ ∈ Democrat/Independent/Republican). Each entry of that matrix has a Normal(0, 1.5) prior — informative on the log-odds scale (about ±3, the realistic range of vote-choice log-odds) but flat on the probability scale (we showed this with prior predictive simulation comparing σ ∈ {0.3, 1.5, 5}). We fit it with PyMC NUTS, validated the recovery of pre-set parameters on simulated data, and report the **post-stratified average causal effect** by averaging the per-stratum probability contrast over the empirical distribution of party ID. This closes the backdoor path $E \leftarrow P \rightarrow V$ in the DAG."

---

## 4. The math you should be able to write on a whiteboard

Likelihood:
$$V_i \mid p_i \sim \text{Bernoulli}(p_i), \qquad i = 1,\dots,N$$

Linear predictor (logit link, full interaction via 2-D index):
$$\text{logit}(p_i) \;=\; \alpha_{E[i],\,P[i]}$$

Priors:
$$\alpha_{e,p} \stackrel{\text{iid}}{\sim} \text{Normal}(0,\,1.5), \quad e=1,2,3,\;p=1,2,3$$

Causal estimand (post-stratification — this is the do-calculus identification):
$$
\text{ATE}_{e_1 \to e_2}
\;=\;
\sum_{p=1}^{3}\widehat\pi_p \,
\bigl[\,\sigma(\alpha_{e_2,p})-\sigma(\alpha_{e_1,p})\bigr],
\qquad
\widehat\pi_p \;=\; \frac{1}{N}\sum_i \mathbb{1}\{P[i]=p\}
$$

where $\sigma(x) = 1/(1+e^{-x})$.

Why this identifies the causal effect: in the DAG $P \to E$ and $P \to V$, $P$ is a fork. Adjusting for $P$ blocks the only backdoor path. Post-stratification gives the population ATE under the standard "no unmeasured confounding | $P$" assumption.

---

## 5. Headline numbers (memorize at least these to the nearest 0.05)

| Quantity                                        | Posterior mean | 89% HDI            | $\Pr(<0)$ |
|:-----------------------------------------------|:---------------|:-------------------|:----------|
| ATE (Worse vs Better), post-stratified over P  | **−0.206**     | [−0.231, −0.182]   | 1.000     |
| ATE (Worse vs Same),   post-stratified over P  | **−0.184**     | [−0.207, −0.162]   | 1.000     |
| Within-stratum effect (Worse−Better, Democrats)| −0.015         | [−0.027, −0.004]   | ~1.0      |
| Within-stratum effect (Worse−Better, Independents) | **−0.502** | [−0.542, −0.461]   | 1.000     |
| Within-stratum effect (Worse−Better, Republicans)  | **−0.240** | [−0.306, −0.173]   | 1.000     |

**Plain-English answer to the question:**
*Yes — perceiving the national economy as having gotten worse causally reduces the probability of voting for the incumbent (Democratic) party's House candidate by about **15-25 percentage points** on average, and the effect remains large and credibly negative even after we condition on partisan identity. The mechanism is concentrated in Independents (a 50-point swing) and Republicans (a 24-point swing); strong Democrats barely move (1.5 points), which is consistent with the political-science literature on "motivated reasoning" by partisans.*

---

## 6. Diagnostic numbers (have these ready)

* All 9 $\alpha$ parameters: $\widehat R \in [1.000, 1.003]$, all under the 1.01 threshold.
* Bulk ESS for every parameter: 6,300 - 8,000 out of 4,000 NUTS draws — i.e. NUTS is more efficient than independent sampling for our model (the indexed-Bernoulli posterior is very Gaussian-ish in the well-populated cells).
* No divergent transitions reported.
* Trace plots: four overlapping caterpillars per parameter. (Section 6.2 figure.)

---

## 7. Decisions you should be able to defend

1. **Why three economy levels (not five)?** The original CC22_302 has 5 levels plus a "Not sure" sixth. Collapsing into Better/Same/Worse (a) sharpens the contrast of interest, (b) gives all 9 $(E, P)$ cells healthy sample sizes (no cell < 200), (c) avoids an ordinality-vs-categorical debate during the 30-min discussion. We can defend collapsing because the substantive theory is about "did the economy improve, hold, or decline" — a 3-bucket question.

2. **Why drop "Not sure" rather than impute?** Because a missing economic perception is plausibly **informative** (e.g. the disengaged), and imputing Better-or-Worse from observables would itself require a model. The clean exclusion is honest about what we're estimating: the ATE among voters who *have* an opinion on the economy.

3. **Why three party-ID levels, dropping "Other" / "Not sure"?** Same logic as #2 plus we want a clean closed-form interaction. The dropped fraction is small.

4. **Why subsample to 8,000?** Pure compute. The full clean sample is ~30 K. Sampling 30 K observations through NUTS would take 5-10× longer with no meaningful change in posterior precision (we already have HDIs ±0.025 wide on the ATE). The subsample is **stratified on (E, P)** so the empirical joint distribution — which we use for post-stratification — is preserved.

5. **Why interaction encoding ($\alpha_{e,p}$) instead of additive ($\beta_e + \gamma_p$)?** Because the literature says (and our data confirms) the effect of $E$ varies by $P$ — Democrats barely budge, Independents swing massively. Additive encoding would impose homogeneity that is empirically false. The interaction model also nests the additive model, so we lose nothing.

6. **Why Normal(0, 1.5) instead of Normal(0, 1) or weak Normal(0, 5)?** Section 3.1 + the prior-predictive plot shows σ = 1.5 yields an approximately **uniform** prior on the probability scale. σ = 1 piles up at 0.5 (artificially shrinks toward chance), σ = 5 piles up at 0/1 (prior asserts certainty before seeing data). σ = 1.5 is the sweet spot recommended by McElreath (2020) §11.1.1.

7. **Why Bernoulli, not Beta-Binomial?** We have one trial per respondent, not aggregated counts with potential overdispersion. Bernoulli is the unique exponential-family choice for a single 0/1 outcome.

8. **Why post-stratification rather than reporting a marginal coefficient?** A marginal coefficient depends on the encoding. Post-stratification gives the **population-level** average causal effect on the **probability scale** — directly interpretable and what most political scientists would want.

9. **Why 89% HDI rather than 95%?** Following McElreath (Statistical Rethinking convention used throughout the course): 89% is a prime, less ritualized than 95%, and avoids the false implication of a frequentist confidence guarantee.

10. **What unmeasured confounders are we worried about?** Education, income, race, age — all plausibly affect both how people view the economy and how they vote. We acknowledge this in §9 and propose an E-value sensitivity analysis as future work.

---

## 8. Likely "gotcha" questions and prepared answers

**Q1. "Why is your treatment endogenous to your confound?"**
Because party identification shapes economic perception (the partisan-perception literature: Bartels 2002; Bisgaard 2015). That's *exactly* why $P$ is the confound — it has arrows to both $E$ and $V$, creating the backdoor path we close by stratification.

**Q2. "Could the arrow go $V \to E$ instead?"**
This would be reverse causation: voting for the incumbent makes you optimistic about the economy. There is some literature on rationalization, but for a midterm House vote and a *retrospective* economy perception ("over the past year"), the temporal ordering is one-sided enough to be defensible as $E \to V$. We acknowledge this as a sensitivity concern in future work.

**Q3. "Why didn't you use the survey weights `commonpostweight`?"**
Survey weights matter for population-level inference about marginal quantities (e.g. "what fraction of America thought…"). They are usually unnecessary for *causal effects estimated within strata* — the weights would shift sample composition but not change $\sigma(\alpha_{e,p})$. We could add them as a robustness check; the headline ATE direction would not change.

**Q4. "What if I told you ESS bulk should be > 1000, not > 400?"**
The 400 threshold is a common rule-of-thumb (Vehtari et al. 2021 recommend >100 per chain × 4 chains = 400). All our parameters have ESS_bulk > 6,000, far above either threshold.

**Q5. "Walk me through the prior predictive simulation."**
We drew α from $\text{Normal}(0, 1.5)$, applied the inverse-logit, and plotted the implied prior on $\Pr(V=1)$. With σ = 1.5 the prior on the probability scale is ~uniform on (0, 1), meaning we are not pre-committing to any vote share. With σ = 0.3 the prior collapses to 0.5; with σ = 5 it bimodally hugs 0 and 1 (asserting certainty). σ = 1.5 is therefore the minimally informative choice that respects our domain knowledge that subgroup vote-shares are bounded away from 0/1 in practice.

**Q6. "How does the simulation in Section 4 differ from the analysis in Section 6?"**
Same model class. In §4 we *generated* synthetic data from a fixed `ALPHA_TRUE` matrix and showed the posterior recovers it (89% HDIs cover the truth). In §6 we feed the model real CES data; we cannot directly check recovery there because the truth is unknown — instead we check posterior predictive fit (§7).

**Q7. "Are you over-fitting with 9 free parameters?"**
With ~8,000 observations and 9 parameters, we're at ~890 observations per parameter — well within Bayesian regularization comfort. The Normal(0, 1.5) prior also pulls toward 0 (toward 50:50 in probability), providing additional shrinkage in any low-N cell. We see no evidence of overfitting in the PPC.

**Q8. "What does the within-stratum heterogeneity tell us substantively?"**
Independents — the canonical "swing" voters — drive most of the aggregate effect. Strong partisans (especially in-party Democrats here) are essentially insensitive to economic perception. This matches the *motivated reasoning* literature and explains why national elections are decided by economic conditions filtered through the small-but-pivotal independent middle.

**Q9. "If the Democrats were not the incumbent, would the sign flip?"**
Yes. The model is about voting for the *incumbent* party. In a counterfactual 2022 with a Republican president, perceiving the economy as worse would push voters away from the Republican House candidates — i.e. *toward* Democratic candidates. The model encodes the asymmetry through the choice of which party defines $V=1$.

**Q10. "Why isn't pre-treatment vs post-treatment partisan ID a problem?"**
Party ID has been shown to be highly stable in adulthood (Green, Palmquist, Schickler 2002 *Partisan Hearts and Minds*). Treating it as pre-treatment to a 1-year retrospective economy perception is therefore a defensible simplification, though not an ironclad one. We flag this as a future-work item.

---

## 9. Section-by-section presenter cheat sheet (~90 s each)

> The screencast must be ≤ 15 minutes total and every group member must speak (per `final_project_details.fa26.v2.pdf`). Suggested division: Member 1 = §§1-3, Member 2 = §§4-6, Member 3 = §§7-9.

* **§1 — The question.** Open with the political-science background (one sentence on Lewis-Beck & Stegmaier's meta-finding). Then state the estimand precisely.
* **§2 — DAG.** Draw the three nodes, name the variables in *English*, point to the highlighted red E→V arrow as the causal edge of interest, and the P-node as the fork that creates the backdoor.
* **§3 — Statistical model.** Show the LaTeX block, narrate each line. Spend at least 20 s on the prior-predictive plot — this is where the rubric's "priors must be justified by *both* domain knowledge *and* prior predictive simulation" lives.
* **§4 — Simulation validation.** Emphasize the *purpose*: "we know this model can detect the truth because we generated data from it and it found the truth." Point to the `ref_val` lines on the posterior plots.
* **§5 — Data prep.** Be transparent about the 5→3 collapse, the dropped categories, and the stratified subsample.
* **§6 — Real-data fit + diagnostics.** State $\widehat R \le 1.003$, ESS in the thousands, no divergences. Show the trace plots in passing.
* **§7 — PPC.** "Observed inside both HDIs in every cell — model fits."
* **§8 — Conclusion.** Headline number + within-stratum picture. Explicitly note that the confound is "handled by stratifying $\alpha$ on $P$ and post-stratifying when reporting the ATE."
* **§9 — Future work.** One sentence each on: more confounders, sensitivity analysis, district-level multilevel extension.
* **§10 — Don't read the contributions table aloud** (the rubric explicitly says not to).

---

## 10. Citations to have on a backup slide

1. Lewis-Beck & Stegmaier (2000) *Annual Review of Political Science.* Meta-review of economic voting.
2. Healy & Lenz (2014) *AJPS.* Subjective economic perceptions → incumbent vote.
3. Bartels (2002) *Political Behavior.* Partisan colouring of economic perceptions.
4. Bisgaard (2015) *AJPS.* Motivated reasoning about the economy.
5. Green, Palmquist, Schickler (2002) *Partisan Hearts and Minds.* Stability of party ID.
6. McElreath (2020) *Statistical Rethinking* §11. Logistic GLM, prior choice for log-odds, post-stratification.
7. Vehtari et al. (2021) *Bayesian Analysis.* Diagnostics ($\widehat R$, ESS).
8. Schaffner, Ansolabehere, Shih (2023). CES 2022 Common Content. Harvard Dataverse, [https://doi.org/10.7910/DVN/PR4L8P](https://doi.org/10.7910/DVN/PR4L8P). **(Dataset citation only — NOT used as the treatment→outcome reference, per proposal rule.)**

---

## 11. Last-minute pre-flight checklist

* [ ] PDF export ≤ 20 pages at 12 pt.
* [ ] `.ipynb` re-runs end-to-end with no errors.
* [ ] All variable symbols are $E$, $V$, $P$ (NOT $T$, $Y$, $Z$).
* [ ] Index-based encoding (no `pd.get_dummies` anywhere).
* [ ] Prior justification cell *and* prior-predictive simulation present.
* [ ] Simulation-recovery cell uses `ref_val=` in `az.plot_posterior`.
* [ ] PPC plots include observed dot + posterior-mean band + posterior-prediction band.
* [ ] Group contributions section filled in with real names (not the placeholder "Member 1/2/3").
* [ ] Citation for treatment-effect mechanism is **Healy & Lenz** (or similar) — *not* the CES dataset paper.
* [ ] Screencast: each member identifies themselves at start of their part; ≤ 15 min total.
