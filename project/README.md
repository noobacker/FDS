# Does Negative National Economic Perception Cause Voters to Defect from the Incumbent Party?

### A Bayesian Causal-Inference Analysis of the 2022 U.S. Midterm Elections

> **Course:** Foundations of Data Science (NYU, Spring 2026)
> **Group Members:** Member 1, Member 2, Member 3
> **Submission:** `FDS_Final_Project.ipynb` (executed) + `README.md` (this file)

---

## 0. Files in this folder & how to run

| File                        | Purpose                                                                                                  | Submit? |
|:----------------------------|:---------------------------------------------------------------------------------------------------------|:-------:|
| `FDS_Final_Project.ipynb`   | The actual deliverable — 33-cell executed Jupyter notebook with all code, math, plots, and outputs.      |   ✅    |
| `README.md`                 | This polished, professor-facing write-up.                                                                |   ✅    |
| `build_notebook.py`         | Generator script that produced the `.ipynb` from Python source. Only needed if you want to regenerate.    |   ❌    |
| `DEVELOPER_README.md`       | Internal cheat-sheet for the group's in-person discussion (math, defended decisions, anticipated Qs).     |   ❌    |

### Just want to read / submit the project?

The notebook is already executed with all outputs filled in. Open it directly:

```bash
cd "/Users/harshalpimpalshende/VSCode and IDE/NYU/FDS/project"
open FDS_Final_Project.ipynb
```

### Want to verify reproducibility (re-execute from scratch, ~85 s)?

```bash
cd "/Users/harshalpimpalshende/VSCode and IDE/NYU/FDS/project"
uv run --quiet \
  --with "pymc>=5" --with arviz --with pandas --with matplotlib \
  --with seaborn --with scipy --with numpy --with networkx \
  --with jupyter --with ipykernel --with nbconvert --with nbformat \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 FDS_Final_Project.ipynb
```

All randomness is seeded (`random_seed=20260501`) so the numbers will reproduce exactly.

### Want to rebuild the notebook from the Python source first (only if you edited `build_notebook.py`)?

```bash
# Step 1: regenerate the .ipynb shell (cells only, no outputs)
uv run --quiet --with nbformat python3 build_notebook.py

# Step 2: execute it to populate outputs (same command as above)
uv run --quiet \
  --with "pymc>=5" --with arviz --with pandas --with matplotlib \
  --with seaborn --with scipy --with numpy --with networkx \
  --with jupyter --with ipykernel --with nbconvert --with nbformat \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 FDS_Final_Project.ipynb
```

**Why two files?** `build_notebook.py` is the recipe; `FDS_Final_Project.ipynb` is the finished cake. The notebook is a 845 KB JSON blob with embedded plot images — painful to edit directly. The Python generator is clean to edit, diff, and code-review. Only the `.ipynb` is graded.

### Required input data

Place the CES 2022 Common Content CSV at `../dataverse_files/CCES22_Common_OUTPUT_vv_topost.csv` (relative to the notebook). Source: [https://doi.org/10.7910/DVN/PR4L8P](https://doi.org/10.7910/DVN/PR4L8P) (192 MB, not redistributed in this repo).

---

## 1. Research Question

> **Does a voter's *negative perception of the national economy* causally reduce the probability that they vote for the *incumbent president's party* in U.S. House races, after accounting for the confounding effect of *partisan identification*?**

The classical *retrospective economic-voting* hypothesis (Key 1966; Fiorina 1981; Lewis-Beck & Stegmaier 2000) predicts that voters punish the incumbent party when they perceive the national economy as having gotten worse. We test this hypothesis at the **individual-voter level** in the 2022 U.S. midterm House elections, where the incumbent president was Joseph Biden (Democratic Party).

The estimand is the **post-stratified average causal effect** (ATE) on the probability of voting for the Democratic House candidate when a voter shifts from perceiving the economy as *better* over the past year to *worse*, marginalized over the empirical distribution of party identification:

$$
\text{ATE}_{\text{Worse} \,\to\, \text{Better}}
\;=\;
\sum_p \widehat\pi_p\,\bigl[\sigma(\alpha_{\text{Worse},p}) - \sigma(\alpha_{\text{Better},p})\bigr]
$$

---

## 2. Headline Result

| Quantity                                                     | Posterior mean | 89% HDI            | Pr(< 0) |
|:-------------------------------------------------------------|:--------------:|:------------------:|:-------:|
| **ATE (Worse vs Better economy), post-stratified over Party ID** | **−0.206**     | [−0.231, −0.182]   |  1.000  |
| ATE (Worse vs Same economy), post-stratified over Party ID   | −0.184         | [−0.207, −0.162]   |  1.000  |

**Plain-English answer.** Yes — perceiving the national economy as having gotten worse causally **reduces the probability of voting for the incumbent party's House candidate by approximately 15-25 percentage points**, and the effect remains large and credibly negative even after we condition on partisan identity. The mechanism is concentrated in **Independent voters** (a 50-percentage-point swing) and **Republican voters** (a 24-point swing); strong **Democrats barely move** (1.5 points). This pattern is consistent with the political-science literature on motivated reasoning by partisans — strong partisans rationalize away evidence that conflicts with their pre-existing vote intention, whereas Independents respond closer to face value.

---

## 3. Data

* **Source.** Cooperative Election Study (CES) 2022 Common Content. Schaffner, Ansolabehere, & Shih (2023). Harvard Dataverse, [https://doi.org/10.7910/DVN/PR4L8P](https://doi.org/10.7910/DVN/PR4L8P).
* **Mode.** YouGov-administered online survey of 60,000 U.S. adults; `commonpostweight` provided for population post-stratification (we use the unweighted sample for *causal* estimation within strata, by design).
* **Analysis sample.** After restricting to registered voters (`votereg == 1`), keeping only respondents who reported a major-party House vote, dropping "Not sure" / "Other" categories on both the treatment and the confound, and stratified down-sampling for tractable PyMC sampling, **N = 8,000**.

### Variables in the causal model

| Role        | Symbol | Source variable             | Operationalization                                                                                                | Levels                                                |
|:------------|:------:|:----------------------------|:------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------|
| Treatment   |  $E$   | `CC22_302`                  | Retrospective national-economy perception (5-pt → collapsed to 3-pt)                                              | 1 = Better, 2 = Same, 3 = Worse                       |
| Outcome     |  $V$   | `CC22_367` + `HouseCand{i}Party` | Voted for the **incumbent (Democratic) party's** House candidate in 2022 (Bernoulli)                          | 0 = No, 1 = Yes                                       |
| Confound    |  $P$   | `pid3`                      | Self-reported partisan identification (3-pt)                                                                      | 1 = Democrat, 2 = Independent, 3 = Republican         |

We deliberately use the symbols **$E$, $V$, $P$** (mnemonic for **E**conomy, **V**ote, **P**arty) rather than the generic $T$, $Y$, $Z$ recommended against in the project template.

---

## 4. Causal Model (DAG)

```
                          P  (Party ID)
                          │  ╲
                          ▼   ╲
              E (Econ Perception) ──→ V (Vote Incumbent)
                            (causal effect of interest)
```

* $P \rightarrow E$: party identification colors economic perception (Bartels 2002; Bisgaard 2015).
* $P \rightarrow V$: party identification is the strongest single predictor of vote choice (Campbell et al. 1960; Bartels 2000).
* $E \rightarrow V$: the effect of interest — retrospective economic perception influences vote choice.

$P$ is a **fork** that opens the backdoor path $E \leftarrow P \rightarrow V$. We close it by stratifying the model on $P$ and post-stratifying when reporting the causal effect.

---

## 5. Statistical Model

We model the binary vote indicator with a Bernoulli likelihood and a logit link, indexing a $3 \times 3$ matrix of log-odds parameters by the categorical levels of treatment and confound. **This is the index-based encoding used throughout the course (e.g. UC Berkeley admissions example, Homework #5)**, NOT dummy/one-hot encoding.

$$
\begin{aligned}
V_i &\sim \text{Bernoulli}(p_i)\\
\text{logit}(p_i) &= \alpha_{E[i],\,P[i]}\\
\alpha_{e,p} &\sim \text{Normal}(0,\,1.5),\quad e\in\{1,2,3\},\;p\in\{1,2,3\}
\end{aligned}
$$

### 5.1 Outcome distribution

The outcome is a single 0/1 trial per respondent, so a **Bernoulli** likelihood is the unique exponential-family choice. The logit link maps the bounded probability $(0,1)$ to the unbounded real line for the linear predictor.

### 5.2 Prior choice and prior predictive simulation

The prior $\alpha_{e,p}\sim\text{Normal}(0,\,1.5)$ was chosen by **two complementary justifications**, both of which the project rubric requires:

1. **Domain knowledge.** Realistic subgroup vote-for-incumbent probabilities lie in $[0.05, 0.95]$, i.e. log-odds in $[-3, +3]$. A Normal(0, 1.5) prior places ~95% of mass in that range — informative enough to rule out absurd subgroup probabilities, flat enough not to bias the posterior toward 50:50.
2. **Prior predictive simulation.** Pushing draws from the prior through the inverse-logit yields an approximately *uniform* prior on the probability scale — i.e. the prior makes no a-priori claim about which cell votes one way or the other. A side-by-side comparison with $\text{Normal}(0, 0.3)$ (too tight: piles up at 0.5) and $\text{Normal}(0, 5)$ (too wide: piles up at 0/1) is included in the notebook to substantiate the choice (per McElreath 2020 §11.1.1).

### 5.3 How the confound is handled

Because $P$ is a fork in the DAG, conditioning on it blocks the only backdoor path from $E$ to $V$ (do-calculus rule 2). The interaction parameterization $\alpha_{e,p}$ stratifies the entire log-odds surface on $P$, and we report the causal effect by **post-stratification** — averaging the per-stratum probability contrast over the empirical distribution of $P$ in the data:

$$
\widehat{\text{ATE}}\;=\;\sum_{p}\widehat\pi_p\,
\bigl[\sigma(\alpha_{\text{Worse},p})-\sigma(\alpha_{\text{Better},p})\bigr]
$$

This is the textbook do-calculus identification of a causal effect under no-unmeasured-confounding-given-$P$.

---

## 6. Model Validation on Simulated Data

Before touching real data we (i) fixed ground-truth values for the 9-cell $\alpha$ matrix, (ii) generated a 6,000-row synthetic dataset from the assumed model, (iii) fit the PyMC model to the simulated data, and (iv) checked recovery using `arviz.plot_posterior(..., ref_val=...)`. Every one of the nine 89% HDIs covers its corresponding ground-truth parameter, with posterior means within ~0.1 log-odds of the truth in all cells. The model is therefore **identified** and **correctly specified** for the assumed data-generating mechanism.

---

## 7. Real-Data Fit and Diagnostics

We fit the model with PyMC's NUTS sampler (4 chains × 1,000 tune + 1,000 draws, target acceptance = 0.9, seeded). Sampling completed in ~15 seconds.

| Diagnostic                              | Threshold       | Observed                  | Pass? |
|:----------------------------------------|:----------------|:--------------------------|:-----:|
| $\widehat R$ (Gelman-Rubin)             | < 1.01          | All 9 params: 1.000-1.003 |   ✅  |
| Bulk ESS                                | > 400           | All 9 params: 6,300-8,000 |   ✅  |
| Divergent transitions                   | 0               | 0                         |   ✅  |
| Trace mixing (visual)                   | overlapping     | overlapping caterpillars  |   ✅  |

The posterior approximation is reliable.

### Posterior parameter estimates ($\alpha_{e,p}$, log-odds)

|              | Democrat (P=1) | Independent (P=2) | Republican (P=3) |
|:-------------|:---------------|:-------------------|:-----------------|
| Better (E=1) | +3.49          | +1.95              | −0.98            |
| Same   (E=2) | +3.53          | +2.31              | −1.54            |
| Worse  (E=3) | +3.05          | −0.53              | −3.30            |

(All HDIs of width ≤ 0.4. Read down a column to see the within-stratum effect of economic perception on vote choice; read across a row to see the partisan baseline.)

---

## 8. Posterior Predictive Checks

Two checks are included in the notebook:

1. **Cell-level PPC** (3×3 grid). For each $(E, P)$ cell we plot the observed vote-share (vertical line), the posterior distribution of the cell-level mean (89% HDI as a band), and the posterior-predictive distribution of new-voter shares (wider band). The observed share lies inside both bands in every cell.
2. **Marginal PPC** by economic perception. We plot the posterior of the post-stratified vote-share for each $E$ level (violin) against the observed marginal share (dot). The model exactly reproduces the empirical Better → Same → Worse gradient.

The Bernoulli + 3×3 interaction model is therefore an adequate description of the data.

---

## 9. Discussion

### 9.1 Answering the question

Yes. The full posterior distribution of the post-stratified ATE lies entirely below zero, with $\Pr(\text{ATE} < 0) = 1$ across 4,000 NUTS draws. Intervening to flip a voter's economic perception from *better* to *worse* — while holding the partisan composition of the electorate at its empirical 2022 mix — would, on average, **reduce the probability of voting for the incumbent (Democratic) House candidate by about 21 percentage points** (89% HDI: 18 to 23 points).

### 9.2 The role of the confound

Stratifying on $P$ revealed substantial **effect heterogeneity by partisan identity**:

| Stratum     | Within-stratum effect (Worse − Better) | 89% HDI            |
|:------------|:----------------------------------------|:-------------------|
| Democrat    | **−0.015**                              | [−0.027, −0.004]   |
| Independent | **−0.502**                              | [−0.542, −0.461]   |
| Republican  | **−0.240**                              | [−0.306, −0.173]   |

* **Independents** drive most of the aggregate effect — they are the most economy-responsive subgroup, consistent with the canonical role of independents as "swing voters."
* **Strong Democrats** are essentially insensitive (1.5-point shift), consistent with motivated-reasoning theory: they rationalize away negative economic information that conflicts with their pre-existing pro-incumbent disposition.
* **Republicans** show a large effect, but their baseline support for the Democratic incumbent is already very low, so the shift is smaller in absolute terms than the Independents'.

Had we **failed** to condition on party ID, the marginal association between $E$ and $V$ would conflate (a) the genuine economic-voting effect with (b) the perception-coloring channel by which Republicans both perceive the economy more negatively *and* vote against the incumbent. The interaction encoding plus post-stratification cleanly separates these channels.

### 9.3 Limitations and future work

1. **More confounders.** The 3-variable cap of this assignment forced us to absorb education, income, race, age, and district characteristics into the unmeasured residual. A natural extension is a multilevel model with these covariates.
2. **Reverse causation.** A small amount of $V \to E$ (rationalization) cannot be ruled out from cross-sectional data; a panel design would help.
3. **Pre-treatment vs post-treatment $P$.** Party ID is highly stable in adulthood (Green, Palmquist, Schickler 2002) but not perfectly so; a longitudinal extension would let us examine whether $P$ itself moves in response to $E$.
4. **District-level heterogeneity.** The current model pools across all districts; a multilevel extension with district-level intercepts (and competitiveness as a moderator) would estimate where the economic-voting mechanism is strongest.
5. **Alternative outcomes.** Replicating on Senate / gubernatorial vote choice would test the generalizability of the effect.
6. **Sensitivity analysis.** An E-value or Rosenbaum-bound style calculation would quantify how strong an unmeasured confounder would need to be to overturn the conclusion.

---

## 10. Reproducibility

```bash
# from the project directory
uv run --quiet --with nbformat python3 build_notebook.py        # rebuild .ipynb
uv run --quiet \
  --with "pymc>=5" --with arviz --with pandas --with matplotlib \
  --with seaborn --with scipy --with numpy --with networkx \
  --with jupyter --with ipykernel --with nbconvert --with nbformat \
  jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 FDS_Final_Project.ipynb
```

* All randomness is seeded (`random_seed = 20260501`).
* Total execution time on Apple-Silicon: ~85 s end-to-end.
* The raw 192 MB CES file is **not** redistributed; it is freely available at the Harvard Dataverse DOI above. Place it at `../dataverse_files/CCES22_Common_OUTPUT_vv_topost.csv` relative to the notebook.

---

## 11. References

* **Treatment → Outcome mechanism (independent of the dataset paper):**
  Healy, A. & Lenz, G. S. (2014). Substituting the End for the Whole: Why Voters Respond Primarily to the Election-Year Economy. *American Journal of Political Science*, 58(1), 31-47.
* **Meta-review:**
  Lewis-Beck, M. S. & Stegmaier, M. (2000). Economic determinants of electoral outcomes. *Annual Review of Political Science*, 3, 183-219.
* **Partisan-perception confounding channel:**
  Bartels, L. M. (2002). Beyond the Running Tally: Partisan Bias in Political Perceptions. *Political Behavior*, 24, 117-150.
  Bisgaard, M. (2015). Bias Will Find a Way: Economic Perceptions, Attribution of Blame, and Partisan-Motivated Reasoning during Crisis. *American Journal of Political Science*, 77(3), 849-860.
* **Stability of party ID:**
  Green, D., Palmquist, B., & Schickler, E. (2002). *Partisan Hearts and Minds.* Yale University Press.
* **Methodology:**
  McElreath, R. (2020). *Statistical Rethinking* (2nd ed.), §11. CRC Press.
  Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner, P.-C. (2021). Rank-Normalization, Folding, and Localization: An Improved $\widehat R$ for Assessing Convergence of MCMC. *Bayesian Analysis*, 16(2), 667-718.
* **Dataset (cited only as the data source, not as the treatment-effect reference, per project rules):**
  Schaffner, B., Ansolabehere, S., & Shih, M. (2023). *Cooperative Election Study Common Content, 2022*. Harvard Dataverse, [https://doi.org/10.7910/DVN/PR4L8P](https://doi.org/10.7910/DVN/PR4L8P).

---

## 12. Group Member Contributions

* **Proposal:** Member 1, Member 2, Member 3
* **Introduction:** Member 1
* **Causal Model + DAG:** Member 2
* **Statistical Model + Prior Predictive Simulation:** Member 3
* **Model Validation on Simulated Data:** Member 1, Member 2
* **Data Preparation:** Member 3
* **Posterior Model + Diagnostics:** Member 1
* **Posterior Predictive Checks:** Member 2
* **Discussion and Conclusion:** Member 3
* **Future Work:** Member 1, Member 2, Member 3
* **Final write-up assembly + formatting:** Member 1
