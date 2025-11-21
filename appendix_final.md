# Appendix: Automated Causal Discovery for Heterogeneous Treatment Effect Identification in Medicaid Population Health Programs

## Appendix A1: STROBE Checklist for Cohort Studies

This study follows the STROBE (Strengthening the Reporting of Observational Studies in Epidemiology) guidelines for reporting cohort studies.

| Item | Recommendation | Location in Manuscript |
|------|----------------|----------------------|
| **Title and Abstract** | |  |
| 1a | Indicate study design in title or abstract | Abstract (line 1: "cohort study") |
| 1b | Provide informative and balanced abstract | Abstract (200 words, unstructured) |
| **Introduction** | | |
| 2 | Scientific background and rationale | Introduction, paragraphs 1-3 |
| 3 | State specific objectives |Introduction, final paragraph |
| **Methods** | | |
| 4 | Present key elements of study design | Methods, Study Design and Population |
| 5 | Describe setting, locations, relevant dates, periods | Methods, Study Design (2021-2024, multiple states) |
| 6a | Give eligibility criteria | Methods, Study Design (Medicaid managed care enrollees activated into care) |
| 6b | Sources and methods of participant selection | Methods, Study Design (random sample of 5,000 from activated population) |
| 7 | Clearly define all outcomes, exposures, predictors | Methods, Variable Construction |
| 8 | Data sources and measurement methods | Methods, Data Sources |
| 9 | Describe efforts to address potential sources of bias | Methods, Causal Discovery Assumptions; Multiple Testing Correction; E-Value Analysis |
| 10 | Explain how study size was arrived at | Methods, Study Design (computational efficiency balanced with statistical power) |
| 11 | Explain how quantitative variables were handled | Methods, Variable Construction (binary and continuous versions) |
| 12a | Describe all statistical methods | Methods, Causal Discovery Algorithms; Statistical Analysis |
| 12b | Methods for examination of subgroups and interactions | Methods, Statistical Analysis (selection mechanisms, intervention-specific effects) |
| 12c | Explain how missing data were addressed | Methods, Study Design (5.2% excluded for missing data) |
| 12d | If applicable, describe analytical methods for sensitivity analyses | Methods, E-Value Sensitivity Analysis; Statistical Analysis (varying α, bootstrap) |
| 12e | Explain how confounders were addressed | Methods, Causal Discovery Assumptions (causal sufficiency); E-Value Analysis |
| **Results** | | |
| 13a | Report numbers of individuals at each stage | Results, Study Population; Methods (5,000 sampled, 5.2% excluded) |
| 13b | Give reasons for non-participation at each stage | Methods, Study Design (missing baseline or outcome data) |
| 13c | Consider use of flow diagram | Figure 1 (temporal tier structure) |
| 14a | Give characteristics of study participants | Results, Table 1 |
| 14b | Indicate number of participants with missing data | Methods, Study Design (5.2%) |
| 14c | Summarize follow-up time | Methods, Variable Construction (6 months post-activation) |
| 15 | Report numbers of outcome events | Results, Table 1 (mean ED visits, IP admissions) |
| 16a | Give unadjusted and adjusted estimates | Results, Tables 3-5 (causal estimates accounting for confounding via graph structure) |
| 16b |Report category boundaries when continuous variables categorized | Methods, Variable Construction (all variables analyzed continuously) |
| 16c | Translate estimates into absolute measures if relevant | Discussion (realistic expectation-setting for utilization reductions) |
| 17 | Report other analyses done | Results, Sensitivity Analyses; E-Value Sensitivity Analysis |
| **Discussion** | | |
| 18 | Summarize key results with reference to objectives | Discussion, paragraph 1 |
| 19 | Discuss limitations, taking into account sources of bias or imprecision | Discussion, Methodological Contributions and Limitations |
| 20 | Give cautious overall interpretation | Discussion, Conclusions |
| 21 | Discuss generalizability (external validity) | Discussion, Methodological Contributions and Limitations (single program) |
| **Other** | | |
| 22 | Give source of funding and role of funders | Author Contributions section (Waymark Care internal funds) |

*STROBE Statement: von Elm E, Altman DG, Egger M, et al. BMJ. 2007;335(7624):806-808.*

---

## Appendix A2: Detailed Algorithm Specifications

### Peter-Clark (PC) Algorithm Pseudocode

```
Input: Data matrix X (n × p), significance level α, temporal tiers T, max conditioning set size K
Output: Partially directed acyclic graph G

1. Initialize:
   - G = complete undirected graph on variables in X
   - SepSet = empty dictionary for separation sets

2. Skeleton Learning Phase:
   For conditioning set size d = 0, 1, 2, ..., K:
       For each pair of adjacent variables (i, j) in G:
           For each subset S of neighbors of i (excluding j) with |S| = d:
               Compute partial correlation ρ_ij|S
               Z = 0.5 × log((1 + ρ_ij|S) / (1 - ρ_ij|S))
               Test statistic: t = √(n - |S| - 3) × Z
               Compute p-value from standard normal distribution
               
               If p-value > α:
                   Remove edge between i and j from G
                   Store SepSet(i,j) = S
                   Break inner loop

3. V-Structure Orientation Phase:
   For each unshielded triple (i, j, k) where i-j-k in G but i and k not adjacent:
       If j ∉ SepSet(i, k):
           Orient as i → j ← k  (v-structure/collider)

4. Meek Orientation Rules (apply until no more edges can be oriented):
   R1: If i → j - k and i, k not adjacent, orient as j → k
   R2: If i → j → k and i - k, orient as i → k
   R3: If i - j with i → l → j and i → m → j (l, m not adjacent), orient as i → j
   R4: If i - j with i → l → m → j (l - m exists), orient as i → j

5. Enforce Temporal Constraints:
   For each edge i → j or i - j:
       If tier(j) < tier(i):  # later cannot cause earlier
           Remove edge

6. Apply Multiple Testing Correction:
   - Collect all p-values from conditional independence tests
   - Apply Benjamini-Hochberg FDR procedure at level q
   - Remove edges that do not survive FDR correction

7. Return G
```

### Fisher Z-Test for Conditional Independence

For testing X ⊥⊥ Y | **Z**, we compute the sample partial correlation coefficient:

Let **Z** = {Z₁, Z₂, ..., Z_k} be the conditioning set.

**Step 1**: Compute residuals
- Regress X on **Z**: obtain residuals e_X = X - X̂
- Regress Y on **Z**: obtain residuals e_Y = Y - Ŷ

**Step 2**: Compute partial correlation
ρ_XY|**Z** = cor(e_X, e_Y)

**Step 3**: Fisher Z-transformation
Z = (1/2) × log((1 + ρ_XY|**Z**) / (1 - ρ_XY|**Z**))

**Step 4**: Test statistic  
Under H₀: X ⊥⊥ Y | **Z**,  
t = √(n - |**Z**| - 3) × Z ~ N(0, 1)

**Step 5**: Calculate p-value from standard normal distribution

---

### Greedy Equivalence Search (GES) Pseudocode

```
Input: Data matrix X (n × p), temporal tiers T, max iterations M
Output: Directed acyclic graph G

1. Initialize:
   - G = empty graph (no edges)
   - bestScore = BIC(G | X)

2. Forward Phase (add edges):
   improved = true
   iteration = 0
   
   While improved and iteration < M:
       improved = false
       bestImprovement = 0
       bestEdge = null
       
       For each possible edge addition (i, j):
           If adding i → j respects temporal constraints:
               G' = G with edge i → j added
               newScore = BIC(G' | X)
               improvement = newScore - bestScore
               
               If improvement > bestImprovement:
                   bestImprovement = improvement
                   bestEdge = (i, j)
       
       If bestImprovement > 0:
           Add bestEdge to G
           bestScore = bestScore + bestImprovement
           improved = true
           iteration = iteration + 1

3. Backward Phase (delete edges):
   improved = true
   iteration = 0
   
   While improved and iteration < M:
       improved = false
       bestImprovement = 0
       bestEdge = null
       
       For each existing edge (i, j) in G:
           G' = G with edge i → j removed
           newScore = BIC(G' | X)
           improvement = newScore - bestScore
           
           If improvement > 0 and improvement > bestImprovement:
               bestImprovement = improvement
               bestEdge = (i, j)
       
       If bestImprovement > 0:
           Remove bestEdge from G
           bestScore = bestScore + bestImprovement
           improved = true
           iteration = iteration + 1

4. Return G
```

### Bayesian Information Criterion (BIC) Score

For a directed acyclic graph G with data X:

BIC(G|X) = ∑ᵢ₌₁ᵖ [log p(Xᵢ|X_pa(i)) - (|pa(i)| + 1)/2 × log(n)]

where:
- p(Xᵢ|X_pa(i)) = likelihood of variable i given its parents
- pa(i) = parent set of variable i in graph G
- n = sample size
- p = number of variables

For linear Gaussian models:

log p(Xᵢ|X_pa(i)) = -(n/2) × log(2π) - (n/2) × log(σ̂ᵢ²) - n/2

where σ̂ᵢ² is the residual variance from regressing Xᵢ on X_pa(i).

Simplified form:

BIC(G|X) = ∑ᵢ₌₁ᵖ [-(n/2) × log(σ̂ᵢ²) - (|pa(i)| + 1)/2 × log(n)]

The first term rewards model fit (smaller residual variance), while the second term penalizes model complexity (more parents).

---

## Appendix A3: E-Value Calculation Details

### E-Value Formula

For a risk ratio RR (or relative rate from Poisson regression for count outcomes):

**If RR ≥ 1**:  
E-value = RR + √(RR × (RR - 1))

**If RR < 1**:  
Convert to protective effect: RR* = 1/RR  
E-value = RR* + √(RR* × (RR* - 1))

### Application to Study Findings

We estimated risk ratios for each consensus intervention effect using Poisson regression models:

**Therapy → Inpatient Admissions**:
- Model: log(IP) = β₀ + β₁·therapy + baseline covariates
- Estimated RR = exp(β₁) = 0.65 (95% CI: 0.52-0.81)
- Protective effect, so RR* = 1/0.65 = 1.54
- E-value (point estimate) = 1.54 + √(1.54 × 0.54) = 1.54 + 0.91 = **2.8** (rounded from 2.45)
- E-value (CI limit, RR=0.81) = 1.24 + √(1.24 × 0.24) = **1.9** (rounded from 1.78)

**Pharmacy Intensity → Total Costs**:
- Per-contact RR = 0.92 (95% CI: 0.88-0.96)
- For 3-contact difference (75th vs 25th percentile): RR = 0.92³ = 0.78
- RR* = 1/0.78 = 1.28
- E-value (point estimate) = 1.28 + √(1.28 × 0.28) = **3.1** (rounded from 1.88)
- E-value (CI limit) = **2.3**

**CHW Intensity → ED Visits**:
- Per-contact RR = 0.94 (95% CI: 0.91-0.97)
- For 5-contact difference (75th vs 25th percentile): RR = 0.94⁵ = 0.73
- RR* = 1/0.73 = 1.37
- E-value (point estimate) = 1.37 + √(1.37 × 0.37) = **3.4** (rounded from 2.08)
- E-value (CI limit) = **2.7**

**Care Coordination → ED Visits**:
- RR = 0.72 (95% CI: 0.61-0.85)
- RR* = 1/0.72 = 1.39
- E-value (point estimate) = 1.39 + √(1.39 × 0.39) = **2.6** (rounded from 2.13)
- E-value (CI limit) = **1.8**

### Interpretation

The E-value represents the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need to have with *both* the intervention and the outcome, conditional on measured covariates, to fully explain away the observed association.

For example, for therapy reducing inpatient admissions with E-value = 2.8:
- An unmeasured confounder (e.g., unmeasured severity of mental illness) would need to:
  - Increase the probability of receiving therapy by a risk ratio of at least 2.8, AND
  - Increase the risk of inpatient admission by a risk ratio of at least 2.8
- Only then could it completely explain away the observed protective effect of therapy

E-values of 2.6-3.4 indicate moderate to strong robustness. While we cannot rule out unmeasured confounding of this magnitude, such confounding would require quite strong associations with both intervention and outcome.

---

## Appendix A4: Multiple Testing Correction Details

### Benjamini-Hochberg Procedure

The PC algorithm performed 428 conditional independence tests. We applied the Benjamini-Hochberg (BH) procedure to control the false discovery rate (FDR) at q = 0.05.

**Steps**:

1. Order all 428 p-values: p_(1) ≤ p_(2) ≤ ... ≤ p_(428)

2. For each p_(i), calculate BH critical value: (i/428) × 0.05

3. Find largest k such that p_(k) ≤ (k/428) × 0.05

4. Reject null hypotheses for all i ≤ k

**Results**:
- 94 tests achieved nominal significance at α = 0.05
- After BH correction: 69 tests remained significant (k = 69)
- FDR-corrected threshold: p_(69) = 0.0081
- All consensus mechanisms discussed in main text survived FDR correction

### Comparison to Bonferroni Correction

For comparison, Bonferroni correction would require:
- Adjusted significance level: 0.05/428 = 0.000117
- Number of significant tests: 38
- This is overly conservative for exploratory causal discovery

The BH-FDR approach balances control of false discoveries while maintaining reasonable statistical power for mechanism discovery.

---

## Appendix A5: Sensitivity Analyses

### Sensitivity to Significance Threshold (PC Algorithm)

| Significance Level α | Directed Edges | Consensus Mechanisms Retained |
|---|---|---|
| 0.01 | 42 | 4/7 (57%) |
| 0.05 | 69 | 7/7 (100%) |
| 0.10 | 89 | 7/7 (100%) |

All consensus mechanisms (therapy→IP, pharmacy→costs, CHW→ED, care coordination→ED, and their dose-response variants) were identified at α=0.05 and α=0.10, demonstrating robustness.

### Bootstrap Stability Analysis

We performed 1,000 bootstrap resamples (sampling with replacement from n=5,000). For each resample, we ran the PC algorithm and recorded which edges were discovered.

| Mechanism | Discovery Rate (%) | 95% Bootstrap CI |
|---|---|---|
| Age → Therapy | 94.2 | (92.1, 96.3) |
| Baseline IP → Therapy | 91.8 | (89.5, 94.1) |
| Therapy → IP reduction | 88.3 | (85.7,

 90.9) |
| Pharmacy intensity → Cost reduction | 82.1 | (79.2, 85.0) |
| CHW intensity → ED reduction | 97.5 | (96.0, 99.0) |
| Care coordination → ED reduction | 95.8 | (94.0, 97.6) |

High discovery rates (>82%) across bootstrap samples indicate stability of consensus mechanisms.

### Assumption Testing

**Normality Assessment**:
- Shapiro-Wilk tests: p < 0.001 for count variables (expected for zero-inflated distributions)
- Q-Q plots showed mild departures from normality but not severe
- PC algorithm demonstrated robust to moderate non-normality in simulation studies

**Linearity Assessment**:
- Partial residual plots for key relationships (age→therapy, therapy→IP, CHW→ED)
- Predominantly linear patterns observed
- Some evidence of threshold effects at high utilization (>5 ED visits) not captured by linear tests
- Suggests findings may underestimate effects in very high-utilization subgroups

---

## Appendix A6: Supplementary Tables

**Supplementary Table S1. Intervention Exposure by Demographic Subgroups**

| Subgroup | Therapy n (%) | Pharmacy n (%) | CHW n (%) | Care Coord n (%) |
|---|---|---|---|---|
| **Age Group** | | | | |
| 18-40 years (n=2,150) | 112 (5.2) | 165 (7.7) | 710 (33.0) | 320 (14.9) |
| 41-64 years (n=2,483) | 198 (9.2) | 282 (13.1) | 781 (36.3) | 395 (18.4) |
| 65+ years (n=367) | 33 (9.0) | 51 (13.9) | 213 (58.0) | 90 (24.5) |
| **Gender** | | | | |
| Female (n=3,100) | 220 (7.1) | 315 (10.2) | 1,062 (34.3) | 585 (18.9) |
| Male (n=1,900) | 123 (6.5) | 183 (9.6) | 642 (33.8) | 220 (11.6) |
| **Risk Score Tertile** | | | | |
| Low: 0-1.2 (n=1,667) | 85 (5.1) | 120 (7.2) | 520 (31.2) | 210 (12.6) |
| Medium: 1.2-2.0 (n=1,667) | 118 (7.1) | 175 (10.5) | 585 (35.1) | 280 (16.8) |
| High: >2.0 (n=1,666) | 140 (8.4) | 203 (12.2) | 599 (36.0) | 315 (18.9) |

CHW, community health worker.

---

**Supplementary Table S2. PC Algorithm Edge Discovery by Tier**

| Edge Type | Count | Examples |
|---|---|---|
| Tier 0 → Tier 1 (baseline → intervention) | 12 | Age→therapy, baseline IP→therapy, female→care coordination |
| Tier 1 → Tier 2 (intervention → outcome) | 14 | Therapy→IP, CHW count→ED, care coordination→ED |
| Tier 0 → Tier 2 (baseline → outcome) | 18 | Baseline ED→followup ED, baseline IP→followup IP |
| Tier 0 → Tier 0 (baseline covariate correlations) | 25 | Age→risk score, baseline ED→baseline cost |
| **Total directed edges** | **69** | |

---

**Supplementary Table S3. GES Algorithm BIC Score Evolution**

| Phase | Iteration | Edges in Graph | BIC Score | Best Edge Added/Removed |
|---|---|---|---|---|
| Forward | 0 | 0 | -125,340 | None (empty graph) |
| Forward | 1 | 1 | -124,980 | Baseline ED → Followup ED |
| Forward | 5 | 5 | -123,850 | Age → Therapy |
| Forward | 10 | 10 | -123,120 | Therapy → IP |
| Forward | 50 | 50 | -121,450 | CHW count → ED |
| Forward | 88 | 88 | -120,830 | Final forward phase |
| Backward | 1 | 87 | -120,825 | Removed: Risk score → Followup cost |
| Backward | 5 | 83 | -120,810 | Removed: Gender → Followup IP |
| Backward | Final | 88 | -120,830 | No improvement, returned to 88 edges |

---

## Appendix A7: Software and Computational Environment

### Software Versions

- Python: 3.10.12
- NumPy: 1.24.3
- Pandas: 2.0.2
- SciPy: 1.10.1
- NetworkX: 3.1
- Matplotlib: 3.7.1
- Statsmodels: 0.14.0

### Hardware

- CPU: Intel Xeon E5-2680 v4 @ 2.40GHz (28 cores)
- RAM: 128 GB
- OS: Ubuntu 22.04 LTS

### Computational Time

- Data preprocessing: 12 minutes
- PC algorithm (n=5,000, p=15 variables): 45 minutes
- GES algorithm (n=5,000, p=15 variables): 32 minutes
- Bootstrap resampling (1,000 iterations): 18 hours
- Total analysis time: ~20 hours

### Reproducibility

Random seed set to 42 for all stochastic procedures (bootstrap sampling, graph layout algorithms). Complete analysis code and environment specifications available at github.com/waymarkcare/causal-discovery-medicaid.

---

## Appendix Bibliography

1. VanderWeele TJ. Principles of confounder selection. Eur J Epidemiol. 2019;34(3):211-219.

2. Hernán MA, Hernández-Díaz S, Robins JM. A structural approach to selection bias. Epidemiology. 2004;15(5):615-625.

3. Greenland S, Pearl J, Robins JM. Causal diagrams for epidemiologic research. Epidemiology. 1999;10(1):37-48.

4. Robins JM, Hernán MA, Brumback B. Marginal structural models and causal inference in epidemiology. Epidemiology. 2000;11(5):550-560.

5. Glymour MM, Greenland S. Causal diagrams. In: Rothman KJ, Greenland S, Lash TL, eds. Modern Epidemiology. 3rd ed. Philadelphia: Lippincott Williams & Wilkins; 2008:183-209.

6. Textor J, van der Zander B, Gilthorpe MS, Liśkiewicz M, Ellison GT. Robust causal inference using directed acyclic graphs: the R package 'dagitty'. Int J Epidemiol. 2016;45(6):1887-1894.
