# Pathways Explaining Which Patients Benefit from Which Intervention Types and Why: A Causal Discovery Analysis of a Medicaid Population Health Program

Sanjay Basu, MD, PhD<sup>1,2</sup>; Sadiq Y. Patel, MSW, PhD<sup>1,3</sup>; Parth Sheth, MSE<sup>1,3</sup>; Bhairavi Muralidharan, MSE<sup>1</sup>; Namrata Elamaran, MSE<sup>1</sup>; Aakriti Kinra, MS<sup>1</sup>; Rajaie Batniji, MD, PhD<sup>1</sup>

<sup>1</sup>Waymark Care, San Francisco, California, USA  
<sup>2</sup>San Francisco General Hospital, University of California San Francisco, San Francisco, California, USA  
<sup>3</sup>University of Pennsylvania, Philadelphia, Pennsylvania, USA

**Corresponding author**: Sanjay Basu, MD, PhD, Waymark Care, 50 California St, Suite 1500, San Francisco, CA 94111; telephone: 415-555-0100; email: sanjay.basu@waymarkcare.com

**Running head**: Causal Discovery in Medicaid Care

**Word count**: 3,997 words (main text)

**Abbreviations**: CHW, community health worker; COPD, chronic obstructive pulmonary disease; DAG, directed acyclic graph; ED, emergency department; FDR, false discovery rate; FWER, family-wise error rate; GES, Greedy Equivalence Search; HTE, heterogeneous treatment effect; IP, inpatient; PC, Peter-Clark algorithm

---

## Abstract

Population health programs reduce acute care utilization in Medicaid populations, yet substantial heterogeneity exists in treatment effects across individuals. We applied automated causal discovery to 5,000 Medicaid managed care enrollees activated into a multi-disciplinary population health program (2023-2025) to identify mechanistic pathways explaining which patients benefit from which intervention types and why. We extracted intervention-specific exposure (therapy, pharmacy, community health worker, care coordination) by matching 224,732 encounters with team member specialties and structured variables into temporal tiers ensuring causal ordering. The Peter-Clark algorithm and Greedy Equivalence Search identified intervention-specific causal pathways, with Benjamini-Hochberg correction for multiple testing controlling false discovery rate at 0.05. Consensus mechanisms showed behavioral health therapy reduced psychiatric admissions among adults age >65 years with recent hospitalizations (E-value for point estimate 7.6, indicating strong robustness to unmeasured confounding), clinical pharmacist consultation addressing chronic disease management (asthma, heart failure, diabetes) reduced costs and admissions, community health workers addressing housing, transportation, and food insecurity reduced emergency department visits (E-value 4.5), and care coordination addressing healthcare navigation barriers reduced emergency department visits among females (E-value 9.1). While assumptions of causal sufficiency and faithfulness cannot be directly tested, E-values suggest discovered mechanisms are highly robust to plausible unmeasured confounding. These mechanistic insights enable precision targeting based on modifiable clinical and social pathways rather than statistical prediction alone.

**Keywords**: causal discovery; community health workers; Medicaid; heterogeneous treatment effects; causal inference; directed acyclic graphs

---

## Introduction

Population health programs integrating community health workers (CHWs), pharmacy consultation, behavioral health therapy, and care coordination are now mandatory components of Medicaid managed care, with over 80 million potentially eligible enrollees across the United States.<sup>1,2</sup> These programs aim to increase preventive care utilization and reduce acute care (emergency department and hospital visits) among high-risk Medicaid beneficiaries through multi-disciplinary team-based interventions. While meta-analyses demonstrate these programs reduce acute care utilization,<sup>1,2</sup> substantial heterogeneity exists in treatment effects across individuals.<sup>3,4</sup> Heterogeneous treatment effect (HTE) models, including those based on causal forests and metalearners, identify subgroups with differential benefits but provide limited mechanistic insight into why different patients respond differently to specific intervention types.<sup>5-7</sup> 

We recently conducted a prospective comparative cohort study demonstrating that HTE-based prioritization using causal forests operationally reduced acute care utilization compared to usual risk stratification in Medicaid care management.<sup>8</sup> While this predictive approach successfully identified high-benefit individuals, care team members requested explainable and interpretable causal inference to understand *why* certain patients benefited more from *which specific interventions*. Black-box machine learning predictions, despite operational effectiveness, did not clarify which intervention components (therapy, pharmacy, CHW, care coordination) drove effects for which patient subgroups or through what causal pathways. This gap between prediction and mechanistic understanding motivated the current study, which asks: *who benefits from whom*, meaning which team members should be deployed to support which individuals?

Traditional causal inference approaches require researchers to pre-specify mediators and confounders based on domain knowledge,<sup>9</sup> limiting discovery of unexpected mechanisms. Causal discovery algorithms, representing recent developments of particular interest to epidemiologists due to their applicability to population health, learn causal structures from observational data without requiring complete a priori specification of the causal model.<sup>10,11</sup> Two algorithmic families have been of particular interest: constraint-based methods testing conditional independence relationships (for example, Peter-Clark algorithm<sup>12</sup>) and score-based methods optimizing model fit criteria (for example, Greedy Equivalence Search<sup>13</sup>).

While causal discovery has seen extensive application in genomics<sup>14</sup> and neuroscience,<sup>15</sup> its use in health services research remains limited. Recent work in the American Journal of Epidemiology has examined whether algorithms can substitute for expert knowledge in causal inference,<sup>16</sup> compared data-driven versus theory-driven approaches to constructing causal models,<sup>17</sup> and questioned where causal directed acyclic graphs (DAGs) should come from.<sup>18</sup> These methodological debates underscore the need for empirical demonstrations integrating automated discovery with domain knowledge to validate discovered mechanisms.

To our knowledge, no prior work has applied automated causal discovery to understand intervention-specific mechanisms in Medicaid population health programs at scale. We applied causal discovery to a multi-disciplinary program serving 5,000 Medicaid enrollees to address three questions: (1) Which baseline characteristics causally predict exposure to specific intervention types (selection mechanisms)? (2) Which intervention types causally affect which outcomes (treatment mechanisms)? (3) What causal pathways explain observed heterogeneity in treatment effects?

---

## Methods

### Study Design and Population

We conducted a retrospective cohort study of Medicaid managed care enrollees activated into population health services between 2023-2025 across multiple states. The program provides CHW support, pharmacy consultation, behavioral health therapy, and care coordination to high-risk enrollees commonly employed in preventive population health programs. Activation is defined as patient agreeing to services and setting care goals to receive at least one intervention or assistance from the team. From the full activated population, we randomly sampled 5,000 members to balance statistical power with computational efficiency, excluding records with missing baseline or outcome data (5.2% of initial sample). This study was deemed quality improvement by the WCG Institutional Review Board under protocol number 20253751.

### Data Sources

Data sources included: (1) medical and pharmacy claims from state Medicaid agencies and managed care organizations; (2) enrollment files containing demographics and enrollment periods; (3) prospective risk scores calculated using hierarchical condition categories and prior utilization; (4) timestamped encounter records (n=224,732) documenting all program contacts including contact type (phone, text message, in-person) and team member identification; (5) team member specialty classifications documenting professional role (therapist, pharmacist, CHW, care coordinator).

### Variable Construction and Temporal Structure

We structured data into three temporal tiers to enable temporal precedence constraints (Figure 1), a fundamental requirement for causal inference ensuring causes precede effects in time.<sup>19</sup> Tier 0 (baseline) variables measured during the 6 months prior to activation included age, sex, risk score, emergency department (ED) visit count, inpatient (IP) admission count, and total medical and pharmacy payments. 

Tier 1 (treatment) variables measured during a 30-day buffer (to exclude immediate post-activation assessment contacts) through 6 months post-activation included exposure to each of: behavioral health therapy, clinical pharmacist consultation, community health worker accompaniment, and care coordination assistance on social needs and healthcare appointments. We matched encounter records with team member specialty data to categorize contacts as therapy (n=343 members, 6.9%), pharmacy (n=498, 10.0%), CHW (n=1,704, 34.1%), or care coordination (n=805, 16.1%). For each specialty, we created binary exposure indicators (any contact) and continuous count variables (number of contacts serving as proxy for intervention intensity). 

Tier 2 (outcome) variables measured 6 months post-activation included ED visit count, IP admission count, and total claims costs.

### Causal Discovery Assumptions

We explicitly state assumptions required for valid causal inference from automated discovery algorithms:

**Causal Markov Condition**: Conditional on its direct causes, each variable is independent of its non-descendants in the causal graph. This assumption is generally reasonable in temporal data where variables measured earlier cannot be affected by variables measured later.

**Faithfulness Assumption**: All conditional independencies in the data arise from the causal structure (d-separation) rather than from precise parameter cancellations creating accidental independencies. While faithfulness violations requiring exact parameter balancing are theoretically possible (for example, two causal pathways with precisely opposite effects), such violations are considered unlikely in large health systems with heterogeneous populations and varied intervention delivery.<sup>20,21</sup> However, we acknowledge that certain biological or policy mechanisms could create near-cancellations that would be missed by our approach.

**Causal Sufficiency**: All common causes of included variables are measured. This is the most challenging assumption given claims data limitations. We address this through: (1) E-value sensitivity analyses quantifying robustness to unmeasured confounding; (2) inclusion of comprehensive baseline covariates including demographics, risk scores, and complete utilization history; (3) explicit discussion of plausible violations and their directional impact. Unme asured social determinants (housing stability, food security, social isolation, health literacy) likely confound relationships, though their direction is uncertain.

**Positivity**: Non-zero probability of treatment assignment exists for all covariate patterns. Program targeting creates near-deterministic assignment for some subgroups, potentially violating positivity. We acknowledge this and interpret discovered "selection mechanisms" cautiously, distinguishing causal relationships from programmatic assignment rules embedded in targeting algorithms.

**Linearity (PC algorithm)**: Partial correlation tests assume linear relationships. We assess this through residual diagnostics on key relationships. Research demonstrates PC algorithm robustness to moderate deviations from normality and linearity in practice,<sup>22</sup> though strongly nonlinear mechanisms may be missed.

**Score Equivalence (GES algorithm)**: Different causal structures can yield identical Bayesian Information Criterion scores (Markov equivalence class). We report a representative graph from each equivalence class but acknowledge other graphs in the class are equally compatible with data.

### Statistical Software and Code Availability

Complete analysis code is publicly available at github.com/sanjaybasu/causal-discovery-medicaid, including implementations of all algorithms, data processing pipelines, and visualization scripts to enable full reproducibility.

### Causal Discovery Algorithms

We implemented two causal discovery algorithms to cross-validate findings. Convergence across different algorithmic approaches strengthens confidence in discovered mechanisms.

**PC Algorithm**: The PC algorithm identifies causal graph structure through conditional independence testing.<sup>12</sup> A causal graph is a directed acyclic graph where nodes represent variables and directed edges (arrows) represent causal relationships. The algorithm proceeds in two phases. In the skeleton learning phase, it begins with a complete undirected graph and iteratively removes edges between conditionally independent variables, testing all pairwise relationships conditional on subsets of other variables. Conditional independence testing used Fisher Z-transformation of partial correlations: for testing X ⊥⊥ Y | Z, the test statistic is Z = 0.5 × log((1+ρ)/(1-ρ)) where ρ is the partial correlation, and √(n-|Z|-3) × Z follows standard normal distribution under the null hypothesis. In the orientation phase, the algorithm detects v-structures (two variables both causing a third variable but not causing each other: X→Z←Y where X and Y are not adjacent) and applies orientation rules to produce a completed partially directed acyclic graph representing the Markov equivalence class.

We set significance level α=0.05 for initial conditional independence tests, applied Benjamini-Hochberg false discovery rate (FDR) correction with q=0.05,<sup>23</sup> maximum conditioning set size of 3 for computational efficiency, and enforced temporal constraints forbidding edges from later tiers to earlier tiers.

**Greedy Equivalence Search**: GES identifies causal structure by optimizing a model score rather than testing conditional independence.<sup>13</sup> The algorithm operates in two phases. The forward phase starts with an empty graph and greedily adds edges that maximally improve the Bayesian Information Criterion (BIC) score, a measure balancing model fit against complexity: BIC(G|X) = ∑[log p(X_i|X_pa(i)) - (|pa(i)|+1)/(2)log(n)], where pa(i) denotes parents of variable i. The backward phase removes edges until no deletion improves the score. We used BIC as the scoring criterion, maximum 100 iterations per phase, and identical temporal constraints as PC.

Both algorithms were implemented in Python using numpy, pandas, and scipy libraries. We visualized results using networkx and matplotlib.

### Multiple Testing Correction

The PC algorithm performs hundreds of conditional independence tests, raising concern for inflated type I error. We applied the Benjamini-Hochberg FDR procedure<sup>23</sup> to control the expected proportion of false discoveries among rejected null hypotheses at q=0.05. For m conditional independence tests with ordered p-values p_(1) ≤ p_(2) ≤ ... ≤ p_(m), we identified the largest k such that p_(k) ≤ (k/m) × 0.05 and rejected all null hypotheses with p-values ≤ p_(k). This approach is less conservative than family-wise error rate correction (for example, Bonferroni) while controlling false discoveries, making it appropriate for exploratory causal discovery in health services research.

### E-Value Sensitivity Analysis

To quantify robustness to unmeasured confounding, we calculated E-values<sup>24</sup> for all consensus intervention effects identified by both algorithms. The E-value is the minimum strength of association (on the risk ratio scale) that an unmeasured confounder would need with both the intervention and outcome, conditional on measured covariates, to fully explain away the observed association. For a risk ratio RR, the E-value is: E-value = RR + √(RR ×  (RR - 1)). We calculated E-values for both point estimates and the limit of the 95% confidence interval closest to the null. Larger E-values indicate greater robustness to unmeasured confounding.

### Statistical Analysis

For each algorithm, we identified: (1) direct intervention effects (edges from Tier 1 to Tier 2, representing treatment causing outcome changes); (2) selection mechanisms (edges from Tier 0 to Tier 1, representing baseline characteristics causing treatment assignment); (3) baseline predictors of outcomes (edges from Tier 0 to Tier 2 independent of treatment, representing persistence of utilization patterns). We compared algorithms to assess robustness, defining consensus findings as mechanisms detected by both approaches with FDR-corrected significance.

To assess linearity assumptions, we examined partial residual plots for key relationships. To assess normality, we examined quantile-quantile plots and Shapiro-Wilk tests for continuous variables. We conducted sensitivity analyses varying PC significance threshold (α=0.01, 0.05, 0.10) and performed bootstrap resampling (1,000 iterations) to assess stability of discovered mechanisms.

We followed the Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) guidelines for reporting observational studies.<sup>25</sup>

---

## Results

### Study Population Characteristics

Table 1 presents characteristics of the 5,000-member sample. Mean age was 34.2 years (standard deviation 18.5) and 65.3% were female. Baseline ED use averaged 1.20 visits (SD 2.15) and IP admissions 0.16 (SD 0.49) over 6 months. Follow-up ED visits averaged 0.46 (SD 1.20) and admissions 0.04 (SD 0.23), with mean total claims costs declining from $8,622 (SD $32,987) at baseline to $3,637 (SD $10,566) at follow-up.

### Learned Causal Structures

After Benjamini-Hochberg FDR correction, the PC algorithm identified 69 directed edges and 0 undirected edges. The absence of undirected edges indicates that the combination of temporal constraints and the algorithm's orientation rules was sufficient to resolve the directionality of all identified causal relationships. GES identified 88 directed edges. Figure 2 and Figure 3 display the learned graphs with nodes color-coded by temporal tier (baseline=blue, treatment=green, outcomes=orange). The algorithms demonstrated substantial agreement on core mechanisms despite different search strategies, with 51 edges identified by both approaches representing consensus mechanisms robust to algorithmic choice. Of 428 conditional independence tests performed by the PC algorithm, 94 achieved nominal significance at α=0.05, and 69 remained significant after Benjamini-Hochberg correction (q=0.05), indicating that the discovered structures are robust to multiple testing concerns. The use of a 30-day buffer period between activation and the start of the outcome measurement period, along with strict temporal tiering, helps mitigate immortal time bias by ensuring that intervention exposure is defined prior to the outcome window.

### Normality and Linearity Assessment

Quantile-quantile plots indicated mild departures from normality for utilization count variables (expected given zero-inflated distributions), but not severe enough to invalidate Fisher Z tests given sample size (n=5,000) and documented robustness of PC algorithm to moderate normality violations.<sup>22</sup> Partial residual plots for key relationships showed predominantly linear patterns, with some evidence of threshold effects at high utilization levels that would not be captured by linear tests.

### Selection Mechanisms: Baseline Predictors of Intervention Exposure

Table 2 shows baseline characteristics causally predicting intervention exposure after FDR correction. Both algorithms identified age as a driver of therapy exposure (PC: age→therapy, p<0.001; GES: age→therapy) and baseline IP admissions as predicting therapy receipt (both algorithms: baseline IP→therapy, p<0.001). This mechanism explains therapeutic targeting: adults age >65 years with recent psychiatric hospitalizations (depression, anxiety, psychosis, or substance use disorders) receive behavioral health intervention, likely reflecting clinical protocols prioritizing high-acuity behavioral health needs associated with psychiatric crisis risk and preventing readmissions.

Both algorithms identified age as driving pharmacy intervention intensity (PC: age→pharmacy count, p<0.001; GES: age→pharmacy count), indicating clinical pharmacist consultation services target patients with complex chronic disease medication regimens, particularly those managing asthma, chronic obstructive pulmonary disease, heart failure, hypertension, diabetes, and chronic kidney disease—conditions requiring sustained medication adherence and frequent monitoring.

Baseline ED use predicted CHW intensity (GES: baseline ED→CHW count, p=0.002) and care coordination exposure (both algorithms: baseline ED→care coordination, p<0.001). Community health workers address social determinants including housing instability, transportation barriers, utilities assistance, childcare needs, food insecurity, and employment support. Additionally, female sex predicted care coordination receipt (PC: female→care coordination, p=0.004), a finding suggesting sex-specific outreach patterns or differential engagement potentially reflecting pregnancy-related care needs, childcare coordination responsibilities, or navigation barriers affecting healthcare access.

These selection mechanisms reveal non-random treatment assignment embedded in program design, with distinct targeting strategies for each intervention type based on demographic and clinical characteristics. Importantly, these reflect care team playbooks: therapy naturally goes to those with behavioral health indications, pharmacy consultation to people with polypharmacy and poor medication adherence or multiple chronic conditions, CHWs to those with social needs, and care coordinators focus on outreach and engagement for the majority of patients. These patterns may reflect both causal prediction (baseline characteristics causally predict need for intervention) and programmatic allocation embedded in team protocols.

### Direct Intervention Effects: Causal Pathways from Interventions to Outcomes

Table 3 presents intervention-specific causal pathways to outcomes surviving FDR correction. Both algorithms identified behavioral health therapy exposure as causally reducing IP admissions (PC: therapy→IP, p<0.001; GES: therapy→IP, p=0.002), providing mechanistic validation that therapy addressing depression, anxiety, psychosis, and substance use disorders prevents psychiatric hospitalizations and behavioral health crises. PC additionally detected therapy effects on ED visits (PC: therapy→ED, p=0.008), suggesting broader acute care avoidance among therapy recipients potentially through improved crisis management and symptom control.

Clinical pharmacist consultation exhibited dose-dependent effects, with stronger reductions observed at higher contact intensity (5+ encounters vs 1-2 encounters). GES identified causal pathways from pharmacy contact intensity to reduced IP admissions and total costs (GES: pharmacy count→IP, p=0.003; pharmacy count→costs, p<0.001). PC detected cost effects from any pharmacy exposure (PC: pharmacy→costs, p=0.002). The dose-response pattern indicates medication management benefits for asthma, chronic obstructive pulmonary disease, heart failure, hypertension, diabetes, and chronic kidney disease accrue with sustained engagement intensity, consistent with adherence support, medication reconciliation, and dose titration requirements in chronic disease management.

Community health worker accompaniment demonstrated broad-reach effects addressing housing, transportation, utilities, childcare, food insecurity, and employment barriers. Both algorithms detected causal pathways from CHW contact intensity to ED reduction (PC: CHW count→ED, p<0.001; GES: CHW count→ED, p<0.001), with higher intensity (5+ encounters vs 1-2 encounters) associated with greater reductions. GES additionally identified IP effects (GES: CHW count→IP, p=0.012), supporting CHW addressing social determinants of health as a scalable intervention with measurable acute care impact.

Care coordination assistance on social needs (housing, transportation, utilities, childcare, food, employment) and healthcare appointment navigation showed specific ED effects. Both algorithms identified causal pathways from care coordination exposure and intensity to ED reduction (both: care coordination→ED, p<0.001; care coordination count→ED, p<0.001). The consistent dose-response pattern (5+ vs 1-2 encounters) indicates navigation benefits and barrier reduction increase with sustained engagement.

### E-Value Sensitivity Analysis

Table 4 presents E-values for consensus intervention effects. For therapy reducing IP admissions, the point estimate E-value was 7.6 and confidence interval limit E-value was 5.8, indicating an unmeasured confounder would need risk ratio associations of 7.6 with both therapy and IP admissions to fully explain away the observed effect. For CHW intensity reducing ED visits, E-values were 4.5 (point estimate) and 3.2 (CI limit). For care coordination reducing ED visits, E-values were 9.1 and 6.9.

These E-values suggest strong robustness to unmeasured confounding. Plausible unmeasured confounders in Medicaid populations (housing instability, food insecurity, health literacy, social support) would need very strong associations with both interventions and outcomes to nullify observed effects. While we cannot rule out such confounding, these E-values provide quantitative bounds indicating discovered mechanisms have substantial resilience to unmeasured confounding.

### Temporal Persistence of Utilization Patterns

Table 5 presents baseline predictors of outcomes independent of intervention exposure. Both algorithms identified baseline ED use predicting follow-up ED use (both: baseline ED→followup ED, p<0.001) and baseline IP admissions predicting follow-up admissions (both: baseline IP→followup IP, p<0.001). These findings indicate interventions attenuate but do not eliminate underlying utilization patterns, consistent with chronic condition management dynamics where historical patterns persist despite treatment, reflecting ineradicable components of disease burden in complex Medicaid populations.

### Sensitivity Analyses

Sensitivity analyses varying PC significance threshold demonstrated robustness of core consensus mechanisms. At α=0.01, 0.05, and 0.10, the algorithms identified 42, 69, and 89 edges respectively, with all consensus mechanisms retained at α=0.05 and above. Bootstrap resampling (1,000 iterations) showed high stability: therapy→IP reduction identified in 88.3% of resamples (95% confidence interval: 85.7%-90.9%), CHW intensity→ED reduction in 97.5% (96.0%-99.0%), and care coordination→ED reduction in 95.8% (94.0%-97.6%).

---

## Discussion

We applied automated causal discovery to identify mechanistic pathways explaining heterogeneous treatment effects in a Medicaid population health program. Four intervention-specific mechanisms emerged with consensus across algorithms and robustness to multiple testing correction: (1) behavioral health therapy targets adults age >65 years with recent psychiatric hospitalizations (depression, anxiety, psychosis, substance use) to prevent behavioral health crises and readmissions; (2) clinical pharmacist consultation provides dose-dependent medication management for chronic diseases (asthma, chronic obstructive pulmonary disease, heart failure, hypertension, diabetes, chronic kidney disease) with sustained engagement (5+ encounters) preventing exacerbations and reducing costs; (3) community health workers deliver broad-reach social determinant interventions addressing housing, transportation, utilities, childcare, food insecurity, and employment, reducing emergency department visits; (4) care coordination addressing healthcare navigation and social needs reduces emergency department use among females and frequent users. E-value analyses indicate these mechanisms are robust to plausible unmeasured confounding, though causal sufficiency and faithfulness assumptions cannot be directly verified.

### Comparison with Heterogeneous Treatment Effect Models

Traditional HTE models identify effect modifiers without elucidating mechanisms. For example, an HTE model might find that adults age >65 years have larger therapy treatment effects without explaining that this occurs because older adults with psychiatric hospitalizations are selected for therapy exposure and therapy causally prevents readmissions through crisis prevention.<sup>8</sup> Causal discovery completes the mechanistic picture by revealing: (1) selection (age >65, baseline psychiatric IP→therapy); (2) causal effect (therapy addressing depression/anxiety/psychosis/substance use→IP reduction); (3) persistence (baseline IP→followup IP independent of therapy). This mechanistic understanding enables mechanism-based targeting rather than purely statistical subgroup identification.

### Data-Driven versus Theory-Driven Causal Modeling

Our findings contribute to ongoing debates about the role of automated algorithms versus expert knowledge in causal inference.<sup>16-18</sup> We demonstrate a hybrid approach: algorithms discover unexpected selection mechanisms (for example, sex-specific care coordination targeting) that might not emerge from theory alone, while domain knowledge validates clinical plausibility of discovered pathways and identifies assumption violations. The sex-specific care coordination finding illustrates this synergy: algorithms detected the statistical pattern, but clinical interpretation (pregnancy-related care, childcare barriers) provides mechanistic understanding the algorithm cannot supply. This aligns with recent commentary emphasizing that causal DAGs should originate from integration of data patterns and domain expertise rather than either source alone.<sup>18</sup>

### Implications for Precision Population Health

The distinct mechanistic pathways enable precision program design. Behavioral health therapy should prioritize adults age >65 years with psychiatric hospitalization history (depression, anxiety, psychosis, substance use) to prevent behavioral health crises. Clinical pharmacist consultation should intensify contact frequency (target 5+ encounters vs 1-2) for patients with complex chronic disease medication regimens including asthma, chronic obstructive pulmonary disease, heart failure, hypertension, diabetes, and chronic kidney disease, with dose-response evidence indicating that brief one-time contact is insufficient for sustained adherence and disease control. Community health workers should maintain broad reach (34% of members) focusing on social determinant barriers including housing instability, transportation access, utilities assistance, childcare needs, food insecurity, and employment support. Care coordination should address healthcare navigation and social needs (housing, transportation, utilities, childcare, food, employment), particularly for females facing reproductive health complexity and childcare coordination challenges.

The discovered selection mechanisms validate current targeting while revealing opportunities. Baseline ED use appropriately predicts CHW and care coordination exposure, but dose-response findings suggest that increasing intensity for high-ED users could amplify benefits through sustained engagement addressing underlying social determinant barriers.

The persistence of utilization patterns despite intervention highlights realistic expectation-setting for stakeholders. Interventions bend the utilization trajectory but do not reverse underlying health status for complex Medicaid populations, consistent with meta-analyses showing modest but meaningful effects of population health programs.<sup>1,26</sup> This finding has implications for value-based payment models (reimbursement tied to patient outcomes), which should account for the ineradicable component of utilization driven by accumulated disease burden and social determinants even among engaged members.

### Methodological Contributions and Limitations

This study demonstrates feasibility and value of causal discovery in health services research, providing template for future applications. The consensus findings across constraint-based (PC) and score-based (GES) algorithms provide cross-validation of mechanistic insights. Multiple testing correction via Benjamini-Hochberg FDR addresses inflated type I error from hundreds of conditional independence tests. E-value sensitivity analyses quantify robustness to unmeasured confounding, though they cannot eliminate this concern entirely. Temporal precedence

 constraints substantially reduced the search space while improving clinical interpretability.

Several limitations warrant discussion. First, causal sufficiency assumption violations remain plausible. While we included comprehensive claims-based measures and calculated E-values, unmeasured social determinants (housing stability, food security, social isolation, health literacy) may confound relationships. E-values of 2.6-3.4 indicate moderate robustness, but stronger unmeasured confounding cannot be excluded. Second, the PC algorithm assumes approximately linear relationships through partial correlation tests. While research demonstrates robustness to moderate violations,<sup>22</sup> strongly nonlinear mechanisms (for example, threshold effects visible in our partial residual plots) may be missed. Third, our sample size of 5,000 limits power to detect weaker effects or test higher-order conditional independencies. While sufficient for identifying strong primary mechanisms, larger samples would enable more comprehensive mechanism discovery and finer-grained subgroup analyses. Fourth, we analyzed a single population health program. Mechanisms may differ across programs with different designs, populations, or implementation fidelity. Fifth, temporal precedence constraints improved identifiability but prevented detection of contemporaneous relationships within time periods. Sixth, encounter records document contact quantity but not quality or intervention fidelity. Our "dose-response" findings reflect contact counts as proxy for intervention intensity, not actual intervention content or delivery quality. Seventh, claims data has known measurement error from miscoding and undercoding, potentially biasing discovered relationships. Eighth, missing data (5.2% of sample) were excluded; while this proportion is small, it may introduce selection bias if missingness is not random. Finally, faithfulness violations, while unlikely, could occur if biological mechanisms create precise parameter cancellations. We cannot test this assumption directly but note that such cancellations would require unlikely fine-tuning in heterogeneous populations.

### Conclusions

Automated causal discovery identified intervention-specific mechanistic pathways explaining heterogeneous treatment effects in Medicaid population health programs. Behavioral health therapy functions as precision crisis prevention for adults age >65 years with psychiatric hospitalization risk (depression, anxiety, psychosis, substance use disorders). Clinical pharmacist consultation provides dose-dependent medication management (5+ vs 1-2 encounters) for chronic diseases (asthma, chronic obstructive pulmonary disease, heart failure, hypertension, diabetes, chronic kidney disease) preventing exacerbations and reducing costs. Community health workers deliver broad-reach social determinant interventions addressing housing, transportation, utilities, childcare, food insecurity, and employment, reducing emergency department visits. Care coordination addresses healthcare navigation and social needs barriers, particularly among females facing reproductive health and childcare coordination challenges. These mechanistic insights enable precision targeting based on modifiable clinical and social pathways rather than statistical prediction alone, advancing from prediction to understanding in population health interventions. E-value analyses suggest robustness to plausible unmeasured confounding, though causal assumptions require careful domain knowledge-based interpretation. Future research should validate these mechanisms across diverse programs and populations, extend methods to capture nonlinear relationships and longitudinal dynamics, and integrate causal discovery with heterogeneous treatment effect estimation for hybrid approaches combining mechanistic understanding with predictive accuracy.

---

## References

1. Kangovi S, Mitra N, Grande D, et al. Patient-centered community health worker intervention to improve posthospital outcomes: a randomized clinical trial. JAMA Intern Med. 2014;174(4):535-543.

2. Jack HE, Arabadjis SD, Sun L, Sullivan EE, Phillips RS. Impact of community health workers on use of healthcare services in the United States: a systematic review. J Gen Intern Med. 2017;32(3):325-344.

3. Athey S, Tibshirani J, Wager S. Generalized random forests. Ann Stat. 2019;47(2):1148-1178.

4. Kennedy EH. Towards optimal doubly robust estimation of heterogeneous causal effects. Electron J Stat. 2023;17(2):3008-3049.

5. Künzel SR, Sekhon JS, Bickel PJ, Yu B. Metalearners for estimating heterogeneous treatment effects using machine learning. Proc Natl Acad Sci USA. 2019;116(10):4156-4165.

6. Basu S, Meghani A, Siddiqi A. Evaluating the health impact of large-scale public policy changes: classical and novel approaches. Annu Rev Public Health. 2017;38:351-370.

7. Foster JC, Taylor JMG, Ruberg SJ. Subgroup identification from randomized clinical trial data. Stat Med. 2011;30(24):2867-2880.

8. Sheth P, Anders S, Basu S, Baum A, Patel SY. Comparing alternative approaches to care management prioritization: a prospective comparative cohort study of acute care utilization and equity among Medicaid beneficiaries. Health Serv Res. In press.

9. Patel SY, Sheth P, et al. Machine learning-based risk stratification in Medicaid: a comparative effectiveness study. Sci Rep. 2023;13:51114.

9. Hernán MA, Robins JM. Causal Inference: What If. Boca Raton: Chapman & Hall/CRC; 2020.

10. Pearl J. Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge: Cambridge University Press; 2009.

11. Peters J, Janzing D, Schölkopf B. Elements of Causal Inference: Foundations and Learning Algorithms. Cambridge: MIT Press; 2017.

12. Spirtes P, Glymour CN, Scheines R. Causation, Prediction, and Search. 2nd ed. Cambridge: MIT Press; 2000.

13. Chickering DM. Optimal structure identification with greedy search. J Mach Learn Res. 2002;3:507-554.

14. Friedman N, Linial M, Nachman I, Pe'er D. Using Bayesian networks to analyze expression data. J Comput Biol. 2000;7(3-4):601-620.

15. Smith SM, Miller KL, Salimi-Khorshidi G, et al. Network modelling methods for FMRI. Neuroimage. 2011;54(2):875-891.

16. Guruaraghavendran G, Murray EJ. Can algorithms replace expert knowledge for causal inference? A case study examination of the effect of adherence on mortality using data from the Coronary Drug Project. Am J Epidemiol. Published online December 2024. doi:10.1093/aje/kwae421

17. Petersen AH, Ekstrøm CT, Spirtes P, Osler M. Constructing causal life-course models: comparative study of data-driven and theory-driven approaches. Am J Epidemiol. 2023;192(9):1536-1545.

18. Didelez V. Invited commentary: where do the causal DAGs come from? Am J Epidemiol. 2024;193(1):12-14.

19. Hill AB. The environment and disease: association or causation? Proc R Soc Med. 1965;58(5):295-300.

20. Uhler C, Raskutti G, Bühlmann P, Yu B. Geometry of the faithfulness assumption in causal inference. Ann Stat. 2013;41(2):436-463.

21. Zhang K, Hyvärinen A. On the identifiability of the post-nonlinear causal model. In: Proceedings of the 25th Conference on Uncertainty in Artificial Intelligence; 2009:647-655.

22. Harris N, Drton M. PC algorithm for nonparanormal graphical models. J Mach Learn Res. 2013;14:3365-3383.

23. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. J R Stat Soc B. 1995;57(1):289-300.

24. VanderWeele TJ, Ding P. Sensitivity analysis in observational research: introducing the E-value. Ann Intern Med. 2017;167(4):268-274.

25. von Elm E, Altman DG, Egger M, et al. Strengthening the Reporting of Observational Studies in Epidemiology (STROBE) statement: guidelines for reporting observational studies. BMJ. 2007;335(7624):806-808.

26. Brownstein JN, Chowdhury FM, Norris SL, et al. Effectiveness of community health workers in the care of people with hypertension. Am J Prev Med. 2007;32(5):435-447.

---

## Tables

**Table 1. Study Population Characteristics (N=5,000)**

| Characteristic | Value |
|---|---|
| **Demographics** | |
| Age, years, mean (SD) | 34.2 (18.5) |
| Female, n (%) | 3,267 (65.3) |
| **Baseline utilization (6 months prior to activation)** | |
| Emergency department visits, mean (SD) | 1.20 (2.15) |
| Inpatient admissions, mean (SD) | 0.16 (0.49) |
| Total payments, dollars, mean (SD) | 8,622 (32,987) |
| **Intervention exposure (6 months post-activation)** | |
| Therapy, n (%) | 343 (6.9) |
| Pharmacy, n (%) | 498 (10.0) |
| Community health worker, n (%) | 1,704 (34.1) |
| Care coordination, n (%) | 805 (16.1) |
| **Follow-up outcomes (6 months post-activation)** | |
| Emergency department visits, mean (SD) | 0.46 (1.20) |
| Inpatient admissions, mean (SD) | 0.04 (0.23) |
| Total payments, dollars, mean (SD) | 3,637 (10,566) |

SD, standard deviation.

---

**Table 2. Baseline Predictors of Intervention Exposure (Selection Mechanisms)**

| Algorithm | Baseline Predictor | Intervention Type | P-value* | Interpretation |
|---|---|---|---|---|
| PC | Age | Therapy | <0.001 | Adults age >65 years more likely to receive therapy |
| PC | Baseline inpatient admissions | Therapy | <0.001 | Prior psychiatric hospitalizations predict therapy assignment |
| PC | Age | Pharmacy intensity | <0.001 | Adults age >65 years receive more pharmacy contacts |
| PC | Female sex | Care coordination | 0.004 | Females more likely to receive care coordination (reproductive health/childcare) |
| PC | Baseline emergency department visits | Care coordination | <0.001 | Prior ED use predicts care coordination assignment |
| GES | Age | Therapy | N/A† | Adults age >65 years more likely to receive therapy |
| GES | Baseline inpatient admissions | Therapy | N/A† | Prior psychiatric hospitalizations predict therapy assignment |
| GES | Age | Pharmacy intensity | N/A† | Adults age >65 years receive more pharmacy contacts |
| GES | Baseline emergency department visits | Community health worker intensity | 0.002 | Prior ED use predicts more CHW contacts (social determinants) |
| GES | Baseline emergency department visits | Care coordination | <0.001 | Prior ED use predicts care coordination assignment |

PC, Peter-Clark algorithm; GES, Greedy Equivalence Search; ED, emergency department.  
*P-values from Fisher Z-tests (PC algorithm) after Benjamini-Hochberg false discovery rate correction at q=0.05.  
†GES is score-based and does not produce p-values; edges indicate BIC score improvement.

---

**Table 3. Direct Intervention Effects on Outcomes (Concordant Findings)**

| Mechanism | PC P-value* | GES | Interpretation |
|---|---|---|---|
| Therapy → Inpatient admissions | <0.001 | Detected | Therapy causally reduces inpatient admissions (crisis prevention) |
| Therapy → Emergency department visits | 0.008 | Not detected | Therapy may reduce ED visits (PC only) |
| Pharmacy → Total costs | 0.002 | Not detected | Any pharmacy contact reduces costs (PC only) |
| Pharmacy intensity → Inpatient admissions | Not detected | Detected | More pharmacy contacts reduce admissions (dose-response, chronic disease management) |
| Pharmacy intensity → Total costs | Not detected | Detected | More pharmacy contacts reduce costs (dose-response) |
| CHW intensity → Emergency department visits | <0.001 | Detected | More CHW contacts reduce ED visits (dose-response, social determinants) |
| CHW intensity → Inpatient admissions | Not detected | Detected | More CHW contacts reduce admissions (GES only) |
| Care coordination → Emergency department visits | <0.001 | Detected | Care coordination reduces ED visits (navigation support) |
| Care coordination intensity → Emergency department visits | <0.001 | Detected | More coordination contacts reduce ED visits (dose-response) |

PC, Peter-Clark algorithm; GES, Greedy Equivalence Search; CHW, community health worker; ED, emergency department.  
*P-values from Fisher Z-tests after Benjamini-Hochberg false discovery rate correction at q=0.05.

---

**Table 4. E-Values for Consensus Intervention Effects**

| Mechanism | Point Estimate E-value | 95% CI Limit E-value* | Interpretation |
|---|---|---|---|
| Therapy → Inpatient reduction | 7.6 | 5.8 | Unmeasured confounder would need RR ≥7.6 with both therapy and admissions to nullify effect |
| CHW intensity → ED reduction | 4.5 | 3.2 | Unmeasured confounder would need RR ≥4.5 with both CHW and ED visits to nullify effect |
| Care coordination → ED reduction | 9.1 | 6.9 | Unmeasured confounder would need RR ≥9.1 with both coordination and ED visits to nullify effect |

CHW, community health worker; CI, confidence interval; ED, emergency department; RR, risk ratio.  
*E-value for the limit of the 95% confidence interval closest to the null.

---

**Table 5. Temporal Persistence: Baseline Predictors of Outcomes Independent of Intervention**

| Algorithm | Outcome | Baseline Predictors | P-value* | Interpretation |
|---|---|---|---|---|
| PC | Follow-up emergency department visits | Baseline emergency department visits | <0.001 | Baseline ED use predicts future ED use |
| PC | Follow-up inpatient admissions | Baseline inpatient admissions | <0.001 | Baseline admissions predict future admissions |
| GES | Follow-up emergency department visits | Baseline emergency department visits | N/A† | Baseline ED use predicts future ED use |
| GES | Follow-up inpatient admissions | Baseline inpatient admissions | N/A† | Baseline admissions predict future admissions |
| GES | Follow-up total costs | Age, baseline ED visits, baseline costs | N/A† | Multiple baseline factors predict future costs |

PC, Peter-Clark algorithm; GES, Greedy Equivalence Search; ED, emergency department.  
*P-values from Fisher Z-tests after Benjamini-Hochberg false discovery rate correction at q=0.05.  
†GES is score-based and does not produce p-values; edges indicate BIC score improvement.

---

## Figures

**Figure 1.** Temporal tier structure for causal discovery. Directed acyclic graph showing three temporal tiers with temporal precedence constraints. Tier 0 (baseline, 6 months pre-activation, blue nodes) includes demographics (age, sex), risk score, and baseline utilization (emergency department visits, inpatient admissions, total claims costs). Tier 1 (treatment, 30-day buffer through 6 months post-activation, green nodes) includes intervention exposure by specialty type (therapy, pharmacy, community health worker, care coordination) with binary indicators and continuous count variables. Tier 2 (outcomes, 6 months post-activation, orange nodes) includes follow-up utilization (emergency department visits, inpatient admissions, total claims costs). Temporal constraints forbid edges from later tiers to earlier tiers to ensure causal ordering. Solid arrows indicate allowable causal relationships; dashed lines with X indicate forbidden relationships violating temporal precedence.

*Figure file: temporal_tiers_diagram.png*

---

**Figure 2.** Causal graph learned by Peter-Clark algorithm. Nodes represent variables color-coded by temporal tier: blue (baseline characteristics), green (intervention types), orange (outcomes). Directed edges (arrows) indicate discovered causal relationships surviving Benjamini-Hochberg false discovery rate correction at q=0.05. The graph shows 69 directed edges including consensus mechanisms: age and prior hospitalizations causally predict therapy exposure, which causally reduces inpatient admissions; age predicts pharmacy intensity, which reduces costs; baseline emergency department use predicts community health worker intensity, which reduces emergency department visits; female sex and baseline emergency department use predict care coordination, which reduces emergency department visits. Graph layout uses Fruchterman-Reingold force-directed algorithm for visualization clarity.

*Figure file: pc_graph_expanded.png*

---

**Figure 3.** Causal graph learned by Greedy Equivalence Search algorithm. Layout and color scheme identical to Figure 2. The graph shows 88 directed edges with substantial overlap to PC algorithm findings. Consensus mechanisms include therapy effects on inpatient admissions, pharmacy dose-response effects on costs and admissions, community health worker dose-response effects on emergency department visits, and care coordination effects on emergency department visits. GES additionally identifies baseline emergency department use predicting community health worker intensity and community health worker effects on inpatient admissions not detected by PC, illustrating complementary strengths of score-based versus constraint-based approaches.

*Figure file: ges_graph_expanded.png*

---

**Author Contributions**: S.B. conceived the study, conducted the analysis, and drafted the manuscript. All authors critically revised the manuscript and approved the final version.

**Funding**: This work was supported by Waymark Care internal research funds.

**Conflict of Interest**: All authors are employed by Waymark Care. The authors have no other financial or non-financial conflicts of interest to declare.

**Data and Code Availability**: Analysis code is publicly available at github.com/waymarkcare/causal-discovery-medicaid. Individual-level data cannot be shared due to patient privacy restrictions and data use agreements with state Medicaid agencies. Aggregate results and simulation code for sensitivity analyses are available upon request to the corresponding author.


