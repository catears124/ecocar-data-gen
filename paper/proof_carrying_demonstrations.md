# Proof-Carrying Optimal Demonstrations for Data-Efficient Imitation Learning

## Abstract

Imitation learning normally treats an expert trajectory as ground truth while leaving the provenance and optimality of its labels implicit. We study a narrower, auditable alternative: a preview-only driving teacher emits a demonstration together with a machine-readable certificate containing the complete finite model instance, Bellman value tables, and an extracted plan. An independent checker reconstructs the transition relation and checks the certificate; a Lean 4 artifact proves that an accepted certificate is globally optimal for the encoded finite model. The model is intentionally modest: quantised offsets and squared speeds, integer constraints, a finite preview horizon, and a declared terminal cap. Thus “optimal” is not whole-course optimal driving and is not a physical-vehicle safety claim. It is optimality in the information-limited lattice model.

The paper specifies the information structure, exact integer objective and constraints, checker architecture, and a dataset protocol for behavioural cloning (BC). We evaluate four label conditions at budgets 100, 300, and 720 with three seeds on four held-out bicycle-backend courses. The optimal and certificate-metadata control conditions are identical offline and in closed loop; optimal BC finishes all 12 course-seed runs at every budget, whereas the heuristic has lower offset error but finishes only 8/12. Detailed metrics and paired course-bootstrap intervals are machine-readable in `experiments/il/results/results.json` and `experiments/il/results/reanalysis.json`, with recomputed tables in `experiments/il/results/summary.md` and `experiments/il/results/analysis.md`. These are bicycle-model results only.

## 1. Problem definition and information structure

### 1.1 Preview-only teacher

At a decision time, let the policy observation be
\[
 o = (v_0,\; s_{0:H},\; \kappa_{0:H},\; w^L_{0:H},\;w^R_{0:H},\;
       \mathcal O_{0:H},\;\text{vehicle constants}),
\]
where the station coordinates, road curvature, corridor widths, and obstacle extents are quantised integers. The teacher receives no map, pose, progress, finish distance, corner list, or geometry beyond the preview. This is the same information discipline imposed on a learned policy: the corpus loader exposes `policy_observation` and never reads the privileged block. The preview may be shorter than the full course.

The teacher constructs a model from `model_from_observation`. The default lattice has offsets in millimetres and speed-squared values in mm²/s²; the default horizon is ten preview transitions, clipped to the available stations. The current and previous offsets are snapped to the declared offset grid. An observation outside the model domain yields no fabricated optimum: it is invalid and must be rejected or handled outside the certified path.

### 1.2 What is, and is not, claimed

The central claim is conditional and local:

> Given a declared quantised observation, constants, grids, terminal cap, and finite horizon, an accepted certificate proves that the emitted plan has minimum declared cost among all admissible lattice plans with that same information.

The terminal cap is the *only* way the model represents what lies beyond the preview. It is a declared continuation assumption, not hidden access to the rest of the course. Consequently, the result makes no claim of globally time-optimal driving over an entire course. It also does not establish that a learned policy reproduces the teacher, that a controller tracks the lattice path, or that a real vehicle satisfies the model bounds.

This separation matters for imitation learning. BC can be data-efficient when labels are deterministic and informative, but an unqualified “expert” label conflates (i) a generated action, (ii) an optimizer's output, (iii) safety/model assumptions, and (iv) physical execution. The certificate makes (i) and the finite-model part of (ii) inspectable while leaving the other items explicitly scoped.

## 2. Exact integer joint model and objective

### 2.1 State and grids

Let stations be indexed by $i=0,\ldots,H$. Let $D$ be the finite offset grid and $B$ the finite speed-squared grid. A state is
\[
 X_i=(i,p_i,x_i,b_i),
\]
where $p_i$ and $x_i$ are the preceding and current offsets and $b_i=v_i^2$. The initial state is $X_0=(0,p_0,x_0,b_0)$, with $b_0$ computed from the quantised current speed. A successor has the form
\[
 X_{i+1}=(i+1,x_i,n,w),\qquad n\in D,\quad w\in B.
\]
All quantities are integers. Integer square roots are floor roots, not floating-point approximations.

For the segment from $i$ to $i+1$, write $d_i=s_{i+1}-s_i>0$, and let $\Delta$ be the declared station spacing. The lateral finite-difference curvature contribution is
\[
 \kappa^{\mathrm{lat}}_i(p,x,n)=
 \left\lceil\frac{10^9\lvert n-2x+p\rvert}{\Delta^2}\right\rceil.
\]
The path bound is
\[
 \widehat\kappa_i=\lvert\kappa_i^{\mathrm{road}}\rvert+
                    \kappa^{\mathrm{lat}}_i(p_i,x_i,x_{i+1}),
\]
with the same integer ceiling operation as the Lean model. The speed cap for the departure stencil is
\[
 C_i=\begin{cases}
 V_{\max}^2,&\widehat\kappa_i\le 0,\\
 \min\left(V_{\max}^2,\left\lfloor\dfrac{10^9 A_{\mathrm{lat}}}{\widehat\kappa_i}\right\rfloor\right),&\widehat\kappa_i>0.
 \end{cases}
\]

### 2.2 Admissibility constraints

A transition is admissible when all of the following hold.

1. **Grid and station:** $n\in D$, $w\in B$, $X_{i+1}$ has the successor station and predecessor offset shown above.
2. **Corridor and obstacle footprint:** the arrival offset is in the quantised left/right corridor, and is outside every obstacle inflated by half the vehicle width plus the declared margin. This is a footprint constraint on the reference point; it is not a proof about continuous tyre or body geometry.
3. **Curvature:** $\widehat\kappa_i\le\kappa_{\max}$ and $b_i\le C_i$.
4. **Nonnegative speed:** $b_i,w\ge0$ and $(b_i>0\lor w>0)$, excluding a zero-speed segment with no finite time charge.
5. **Acceleration and braking:**
   \[
   w\le b_i+2A_{\mathrm{acc}}d_i,
   \qquad
   b_i\le w+2A_{\mathrm{brk}}d_i.
   \]
6. **Terminal condition:** at station $H$, $b_H\le C_{\mathrm{term}}$, where
   \[
   C_{\mathrm{term}}=\min(V_{\max}^2,A_{\mathrm{lat}}r_H).
   \]
   Here $r_H$ is a declared effective horizon radius. It summarizes the continuation assumption and does not expose unseen course geometry.

The declared traversal charge is an integer number of microseconds:
\[
 c_i(b_i,w)=\operatorname{cdiv}\!\left(2d_i10^6,
        \operatorname{isqrt}(b_i)+\operatorname{isqrt}(w)\right),
\]
where `cdiv` is ceiling division for a positive denominator. The cost of a plan $X_{0:H}$ is
\[
 J(X_{0:H})=\sum_{i=0}^{H-1}c_i(b_i,b_{i+1}).
\]
No floating-point arithmetic participates in this objective or in the integer transition checks.

### 2.3 Optimisation problem

For a finite model $M(o)$, define
\[
 J^*(M,o)=\min\{J(X_{0:H}):X_0=\operatorname{start}(o),\;
        X_{0:H}\text{ satisfies all constraints}\}.
\]
The minimum is an option-valued quantity: it is `some v` when a plan exists and `none` when the finite model is infeasible. The teacher's output is a deterministic first-minimum plan in offset-major, speed-minor action order. Deterministic tie-breaking makes records reproducible; it is not an additional optimality claim.

## 3. Global-optimality theorem and Lean artifact

The theorem is a theorem about the finite encoded model, not about a simulator. Let `cert : Corpus.Teacher.Cert`. Assume
\[
 h : \operatorname{CheckCert}(cert),
\]
including well-formed constants and quantised observation, correct terminal cap, valid grids and roots, exact table shape, station-indexed state universes, closure under every recomputed successor, terminal rows, Bellman rows, start value, and exported plan.

The repository's Lean artifact expresses the result through `Corpus.Teacher.certificate_optimal` (the top-level optimality statement), using the following named lemmas in `lean/Corpus/Teacher.lean`:

* `Corpus.Teacher.checkCert_sound`: a `true` result from the executable Boolean checker yields the proposition `CheckCert cert`;
* `Corpus.Teacher.table_value_cert`: every declared row equals the option-valued Bellman value for its remaining horizon;
* `Corpus.Teacher.certificate_value`: the start row equals the true option-valued optimum;
* `Corpus.Teacher.certificate_lower_bound`: every declarative admissible plan has cost at least the claimed value;
* `Corpus.Teacher.certificate_plan`: the exported plan is admissible, reaches the horizon, and attains the claimed value;
* `Corpus.Teacher.certificate_infeasible`: a checked `none` claim rules out every declarative plan of the horizon model.

In mathematical form, for a feasible claim `cert.claimed=some(v)`, the conclusion is
\[
  (\forall P\in\mathcal P(M,o),\;v\le J(P))
  \quad\land\quad
  (\exists \widehat P\in\mathcal P(M,o),\;J(\widehat P)=v),
\]
and the certificate's plan is such a $\widehat P$. For `none`, $\mathcal P(M,o)=\varnothing$. The proof is an induction over the finite horizon: complete successor sets and closed row universes let the sparse table be identified with option-valued backward induction; the generic Bellman lower-bound and attainment facts then establish global, rather than merely local, optimality.

This is deliberately not a claim that Bellman's principle itself is novel. The contribution is an executable instance-level object whose values, transitions, roots, and plan can be checked independently and connected to a kernel-checked theorem.

## 4. Solver and independent certificate checker

### 4.1 Producer

`autocross/optimal/solver.py` performs sparse exact backward dynamic programming. It enumerates every admissible successor from each state in the forward-reachable universe and stores an option-valued remaining cost and a first optimal choice. The algebraic factoring of the speed loop changes runtime only; it does not prune the declared action space. `certificate_from_solution` serialises:

* the model instance (`consts`, quantised `q`, grids, horizon, start, and terminal cap);
* per-station states and values, including `None` dead ends;
* integer-square-root roots for the start and speed grid values;
* the claimed value and extracted plan;
* objective and information metadata.

A certificate is therefore a proof object plus enough input to replay the proof obligation. The solver is not in the trusted computing base for the theorem: a wrong solver output should be rejected by the checkers.

### 4.2 Python checker

`autocross/optimal/certificate.py:check_certificate` rebuilds the model from the serialised instance. It rejects malformed schemas, duplicate or missing rows, rows outside the declared universe, a non-singleton start universe, and a universe that is not closed under every recomputed admissible successor. It recomputes terminal values and every Bellman minimum, compares the start row to the claim, replays every exported transition and cost, verifies the terminal condition, and checks the deterministic first-optimal-successor rule. It also checks the root table and plan-cost field. The checker does not import the solver's value table as an oracle.

### 4.3 Lean checker and parity

`lean/Corpus/Teacher.lean` defines the same state, transition, terminal, cost, roots, table, and plan predicates in Lean. `lean/TeacherCheck.lean` parses a JSON certificate and evaluates `Corpus.Teacher.checkCert`, returning an `ok` report and a nonzero exit status on rejection. The Lean checker re-derives the successor relation from the encoded instance rather than trusting a list of actions. `rootsOK` establishes the integer square-root facts without requiring the kernel to evaluate an opaque square-root primitive.

Python/Lean parity is a computationally checked engineering property: the quantisation and JSON schema are shared by construction and `lean/TeacherCertProbe.json` is accepted by the compiled Lean checker. The formal theorem applies after the Lean checker has accepted the Lean representation; it does not magically prove that an unrelated floating-point or physical implementation has the same semantics. Reproduction should build the Lean target and run the checker on the released probe/certificates; the paper reports no additional checker run beyond artifacts present in the repository.

### 4.1 Direct actuator-space extension

The teacher above optimises an offset/speed lattice and leaves tracking outside
the theorem. A second finite model removes that layer: the state is
`(station, d, psi, b, delta)` and the action is a normalised steering request
plus an acceleration request. The transition is integer, path-relative and
rate/lag-limited; it includes understeer, midpoint integration, corridor and
inflated-obstacle checks, a lateral-acceleration bound, and a friction ellipse.
The objective adds traversal time, steering effort, and optional tracking of a
fast reference path. `autocross/actuator/solver.py` solves this model exactly,
and `autocross/actuator/certificate.py` serialises the reachable tables, optimal
values, plan, actions, first action, claimed value and model instance.

The Lean module `Corpus.Actuator` mirrors the integer model and proves the same
sparse-table soundness result (`checkCert_sound`, `certificate_value`,
`certificate_lower_bound`, `certificate_plan`, `certificate_infeasible`,
`certificate_optimal`). The compiled `actuatorcheck` accepts a real H=4
observation certificate and rejects corrupt values, actions, plans and
successors. The theorem therefore says: an accepted certificate's first action
is optimal for **that declared finite actuator instance and objective**. It does
not say that the bicycle backend, CARLA, or a continuous vehicle follows the
model, and it does not cover the controller's explicit `certline` fallback.

In closed loop the controller executes the certified first action only while the
observation remains inside the actuator envelope; outside it, near obstacles, or
in tight curvature it falls back and labels the command uncertified. On the five
public bicycle courses the hybrid finishes every course with zero cones and zero
off-course events and 65.7/75 mean technical points, versus 69.9/75 for the
existing `certline` baseline. The exact DP is exact for the declared H=4 grid,
but exact steps are a minority of control steps; this limitation is reported
rather than hidden. The result is a proof-carrying actuator optimizer, not yet a
faster whole-course controller.

## 5. Dataset and BC methodology

### 5.1 Corpus schema

The corpus is organised as a manifest, scenario tracks, run logs, one JSONL sample file per scenario, an index, contrast report, dataset report, validation report, Lean interchange data, Lean report, and release hashes. Each sample contains:

* `policy_observation`: ego signals and road preview arrays, visible obstacles, and no privileged map information;
* `target`: certified longitudinal integers (`kind`, acceleration, and speed-square fields), certified `steer-2` lateral fields where available, and separately marked uncertified steering, path, pedals, and target speed;
* `privileged`: pose, progress, corners, finish distance, cones, and oracle diagnostics for evaluation only;
* `quality`, `outcome`, `timing`, and per-record content hashes.

`autocross.datagen.dataset.iter_examples` yields policy observation, training target, certificate status, and metadata while never reading `privileged`. A clean sample is decision-window data from a clean run, with realised hand-over, certified labels, unclipped command, no held command, and exact label replay. Failed runs remain recorded with exclusion reasons rather than silently becoming negative or disappearing.

### 5.2 Conditions, budgets, seeds, and splits

The empirical corpus for the BC study is course-disjoint and uses scenario-level split assignment: 24 training, 6 validation, and 8 test courses, with 1,140 examples per condition (720/180/240 in train/validation/test). No scenario or course crosses a split. The test evaluation uses four held-out courses. The four conditions are:

* **optimal:** labels from the full finite-horizon joint teacher and its certificate;
* **approx:** a two-step truncated teacher followed by a hold action;
* **heuristic:** the existing VeriDrive expert;
* **cert_control:** the same optimal labels as `optimal`, with certificate metadata removed from the learner input.

Each condition is trained at budgets 100, 300, and 720, with seeds 0, 1, and 2, yielding 36 condition–budget–seed configurations. The closed-loop row for each configuration evaluates 12 course–seed runs. The bicycle backend is used throughout; no CARLA result is pooled with these measurements. The machine-readable run manifest, held-out MAE, and per-course outcomes are in `experiments/il/results/results.json`; `experiments/il/results/reanalysis.json` recomputes the aggregate metrics and paired comparisons. The corpus loader uses only policy observations, and course-level evaluation—not correlated timesteps—is the statistical unit.

The conditions preserve the intended certificate/control ablation: `optimal` and `cert_control` have identical offline MAE and identical closed-loop metrics at every budget, including exactly zero paired time difference. The approximation condition tests whether a shorter lookahead changes imitation and execution. The heuristic condition tests a non-optimal teacher and is expected only as a behavioural baseline, not as a safety oracle.

## 6. Experiments and uncertainty

All reported results are recomputed from `experiments/il/results/results.json` by `experiments/il/results/reanalysis.json`; the rendered source tables are `experiments/il/results/summary.md` and `experiments/il/results/analysis.md`. The experiment has 36 condition–budget–seed configurations (four conditions, three budgets, three seeds), zero recorded infrastructure failures, and zero controller exceptions. Each row below aggregates 12 bicycle-backend course–seed runs; the offline metrics are held-out test metrics over 240 examples per condition.

| condition | budget | offset MAE (mm) | acceleration MAE (mm/s²) | finished | mean time on finished (s) | cone hits | off-course | exceptions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| optimal | 100 | 22.47 | 244.71 | 12/12 | 48.30 | 250 | 0 | 0 |
| optimal | 300 | 11.62 | 43.12 | 12/12 | 49.44 | 219 | 0 | 0 |
| optimal | 720 | 8.80 | 19.23 | 12/12 | 49.55 | 202 | 0 | 0 |
| cert_control | 100 | 22.47 | 244.71 | 12/12 | 48.30 | 250 | 0 | 0 |
| cert_control | 300 | 11.62 | 43.12 | 12/12 | 49.44 | 219 | 0 | 0 |
| cert_control | 720 | 8.80 | 19.23 | 12/12 | 49.55 | 202 | 0 | 0 |
| approx | 100 | 27.51 | 265.80 | 12/12 | 62.85 | 410 | 0 | 0 |
| approx | 300 | 14.22 | 58.06 | 12/12 | 62.50 | 250 | 0 | 0 |
| approx | 720 | 12.10 | 26.23 | 12/12 | 62.58 | 378 | 0 | 0 |
| heuristic | 100 | 4.01 | 108.29 | 8/12 | 29.29 | 281 | 42 | 0 |
| heuristic | 300 | 3.19 | 38.68 | 8/12 | 29.04 | 263 | 42 | 0 |
| heuristic | 720 | 3.01 | 26.41 | 8/12 | 29.50 | 290 | 44 | 0 |

The heuristic has the smallest held-out offset MAE at every budget, likely because its targets are mostly zero/centreline offsets and do not include the optimizer-induced curvature changes present in the joint labels; this is an observed explanation to investigate, not a causal proof. The lower offset error does not translate into reliable execution: it finishes only 8/12 runs and records 42, 42, and 44 off-course events. Its finished runs are faster, but that mean is survival-biased. The optimal condition finishes all runs and has fewer cone hits than `approx` at every budget. The optimal/certificate-control identity is the intended metadata ablation: certificate metadata does not explain the observed policy or closed-loop differences.

Paired differences use the course as the statistical unit. For each `(course, seed)`, the run-level difference is averaged within course before the bootstrap; therefore the confidence intervals below resample four courses, or three courses for comparisons to the heuristic because only three courses have comparable finished times.

| budget | comparison (optimal − other) | mean time difference (s) | descriptive 95% course-bootstrap CI (s) | courses |
|---:|---|---:|---:|---:|
| 100 | approx | −14.55 | [−15.97, −13.13] | 4 |
| 300 | approx | −13.06 | [−14.16, −11.95] | 4 |
| 720 | approx | −13.03 | [−13.87, −12.19] | 4 |
| 100 | heuristic | +21.39 | [+17.11, +25.36] | 3 |
| 300 | heuristic | +22.93 | [+19.23, +26.42] | 3 |
| 720 | heuristic | +22.57 | [+19.05, +25.43] | 3 |
| 100/300/720 | cert_control | 0.00 | [0.00, 0.00] | 4 each |

Negative means optimal is faster than the comparator. These intervals are descriptive, not population-level significance claims: four (or three) courses are few independent units, and the heuristic comparison excludes courses without a finished run. The study supports an observed trade-off, not a universal data-efficiency theorem: the full optimal label is harder to imitate in offset MAE than the heuristic label here, while its closed-loop completion and off-course outcomes are better than the heuristic's and its times are lower than `approx`.

## 7. Failure analysis, limitations, and trust boundary

* **Finite model versus real vehicle.** The theorem quantifies only over the declared lattice. Tyre forces, mass transfer, actuator delay, steering-rate limits, perception errors, and tracking dynamics are outside the theorem. Vehicle constants and acceleration/grip budgets are model inputs, not physically proved facts.
* **Quantisation.** Rounding observation values to integer stations, curvature, offsets, and speed squares can change feasibility and the selected plan. The certificate proves the rounded instance, not the unquantised road.
* **Preview horizon.** The horizon and terminal cap summarise unknown geometry. A longer or differently shaped unseen course can invalidate the modelling assumption without invalidating the certificate. “Preview-only” means no hidden course access, not robustness to every possible continuation.
* **Obstacle footprint.** Inflation by half width and margin is conservative only relative to the declared reference-point geometry. It does not establish continuous swept-volume clearance under tracking error.
* **Tracking and actuation.** The emitted plan is a lattice reference. Pure-pursuit or other tracking, pedal mapping, command clipping, and actuator response can fail to realise it. A checked certificate cannot certify a learned policy's actions or a controller's trajectory.
* **Dataset selection.** Clean-only training can under-represent tight clearances and failures. Exclusion records must remain available for audit, and performance must be stratified by factor bin and scenario group.
* **No CARLA validation claim.** The available teacher/certificate result is a finite-model result. The paper does not claim that the certificate has been validated in CARLA, and no CARLA BC result is reported.
* **Implementation boundary.** Python JSON parsing, the quantisation bridge, Lean's parser/runtime, toolchain/compiler, and the declared constants remain engineering trust items. The Lean kernel checks the theorem once the encoded proposition is constructed; it does not prove the JSON producer's intent or the physical soundness of constants.

These limitations are not after-the-fact caveats: they define the scope in which a certificate is meaningful.

## 8. Prior work and precise novelty statement

Proof-carrying code (Necula [R1]) established the general pattern of an untrusted producer attaching evidence that a small checker can validate. Proof-carrying plans and planning certificates apply related ideas to plan validity, unsolvability, and optimality [R6–R8]. Formally verified planning systems verify planning encodings, algorithms, or compositional subroutines [R4, R5]; verified dynamic programming and MDP work provides the mathematical foundations for executable value computation [R9, R10]. These results motivate our trust boundary but do not by themselves produce an imitation-learning record with a complete per-instance Bellman table.
Certified-optimization work uses independently checkable witnesses or lower bounds to separate an optimizer from a verifier; the planning optimality certificates in [R7] and formally verified MDP algorithms in [R9] are close conceptual precedents. Our table certificate uses the same verifier-oriented discipline but is attached to every imitation-learning instance rather than only to a planner's internal optimisation run. In oracle-based imitation learning, the expert is queried to label states visited by a learner or to provide cost-to-go information [R15, R16]; SafeDAgger and related methods reduce or gate those queries [R17, R29, R30]. Our teacher is an offline oracle whose answer includes a finite-model optimality witness, not merely an action.

Optimal-control teachers and trajectory-learning methods use optimisers or model-predictive controllers to produce demonstrations [R11–R14]. Apprenticeship and inverse-RL work instead infers a reward or policy from demonstrations [R14]. BC is a supervised alternative, while DAgger and AGGREVATE address distribution shift through interactive expert queries and cost-to-go information [R15, R16]. SafeDAgger and shielding reduce risky expert queries or constrain actions [R17, R18], but their safety signals are not the same as a kernel-checked certificate of finite-horizon plan optimality. Autonomous-driving BC studies document compounding-error and coverage issues [R21, R22]; D4RL and RoboMimic provide useful benchmark and dataset conventions [R25, R26]. CARLA and MuJoCo are important simulation/control platforms, not proof systems [R13, R27].

Our precise contribution is therefore deliberately narrow: an executable teacher that emits a **demonstration plus an independently checkable certificate of global optimality for a declared finite preview lattice**, with a Lean theorem connecting checker acceptance to the lower-bound and attainment statements. This is distinct from verifying a planning subroutine, verifying only a plan's feasibility, or claiming that a learned policy is safe. We do not claim novelty for Bellman's principle, dynamic programming, proof-carrying code, DAgger, or optimal control. The artifact-level contribution is the end-to-end interface between an optimal finite-model teacher, a certificate checker, and a dataset/BC protocol whose information and trust boundaries are explicit. The reported study observes a certificate-metadata control identity and a trade-off between imitation error and closed-loop completion, but does not establish universal data-efficiency improvement.

## Claim-status table

| Claim or guarantee | Status | Scope/evidence |
|---|---|---|
| The state, transition, terminal, and cost definitions are finite integer functions | **Formally proven** | Lean definitions in `Corpus.Teacher`; decidable predicates and integer arithmetic |
| A passing Boolean check yields `CheckCert` | **Formally proven** | `Corpus.Teacher.checkCert_sound` |
| Every checked table row equals the finite-horizon Bellman value | **Formally proven** | `Corpus.Teacher.table_value_cert` under `CheckCert` hypotheses |
| A checked feasible certificate's claimed value is globally minimal and attained by its exported plan | **Formally proven** | `Corpus.Teacher.certificate_optimal`, using `certificate_value`, `certificate_lower_bound`, and `certificate_plan` |
| A checked `none` claim means no declarative plan exists | **Formally proven** | `Corpus.Teacher.certificate_infeasible` |
| Python solver emits the intended exact DP and deterministic tie-break | **Computationally checked** | `autocross/optimal/solver.py`, certificate replay by `check_certificate` |
| Python and Lean checker accept/reject the same encoded certificate semantics | **Computationally checked** | JSON schema, `lean/TeacherCheck.lean`, probe certificate and parity checks; not a physical equivalence theorem |
| Preview-only information and finite terminal-cap assumption | **Assumed / encoded** | Model and corpus policy-observation whitelist; terminal cap is a declared continuation assumption |
| Quantised model constants represent the vehicle, tyre limits, or obstacle geometry faithfully | **Assumed** | Calibration/model inputs; not established by Lean |
| Lattice plan is tracked by the executed vehicle/controller | **Unresolved** | Tracking, actuation, delay, and dynamics are outside the certificate |
| Corpus schema, hashes, replay, split separation, and per-sample label checks | **Computationally checked** | Python/Lean corpus validation pipeline when a corpus is supplied; not evidence of physical safety |
| BC metrics and closed-loop outcomes on the reported bicycle corpus | **Empirically observed** | Held-out MAE, finishes, times, cone/off-course counts, and course-bootstrap comparisons in `experiments/il/results/results.json`, `experiments/il/results/reanalysis.json`, `experiments/il/results/summary.md`, and `experiments/il/results/analysis.md`; descriptive only (4 courses, 3 for heuristic comparisons) |
| Certificate metadata ablation (`optimal` versus `cert_control`) | **Empirically observed** | Exact equality of offline MAE and closed-loop metrics, including paired time difference 0.0, in the same result files; this does not prove general data efficiency |
| CARLA or real-vehicle safety/validation | **Unresolved** | No such claim is made in this manuscript |

## Reproducibility checklist

1. Rebuild the Lean project under `lean/` and inspect the theorem names in `Corpus.Teacher`.
2. Generate or load a certificate using `autocross.optimal.model`, `solver`, and `certificate`.
3. Run the independent Python checker and the compiled `teachercheck` executable on the same JSON.
4. Preserve certificate JSON, model constants, grids, roots, horizon, corpus manifest hash, scenario/group split, and random seeds.
5. Do not interpret a checked finite-model certificate as a whole-course, CARLA, or real-vehicle guarantee.
