/-
Kernel-checked shape facts for steer-2: exact smoothstep finite-difference
identities, monotone first differences, and signed-road mirror definitions.
Supplier-switching curvature bounds are intentionally not asserted here.
-/
import Corpus.SteerProofs

namespace Corpus.SteerShape
open Corpus Corpus.Steer Corpus.SteerRamp

/-- Exact first-difference identity for the cubic numerator. -/
theorem n_diff_identity (x h L : Int) :
    n (x+h) L - n x L =
      h * (3*L*(2*x+h) - 2*(3*x*x + 3*x*h + h*h)) := by
  unfold n
  grind

/-- Polynomial first-difference bound (any `x`, `h ≥ 0`):
`4·(n(x+h) - n(x)) = 6hL² - 2h³ - 6h(L-2x-h)² ≤ 6hL² - 2h³`. -/
theorem n_diff_bound (x h L : Int) (hh : 0 ≤ h) :
    4 * (n (x+h) L - n x L) ≤ 6*h*L*L - 2*h*h*h := by
  have hid : 4 * (n (x+h) L - n x L) = 6*h*L*L - 2*h*h*h - 6*(h*((L-2*x-h)*(L-2*x-h))) := by
    unfold n; grind
  have hsq : 0 ≤ (L-2*x-h)*(L-2*x-h) := by
    rcases Int.le_total 0 (L-2*x-h) with h0 | h0
    · exact Int.mul_nonneg h0 h0
    · have := Int.mul_nonneg (Int.neg_nonneg_of_nonpos h0) (Int.neg_nonneg_of_nonpos h0)
      rwa [Int.neg_mul_neg] at this
  have := Int.mul_nonneg hh hsq
  omega

theorem floor_bounds (N D : Int) (hD : 0 < D) : N / D * D ≤ N ∧ N < N / D * D + D := by
  have h1 := Int.ediv_mul_le N (by omega : D ≠ 0)
  have h2 := Int.lt_ediv_add_one_mul_self N hD
  rw [Int.add_mul, Int.one_mul] at h2
  exact ⟨h1, h2⟩

theorem ramp_eq (A x L : Int) : ramp A x L = (A * n x L) / (L * L * L) := by
  unfold ramp n; rfl

/-- **Discrete slope bound of the floored smoothstep** (exact integer rounding):
`ramp(x+h) - ramp(x) < 3Ah/(2L) + 1`, for every `x`, `h ≥ 0`, `A ≥ 0`, `L > 0`. -/
theorem ramp_slope_bound (A x h L : Int) (hA : 0 ≤ A) (hL : 0 < L) (hh : 0 ≤ h) :
    2 * L * (ramp A (x+h) L - ramp A x L) < 3 * A * h + 2 * L := by
  have hD : 0 < L * L * L := Int.mul_pos (Int.mul_pos hL hL) hL
  rw [ramp_eq, ramp_eq]
  obtain ⟨a1, b1⟩ := floor_bounds (A * n (x+h) L) (L*L*L) hD
  obtain ⟨a0, b0⟩ := floor_bounds (A * n x L) (L*L*L) hD
  have hpoly := n_diff_bound x h L hh
  have hAp := Int.mul_le_mul_of_nonneg_left hpoly hA
  have hh3 : 0 ≤ A * (h*h*h) := Int.mul_nonneg hA (Int.mul_nonneg (Int.mul_nonneg hh hh) hh)
  have e1 : A * (4 * (n (x+h) L - n x L)) = 4 * (A * n (x+h) L) - 4 * (A * n x L) := by grind
  have e2 : A * (6*h*L*L - 2*h*h*h) = 6 * A * h * (L * L) - 2 * (A * (h*h*h)) := by grind
  generalize hq1 : A * n (x+h) L / (L*L*L) = q1 at *
  generalize hq0 : A * n x L / (L*L*L) = q0 at *
  have lhs : L * L * (4 * L * (q1 - q0)) = 4 * (q1 * (L*L*L)) - 4 * (q0 * (L*L*L)) := by grind
  have rhs : L * L * (6 * A * h + 4 * L) = 6 * A * h * (L * L) + 4 * (L*L*L) := by grind
  have key : L * L * (4 * L * (q1 - q0)) < L * L * (6 * A * h + 4 * L) := by
    rw [lhs, rhs]
    generalize q1 * (L*L*L) = X1 at *
    generalize q0 * (L*L*L) = X0 at *
    generalize A * n (x+h) L = N1 at *
    generalize A * n x L = N0 at *
    generalize A * (4 * (n (x+h) L - n x L)) = W at *
    generalize A * (6*h*L*L - 2*h*h*h) = V at *
    generalize A * (h*h*h) = H3 at *
    generalize 6 * A * h * (L * L) = S at *
    generalize L*L*L = D at *
    omega
  have := Int.lt_of_mul_lt_mul_left key (Int.le_of_lt (Int.mul_pos hL hL))
  have f1 : 4 * L * (q1 - q0) = 2 * (2 * L * (q1 - q0)) := by grind
  have f2 : 6 * A * h = 2 * (3 * A * h) := by grind
  rw [f1, f2] at this
  omega

/-- **Discrete second-difference bound of the floored smoothstep** inside the ramp
(`0 ≤ x ≤ L`, any step `h`): `L²·|Δ²ramp| < 6Ah² + 2L²`. -/
theorem ramp_second_bound (A x h L : Int) (hA : 0 ≤ A) (hL : 0 < L) (hx0 : 0 ≤ x) (hxL : x ≤ L) :
    L * L * (ramp A (x+h) L - 2 * ramp A x L + ramp A (x-h) L) < 6 * A * h * h + 2 * L * L ∧
    -(6 * A * h * h + 2 * L * L) < L * L * (ramp A (x+h) L - 2 * ramp A x L + ramp A (x-h) L) := by
  have hD : 0 < L * L * L := Int.mul_pos (Int.mul_pos hL hL) hL
  rw [ramp_eq, ramp_eq, ramp_eq]
  obtain ⟨a2, b2⟩ := floor_bounds (A * n (x+h) L) (L*L*L) hD
  obtain ⟨a1, b1⟩ := floor_bounds (A * n x L) (L*L*L) hD
  obtain ⟨a0, b0⟩ := floor_bounds (A * n (x-h) L) (L*L*L) hD
  have hsec : A * n (x+h) L - 2 * (A * n x L) + A * n (x-h) L = A * (h*h) * (6*L - 12*x) := by
    unfold n; grind
  have hAh : 0 ≤ A * (h*h) := Int.mul_nonneg hA (by
    rcases Int.le_total 0 h with h0 | h0
    · exact Int.mul_nonneg h0 h0
    · have := Int.mul_nonneg (Int.neg_nonneg_of_nonpos h0) (Int.neg_nonneg_of_nonpos h0)
      rwa [Int.neg_mul_neg] at this)
  have up := Int.mul_le_mul_of_nonneg_left (show 6*L - 12*x ≤ 6*L by omega) hAh
  have dn := Int.mul_le_mul_of_nonneg_left (show -(6*L) ≤ 6*L - 12*x by omega) hAh
  have e1 : A * (h*h) * (6*L) = L * (6 * A * h * h) := by grind
  have e2 : A * (h*h) * (-(6*L)) = -(L * (6 * A * h * h)) := by grind
  rw [e1] at up; rw [e2] at dn
  generalize hq2 : A * n (x+h) L / (L*L*L) = q2 at *
  generalize hq1 : A * n x L / (L*L*L) = q1 at *
  generalize hq0 : A * n (x-h) L / (L*L*L) = q0 at *
  have lhs : L * (L * L * (q2 - 2 * q1 + q0)) = q2 * (L*L*L) - 2 * (q1 * (L*L*L)) + q0 * (L*L*L) := by grind
  have rhs : L * (6 * A * h * h + 2 * L * L) = L * (6 * A * h * h) + 2 * (L*L*L) := by grind
  have rhs' : L * (-(6 * A * h * h + 2 * L * L)) = -(L * (6 * A * h * h)) - 2 * (L*L*L) := by grind
  have k1 : L * (L * L * (q2 - 2 * q1 + q0)) < L * (6 * A * h * h + 2 * L * L) := by
    rw [lhs, rhs]
    generalize q2 * (L*L*L) = X2 at *
    generalize q1 * (L*L*L) = X1 at *
    generalize q0 * (L*L*L) = X0 at *
    generalize A * n (x+h) L = N2 at *
    generalize A * n x L = N1 at *
    generalize A * n (x-h) L = N0 at *
    generalize A * (h*h) * (6*L - 12*x) = W at *
    generalize L * (6 * A * h * h) = S at *
    generalize L*L*L = D at *
    omega
  have k2 : L * (-(6 * A * h * h + 2 * L * L)) < L * (L * L * (q2 - 2 * q1 + q0)) := by
    rw [lhs, rhs']
    generalize q2 * (L*L*L) = X2 at *
    generalize q1 * (L*L*L) = X1 at *
    generalize q0 * (L*L*L) = X0 at *
    generalize A * n (x+h) L = N2 at *
    generalize A * n x L = N1 at *
    generalize A * n (x-h) L = N0 at *
    generalize A * (h*h) * (6*L - 12*x) = W at *
    generalize L * (6 * A * h * h) = S at *
    generalize L*L*L = D at *
    omega
  exact ⟨Int.lt_of_mul_lt_mul_left k1 (Int.le_of_lt hL),
    Int.lt_of_mul_lt_mul_left k2 (Int.le_of_lt hL)⟩

/-- Monotonicity supplies nonnegative first differences of the floor ramp. -/
theorem ramp_diff_nonneg (A x h L : Int) (hA : 0 ≤ A) (hx : 0 ≤ x)
    (hh : 0 ≤ h) (hxl : x+h ≤ L) (hL : 0 < L) :
    0 ≤ ramp A (x+h) L - ramp A x L := by
  have hm := ramp_mono A x (x+h) L hA hx (by omega) hxl hL
  omega

/-- Exact second-difference identity for the cubic numerator. -/
theorem n_second_identity (x h L : Int) :
    n (x+h) L - 2*n x L + n (x-h) L = h*h*(6*L - 12*x) := by
  unfold n
  grind

/-- Signed mirror of a quantised observation.  Lateral magnitude is retained,
matching the corpus's absolute-latitude side tie rule. -/
def mirrorQ (q : Q) : Q :=
  { q with
    base := { q.base with
      k := q.base.k.map (fun z => (-z))
      wl := q.base.wr
      wr := q.base.wl
      obstacles := q.base.obstacles.map (fun ob =>
        match Cert.obsTriple ob with
        | some (d,lmin,lmax) => [d, -lmax, -lmin]
        | none => ob) } }

/-- Obstacle offset is odd in the signed side. -/
theorem offOb_neg_side (c : Consts) (d side amp si : Int) :
    offOb c d (-side) amp si = -offOb c d side amp si := by
  by_cases h0 : si < d - c.lon.hold - c.lon.ramp_in <;>
    by_cases h1 : si < d - c.lon.hold <;>
    by_cases h2 : si ≤ d + c.lon.hold <;>
    by_cases h3 : si ≤ d + c.lon.hold + c.lon.ramp_out <;>
    simp [offOb, h0, h1, h2, h3] <;> grind

/-- The corpus tie rule is not mirror-equivariant: equal gaps choose left for
nonnegative (including zero) corpus latitude. -/
theorem mirror_tie_counterexample :
    let q : Q :=
      { base :=
          { v0 := 0, s := [0], k := [0], wl := [3000], wr := [3000],
            obstacles := [[1000, -500, 500]] },
        lat := 0 }
    (sideAmp (mirrorQ q) 1000 (-500) 500).1 =
      (sideAmp q 1000 (-500) 500).1 := by decide

/-! ## Mirror equivariance of the actual lateral decision functions

Hypotheses: no obstacle has equal left and right gaps (`noTies`; the tie rule is
not equivariant, `mirror_tie_counterexample`), and the corridor is nonempty at the
stations considered (the empty-corridor midpoint `(lo+hi)/2` rounds toward −∞). -/

def mirrorOb (ob : List Int) : List Int :=
  match Cert.obsTriple ob with
  | some (d, lmin, lmax) => [d, -lmax, -lmin]
  | none => ob

theorem mirrorQ_obstacles (q : Q) : (mirrorQ q).base.obstacles = q.base.obstacles.map mirrorOb := rfl

def flipPlan (p : Plan) : Plan :=
  { d := p.d, lmin := -p.lmax, lmax := -p.lmin, side := -p.side, amp := p.amp }

def noTies (q : Q) : Bool :=
  q.base.obstacles.all fun ob => match Cert.obsTriple ob with
    | some (d, lmin, lmax) => gapLeft q d lmax != gapRight q d lmin
    | none => true

theorem iabs_neg (x : Int) : Cert.iabs (-x) = Cert.iabs x := by
  unfold Cert.iabs; split <;> split <;> omega

theorem sideAmp_core (lat wl wr lmin lmax : Int) (hne : wl - lmax ≠ lmin + wr) :
    (if wr - -lmin > -lmax + wl then (1, Cert.cdiv (Cert.iabs (-lmin + wr)) 2)
      else if -lmax + wl > wr - -lmin then (-1, Cert.cdiv (Cert.iabs (-lmax - wl)) 2)
      else if 0 ≤ lat then (1, Cert.cdiv (Cert.iabs (-lmin + wr)) 2)
      else (-1, Cert.cdiv (Cert.iabs (-lmax - wl)) 2)) =
    (-(if wl - lmax > lmin + wr then ((1 : Int), Cert.cdiv (Cert.iabs (lmax + wl)) 2)
        else if lmin + wr > wl - lmax then (-1, Cert.cdiv (Cert.iabs (lmin - wr)) 2)
        else if 0 ≤ lat then (1, Cert.cdiv (Cert.iabs (lmax + wl)) 2)
        else (-1, Cert.cdiv (Cert.iabs (lmin - wr)) 2)).1,
     (if wl - lmax > lmin + wr then ((1 : Int), Cert.cdiv (Cert.iabs (lmax + wl)) 2)
        else if lmin + wr > wl - lmax then (-1, Cert.cdiv (Cert.iabs (lmin - wr)) 2)
        else if 0 ≤ lat then (1, Cert.cdiv (Cert.iabs (lmax + wl)) 2)
        else (-1, Cert.cdiv (Cert.iabs (lmin - wr)) 2)).2) := by
  have e1 : Cert.iabs (-lmin + wr) = Cert.iabs (lmin - wr) := by
    rw [show -lmin + wr = -(lmin - wr) by omega, iabs_neg]
  have e2 : Cert.iabs (-lmax - wl) = Cert.iabs (lmax + wl) := by
    rw [show -lmax - wl = -(lmax + wl) by omega, iabs_neg]
  rw [e1, e2]
  by_cases h1 : wl - lmax > lmin + wr
  · have m1 : ¬ (wr - -lmin > -lmax + wl) := by omega
    have m2 : -lmax + wl > wr - -lmin := by omega
    rw [if_pos h1, if_neg m1, if_pos m2]
  · have h2 : lmin + wr > wl - lmax := by omega
    have m1 : wr - -lmin > -lmax + wl := by omega
    rw [if_neg h1, if_pos h2, if_pos m1]
    rfl

theorem sideAmp_mirror (q : Q) (d lmin lmax : Int) (hne : gapLeft q d lmax ≠ gapRight q d lmin) :
    sideAmp (mirrorQ q) d (-lmax) (-lmin) =
      (-(sideAmp q d lmin lmax).1, (sideAmp q d lmin lmax).2) := by
  unfold gapLeft gapRight at hne
  exact sideAmp_core q.lat _ _ lmin lmax hne

theorem lo_mirror (c : Consts) (q : Q) (i : Nat) : lo c (mirrorQ q) i = -hi c q i := by
  simp [lo, hi, mirrorQ]
theorem hi_mirror (c : Consts) (q : Q) (i : Nat) : hi c (mirrorQ q) i = -lo c q i := by
  simp [lo, hi, mirrorQ]

theorem clampAt_mirror (c : Consts) (q : Q) (i : Nat) (x : Int) (hc : lo c q i ≤ hi c q i) :
    clampAt c (mirrorQ q) i (-x) = -clampAt c q i x := by
  unfold clampAt
  simp only [lo_mirror, hi_mirror]
  have hc' : -hi c q i ≤ -lo c q i := by omega
  rw [if_pos hc', if_pos hc]
  omega

theorem obsTriple_mirror (ob : List Int) :
    Cert.obsTriple (mirrorOb ob) = (Cert.obsTriple ob).map (fun t => (t.1, -t.2.2, -t.2.1)) := by
  match ob with
  | [d, lmin, lmax] => rfl
  | [] => rfl
  | [_] => rfl
  | [_, _] => rfl
  | _ :: _ :: _ :: _ :: _ => rfl

theorem go_mirror (c : Consts) (q : Q) (si : Int) :
    ∀ (l : List (List Int)) (bestW : Int) (best : Option Plan),
      (∀ ob ∈ l, ∀ d lmin lmax, Cert.obsTriple ob = some (d, lmin, lmax) →
        gapLeft q d lmax ≠ gapRight q d lmin) →
      rawPlanAt.go c (mirrorQ q) si bestW (best.map flipPlan) (l.map mirrorOb) =
        (rawPlanAt.go c q si bestW best l).map flipPlan := by
  intro l
  induction l with
  | nil => intro _ _ _; rfl
  | cons ob tl ih =>
      intro bestW best hnt
      have htl : ∀ ob' ∈ tl, ∀ d lmin lmax, Cert.obsTriple ob' = some (d, lmin, lmax) →
          gapLeft q d lmax ≠ gapRight q d lmin := fun ob' h' => hnt ob' (List.mem_cons_of_mem _ h')
      simp only [List.map_cons, rawPlanAt.go]
      rw [obsTriple_mirror]
      cases ht : Cert.obsTriple ob with
      | none => simp only [Option.map_none]; exact ih bestW best htl
      | some t =>
          obtain ⟨d, lmin, lmax⟩ := t
          simp only [Option.map_some]
          by_cases hw : bestW < wOb c d si
          · rw [if_pos hw, if_pos hw, sideAmp_mirror q d lmin lmax (hnt ob (by simp) d lmin lmax ht)]
            exact ih (wOb c d si) (some (Plan.mk d lmin lmax (sideAmp q d lmin lmax).1 (sideAmp q d lmin lmax).2)) htl
          · rw [if_neg hw, if_neg hw]
            exact ih bestW best htl

theorem noTies_sound (q : Q) (h : noTies q = true) :
    ∀ ob ∈ q.base.obstacles, ∀ d lmin lmax, Cert.obsTriple ob = some (d, lmin, lmax) →
      gapLeft q d lmax ≠ gapRight q d lmin := by
  intro ob hob d lmin lmax ht
  have := List.all_eq_true.mp h ob hob
  rw [ht] at this
  simpa using this

theorem rawPlanAt_mirror (c : Consts) (q : Q) (si : Int) (h : noTies q = true) :
    rawPlanAt c (mirrorQ q) si = (rawPlanAt c q si).map flipPlan := by
  unfold rawPlanAt
  rw [mirrorQ_obstacles]
  exact go_mirror c q si q.base.obstacles 0 none (noTies_sound q h)

/-- **Mirror equivariance of every published offset.** -/
theorem offAt_mirror (c : Consts) (q : Q) (si : Int) (i : Nat) (h : noTies q = true)
    (hc : lo c q i ≤ hi c q i) : offAt c (mirrorQ q) si i = -offAt c q si i := by
  unfold offAt rawOffAt
  rw [rawPlanAt_mirror c q si h]
  cases hp : rawPlanAt c q si with
  | none => simp only [Option.map_none]; rw [show (0 : Int) = -0 by rfl]; exact clampAt_mirror c q i 0 hc
  | some p =>
      simp only [Option.map_some, flipPlan]
      rw [offOb_neg_side]
      exact clampAt_mirror c q i _ hc

theorem offs_mirror (c : Consts) (q : Q) (h : noTies q = true)
    (hc : ∀ i, i < q.base.s.length → lo c q i ≤ hi c q i) :
    offs c (mirrorQ q) = (offs c q).map (fun x => -x) := by
  unfold offs
  simp only [List.map_map]
  apply List.map_congr_left
  intro i hi
  simp only [Function.comp, mirrorQ]
  exact offAt_mirror c q _ i h (hc i (List.mem_range.mp hi))

/-- The certified clearance of the mirrored plan equals the original clearance. -/
theorem clearAt_mirror (c : Consts) (q : Q) (si : Int) (i : Nat) (x : Int) (h : noTies q = true)
    (hside : ∀ p, rawPlanAt c q si = some p → p.side = 1 ∨ p.side = -1) :
    clearAt c (mirrorQ q) si i (-x) = clearAt c q si i x := by
  unfold clearAt
  rw [rawPlanAt_mirror c q si h, lo_mirror, hi_mirror]
  cases hp : rawPlanAt c q si with
  | none => simp only [Option.map_none]; omega
  | some p =>
      simp only [Option.map_some, flipPlan]
      rcases hside p hp with hs | hs
      · simp only [hs]
        by_cases hin : p.d - c.lon.hold ≤ si ∧ si ≤ p.d + c.lon.hold
        · simp only [hin, and_self, if_true]; simp; omega
        · simp only [hin, if_false]; omega
      · simp only [hs]
        by_cases hin : p.d - c.lon.hold ≤ si ∧ si ≤ p.d + c.lon.hold
        · simp only [hin, and_self, if_true]; simp; omega
        · simp only [hin, if_false]; omega

theorem secondAt_neg (xs : List Int) (i : Nat) : secondAt (xs.map (fun x => -x)) i = secondAt xs i := by
  unfold secondAt
  simp only [List.length_map]
  split
  · rfl
  · have g : ∀ j, (xs.map (fun x => -x)).getD j 0 = -(xs.getD j 0) := by
      intro j; simp [List.getD_eq_getElem?_getD]; cases xs[j]? <;> simp
    rw [g, g, g, show -(xs.getD (i+1) 0) - 2 * -(xs.getD i 0) + -(xs.getD (i-1) 0) =
      -(xs.getD (i+1) 0 - 2 * xs.getD i 0 + xs.getD (i-1) 0) by omega, iabs_neg]

/-- Curvature demand and steering feed-forward are mirror-invariant. -/
theorem curvs_mirror (c : Consts) (q : Q) (h : noTies q = true)
    (hc : ∀ i, i < q.base.s.length → lo c q i ≤ hi c q i) :
    curvs (mirrorQ q) (offs c (mirrorQ q)) = curvs q (offs c q) ∧
    steers c (curvs (mirrorQ q) (offs c (mirrorQ q))) = steers c (curvs q (offs c q)) := by
  have hk : ∀ i, Cert.iabs ((mirrorQ q).base.k.getD i 0) = Cert.iabs (q.base.k.getD i 0) := by
    intro i
    simp only [mirrorQ]
    have : (q.base.k.map (fun z => -z)).getD i 0 = -(q.base.k.getD i 0) := by
      simp [List.getD_eq_getElem?_getD]; cases q.base.k[i]? <;> simp
    rw [this, iabs_neg]
  have hcv : curvs (mirrorQ q) (offs c (mirrorQ q)) = curvs q (offs c q) := by
    rw [offs_mirror c q h hc]
    unfold curvs curvAt
    simp only [List.length_map, secondAt_neg, hk]
  exact ⟨hcv, by rw [hcv]⟩

/-- Pass sides flip and amplitudes are preserved. -/
theorem sides_amps_mirror (q : Q) (h : noTies q = true) :
    sides (mirrorQ q) = (sides q).map (fun x => -x) ∧ amps (mirrorQ q) = amps q := by
  have hnt := noTies_sound q h
  have key : ∀ ob ∈ q.base.obstacles, sideAmpOb (mirrorQ q) (mirrorOb ob) =
      (-(sideAmpOb q ob).1, (sideAmpOb q ob).2) := by
    intro ob hob
    unfold sideAmpOb
    rw [obsTriple_mirror]
    cases ht : Cert.obsTriple ob with
    | none => simp
    | some t =>
        obtain ⟨d, lmin, lmax⟩ := t
        simp only [Option.map_some]
        exact sideAmp_mirror q d lmin lmax (hnt ob hob d lmin lmax ht)
  unfold sides amps
  rw [mirrorQ_obstacles]
  simp only [List.map_map]
  constructor
  · apply List.map_congr_left; intro ob hob; simp [Function.comp, key ob hob]
  · apply List.map_congr_left; intro ob hob; simp [Function.comp, key ob hob]

/-- Non-vacuity: the opposing-obstacle layout has no ties. -/
def mirrorInst : Q :=
  { base := { v0 := 0, s := [0, 1000, 2000], k := [0, 0, 0], wl := [3000, 3000, 3000],
              wr := [3000, 3000, 3000], obstacles := [[1000, 500, 1500]] },
    lat := 0 }

theorem mirror_hypotheses_instance : noTies mirrorInst = true ∧
    offs ⟨⟨6000, 2000, 3000, 9000, 20000, 2000, 300, 1000, 1000, 15, 2000⟩, 200, 1000, 400000⟩
      (mirrorQ mirrorInst) =
    (offs ⟨⟨6000, 2000, 3000, 9000, 20000, 2000, 300, 1000, 1000, 15, 2000⟩, 200, 1000, 400000⟩
      mirrorInst).map (fun x => -x) := by decide

end Corpus.SteerShape
