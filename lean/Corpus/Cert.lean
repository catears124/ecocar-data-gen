/-
Certified longitudinal decision, model `cert-1` (mirrors `autocross/datagen/cert.py`).

Part 1 proves, for ANY list of stations, caps and budgets:
  * `le_bpass`      : every feasible speed² profile lies below the backward pass;
  * `fpass_greatest`: every feasible profile starting at or below x lies below `fpass … x`;
  * `fpass_feasible`: `fpass … b0` is itself feasible (non-negative budgets and gaps);
  * `brake_least`   : full braking gives the pointwise-least dynamically possible profile.
Part 2 computes the model's inputs (caps, horizon) from a quantised observation exactly as the
generator does, and `decide` returns the certified decision. `decide_greatest_safe` ties the
two together: when the decision is `greatest_safe`, no feasible profile from the current
speed reaches the next station faster.
-/
import Lean.Data.Json

open Lean

namespace Corpus.Cert

/-! ## Part 1: profiles -/

/-- Backward pass over segments `(cap_i, gap_i)` (station i to i+1) and final cap `cN`. -/
def bpass (aB : Int) : List (Int × Int) → Int → List Int
  | [], cN => [cN]
  | (c, d) :: rest, cN => min c ((bpass aB rest cN).headD 0 + 2 * aB * d) :: bpass aB rest cN

/-- Forward pass from start value `x` (speed² at station 0). -/
def fpass (aA aB : Int) : List (Int × Int) → Int → Int → List Int
  | [], _, x => [x]
  | (_, d) :: rest, cN, x => x :: fpass aA aB rest cN (min ((bpass aB rest cN).headD 0) (x + 2 * aA * d))

/-- A speed² profile respecting caps, the acceleration budget `aA` and the braking budget `aB`. -/
def Feasible (aA aB : Int) : List (Int × Int) → Int → List Int → Prop
  | [], cN, [x] => x ≤ cN
  | (c, d) :: rest, cN, x :: y :: t =>
      x ≤ c ∧ y ≤ x + 2 * aA * d ∧ x ≤ y + 2 * aB * d ∧ Feasible aA aB rest cN (y :: t)
  | _, _, _ => False

/-- Pointwise ≤ for equal-length lists. -/
def LeAll : List Int → List Int → Prop
  | [], [] => True
  | x :: xs, y :: ys => x ≤ y ∧ LeAll xs ys
  | _, _ => False

theorem LeAll_head {y : Int} {t L : List Int} (h : LeAll (y :: t) L) : y ≤ L.headD 0 := by
  cases L with
  | nil => exact False.elim h
  | cons z zs => exact h.1

theorem fpass_head (aA aB : Int) (segs : List (Int × Int)) (cN x : Int) :
    ∃ t, fpass aA aB segs cN x = x :: t := by
  cases segs with
  | nil => exact ⟨[], rfl⟩
  | cons p rest => obtain ⟨c, d⟩ := p; exact ⟨_, rfl⟩

theorem le_bpass (aA aB : Int) : ∀ (segs : List (Int × Int)) (cN : Int) (b : List Int),
    Feasible aA aB segs cN b → LeAll b (bpass aB segs cN) := by
  intro segs
  induction segs with
  | nil =>
    intro cN b h
    match b, h with
    | [x], h => exact ⟨h, trivial⟩
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cN b h
    match b, h with
    | x :: y :: t, ⟨hc, _, hb, hf⟩ =>
      have hl := ih cN (y :: t) hf
      have hy := LeAll_head hl
      refine ⟨?_, hl⟩
      show x ≤ min c ((bpass aB rest cN).headD 0 + 2 * aB * d)
      generalize (bpass aB rest cN).headD 0 = H at *
      generalize 2 * aB * d = e at *
      omega

theorem fpass_greatest (aA aB : Int) : ∀ (segs : List (Int × Int)) (cN x x' : Int) (t : List Int),
    Feasible aA aB segs cN (x' :: t) → x' ≤ x → LeAll (x' :: t) (fpass aA aB segs cN x) := by
  intro segs
  induction segs with
  | nil =>
    intro cN x x' t h hx
    match t, h with
    | [], _ => exact ⟨hx, trivial⟩
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cN x x' t h hx
    match t, h with
    | y :: t', ⟨_, ha, _, hf⟩ =>
      have hy := LeAll_head (le_bpass aA aB rest cN (y :: t') hf)
      refine ⟨hx, ih cN _ y t' hf ?_⟩
      generalize 2 * aA * d = e at *
      omega

theorem fpass_feasible (aA aB : Int) (haA : 0 ≤ aA) (haB : 0 ≤ aB) :
    ∀ (segs : List (Int × Int)) (cN x : Int), (∀ p ∈ segs, 0 ≤ p.2) →
    x ≤ (bpass aB segs cN).headD 0 → Feasible aA aB segs cN (fpass aA aB segs cN x) := by
  intro segs
  induction segs with
  | nil => intro cN x _ hx; simpa [fpass, bpass, Feasible] using hx
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cN x hd hx
    have hd0 : 0 ≤ d := hd (c, d) (by simp)
    have hrest : ∀ q ∈ rest, 0 ≤ q.2 := fun q hq => hd q (by simp [hq])
    have hA : 0 ≤ 2 * aA * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haA) hd0
    have hB : 0 ≤ 2 * aB * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haB) hd0
    obtain ⟨t, ht⟩ := fpass_head aA aB rest cN (min ((bpass aB rest cN).headD 0) (x + 2 * aA * d))
    have hnext := ih cN (min ((bpass aB rest cN).headD 0) (x + 2 * aA * d)) hrest (Int.min_le_left _ _)
    simp only [fpass, ht, Feasible]
    rw [ht] at hnext
    change x ≤ min c ((bpass aB rest cN).headD 0 + 2 * aB * d) at hx
    refine ⟨?_, ?_, ?_, hnext⟩
    · omega
    · exact Int.min_le_right _ _
    · generalize (bpass aB rest cN).headD 0 = H at *
      generalize 2 * aA * d = eA at *
      generalize 2 * aB * d = eB at *
      omega

/-- Speed² profile of full braking at `aM` (a speed² can not drop below 0). -/
def brakeProfile (aM : Int) : List Int → Int → List Int
  | [], x => [x]
  | d :: ds, x => x :: brakeProfile aM ds (max 0 (x - 2 * aM * d))

/-- Physically possible profile: speed² ≥ 0 and no deceleration beyond `aM`. -/
def DynPossible (aM : Int) : List Int → List Int → Prop
  | [], [x] => 0 ≤ x
  | d :: ds, x :: y :: t => 0 ≤ x ∧ x - 2 * aM * d ≤ y ∧ DynPossible aM ds (y :: t)
  | _, _ => False

theorem DynPossible_head_nonneg {aM : Int} : ∀ {ds : List Int} {y : Int} {t : List Int},
    DynPossible aM ds (y :: t) → 0 ≤ y := by
  intro ds y t h
  cases ds with
  | nil => match t, h with | [], h => exact h
  | cons d ds => match t, h with | _ :: _, h => exact h.1

theorem brake_least (aM : Int) : ∀ (ds : List Int) (x x' : Int) (t : List Int),
    DynPossible aM ds (x' :: t) → x ≤ x' → LeAll (brakeProfile aM ds x) (x' :: t) := by
  intro ds
  induction ds with
  | nil => intro x x' t h hx; match t, h with | [], _ => exact ⟨hx, trivial⟩
  | cons d ds ih =>
    intro x x' t h hx
    match t, h with
    | y :: t', ⟨_, hy, hf⟩ =>
      have h0 := DynPossible_head_nonneg hf
      refine ⟨hx, ih _ y t' hf ?_⟩
      generalize 2 * aM * d = e at *
      omega

/-- Any objective that does not increase when speeds rise pointwise (e.g. traversal time)
is minimised by a pointwise-greatest feasible profile. -/
theorem greatest_minimises {T : List Int → Int} (hT : ∀ a b, LeAll a b → T b ≤ T a)
    {b F : List Int} (h : LeAll b F) : T F ≤ T b := hT b F h

/-! ## Part 2: model `cert-1` on a quantised observation -/

structure Consts where
  a_lat : Int
  a_acc : Int
  a_brk : Int
  a_max : Int
  v_max : Int
  r_h : Int
  hold : Int
  ramp_in : Int
  ramp_out : Int
  r_sw_m : Int
  gap_base : Int
  gap_per_m : Int
  veh_width : Int
  gap_margin : Int
deriving Repr, DecidableEq, FromJson

/-- Quantised observation: speed (mm/s), preview stations (mm), curvature (1e-6 / m), widths (mm),
obstacles `[distance, lateral_min, lateral_max]` (mm). -/
structure Q where
  v0 : Int
  s : List Int
  k : List Int
  wl : List Int
  wr : List Int
  obstacles : List (List Int)
deriving Repr, FromJson

def iabs (x : Int) : Int := if x < 0 then -x else x

/-- Ceiling division for a positive divisor (`/` on `Int` is Euclidean, i.e. floor here). -/
def cdiv (a b : Int) : Int := -((-a) / b)

def nearestIndex (s : List Int) (d : Int) : Nat :=
  (s.findIdx? (fun x => decide (d ≤ x))).getD (s.length - 1)

/-- (side, |target| bound, gap) of one obstacle, from the widths at the first station at or beyond it. -/
def plan (q : Q) (d lmin lmax : Int) : Int × Int × Int :=
  let i := nearestIndex q.s d
  let wl := q.wl.getD i 0
  let wr := q.wr.getD i 0
  let gl := wl - lmax
  let gr := lmin + wr
  let yl := cdiv (iabs (lmax + wl)) 2
  let yr := cdiv (iabs (lmin - wr)) 2
  if gl > gr then (1, yl, gl) else if gr > gl then (-1, yr, gr) else (0, max yl yr, gl)

def obsTriple : List Int → Option (Int × Int × Int)
  | [d, lmin, lmax] => some (d, lmin, lmax)
  | _ => none

def stationY (c : Consts) (q : Q) (si : Int) : Int :=
  q.obstacles.foldl (fun acc ob => match obsTriple ob with
    | some (d, lmin, lmax) =>
        if d - c.hold - c.ramp_in ≤ si ∧ si ≤ d + c.hold + c.ramp_out then max acc (plan q d lmin lmax).2.1 else acc
    | none => acc) 0

def stationGapCap (c : Consts) (q : Q) (si : Int) : Option Int :=
  q.obstacles.foldl (fun acc ob => match obsTriple ob with
    | some (d, lmin, lmax) =>
        if d - c.hold ≤ si ∧ si ≤ d + c.hold then
          let vg := c.gap_base + c.gap_per_m * max ((plan q d lmin lmax).2.2 - c.veh_width) 0
          some (match acc with | some g => min g (vg * vg) | none => vg * vg)
        else acc
    | none => acc) none

def gapsFeasible (c : Consts) (q : Q) : Bool :=
  q.obstacles.all fun ob => match obsTriple ob with
    | some (d, lmin, lmax) => decide ((plan q d lmin lmax).2.2 ≥ c.veh_width + 2 * c.gap_margin)
    | none => false

def capAt (c : Consts) (q : Q) (i : Nat) : Int :=
  let si := q.s.getD i 0
  let y := stationY c q si
  let ka := iabs (q.k.getD i 0)
  let ksw := cdiv (6000 * y) (c.r_sw_m * c.r_sw_m)
  let den := max (1000000 - cdiv (ka * y) 1000) 200000
  let kbar := cdiv ((ka + ksw) * 1000000) den
  let vmax2 := c.v_max * c.v_max
  let cap := if kbar = 0 then vmax2 else min vmax2 (c.a_lat * 1000000000 / kbar)
  match stationGapCap c q si with
  | some g => min cap g
  | none => cap

/-- Segments `(cap_i, s_{i+1} - s_i)` for i < n-1, and the final cap with the horizon bound. -/
def segs (c : Consts) (q : Q) : List (Int × Int) :=
  (List.range (q.s.length - 1)).map fun i => (capAt c q i, q.s.getD (i + 1) 0 - q.s.getD i 0)

def lastCap (c : Consts) (q : Q) : Int := min (capAt c q (q.s.length - 1)) (c.a_lat * c.r_h)

def strictlyIncreasing : List Int → Bool
  | x :: y :: t => decide (x < y) && strictlyIncreasing (y :: t)
  | _ => true

def wellFormed (q : Q) : Bool :=
  decide (2 ≤ q.s.length) && q.s.headD 1 == 0 && strictlyIncreasing q.s &&
  q.k.length == q.s.length && q.wl.length == q.s.length && q.wr.length == q.s.length &&
  q.obstacles.all (fun ob => ob.length == 3)

structure Decision where
  kind : Nat
  a : Int
  b1 : Option Int
deriving Repr, DecidableEq, FromJson

def kindGreatestSafe : Nat := 0
def kindMinViolation : Nat := 1

/-- The certified decision (same branches and arithmetic as `cert.decide`). -/
def decide (c : Consts) (q : Q) : Decision :=
  if !wellFormed q then ⟨3, -c.a_max, none⟩
  else if !gapsFeasible c q then ⟨2, -c.a_max, none⟩
  else if q.v0 * q.v0 > (bpass c.a_brk (segs c q) (lastCap c q)).headD 0 then ⟨1, -c.a_max, none⟩
  else
    ⟨0, ((fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0 - q.v0 * q.v0) /
          (2 * (q.s.getD 1 0 - q.s.getD 0 0)),
     some ((fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0)⟩

/-- Element 1 of a pointwise-dominated list is dominated. -/
theorem LeAll_get1 : ∀ {a b : List Int}, LeAll a b → a.getD 1 0 ≤ b.getD 1 0
  | [], [], _ => Int.le_refl 0
  | [_], [_], _ => Int.le_refl 0
  | _ :: a1 :: _, _ :: b1 :: _, ⟨_, h1, _⟩ => h1
  | [], _ :: _, h => by simp [LeAll] at h
  | _ :: _, [], h => by simp [LeAll] at h
  | [_], _ :: _ :: _, h => by simp [LeAll] at h
  | _ :: _ :: _, [_], h => by simp [LeAll] at h

/-- When `decide` returns `greatest_safe`, its next-station speed² `b1` is at least that of every
feasible profile from the current speed: no command that keeps the plan feasible for all
continuations beyond the preview can be faster at the next station. -/
theorem decide_greatest_safe (c : Consts) (q : Q) (b1 : Int)
    (hd : (decide c q).kind = kindGreatestSafe) (hb : (decide c q).b1 = some b1)
    (t : List Int) (hf : Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: t)) :
    (q.v0 * q.v0 :: t).getD 1 0 ≤ b1 := by
  have hle := LeAll_get1 (fpass_greatest c.a_acc c.a_brk (segs c q) (lastCap c q)
    (q.v0 * q.v0) (q.v0 * q.v0) t hf (Int.le_refl _))
  unfold decide at hb
  split at hb
  · simp at hb
  split at hb
  · simp at hb
  split at hb
  · simp at hb
  simp at hb
  rw [← hb]
  exact hle

/-- When `decide` returns `min_violation`, the label is full braking at `a_max`, and the
full-braking profile is below every physically possible profile from the current speed. -/
theorem decide_min_violation (c : Consts) (q : Q) (hd : (decide c q).kind = kindMinViolation)
    (ds : List Int) (t : List Int) (hp : DynPossible c.a_max ds (q.v0 * q.v0 :: t)) :
    (decide c q).a = -c.a_max ∧ LeAll (brakeProfile c.a_max ds (q.v0 * q.v0)) (q.v0 * q.v0 :: t) := by
  refine ⟨?_, brake_least c.a_max ds _ _ t hp (Int.le_refl _)⟩
  unfold decide at hd ⊢
  split
  · rfl
  split
  · rfl
  split
  · rfl
  rename_i h1 h2 h3
  rw [if_neg h1, if_neg h2, if_neg h3] at hd
  exact absurd hd (by simp [kindMinViolation])

end Corpus.Cert
