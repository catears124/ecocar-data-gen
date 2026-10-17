/-
Certified longitudinal decision, model `cert-2` (mirrors `autocross/datagen/cert.py` line by line).

`cert-2` is the `cert-1` decision arithmetic with explicit domain guards (non-negative speed,
sane constants, non-negative caps): on every input inside that domain the two models return the
same decision; outside it `cert-2` returns `invalid_input` instead of a number.

## What is modelled (all integers: mm, mm/s, mm²/s², mm/s², 1e-6 / m)

Stations `s₀ = 0 < s₁ < … < sₙ` of the visible preview, unknowns `bᵢ = vᵢ²`, and a list of
segments `(capᵢ, dᵢ = sᵢ₊₁ − sᵢ)` plus a final cap. A speed² profile is `Feasible` iff
  * every entry is ≥ 0 (speeds are real),
  * `bᵢ ≤ capᵢ` (grip / gap / speed-limit caps computed from the observation),
  * `bᵢ₊₁ ≤ bᵢ + 2·a_acc·dᵢ` (acceleration budget) and `bᵢ ≤ bᵢ₊₁ + 2·a_brk·dᵢ` (braking budget),
  * `bₙ ≤ min(capₙ, a_lat·R_H)`: the DECLARED terminal constraint (be able to take a corner of
    radius `R_H` that could start right after the preview). It is an assumption of the model,
    not a theorem about every possible unseen road.
The start `b₀ = v₀²` is fixed by the observation.

## What is proved (Part 1 for arbitrary lists, Part 2 for `decide` on a quantised observation)

  * `le_bpass`, `fpass_greatest`: every feasible profile with `b₀' ≤ x` is pointwise ≤ `fpass … x`.
  * `fpass_feasible`: `fpass … x` is itself feasible (non-negative budgets, gaps and caps, `0 ≤ x`).
  * `accel_greatest`: the emitted INTEGER acceleration `a = ⌊(F₁ − b₀) / (2 d₀)⌋` (mm/s²) lies in
    `[−a_brk, a_acc]`, its constant-acceleration end point `max 0 (b₀ + 2·a·d₀)` starts a feasible
    profile, and every integer `a'` whose end point starts a feasible profile satisfies `a' ≤ a`.
    Rounding is by floor, so the command never exceeds the certified `F₁` and, because
    `F₁ ≥ b₀ − 2·a_brk·d₀`, never brakes harder than the braking budget.
  * `greatest_minimises`: an objective that does not increase when speeds rise pointwise
    (e.g. traversal time) is minimised by the pointwise-greatest feasible profile.
  * `brake_least`: full braking at `a_max` gives the pointwise-least physically possible profile;
    `brake_minimises_excess`: it therefore minimises the total excess over ANY upper speed caps
    (and any other objective that is monotone non-decreasing in the speeds). It does NOT
    minimise arbitrary constraint violations (e.g. violations of lower speed bounds).
  * Part 2: `decide_greatest_safe`, `decide_greatest_safe_feasible`, `decide_accel_greatest`,
    `decide_greatest_safe_minimises`, `decide_min_violation`, `decide_min_violation_infeasible`.

Nothing here is about steering, the lateral path, a simulator's dynamics or a learned policy.
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

/-- A non-negative speed² profile respecting caps, the acceleration budget `aA` and the braking
budget `aB`; the last entry respects the terminal cap `cN`. -/
def Feasible (aA aB : Int) : List (Int × Int) → Int → List Int → Prop
  | [], cN, [x] => 0 ≤ x ∧ x ≤ cN
  | (c, d) :: rest, cN, x :: y :: t =>
      0 ≤ x ∧ x ≤ c ∧ y ≤ x + 2 * aA * d ∧ x ≤ y + 2 * aB * d ∧ Feasible aA aB rest cN (y :: t)
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
    | [x], h => exact ⟨h.2, trivial⟩
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cN b h
    match b, h with
    | x :: y :: t, ⟨_, hc, _, hb, hf⟩ =>
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
    | y :: t', ⟨_, _, ha, _, hf⟩ =>
      have hy := LeAll_head (le_bpass aA aB rest cN (y :: t') hf)
      refine ⟨hx, ih cN _ y t' hf ?_⟩
      generalize 2 * aA * d = e at *
      omega

/-- Hypotheses on the segment data: non-negative caps and gaps. -/
def SegsOK (segs : List (Int × Int)) (cN : Int) : Prop := (∀ p ∈ segs, 0 ≤ p.1 ∧ 0 ≤ p.2) ∧ 0 ≤ cN

theorem SegsOK.tail {p : Int × Int} {rest : List (Int × Int)} {cN : Int} (h : SegsOK (p :: rest) cN) :
    SegsOK rest cN :=
  ⟨fun q hq => h.1 q (by simp [hq]), h.2⟩

theorem bpass_head_nonneg (aB : Int) (haB : 0 ≤ aB) : ∀ (segs : List (Int × Int)) (cN : Int),
    SegsOK segs cN → 0 ≤ (bpass aB segs cN).headD 0 := by
  intro segs
  induction segs with
  | nil => intro cN h; exact h.2
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cN h
    have hp := h.1 (c, d) (by simp)
    have hr := ih cN h.tail
    have hB : 0 ≤ 2 * aB * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haB) hp.2
    show 0 ≤ min c ((bpass aB rest cN).headD 0 + 2 * aB * d)
    generalize (bpass aB rest cN).headD 0 = H at *
    generalize 2 * aB * d = e at *
    simp only at hp
    omega

theorem fpass_feasible (aA aB : Int) (haA : 0 ≤ aA) (haB : 0 ≤ aB) :
    ∀ (segs : List (Int × Int)) (cN x : Int), SegsOK segs cN → 0 ≤ x →
    x ≤ (bpass aB segs cN).headD 0 → Feasible aA aB segs cN (fpass aA aB segs cN x) := by
  intro segs
  induction segs with
  | nil => intro cN x _ h0 hx; exact ⟨h0, hx⟩
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cN x hs h0 hx
    have hd0 : 0 ≤ d := (hs.1 (c, d) (by simp)).2
    have hA : 0 ≤ 2 * aA * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haA) hd0
    have hB : 0 ≤ 2 * aB * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haB) hd0
    have hH := bpass_head_nonneg aB haB rest cN hs.tail
    obtain ⟨t, ht⟩ := fpass_head aA aB rest cN (min ((bpass aB rest cN).headD 0) (x + 2 * aA * d))
    have hnext := ih cN (min ((bpass aB rest cN).headD 0) (x + 2 * aA * d)) hs.tail
      (by generalize (bpass aB rest cN).headD 0 = H at *; generalize 2 * aA * d = e at *; omega)
      (Int.min_le_left _ _)
    simp only [fpass, ht, Feasible]
    rw [ht] at hnext
    change x ≤ min c ((bpass aB rest cN).headD 0 + 2 * aB * d) at hx
    refine ⟨h0, ?_, ?_, ?_, hnext⟩
    · omega
    · exact Int.min_le_right _ _
    · generalize (bpass aB rest cN).headD 0 = H at *
      generalize 2 * aA * d = eA at *
      generalize 2 * aB * d = eB at *
      omega

/-! ### Robustness: what the terminal constraint buys (the same-information comparison) -/

/-- Last entry of a list (0 for `[]`). -/
def lastD : List Int → Int
  | [] => 0
  | [x] => x
  | _ :: y :: t => lastD (y :: t)

/-- Holding speed² `x` is feasible through any segments whose caps are ≥ x and gaps are ≥ 0. -/
theorem hold_feasible (aA aB : Int) (haA : 0 ≤ aA) (haB : 0 ≤ aB) (x : Int) (h0 : 0 ≤ x) :
    ∀ (cont : List (Int × Int)) (cE : Int), (∀ p ∈ cont, x ≤ p.1 ∧ 0 ≤ p.2) → x ≤ cE →
    Feasible aA aB cont cE (x :: List.replicate cont.length x) := by
  intro cont
  induction cont with
  | nil => intro cE _ hE; exact ⟨h0, hE⟩
  | cons p rest ih =>
    obtain ⟨c, d⟩ := p
    intro cE hp hE
    have hc := hp (c, d) (by simp)
    have hA : 0 ≤ 2 * aA * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haA) hc.2
    have hB : 0 ≤ 2 * aB * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haB) hc.2
    have hrest := ih cE (fun q hq => hp q (by simp [hq])) hE
    simp only [List.length_cons, List.replicate_succ]
    exact ⟨h0, hc.1, by omega, by omega, hrest⟩

/-- Sufficiency. A profile that is feasible over the preview with terminal cap `min cP cH`
continues safely (holding its last speed) through EVERY continuation of the road: the segment from
the last preview station (own cap `cP`) of any length `d ≥ 0`, then any segments whose caps are all
`≥ cH`, ending at any cap `≥ cH`. -/
theorem robust_extend (aA aB cP cH : Int) (haA : 0 ≤ aA) (haB : 0 ≤ aB) :
    ∀ (segs : List (Int × Int)) (b : List Int), Feasible aA aB segs (min cP cH) b →
    ∀ (d : Int) (cont : List (Int × Int)) (cE : Int), 0 ≤ d → (∀ p ∈ cont, cH ≤ p.1 ∧ 0 ≤ p.2) → cH ≤ cE →
    Feasible aA aB (segs ++ (cP, d) :: cont) cE (b ++ List.replicate (cont.length + 1) (lastD b)) := by
  intro segs
  induction segs with
  | nil =>
    intro b hb d cont cE hd hc hE
    match b, hb with
    | [x], ⟨h0, hx⟩ =>
      have hx1 : x ≤ cP := Int.le_trans hx (Int.min_le_left _ _)
      have hx2 : x ≤ cH := Int.le_trans hx (Int.min_le_right _ _)
      have := hold_feasible aA aB haA haB x h0 ((cP, d) :: cont) cE
        (fun p hp => by
          rcases List.mem_cons.mp hp with rfl | hp'
          · exact ⟨hx1, hd⟩
          · exact ⟨Int.le_trans hx2 (hc p hp').1, (hc p hp').2⟩) (Int.le_trans hx2 hE)
      simpa [lastD] using this
  | cons p rest ih =>
    obtain ⟨c, d'⟩ := p
    intro b hb d cont cE hd hc hE
    match b, hb with
    | x :: y :: t, ⟨h0, h1, h2, h3, hf⟩ =>
      have := ih (y :: t) hf d cont cE hd hc hE
      exact ⟨h0, h1, h2, h3, this⟩

/-- Necessity. The worst in-domain continuation puts a station with cap `cH` immediately after
the last preview station (gap 0). Any profile safe on that road satisfies the terminal cap. -/
theorem robust_necessary (aA aB cP cH : Int) : ∀ (segs : List (Int × Int)) (b : List Int) (y : Int),
    Feasible aA aB (segs ++ [(cP, 0)]) cH (b ++ [y]) → Feasible aA aB segs (min cP cH) b := by
  intro segs
  induction segs with
  | nil =>
    intro b y h
    match b, h with
    | [x], ⟨h0, h1, _, h3, h4⟩ =>
      have e : (2 * aB * 0 : Int) = 0 := by simp
      rw [e] at h3
      exact ⟨h0, by have := h4.2; omega⟩
    | [], h => exact h.elim
    | _ :: _ :: t, h =>
      cases t with
      | nil => exact h.2.2.2.2.elim
      | cons _ _ => exact h.2.2.2.2.elim
  | cons p rest ih =>
    obtain ⟨c, d'⟩ := p
    intro b y h
    match b, h with
    | x :: z :: t, ⟨h0, h1, h2, h3, hf⟩ => exact ⟨h0, h1, h2, h3, ih (z :: t) y hf⟩
    | [], h => exact h.elim
    | [_], h =>
      cases rest with
      | nil => exact h.2.2.2.2.elim
      | cons _ _ => exact h.2.2.2.2.elim

/-- Second entry of the forward pass from `b0` over a non-empty segment list. -/
theorem fpass_cons_get1 (aA aB c0 d cN b0 : Int) (rest : List (Int × Int)) :
    (fpass aA aB ((c0, d) :: rest) cN b0).getD 1 0 =
      min ((bpass aB rest cN).headD 0) (b0 + 2 * aA * d) := by
  obtain ⟨t, ht⟩ := fpass_head aA aB rest cN (min ((bpass aB rest cN).headD 0) (b0 + 2 * aA * d))
  show (b0 :: fpass aA aB rest cN (min ((bpass aB rest cN).headD 0) (b0 + 2 * aA * d))).getD 1 0 = _
  rw [ht]; rfl

/-- Element 1 of a pointwise-dominated list is dominated. -/
theorem LeAll_get1 : ∀ {a b : List Int}, LeAll a b → a.getD 1 0 ≤ b.getD 1 0
  | [], [], _ => Int.le_refl 0
  | [_], [_], _ => Int.le_refl 0
  | _ :: _ :: _, _ :: _ :: _, ⟨_, h1, _⟩ => h1
  | [], _ :: _, h => by simp [LeAll] at h
  | _ :: _, [], h => by simp [LeAll] at h
  | [_], _ :: _ :: _, h => by simp [LeAll] at h
  | _ :: _ :: _, [_], h => by simp [LeAll] at h

/-- The integer acceleration emitted by the planner: floor of `(F₁ − b₀) / (2 d)`. -/
def accelOf (F1 b0 d : Int) : Int := (F1 - b0) / (2 * d)

/-- Integer-action optimality of the emitted command over the first segment (see module doc). -/
theorem accel_greatest (aA aB c0 d cN b0 : Int) (rest : List (Int × Int))
    (haA : 0 ≤ aA) (haB : 0 ≤ aB) (hd : 0 < d) (hs : SegsOK ((c0, d) :: rest) cN) (hb0 : 0 ≤ b0)
    (hB : b0 ≤ (bpass aB ((c0, d) :: rest) cN).headD 0) :
    let F1 := (fpass aA aB ((c0, d) :: rest) cN b0).getD 1 0
    let a := accelOf F1 b0 d
    (-aB) ≤ a ∧ a ≤ aA ∧
    Feasible aA aB ((c0, d) :: rest) cN (b0 :: fpass aA aB rest cN (max 0 (b0 + a * (2 * d)))) ∧
    (∀ (a' : Int) (t : List Int),
      Feasible aA aB ((c0, d) :: rest) cN (b0 :: max 0 (b0 + a' * (2 * d)) :: t) → a' ≤ a) := by
  intro F1 a
  have h2d : 0 < 2 * d := by omega
  have hF1 : F1 = min ((bpass aB rest cN).headD 0) (b0 + 2 * aA * d) := fpass_cons_get1 aA aB c0 d cN b0 rest
  have hH := bpass_head_nonneg aB haB rest cN hs.tail
  have hA : 0 ≤ 2 * aA * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haA) (by omega)
  have hBd : 0 ≤ 2 * aB * d := Int.mul_nonneg (Int.mul_nonneg (by decide) haB) (by omega)
  have hc0 := (hs.1 (c0, d) (by simp)).1
  change b0 ≤ min c0 ((bpass aB rest cN).headD 0 + 2 * aB * d) at hB
  -- normal forms of the products
  have eA : aA * (2 * d) = 2 * aA * d := by
    rw [Int.mul_comm aA, Int.mul_assoc, Int.mul_comm d aA, ← Int.mul_assoc]
  have eB : -aB * (2 * d) = -(2 * aB * d) := by
    rw [Int.neg_mul, Int.mul_comm aB, Int.mul_assoc, Int.mul_comm d aB, ← Int.mul_assoc]
  -- floor properties of `a`
  have hlo : a * (2 * d) ≤ F1 - b0 := Int.ediv_mul_le _ (by omega)
  have hlow : -aB ≤ a := by
    show -aB ≤ (F1 - b0) / (2 * d)
    apply Int.le_ediv_of_mul_le h2d
    rw [eB, hF1]
    generalize (bpass aB rest cN).headD 0 = H at *
    generalize 2 * aB * d = e at *
    generalize 2 * aA * d = e' at *
    omega
  have hup : a ≤ aA := by
    show (F1 - b0) / (2 * d) ≤ aA
    apply Int.ediv_le_of_le_mul h2d
    rw [eA, hF1]
    generalize (bpass aB rest cN).headD 0 = H at *
    generalize 2 * aA * d = e' at *
    omega
  have hmono : -aB * (2 * d) ≤ a * (2 * d) := Int.mul_le_mul_of_nonneg_right hlow (by omega)
  rw [eB] at hmono
  refine ⟨hlow, hup, ?_, ?_⟩
  · -- the clamped end point starts a feasible profile
    obtain ⟨t, ht⟩ := fpass_head aA aB rest cN (max 0 (b0 + a * (2 * d)))
    have hy0 : 0 ≤ max 0 (b0 + a * (2 * d)) := Int.le_max_left _ _
    have hyH : max 0 (b0 + a * (2 * d)) ≤ (bpass aB rest cN).headD 0 := by
      rw [hF1] at hlo
      generalize (bpass aB rest cN).headD 0 = H at *
      generalize a * (2 * d) = m at *
      generalize 2 * aA * d = e' at *
      omega
    have hrest := fpass_feasible aA aB haA haB rest cN _ hs.tail hy0 hyH
    rw [ht] at hrest ⊢
    refine ⟨hb0, ?_, ?_, ?_, hrest⟩
    · omega
    · rw [hF1] at hlo
      generalize (bpass aB rest cN).headD 0 = H at *
      generalize a * (2 * d) = m at *
      generalize 2 * aA * d = e' at *
      omega
    · generalize a * (2 * d) = m at *
      generalize 2 * aB * d = e at *
      omega
  · -- greatest among integer commands
    intro a' t hf
    have hle := LeAll_get1 (fpass_greatest aA aB ((c0, d) :: rest) cN b0 b0 _ hf (Int.le_refl _))
    show a' ≤ (F1 - b0) / (2 * d)
    apply Int.le_ediv_of_mul_le h2d
    have : b0 + a' * (2 * d) ≤ max 0 (b0 + a' * (2 * d)) := Int.le_max_right _ _
    change max 0 (b0 + a' * (2 * d)) ≤ F1 at hle
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

/-- Total excess of a speed² profile over upper caps (missing caps count as no cap). -/
def capExcess : List Int → List Int → Int
  | c :: cs, x :: xs => max 0 (x - c) + capExcess cs xs
  | _, _ => 0

theorem capExcess_mono : ∀ (cs a b : List Int), LeAll a b → capExcess cs a ≤ capExcess cs b
  | [], _, _, _ => by simp [capExcess]
  | _ :: _, [], [], _ => by simp [capExcess]
  | c :: cs, x :: xs, y :: ys, ⟨h, hs⟩ => by
    have := capExcess_mono cs xs ys hs
    simp only [capExcess]
    have : max 0 (x - c) ≤ max 0 (y - c) := by omega
    omega
  | _ :: _, [], _ :: _, h => by simp [LeAll] at h
  | _ :: _, _ :: _, [], h => by simp [LeAll] at h

/-- Full braking minimises the excess over any upper speed caps among all physically possible
profiles from the current speed (an objective monotone in the speeds; not arbitrary violations). -/
theorem brake_minimises_excess (aM : Int) (cs ds : List Int) (x : Int) (t : List Int)
    (h : DynPossible aM ds (x :: t)) :
    capExcess cs (brakeProfile aM ds x) ≤ capExcess cs (x :: t) :=
  capExcess_mono cs _ _ (brake_least aM ds x x t h (Int.le_refl _))

/-! ## Part 2: model `cert-2` on a quantised observation -/

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
  veh_width : Int
deriving Repr, DecidableEq, FromJson, ToJson

/-- Quantised observation: speed (mm/s), preview stations (mm), curvature (1e-6 / m), widths (mm),
obstacles `[distance, lateral_min, lateral_max]` (mm). -/
structure Q where
  v0 : Int
  s : List Int
  k : List Int
  wl : List Int
  wr : List Int
  obstacles : List (List Int)
deriving Repr, DecidableEq, FromJson, ToJson

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

def gapsFeasible (c : Consts) (q : Q) : Bool :=
  q.obstacles.all fun ob => match obsTriple ob with
    | some (d, lmin, lmax) => decide ((plan q d lmin lmax).2.2 ≥ c.veh_width)
    | none => false

/-- Model bound (1e-6 / m) on the planned path's curvature for road curvature `ka` and sidestep `y`. -/
def kbarOf (c : Consts) (ka y : Int) : Int :=
  let ksw := cdiv (6000 * y) (c.r_sw_m * c.r_sw_m)
  let den := max (1000000 - cdiv (ka * y) 1000) 200000
  cdiv ((ka + ksw) * 1000000) den

def capAt (c : Consts) (q : Q) (i : Nat) : Int :=
  let kbar := kbarOf c (iabs (q.k.getD i 0)) (stationY c q (q.s.getD i 0))
  let vmax2 := c.v_max * c.v_max
  if kbar = 0 then vmax2 else min vmax2 (c.a_lat * 1000000000 / kbar)

/-- Segments `(cap_i, s_{i+1} - s_i)` for i < n-1, and the final cap with the horizon bound. -/
def segs (c : Consts) (q : Q) : List (Int × Int) :=
  (List.range (q.s.length - 1)).map fun i => (capAt c q i, q.s.getD (i + 1) 0 - q.s.getD i 0)

def lastCap (c : Consts) (q : Q) : Int := min (capAt c q (q.s.length - 1)) (c.a_lat * c.r_h)

def strictlyIncreasing : List Int → Bool
  | x :: y :: t => decide (x < y) && strictlyIncreasing (y :: t)
  | _ => true

/-- Domain of the model: at least two stations starting at 0, strictly increasing, aligned
vectors, obstacle triples, and a non-negative current speed. -/
def wellFormed (q : Q) : Bool :=
  decide (2 ≤ q.s.length) && q.s.headD 1 == 0 && strictlyIncreasing q.s &&
  q.k.length == q.s.length && q.wl.length == q.s.length && q.wr.length == q.s.length &&
  q.obstacles.all (fun ob => ob.length == 3) && decide (0 ≤ q.v0)

/-- Constants for which the theorems hold: non-negative budgets, positive full-brake deceleration
and ramp length. -/
def constsOK (c : Consts) : Bool :=
  decide (0 ≤ c.a_acc) && decide (0 ≤ c.a_brk) && decide (0 < c.a_max) && decide (0 < c.r_sw_m)

/-- Every cap (and the terminal cap) is non-negative. -/
def capsNonneg (c : Consts) (q : Q) : Bool :=
  (segs c q).all (fun p => decide (0 ≤ p.1)) && decide (0 ≤ lastCap c q)

structure Decision where
  kind : Nat
  a : Int
  b1 : Option Int
  b0 : Option Int
  B0 : Option Int
deriving Repr, DecidableEq, FromJson, ToJson

def kindGreatestSafe : Nat := 0
def kindMinViolation : Nat := 1
def kindNoGap : Nat := 2
def kindInvalid : Nat := 3

/-- The certified decision (same branches and arithmetic as `cert.decide`). -/
def decide (c : Consts) (q : Q) : Decision :=
  if !(wellFormed q && constsOK c && capsNonneg c q) then ⟨kindInvalid, -c.a_max, none, none, none⟩
  else if !gapsFeasible c q then ⟨kindNoGap, -c.a_max, none, some (q.v0 * q.v0), none⟩
  else if q.v0 * q.v0 > (bpass c.a_brk (segs c q) (lastCap c q)).headD 0 then
    ⟨kindMinViolation, -c.a_max, none, some (q.v0 * q.v0), some ((bpass c.a_brk (segs c q) (lastCap c q)).headD 0)⟩
  else
    ⟨kindGreatestSafe,
     accelOf ((fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0) (q.v0 * q.v0)
       (q.s.getD 1 0 - q.s.getD 0 0),
     some ((fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0),
     some (q.v0 * q.v0), some ((bpass c.a_brk (segs c q) (lastCap c q)).headD 0)⟩


/-! ### Facts about the domain -/

theorem strictlyIncreasing_first : ∀ {s : List Int}, strictlyIncreasing s = true → 2 ≤ s.length →
    s.getD 0 0 < s.getD 1 0
  | x :: y :: t, h, _ => by
    simp [strictlyIncreasing] at h
    simpa using h.1

theorem segs_cons (c : Consts) (q : Q) (h : 2 ≤ q.s.length) :
    ∃ rest, segs c q = (capAt c q 0, q.s.getD 1 0 - q.s.getD 0 0) :: rest := by
  unfold segs
  obtain ⟨m, hm⟩ : ∃ m, q.s.length - 1 = m + 1 := ⟨q.s.length - 2, by omega⟩
  rw [hm, List.range_succ_eq_map]
  exact ⟨_, rfl⟩

/-- Everything the in-domain branches of `decide` give the theorems. -/
structure InDomain (c : Consts) (q : Q) : Prop where
  two : 2 ≤ q.s.length
  gap : 0 < q.s.getD 1 0 - q.s.getD 0 0
  v0 : 0 ≤ q.v0
  aA : 0 ≤ c.a_acc
  aB : 0 ≤ c.a_brk
  aM : 0 < c.a_max
  segsOK : SegsOK (segs c q) (lastCap c q)

theorem segs_gaps_pos (c : Consts) (q : Q) (hi : strictlyIncreasing q.s = true) :
    ∀ p ∈ segs c q, 0 < p.2 := by
  intro p hp
  unfold segs at hp
  simp only [List.mem_map, List.mem_range] at hp
  obtain ⟨i, hi', rfl⟩ := hp
  show 0 < q.s.getD (i + 1) 0 - q.s.getD i 0
  suffices ∀ (s : List Int) (i : Nat), strictlyIncreasing s = true → i + 1 < s.length →
      s.getD i 0 < s.getD (i + 1) 0 by have := this q.s i hi (by omega); omega
  intro s
  induction s with
  | nil => intro i _ h; simp at h
  | cons x t ih =>
    intro i hs hl
    match t, i, hs, hl with
    | y :: t', 0, hs, _ => simp [strictlyIncreasing] at hs; simpa using hs.1
    | y :: t', j + 1, hs, hl =>
      simp only [strictlyIncreasing, Bool.and_eq_true] at hs
      have := ih j hs.2 (by simp at hl ⊢; omega)
      simpa using this
    | [], _, _, hl => simp at hl

theorem inDomain_of_valid (c : Consts) (q : Q) (h : (wellFormed q && constsOK c && capsNonneg c q) = true) :
    InDomain c q := by
  simp only [wellFormed, constsOK, capsNonneg, Bool.and_eq_true, decide_eq_true_eq, List.all_eq_true] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨h2, _⟩, hinc⟩, _⟩, _⟩, _⟩, _⟩, hv⟩, ⟨⟨⟨hA, hB⟩, hM⟩, _⟩⟩, hcaps, hlast⟩ := h
  have hgap := segs_gaps_pos c q hinc
  exact { two := h2, gap := by have := strictlyIncreasing_first hinc h2; omega, v0 := hv, aA := hA, aB := hB,
          aM := hM, segsOK := ⟨fun p hp => ⟨by simpa using hcaps p hp, Int.le_of_lt (hgap p hp)⟩, hlast⟩ }

/-- In the `greatest_safe` branch the observation is in the domain and the start is within budget. -/
theorem decide_gs_facts (c : Consts) (q : Q) (hd : (decide c q).kind = kindGreatestSafe) :
    InDomain c q ∧ q.v0 * q.v0 ≤ (bpass c.a_brk (segs c q) (lastCap c q)).headD 0 ∧
    decide c q = ⟨kindGreatestSafe,
      accelOf ((fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0) (q.v0 * q.v0)
        (q.s.getD 1 0 - q.s.getD 0 0),
      some ((fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)).getD 1 0),
      some (q.v0 * q.v0), some ((bpass c.a_brk (segs c q) (lastCap c q)).headD 0)⟩ := by
  unfold decide at hd ⊢
  split at hd
  · simp [kindInvalid, kindGreatestSafe] at hd
  rename_i h1
  split at hd
  · simp [kindNoGap, kindGreatestSafe] at hd
  rename_i h2
  split at hd
  · simp [kindMinViolation, kindGreatestSafe] at hd
  rename_i h3
  simp only [Bool.not_eq_true', Bool.not_eq_false] at h1
  refine ⟨inDomain_of_valid c q (by simpa using h1), by omega, ?_⟩
  rw [ite_eq_right (by simpa using h1), ite_eq_right h2, ite_eq_right h3]

/-- When `decide` returns `greatest_safe`, its next-station speed² `b1` is at least that of every
feasible profile from the current speed (under the declared terminal constraint). -/
theorem decide_greatest_safe (c : Consts) (q : Q) (b1 : Int)
    (hd : (decide c q).kind = kindGreatestSafe) (hb : (decide c q).b1 = some b1)
    (t : List Int) (hf : Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: t)) :
    (q.v0 * q.v0 :: t).getD 1 0 ≤ b1 := by
  have hle := LeAll_get1 (fpass_greatest c.a_acc c.a_brk (segs c q) (lastCap c q)
    (q.v0 * q.v0) (q.v0 * q.v0) t hf (Int.le_refl _))
  obtain ⟨_, _, he⟩ := decide_gs_facts c q hd
  rw [he] at hb
  simp only [Option.some.injEq] at hb
  rw [← hb]
  exact hle

/-- ... and the profile it is taken from is itself feasible. -/
theorem decide_greatest_safe_feasible (c : Consts) (q : Q) (hd : (decide c q).kind = kindGreatestSafe) :
    Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)) := by
  obtain ⟨hD, hB, _⟩ := decide_gs_facts c q hd
  exact fpass_feasible _ _ hD.aA hD.aB _ _ _ hD.segsOK (Int.mul_nonneg hD.v0 hD.v0) hB

/-- ... so the feasible profile it is taken from minimises every objective that does not increase
when speeds rise pointwise (e.g. traversal time) among all feasible profiles. -/
theorem decide_greatest_safe_minimises (c : Consts) (q : Q) (hd : (decide c q).kind = kindGreatestSafe)
    {T : List Int → Int} (hT : ∀ a b, LeAll a b → T b ≤ T a)
    (t : List Int) (hf : Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: t)) :
    Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)) ∧
    T (fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)) ≤ T (q.v0 * q.v0 :: t) :=
  ⟨decide_greatest_safe_feasible c q hd, greatest_minimises hT (fpass_greatest _ _ _ _ _ _ t hf (Int.le_refl _))⟩

/-- Integer-action optimality of the recorded acceleration `a` (mm/s²) in the `greatest_safe`
branch: `a ∈ [−a_brk, a_acc]`; holding `a` over the first segment `d₀ = s₁ − s₀` reaches a
next-station speed² that starts a feasible profile; and no larger integer acceleration does. -/
theorem decide_accel_greatest (c : Consts) (q : Q) (hd : (decide c q).kind = kindGreatestSafe) :
    let d := q.s.getD 1 0 - q.s.getD 0 0
    let a := (decide c q).a
    (-c.a_brk) ≤ a ∧ a ≤ c.a_acc ∧
    (∃ rest, segs c q = (capAt c q 0, d) :: rest ∧
      Feasible c.a_acc c.a_brk (segs c q) (lastCap c q)
        (q.v0 * q.v0 :: fpass c.a_acc c.a_brk rest (lastCap c q) (max 0 (q.v0 * q.v0 + a * (2 * d))))) ∧
    (∀ (a' : Int) (t : List Int),
      Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: max 0 (q.v0 * q.v0 + a' * (2 * d)) :: t) →
      a' ≤ a) := by
  intro d a
  obtain ⟨hD, hB, he⟩ := decide_gs_facts c q hd
  obtain ⟨rest, hs⟩ := segs_cons c q hD.two
  have ha : a = accelOf ((fpass c.a_acc c.a_brk ((capAt c q 0, d) :: rest) (lastCap c q) (q.v0 * q.v0)).getD 1 0)
      (q.v0 * q.v0) d := by
    show (decide c q).a = _
    rw [he]; simp only; rw [hs]
  have hsOK := hD.segsOK
  rw [hs] at hB hsOK
  have := accel_greatest c.a_acc c.a_brk (capAt c q 0) d (lastCap c q) (q.v0 * q.v0) rest hD.aA hD.aB hD.gap
    hsOK (Int.mul_nonneg hD.v0 hD.v0) hB
  simp only at this
  rw [← ha] at this
  obtain ⟨h1, h2, h3, h4⟩ := this
  refine ⟨h1, h2, ⟨rest, hs, by rw [hs]; exact h3⟩, ?_⟩
  intro a' t hf
  rw [hs] at hf
  exact h4 a' t hf

/-- When `decide` returns `min_violation`, no feasible profile from the current speed exists. -/
theorem decide_min_violation_infeasible (c : Consts) (q : Q) (hd : (decide c q).kind = kindMinViolation)
    (t : List Int) : ¬ Feasible c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0 :: t) := by
  intro hf
  have hle := LeAll_head (le_bpass c.a_acc c.a_brk (segs c q) (lastCap c q) _ hf)
  unfold decide at hd
  split at hd
  · simp [kindInvalid, kindMinViolation] at hd
  split at hd
  · simp [kindNoGap, kindMinViolation] at hd
  split at hd
  · rename_i h; omega
  · simp [kindGreatestSafe, kindMinViolation] at hd

/-- When `decide` returns `min_violation`, the label is full braking at `a_max`, and the
full-braking profile is pointwise below every physically possible profile from the current
speed; hence it minimises the excess over any upper caps (`brake_minimises_excess`). -/
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
  rw [ite_eq_right h1, ite_eq_right h2, ite_eq_right h3] at hd
  exact absurd hd (by simp [kindMinViolation, kindGreatestSafe])

/-! ### The same-information comparison for `decide`

A "road continuation in the domain" is: the segment from the last preview station (whose cap the
car can see) of any length `d ≥ 0`, followed by any segments whose caps are all `≥ a_lat * r_h`
(the domain's tightest planned path, `cert.domain_horizon_mm`), ending at any cap `≥ a_lat * r_h`.
An oracle with the same information as the car knows the preview but not which continuation
follows. -/

/-- When `decide` returns `greatest_safe`, its profile `F` (held at its last speed afterwards)
is feasible on EVERY continuation in the domain. -/
theorem decide_robust_extend (c : Consts) (q : Q) (hd : (decide c q).kind = kindGreatestSafe)
    (d : Int) (cont : List (Int × Int)) (cE : Int) (hd0 : 0 ≤ d)
    (hc : ∀ p ∈ cont, c.a_lat * c.r_h ≤ p.1 ∧ 0 ≤ p.2) (hE : c.a_lat * c.r_h ≤ cE) :
    let F := fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)
    Feasible c.a_acc c.a_brk (segs c q ++ (capAt c q (q.s.length - 1), d) :: cont) cE
      (F ++ List.replicate (cont.length + 1) (lastD F)) := by
  intro F
  obtain ⟨hD, _, _⟩ := decide_gs_facts c q hd
  exact robust_extend _ _ _ _ hD.aA hD.aB (segs c q) F (decide_greatest_safe_feasible c q hd) d cont cE hd0 hc hE

/-- ... and no planner with the same information does better: any speed profile from the current
speed that is safe on the WORST continuation in the domain (the tightest planned path starting
right after the preview) is pointwise `≤ F` over the preview; in particular its next-station
speed² is `≤ b1`, and (by `decide_accel_greatest`) no larger integer acceleration is safe. -/
theorem decide_robust_greatest (c : Consts) (q : Q) (_hd : (decide c q).kind = kindGreatestSafe)
    (t : List Int) (y : Int)
    (hw : Feasible c.a_acc c.a_brk (segs c q ++ [(capAt c q (q.s.length - 1), 0)]) (c.a_lat * c.r_h)
      ((q.v0 * q.v0 :: t) ++ [y])) :
    LeAll (q.v0 * q.v0 :: t) (fpass c.a_acc c.a_brk (segs c q) (lastCap c q) (q.v0 * q.v0)) :=
  fpass_greatest _ _ _ _ _ _ t (robust_necessary _ _ _ _ (segs c q) _ y hw) (Int.le_refl _)

/-- When `decide` returns `min_violation`, even the same-information optimum fails: no profile
from the current speed is safe on the worst in-domain continuation. -/
theorem decide_min_violation_robust (c : Consts) (q : Q) (hd : (decide c q).kind = kindMinViolation)
    (t : List Int) (y : Int) :
    ¬ Feasible c.a_acc c.a_brk (segs c q ++ [(capAt c q (q.s.length - 1), 0)]) (c.a_lat * c.r_h)
      ((q.v0 * q.v0 :: t) ++ [y]) :=
  fun hw => decide_min_violation_infeasible c q hd t (robust_necessary _ _ _ _ (segs c q) _ y hw)

end Corpus.Cert
