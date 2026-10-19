/- 
Certified lateral plan and derived steering, model `steer-2`.
The integer definitions and decision branches are kernel-reduced.  The choice of
vehicle constants, the interpretation of lateral widths/obstacles, and the
physics bridge from the integer curvature law to a bicycle model remain
trusted declarations, as in `Corpus.Cert`.  The elementary list, branch, and
integer facts below are proved in Lean; no simulator dynamics are modelled.
-/
import Corpus.Cert
import Lean.Data.Json

open Lean
namespace Corpus.Steer

structure Consts where
  lon : Cert.Consts
  margin : Int
  step : Int
  kappa_max : Int
deriving Repr, DecidableEq, FromJson, ToJson

structure Q where
  base : Cert.Q
  lat : Int
deriving Repr, DecidableEq, FromJson, ToJson

structure Plan where
  d : Int
  lmin : Int
  lmax : Int
  side : Int
  amp : Int

structure Decision where
  kind : Nat
  side : List Int
  amp : List Int
  off : List Int
  clear : List Int
  curv : List Int
  steer : List Int
deriving Repr, DecidableEq, FromJson, ToJson

def kindPlanOk : Nat := 0
def kindNoGap : Nat := 1
def kindInvalid : Nat := 2

def uniformGaps : List Int → Int → Bool
  | _ :: [], _ => true
  | x :: y :: xs, step => decide (y - x = step) && uniformGaps (y :: xs) step
  | _, _ => true

def disjointAt (hold d : Int) : List (List Int) → Bool
  | [] => true
  | ob :: xs => match Cert.obsTriple ob with
    | some (d', _, _) => decide (d' - hold > d + hold ∨ d - hold > d' + hold) && disjointAt hold d xs
    | none => false

def pairwiseDisjoint (hold : Int) : List (List Int) → Bool
  | [] => true
  | ob :: xs => match Cert.obsTriple ob with
    | some (d, _, _) => disjointAt hold d xs && pairwiseDisjoint hold xs
    | none => false

def wellFormed (c : Consts) (q : Q) : Bool :=
  Cert.wellFormed q.base && Cert.constsOK c.lon && decide (0 ≤ q.lat) &&
  uniformGaps q.base.s c.step && pairwiseDisjoint c.lon.hold q.base.obstacles &&
  decide (0 ≤ c.margin) && decide (0 < c.step) && decide (0 ≤ c.kappa_max)

def gapLeft (q : Q) (d lmax : Int) : Int :=
  q.base.wl.getD (Cert.nearestIndex q.base.s d) 0 - lmax

def gapRight (q : Q) (d lmin : Int) : Int :=
  lmin + q.base.wr.getD (Cert.nearestIndex q.base.s d) 0

def sideAmp (q : Q) (d lmin lmax : Int) : Int × Int :=
  let i := Cert.nearestIndex q.base.s d
  let wl := q.base.wl.getD i 0
  let wr := q.base.wr.getD i 0
  let yl := Cert.cdiv (Cert.iabs (lmax + wl)) 2
  let yr := Cert.cdiv (Cert.iabs (lmin - wr)) 2
  let gl := wl - lmax
  let gr := lmin + wr
  if gl > gr then (1, yl) else if gr > gl then (-1, yr) else if 0 ≤ q.lat then (1, yl) else (-1, yr)

def sideAmpOb (q : Q) (ob : List Int) : Int × Int :=
  match Cert.obsTriple ob with
  | some (d, lmin, lmax) => sideAmp q d lmin lmax
  | none => (0, 0)

def ramp (A x L : Int) : Int :=
  (A * (3 * x * x * L - 2 * x * x * x)) / (L * L * L)

def offOb (c : Consts) (d side amp si : Int) : Int :=
  if si < d - c.lon.hold - c.lon.ramp_in then 0
  else if si < d - c.lon.hold then side * ramp amp (si - d + c.lon.hold + c.lon.ramp_in) c.lon.ramp_in
  else if si ≤ d + c.lon.hold then side * amp
  else if si ≤ d + c.lon.hold + c.lon.ramp_out then
    side * (amp - ramp amp (si - d - c.lon.hold) c.lon.ramp_out)
  else 0

def wOb (c : Consts) (d si : Int) : Int :=
  if si < d - c.lon.hold - c.lon.ramp_in then 0
  else if si < d - c.lon.hold then ramp 1000 (si - d + c.lon.hold + c.lon.ramp_in) c.lon.ramp_in
  else if si ≤ d + c.lon.hold then 1000
  else if si ≤ d + c.lon.hold + c.lon.ramp_out then
    min 999 (1000 - ramp 1000 (si - d - c.lon.hold) c.lon.ramp_out)
  else 0

def rawPlanAt (c : Consts) (q : Q) (si : Int) : Option Plan :=
  let rec go : Int → Option Plan → List (List Int) → Option Plan
    | _, best, [] => best
    | bestW, best, ob :: tl => match Cert.obsTriple ob with
      | some (d, lmin, lmax) =>
          let w := wOb c d si
          let sa := sideAmp q d lmin lmax
          if bestW < w then
            go w (some { d := d, lmin := lmin, lmax := lmax, side := sa.1, amp := sa.2 }) tl
          else go bestW best tl
      | none => go bestW best tl
  go 0 none q.base.obstacles

def rawOffAt (c : Consts) (q : Q) (si : Int) : Int :=
  match rawPlanAt c q si with
  | some p => offOb c p.d p.side p.amp si
  | none => 0

def halfw (c : Consts) : Int := c.lon.veh_width / 2
def lo (c : Consts) (q : Q) (i : Nat) : Int :=
  -(q.base.wr.getD i 0 - halfw c - c.margin)
def hi (c : Consts) (q : Q) (i : Nat) : Int :=
  q.base.wl.getD i 0 - halfw c - c.margin
def clampAt (c : Consts) (q : Q) (i : Nat) (x : Int) : Int :=
  let l := lo c q i
  let h := hi c q i
  if l ≤ h then min (max x l) h else (l + h) / 2
def offAt (c : Consts) (q : Q) (si : Int) (i : Nat) : Int :=
  clampAt c q i (rawOffAt c q si)
def offs (c : Consts) (q : Q) : List Int :=
  (List.range q.base.s.length).map (fun i => offAt c q (q.base.s.getD i 0) i)
def clearAt (c : Consts) (q : Q) (si : Int) (i : Nat) (x : Int) : Int :=
  let low := lo c q i
  let high := hi c q i
  match rawPlanAt c q si with
  | some p =>
      if p.d - c.lon.hold ≤ si ∧ si ≤ p.d + c.lon.hold then
        if p.side = 1 then min (x - (p.lmax + halfw c + c.margin)) (high - x)
        else if p.side = -1 then min ((p.lmin - halfw c - c.margin) - x) (x - low)
        else min (x - low) (high - x)
      else min (x - low) (high - x)
  | none => min (x - low) (high - x)

def clears (c : Consts) (q : Q) (xs : List Int) : List Int :=
  (List.range q.base.s.length).map (fun i =>
    clearAt c q (q.base.s.getD i 0) i (xs.getD i 0))

def secondAt (xs : List Int) (i : Nat) : Int :=
  if i = 0 ∨ i + 1 ≥ xs.length then 0
  else Cert.iabs (xs.getD (i + 1) 0 - 2 * xs.getD i 0 + xs.getD (i - 1) 0)

def curvAt (q : Q) (xs : List Int) (i : Nat) : Int :=
  Cert.iabs (q.base.k.getD i 0) + 1000 * secondAt xs i

def curvs (q : Q) (xs : List Int) : List Int :=
  (List.range xs.length).map (curvAt q xs)

/-- Feed-forward law: curvature is measured in micro/m and is mapped linearly to
one thousandths of the available steering range, with clipping. -/
def steerAt (c : Consts) (κ : Int) : Int :=
  max (-1000) (min 1000 (Cert.cdiv (κ * 1000) (max c.kappa_max 1)))

def steers (c : Consts) (ks : List Int) : List Int := ks.map (steerAt c)

def sides (q : Q) : List Int := q.base.obstacles.map (fun ob => (sideAmpOb q ob).1)
def amps (q : Q) : List Int := q.base.obstacles.map (fun ob => (sideAmpOb q ob).2)

def hasNoGap (c : Consts) (q : Q) : Bool :=
  q.base.obstacles.any (fun ob => match Cert.obsTriple ob with
    | some (d, lmin, lmax) => decide (max (gapLeft q d lmax) (gapRight q d lmin) < c.lon.veh_width + 2 * c.margin)
    | none => true)

def decide (c : Consts) (q : Q) : Decision :=
  if !(wellFormed c q) then ⟨kindInvalid, [], [], [], [], [], []⟩
  else
    let os := offs c q
    let cl := clears c q os
    let cs := curvs q os
    let ss := steers c cs
    if hasNoGap c q then ⟨kindNoGap, sides q, amps q, os, cl, cs, ss⟩
    else ⟨kindPlanOk, sides q, amps q, os, cl, cs, ss⟩

/-! Domain and branch facts. -/
theorem wellFormed_off_length (c : Consts) (q : Q) (h : wellFormed c q = true) :
    (offs c q).length = q.base.s.length := by simp [offs]

theorem sides_length (q : Q) : (sides q).length = q.base.obstacles.length := by simp [sides]
theorem amps_length (q : Q) : (amps q).length = q.base.obstacles.length := by simp [amps]

theorem decide_kind_invalid (c : Consts) (q : Q) :
    (decide c q).kind = kindInvalid ↔ wellFormed c q = false := by
  by_cases hw : wellFormed c q = false
  · simp [decide, hw]
  · have hw' : wellFormed c q = true := by
      cases hq : wellFormed c q <;> simp_all
    cases hg : hasNoGap c q <;> simp [decide, hw, hw', hg, kindInvalid, kindNoGap, kindPlanOk]

theorem decide_no_gap_iff (c : Consts) (q : Q) (h : wellFormed c q = true) :
    (decide c q).kind = kindNoGap ↔ hasNoGap c q = true := by
  cases hg : hasNoGap c q <;> simp [decide, h, hg, kindNoGap, kindPlanOk]

theorem decide_plan_ok_iff (c : Consts) (q : Q) (h : wellFormed c q = true) :
    (decide c q).kind = kindPlanOk ↔ hasNoGap c q = false := by
  cases hg : hasNoGap c q <;> simp [decide, h, hg, kindNoGap, kindPlanOk]

/-! Side and amplitude. -/

theorem clamp_abs_le (lo hi x : Int) (hlo : lo ≤ 0) (hhi : 0 ≤ hi) :
    Cert.iabs (min (max x lo) hi) ≤ Cert.iabs x := by
  unfold Cert.iabs
  by_cases hx : x < 0 <;> by_cases hlox : x < lo <;> by_cases hxhi : hi < x <;> simp_all <;> omega

theorem clamp_noop_in_hold (l h x : Int) (hl : l ≤ x) (hx : x ≤ h) :
    min (max x l) h = x := by omega


/-! Feasibility and ramp shape. -/
theorem ramp_zero (A L : Int) : ramp A 0 L = 0 := by simp [ramp]


theorem wOb_zero_before (c : Consts) (d si : Int)
    (h : si < d - c.lon.hold - c.lon.ramp_in) : wOb c d si = 0 := by simp [wOb, h]
theorem off_eq_amp_in_hold (c : Consts) (d side amp si : Int)
    (h0 : d - c.lon.hold - c.lon.ramp_in ≤ si)
    (h1 : d - c.lon.hold ≤ si) (h2 : si ≤ d + c.lon.hold) :
    offOb c d side amp si = side * amp := by
  simp [offOb]
  omega




theorem steer_within_range (c : Consts) (κ : Int) :
    Cert.iabs (steerAt c κ) ≤ 1000 := by
  simp [steerAt, Cert.iabs]
  omega

theorem steer_unsaturated (c : Consts) (κ : Int)
    (hκ : 0 ≤ κ) (hk : 0 < c.kappa_max)
    (hdiv0 : 0 ≤ Cert.cdiv (κ * 1000) c.kappa_max)
    (hdiv : Cert.cdiv (κ * 1000) c.kappa_max < 1000) :
    Cert.iabs (steerAt c κ) < 1000 := by
  have hmax : max c.kappa_max 1 = c.kappa_max := by omega
  simp [steerAt, hmax, Cert.iabs]
  omega

end Corpus.Steer
