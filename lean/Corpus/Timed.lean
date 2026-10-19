/-
Executable, proof-carrying checker for the time-indexed integer model ``timed-1``.

This module is the Lean counterpart of ``autocross/timed/model.py`` and
``autocross/timed/certificate.py``.  ``checkCert`` is a decidable checker for a
certificate of optimality; ``checkCert_sound`` turns a passing check into a
kernel-checked statement about the declared model:

  * the declared instance is in the model's domain (``wellFormed``);
  * the plan is admissible and its recomputed objective equals the claim;
  * every admissible plan of the declared model costs at least the claim
    (lexicographically);
  * every plan enumerated strictly before the claimed plan costs strictly more,
    so the claim is the FIRST minimiser (the solver's tie-break);
  * the published first command is the one the model derives from block 0.

The checker never trusts the solver.  Integer division here is Lean's ``Int``
Euclidean division, which is floor division for strictly positive divisors;
every divisor of the model is positive on a well-formed instance (and the few
guards where a divisor can become <= 0 in the model are mirrored here).  The
model's ``tyre_force`` uses ``Nat.sqrt`` (floor square root), which matches
``math.isqrt`` for non-negative integers.
-/
import Lean.Data.Json

open Lean

namespace Corpus.Timed

/-- Decidability of bounded universal quantification over a list, even when the
element type is infinite. -/
local instance listForallDecidable {α : Type} (l : List α) (p : α → Prop)
    [DecidablePred p] : Decidable (∀ x ∈ l, p x) :=
  decidable_of_iff (l.all p) (by simp [List.all_eq_true])

/-! ## Declared constants -/

/-- The declared constant set of ``timed-1`` (exactly ``model.DEFAULT_CONSTS``). -/
structure TimedConsts where
  H_US : Int
  TICKS : Int
  SLEW : Int
  MAX_STEER : Int
  LAG_NUM : Int
  LAG_DEN : Int
  L : Int
  LF : Int
  LR : Int
  HCG : Int
  MASS : Int
  IZ : Int
  CF : Int
  CR : Int
  FD : Int
  PW : Int
  FB : Int
  FBMAX : Int
  FDF : Int
  FBF : Int
  CD : Int
  FRR : Int
  G : Int
  MU : Int
  MU_T : Int
  V_KIN : Int
  V_FLOOR : Int
  HL : Int
  HW : Int
  A_LT : Int
  A_BT : Int
  VMAX : Int
  RMIN : Int
  KSTEER : Int
  STEP : Int
  T_TAIL : Int
  PQ : Int
deriving Repr, DecidableEq, FromJson, ToJson

/-! ## Instance data -/

structure Block where
  steps : Nat
  steer : List Int
  pedal : List Int
deriving Repr, DecidableEq, FromJson, ToJson

structure State where
  s : Int
  d : Int
  psi : Int
  v : Int
  vy : Int
  r : Int
  sigma : Int
  delta : Int
deriving Repr, DecidableEq

def State.toList (st : State) : List Int :=
  [st.s, st.d, st.psi, st.v, st.vy, st.r, st.sigma, st.delta]

instance : ToJson State := ⟨fun st => toJson st.toList⟩

instance : FromJson State where
  fromJson? j := do
    let xs : List Int ← fromJson? j
    match xs with
    | [s, d, psi, v, vy, r, sigma, delta] => pure ⟨s, d, psi, v, vy, r, sigma, delta⟩
    | _ => throw "start state must have exactly 8 integer entries"

structure Instance where
  consts : TimedConsts
  k : List Int
  wl : List Int
  wr : List Int
  obstacles : List (List Int)
  start : State
  p_prev : Int
  blocks : List Block
deriving Repr

instance : FromJson Instance where
  fromJson? j := do
    let schema ← j.getObjValAs? String "schema"
    let version ← j.getObjValAs? Int "schema_version"
    let model ← j.getObjValAs? String "model"
    if schema ≠ "autocross.timed.instance" then throw "not a timed-1 instance"
    if version ≠ 1 then throw "unsupported instance schema_version"
    if model ≠ "timed-1" then throw "unknown model"
    let consts ← j.getObjValAs? TimedConsts "consts"
    let k ← j.getObjValAs? (List Int) "k"
    let wl ← j.getObjValAs? (List Int) "wl"
    let wr ← j.getObjValAs? (List Int) "wr"
    let obstacles ← j.getObjValAs? (List (List Int)) "obstacles"
    let start ← j.getObjValAs? State "start"
    let p_prev ← j.getObjValAs? Int "p_prev"
    let blocks ← j.getObjValAs? (List Block) "blocks"
    pure { consts, k, wl, wr, obstacles, start, p_prev, blocks }

instance : ToJson Instance where
  toJson inst :=
    Json.mkObj [
      ("schema", toJson "autocross.timed.instance"),
      ("schema_version", toJson (1 : Int)),
      ("model", toJson "timed-1"),
      ("consts", toJson inst.consts),
      ("k", toJson inst.k), ("wl", toJson inst.wl), ("wr", toJson inst.wr),
      ("obstacles", toJson inst.obstacles), ("start", toJson inst.start),
      ("p_prev", toJson inst.p_prev), ("blocks", toJson inst.blocks)]

def Instance.n (inst : Instance) : Nat := inst.k.length

def Block.n (b : Block) : Nat := b.steer.length * b.pedal.length

/-- Block action as in ``model.Block.action``: steer index then pedal index. -/
def Block.action (b : Block) (a : Int) : Int × Int :=
  (b.steer.getD ((a / (b.pedal.length : Int)).toNat) 0,
   b.pedal.getD ((a % (b.pedal.length : Int)).toNat) 0)

/-! ## Integer primitives (exactly ``model.py``) -/

def E6 : Int := 1_000_000
def E9 : Int := 1_000_000_000
def BIG : Int := 1_000_000_000
def PSI_CLAMP : Int := 1_500_000

def cdiv (a b : Int) : Int := -((-a) / b)

def clamp (x lo hi : Int) : Int := if x < lo then lo else if hi < x then hi else x

def iabs (x : Int) : Int := if x < 0 then -x else x

/-- Binary-search floor square root, structurally recursive on fuel so the
kernel can evaluate it (``Nat.sqrt`` is well-founded and opaque to ``decide``).
The fuel is 80 halvings; the model's magnitudes are far below 2^80. -/
private def sqrtSearch (n lo hi : Nat) : Nat → Nat
  | 0 => lo
  | fuel + 1 =>
      if lo + 1 < hi then
        let mid := (lo + hi) / 2
        if mid * mid ≤ n then sqrtSearch n mid hi fuel
        else sqrtSearch n lo mid fuel
      else lo

def isqrtNat (n : Nat) : Nat := if n = 0 then 0 else max 1 (sqrtSearch n 0 (n + 1) 80)

def isqrt (n : Int) : Int := if 0 < n then (isqrtNat n.toNat : Int) else 0

def trig (psi : Int) : Int × Int :=
  let p := clamp psi (-PSI_CLAMP) PSI_CLAMP
  let q := p * p / E6
  (p - p * q / (6 * E6), E6 - q / 2 + q * q / (24 * E6))

def tanU (delta : Int) : Int :=
  let q := delta * delta / E6
  delta + delta * q / (3 * E6) + 2 * delta * q * q / (15 * E6 * E6)

/-- Saturating lateral tyre force; the (3,2) Padé approximation of ``tanh``. -/
def tyreForce (fyLin fx cap : Int) : Int :=
  let av2 := cap * cap - fx * fx
  let avail := if 0 < av2 then (isqrtNat av2.toNat : Int) else 1
  let x := clamp (fyLin * 1000 / avail) (-3000) 3000
  let num := 27_000_000 + x * x
  avail * (x * num / (27_000_000 + 9 * (x * x))) / 1000

/-- `avail` is at least 1, so the divisions inside `tyreForce` have positive
divisors (Euclidean division = floor division). -/
private theorem tyreForce_avail_pos (cap fx : Int) :
    1 ≤ (if 0 < cap * cap - fx * fx then (isqrtNat (cap * cap - fx * fx).toNat : Int) else 1) := by
  split
  · rename_i h
    have hn : 0 < (cap * cap - fx * fx).toNat := by omega
    have hpos : 1 ≤ isqrtNat (cap * cap - fx * fx).toNat := by
      unfold isqrtNat
      split
      · rename_i hz
        omega
      · exact Nat.le_max_left 1 _
    omega
  · omega

private theorem int_mul_self_nonneg (x : Int) : 0 ≤ x * x := by
  by_cases h : 0 ≤ x
  · exact Int.mul_nonneg h h
  · have hx : x ≤ 0 := by omega
    have h1 : 0 ≤ -x := by omega
    have h2 : 0 ≤ (-x) * (-x) := Int.mul_nonneg h1 h1
    rwa [Int.neg_mul_neg] at h2

/-- The Padé denominator `27_000_000 + 9 x²` is strictly positive. -/
private theorem pade_den_pos (x : Int) : 0 < 27_000_000 + 9 * (x * x) := by
  have hx := int_mul_self_nonneg x
  have h9 : 0 ≤ 9 * (x * x) := Int.mul_nonneg (by decide) hx
  omega

def idx (inst : Instance) (x : Int) : Nat :=
  (clamp (x / inst.consts.STEP) 0 (inst.n - 1)).toNat

def kappaAt (inst : Instance) (x : Int) : Int :=
  let step := inst.consts.STEP
  let j := idx inst x
  let j1 := min (j + 1) (inst.n - 1)
  let fr := clamp (x - (j : Int) * step) 0 step
  inst.k.getD j 0 + (inst.k.getD j1 0 - inst.k.getD j 0) * fr / step

/-! ## Domain guard (`model.well_formed`) -/

def absLe (x b : Int) : Prop := -b ≤ x ∧ x ≤ b

def obstacleOK (o : List Int) : Prop :=
  o.length = 4 ∧ o.getD 0 0 ≤ o.getD 1 0 ∧ o.getD 2 0 ≤ o.getD 3 0

def firstBlockOK : List Block → Prop
  | [] => False
  | b :: _ => b.steps = 1

def blockOK (b : Block) : Prop :=
  0 < b.steps ∧ 0 < b.steer.length ∧ 0 < b.pedal.length ∧
  (∀ x ∈ b.steer, absLe x 2000) ∧ (∀ x ∈ b.pedal, absLe x 1000)

instance absLeDecidable (x b : Int) : Decidable (absLe x b) := by
  unfold absLe; infer_instance

instance obstacleOKDecidable (o : List Int) : Decidable (obstacleOK o) := by
  unfold obstacleOK; infer_instance

instance firstBlockOKDecidable (bs : List Block) : Decidable (firstBlockOK bs) := by
  unfold firstBlockOK; split <;> infer_instance

instance blockOKDecidable (b : Block) : Decidable (blockOK b) := by
  unfold blockOK; infer_instance

/-- The declared model's domain guard, mirroring ``model.well_formed``.  The
constant-set check of the Python guard is structural here: the record type is
exactly the declared set. -/
def wellFormed (inst : Instance) : Prop :=
  let c := inst.consts
  0 < c.H_US ∧ 0 < c.TICKS ∧ 0 < c.LAG_DEN ∧ 0 < c.L ∧ 0 < c.MASS ∧ 0 < c.IZ ∧
  0 < c.STEP ∧ 0 < c.PQ ∧ 0 < c.A_BT ∧ 0 < c.V_FLOOR ∧ 0 < c.MU_T ∧
  0 ≤ c.LAG_NUM ∧ c.LAG_NUM ≤ c.LAG_DEN ∧
  c.V_FLOOR < c.V_KIN ∧
  2 ≤ inst.n ∧ inst.wl.length = inst.n ∧ inst.wr.length = inst.n ∧
  (∀ x ∈ inst.k, absLe x 1_000_000) ∧
  (∀ x ∈ inst.wl, 0 ≤ x ∧ x ≤ 50_000) ∧
  (∀ x ∈ inst.wr, 0 ≤ x ∧ x ≤ 50_000) ∧
  (0 ≤ inst.start.v ∧ inst.start.v ≤ 60_000 ∧ absLe inst.start.vy 30_000 ∧
   absLe inst.start.r 5_000_000 ∧ absLe inst.start.d 60_000 ∧
   absLe inst.start.psi 4_000_000 ∧ absLe inst.start.sigma 1000 ∧
   absLe inst.start.delta 1_000_000 ∧ inst.start.s = 0) ∧
  firstBlockOK inst.blocks ∧
  (∀ b ∈ inst.blocks, blockOK b) ∧
  (∀ o ∈ inst.obstacles, obstacleOK o)

instance wellFormedDecidable (inst : Instance) : Decidable (wellFormed inst) := by
  unfold wellFormed; infer_instance

/-! ## Tick, violations, rollout -/

def bodyPoints (inst : Instance) (st : State) : List (Int × Int) :=
  let hl := inst.consts.HL
  let kap := kappaAt inst st.s
  let sn := (trig st.psi).1
  [(st.s + (-hl), st.d + (-hl) * sn / E6 - kap * (-hl) * (-hl) / (2 * E9)),
   (st.s + 0, st.d + 0 * sn / E6 - kap * 0 * 0 / (2 * E9)),
   (st.s + hl, st.d + hl * sn / E6 - kap * hl * hl / (2 * E9))]

def footprintViol (inst : Instance) (st : State) : Int :=
  let hw := inst.consts.HW
  let hl := inst.consts.HL
  let pts := bodyPoints inst st
  let corridor :=
    pts.foldl (fun acc p =>
      let lat := p.2
      let j := idx inst p.1
      acc + max 0 (lat + hw - inst.wl.getD j 0) +
        max 0 (-inst.wr.getD j 0 - (lat - hw))) 0
  let lo := (pts.map (·.2)).foldl min (pts.headD (0, 0)).2 - hw
  let hi := (pts.map (·.2)).foldl max (pts.headD (0, 0)).2 + hw
  inst.obstacles.foldl (fun acc o =>
    let sLo := o.getD 0 0
    let sHi := o.getD 1 0
    let lLo := o.getD 2 0
    let lHi := o.getD 3 0
    if sLo ≤ st.s + hl ∧ st.s - hl ≤ sHi then
      acc + max 0 (min (hi - lLo) (lHi - lo))
    else acc) corridor

def frictionViol (inst : Instance) (fyf fyr fxf fxr ax : Int) : Int :=
  let c := inst.consts
  let fzf := c.MASS * (c.G * c.LR - ax * c.HCG) / (c.L * 1000)
  let fzr := c.MASS * c.G / 1000 - fzf
  let one (fy fx fz : Int) : Int :=
    let cap := max (c.MU * fz / 1000) 1
    let ex := fy * fy + fx * fx - cap * cap
    if 0 < ex then cdiv ex cap else 0
  one fyf fxf fzf + one fyr fxr fzr

def terminalViol (inst : Instance) (st : State) : Int :=
  let c := inst.consts
  let aLt := c.A_LT
  let aBt := c.A_BT
  let step := c.STEP
  let n : Nat := inst.n
  let v2 := st.v * st.v
  let vmax2 := c.VMAX * c.VMAX
  let worst0 := v2 - (aLt * c.RMIN + 2 * aBt * (((n : Int) - 1) * step - st.s))
  let worst := (List.range n).foldl (fun worst (j : Nat) =>
    if j = 0 then worst
    else
      let sj := (j : Int) * step
      if sj ≤ st.s then worst
      else
        let kj := iabs (inst.k.getD j 0)
        let cap := if kj = 0 then vmax2 else min vmax2 (aLt * E9 / kj)
        max worst (v2 - (cap + 2 * aBt * (sj - st.s)))) worst0
  let viol := if 0 < worst then cdiv worst (2 * aBt) else 0
  let kcap := min (aLt * E9 / max v2 1) c.KSTEER
  let kap := kappaAt inst st.s
  let j := idx inst st.s
  let psi := clamp st.psi (-PSI_CLAMP) PSI_CLAMP
  let hw := c.HW
  let dk := if 0 < psi then kcap + kap else if psi < 0 then kcap - kap else 1
  let room := if 0 < psi then inst.wl.getD j 0 - hw - st.d
              else if psi < 0 then st.d + inst.wr.getD j 0 - hw else 0
  let req := if dk ≤ 0 then BIG else cdiv (psi * psi) (2 * dk * 1000)
  viol + (if psi ≠ 0 then max 0 (req - room) else 0)

def tick (inst : Instance) (st : State) (u p : Int) : State × Int :=
  let c := inst.consts
  let H := c.H_US
  let sigma := clamp (st.sigma + clamp (u - st.sigma) (-c.SLEW) c.SLEW) (-1000) 1000
  let tgt := sigma * c.MAX_STEER / 1000
  let delta := tgt + (st.delta - tgt) * c.LAG_NUM / c.LAG_DEN
  let v := st.v
  let fdrv := min c.FD (c.PW * 1000 / max v 1000)
  let fx := if 0 ≤ p then p * fdrv / 1000 else max (p * c.FB / 1000) (-c.FBMAX)
  let fxf := if 0 ≤ p then fx * c.FDF / 1000 else fx * c.FBF / 1000
  let fxr := fx - fxf
  let fres := c.CD * v * v / E9 + c.FRR
  let ax := (fx - fres) * 1000 / c.MASS
  let v1 := max (v + ax * H / E6) 0
  let fzf := c.MASS * (c.G * c.LR - ax * c.HCG) / (c.L * 1000)
  let fzr := c.MASS * c.G / 1000 - fzf
  let vs := max v c.V_FLOOR
  let af := (st.vy + c.LF * st.r / E6) * E6 / vs - delta
  let ar := (st.vy - c.LR * st.r / E6) * E6 / vs
  -- dynamics: the saturated force; the grip budget below is the *demand*
  let demandF := -(c.CF * af / E6)
  let demandR := -(c.CR * ar / E6)
  let fyf := tyreForce demandF fxf (c.MU_T * fzf / 1000)
  let fyr := tyreForce demandR fxr (c.MU_T * fzr / 1000)
  let vyDot := (fyf + fyr) * 1000 / c.MASS - st.r * v / E6
  let rDot := (c.LF * fyf - c.LR * fyr) * 1000 / c.IZ
  let vyD := st.vy + vyDot * H / E6
  let rD := st.r + rDot * H / E6
  let w := clamp ((c.V_KIN - v1) * 1000 / (c.V_KIN - c.V_FLOOR)) 0 1000
  let rK := v1 * tanU delta / c.L
  let vyK := c.LR * rK / E6
  let r1 := (w * rK + (1000 - w) * rD) / 1000
  let vy1 := (w * vyK + (1000 - w) * vyD) / 1000
  let sn := (trig st.psi).1
  let cs := (trig st.psi).2
  let vt := (v1 * cs - vy1 * sn) / E6
  let kap := kappaAt inst (st.s + (vt * H / E6) / 2)
  let den := max (E9 - kap * st.d) (E9 / 10)
  let sdot := vt * E9 / den
  let ddot := (v1 * sn + vy1 * cs) / E6
  let psidot := r1 - kap * sdot / 1000
  let s1 := st.s + sdot * H / E6
  let d1 := st.d + ddot * H / E6
  let psi1 := st.psi + psidot * H / E6
  let out := { s := s1, d := d1, psi := psi1, v := v1, vy := vy1, r := r1,
               sigma := sigma, delta := delta }
  (out, footprintViol inst out + frictionViol inst demandF demandR fxf fxr ax)

/-- Run `n` ticks, returning the final state and the per-tick violations in
chronological order (the accumulator is reversed at the end). -/
def runTicks (inst : Instance) (u p : Int) :
    Nat → State → List Int → State × List Int
  | 0, st, acc => (st, acc.reverse)
  | n + 1, st, acc =>
      let (st', v) := tick inst st u p
      runTicks inst u p n st' (v :: acc)

/-- Rollout: per-tick violations (without the terminal row), final state and
accumulated command effort. -/
structure Rollout where
  viols : List Int
  final : State
  effort : Int
deriving Repr

def rolloutAux (inst : Instance) :
    List Block → List Int → State → Int → List Int → Int → Rollout
  | [], _, st, _, viols, eff => ⟨viols, st, eff⟩
  | _ :: _, [], st, _, viols, eff => ⟨viols, st, eff⟩
  | b :: bs, a :: as, st, pLast, viols, eff =>
      let (off, p) := b.action a
      let u := clamp (st.sigma + off) (-1000) 1000
      let eff' := eff + iabs (u - st.sigma) + iabs (p - pLast)
      let ticks := ((b.steps : Int) * inst.consts.TICKS).toNat
      let (st', vs) := runTicks inst u p ticks st []
      rolloutAux inst bs as st' p (viols ++ vs) eff'

def rollout (inst : Instance) (plan : List Int) : Rollout :=
  rolloutAux inst inst.blocks plan inst.start inst.p_prev [] 0

def terminalViolation (inst : Instance) (plan : List Int) : Int :=
  terminalViol inst (rollout inst plan).final

def cost (inst : Instance) (plan : List Int) : Int × Int × Int :=
  let r := rollout inst plan
  let st := r.final
  let viol := r.viols.foldl (fun acc v => acc + v) 0
  let prog := st.s + st.v * inst.consts.T_TAIL / 1000
  (viol + terminalViol inst st, -(prog / inst.consts.PQ), r.effort)

def firstCommand (inst : Instance) (plan : List Int) : Int × Int :=
  match inst.blocks, plan with
  | b :: _, a :: _ =>
      let (off, p) := b.action a
      (clamp (inst.start.sigma + off) (-1000) 1000, p)
  | _, _ => (0, 0)

/-! ## Plans: the Cartesian product in Python's enumeration order -/

/-- Cartesian product of the block action ranges.  Block 0 varies slowest and
the action index ascends, exactly like ``itertools.product``. -/
def productAux : List Block → List (List Int)
  | [] => [[]]
  | b :: bs =>
      (List.range b.n).flatMap fun (a : Nat) => (productAux bs).map fun tail => (a : Int) :: tail

def allPlans (inst : Instance) : List (List Int) := productAux inst.blocks

def planInRange : List Block → List Int → Prop
  | [], [] => True
  | b :: bs, a :: as => 0 ≤ a ∧ a < (b.n : Int) ∧ planInRange bs as
  | _, _ => False

instance planInRangeDecidable : ∀ (bs : List Block) (xs : List Int),
    Decidable (planInRange bs xs)
  | [], [] => isTrue trivial
  | [], _ :: _ => isFalse (by simp [planInRange])
  | _ :: _, [] => isFalse (by simp [planInRange])
  | b :: bs, a :: as => by
      haveI : Decidable (planInRange bs as) := planInRangeDecidable bs as
      unfold planInRange
      infer_instance

/-- Position of a plan in the enumeration (``length`` if absent). -/
def planIndex (inst : Instance) (plan : List Int) : Nat := List.idxOf plan (allPlans inst)

/-! ## Lexicographic objective -/

/-- Componentwise lexicographic `≤` on triples; the first component is most
significant.  Divisions and comparisons are the model's integer ones. -/
def leLex3 (a1 a2 a3 b1 b2 b3 : Int) : Bool :=
  if a1 < b1 then true
  else if b1 < a1 then false
  else if a2 < b2 then true
  else if b2 < a2 then false
  else a3 ≤ b3

def leLex (a b : Int × Int × Int) : Bool := leLex3 a.1 a.2.1 a.2.2 b.1 b.2.1 b.2.2

/-- Strict lexicographic `<`. -/
def ltLex (a b : Int × Int × Int) : Bool := leLex a b && !leLex b a

private theorem leLex3_true_iff (a1 a2 a3 b1 b2 b3 : Int) :
    leLex3 a1 a2 a3 b1 b2 b3 = true ↔
      a1 < b1 ∨ (a1 = b1 ∧ a2 < b2) ∨ (a1 = b1 ∧ a2 = b2 ∧ a3 ≤ b3) := by
  simp only [leLex3]
  constructor
  · intro h
    by_cases h1 : a1 < b1
    · exact Or.inl h1
    · by_cases h2 : b1 < a1
      · have hz : (if a1 < b1 then true else if b1 < a1 then false else if a2 < b2 then true else if b2 < a2 then false else decide (a3 ≤ b3)) = false := by
          simp [h1, h2]
        rw [hz] at h; cases h
      · have e1 : a1 = b1 := by omega
        by_cases h3 : a2 < b2
        · exact Or.inr (Or.inl ⟨e1, h3⟩)
        · by_cases h4 : b2 < a2
          · have hz : (if a1 < b1 then true else if b1 < a1 then false else if a2 < b2 then true else if b2 < a2 then false else decide (a3 ≤ b3)) = false := by
              simp [h1, h2, h3, h4]
            rw [hz] at h; cases h
          · have e2 : a2 = b2 := by omega
            have hle : a3 ≤ b3 := by
              simpa [h1, h2, h3, h4] using h
            exact Or.inr (Or.inr ⟨e1, e2, hle⟩)
  · intro h
    rcases h with h1 | ⟨e1, h3⟩ | ⟨e1, e2, hle⟩
    · simp [h1]
    · subst e1; simp [h3]
    · subst e1; subst e2; simp [hle]

private theorem ltLex_true {a b : Int × Int × Int}
    (h1 : leLex a b = true) (h2 : leLex b a = false) : ltLex a b = true := by
  simp [ltLex, h1, h2]

theorem leLex_refl (a : Int × Int × Int) : leLex a a = true := by
  obtain ⟨a1, a2, a3⟩ := a
  simp [leLex, leLex3]

theorem leLex_antisymm {a b : Int × Int × Int}
    (h1 : leLex a b = true) (h2 : leLex b a = true) : a = b := by
  obtain ⟨a1, a2, a3⟩ := a
  obtain ⟨b1, b2, b3⟩ := b
  have h1' := (leLex3_true_iff a1 a2 a3 b1 b2 b3).mp h1
  have h2' := (leLex3_true_iff b1 b2 b3 a1 a2 a3).mp h2
  simp only [Prod.mk.injEq]
  rcases h1' with h | ⟨e1, h⟩ | ⟨e1, e2, h⟩ <;>
  rcases h2' with h' | ⟨e1', h'⟩ | ⟨e1', e2', h'⟩ <;>
  omega

theorem leLex_total (a b : Int × Int × Int) :
    leLex a b = true ∨ leLex b a = true := by
  obtain ⟨a1, a2, a3⟩ := a
  obtain ⟨b1, b2, b3⟩ := b
  rw [show leLex (a1, a2, a3) (b1, b2, b3) = leLex3 a1 a2 a3 b1 b2 b3 from rfl,
      show leLex (b1, b2, b3) (a1, a2, a3) = leLex3 b1 b2 b3 a1 a2 a3 from rfl]
  rw [leLex3_true_iff, leLex3_true_iff]
  by_cases h1 : a1 < b1
  · exact Or.inl (Or.inl h1)
  · by_cases h2 : b1 < a1
    · exact Or.inr (Or.inl h2)
    · have e1 : a1 = b1 := by omega
      by_cases h3 : a2 < b2
      · exact Or.inl (Or.inr (Or.inl ⟨e1, h3⟩))
      · by_cases h4 : b2 < a2
        · exact Or.inr (Or.inr (Or.inl ⟨by omega, h4⟩))
        · have e2 : a2 = b2 := by omega
          by_cases h5 : a3 ≤ b3
          · exact Or.inl (Or.inr (Or.inr ⟨e1, e2, h5⟩))
          · exact Or.inr (Or.inr (Or.inr ⟨by omega, by omega, by omega⟩))

theorem leLex_trans {a b c : Int × Int × Int}
    (h1 : leLex a b = true) (h2 : leLex b c = true) : leLex a c = true := by
  obtain ⟨a1, a2, a3⟩ := a
  obtain ⟨b1, b2, b3⟩ := b
  obtain ⟨c1, c2, c3⟩ := c
  have h1' := (leLex3_true_iff a1 a2 a3 b1 b2 b3).mp h1
  have h2' := (leLex3_true_iff b1 b2 b3 c1 c2 c3).mp h2
  rw [show leLex (a1, a2, a3) (c1, c2, c3) = leLex3 a1 a2 a3 c1 c2 c3 from rfl,
      leLex3_true_iff]
  rcases h1' with h | ⟨e1, h⟩ | ⟨e1, e2, h⟩ <;>
  rcases h2' with h' | ⟨f1, h'⟩ | ⟨f1, f2, h'⟩
  all_goals first
    | (left; omega)
    | (right; left; exact ⟨by omega, by omega⟩)
    | (right; right; exact ⟨by omega, by omega, by omega⟩)

/-! ## Certificates -/

structure Cert where
  schema : String
  schemaVersion : Int
  inst : Instance
  instanceSha256 : String
  plan : List Int
  claimed : Int × Int × Int
  first : Int × Int
  objLex : List String
  objTieBreak : String
deriving Repr

private def parseTriple (j : Json) : Except String (Int × Int × Int) := do
  let xs : List Int ← fromJson? j
  match xs with
  | [a, b, c] => pure (a, b, c)
  | _ => throw "expected an array of three integers"

private def parsePair (j : Json) : Except String (Int × Int) := do
  let xs : List Int ← fromJson? j
  match xs with
  | [a, b] => pure (a, b)
  | _ => throw "expected an array of two integers"

instance : FromJson Cert where
  fromJson? j := do
    let schema ← j.getObjValAs? String "schema"
    let version ← j.getObjValAs? Int "schema_version"
    let inst ← j.getObjValAs? Instance "instance"
    let sha ← j.getObjValAs? String "instance_sha256"
    let plan ← j.getObjValAs? (List Int) "plan"
    let claimed ← parseTriple (← j.getObjVal? "claimed")
    let first ← parsePair (← j.getObjVal? "first")
    let objective ← j.getObjVal? "objective"
    let lex ← objective.getObjValAs? (List String) "lex"
    let tie ← objective.getObjValAs? String "tie_break"
    pure { schema, schemaVersion := version, inst, instanceSha256 := sha, plan,
           claimed, first, objLex := lex, objTieBreak := tie }

def schemaOK (cert : Cert) : Prop :=
  cert.schema = "autocross.timed.certificate" ∧ cert.schemaVersion = 1 ∧
  cert.objLex = ["violation", "neg_progress", "effort"] ∧
  cert.objTieBreak = "first plan in enumeration order"

def planOK (cert : Cert) : Prop :=
  cert.plan.length = cert.inst.blocks.length ∧ planInRange cert.inst.blocks cert.plan

def firstOK (cert : Cert) : Prop :=
  firstCommand cert.inst cert.plan = cert.first

def claimedOK (cert : Cert) : Prop :=
  cost cert.inst cert.plan = cert.claimed

def optimalOK (cert : Cert) : Prop :=
  ∀ p ∈ allPlans cert.inst, leLex cert.claimed (cost cert.inst p) = true

def tieBreakOK (cert : Cert) : Prop :=
  ∀ p ∈ allPlans cert.inst,
    planIndex cert.inst p < planIndex cert.inst cert.plan →
      leLex (cost cert.inst p) cert.claimed = false

/-- The full decidable certificate-checking predicate. -/
def CheckCert (cert : Cert) : Prop :=
  schemaOK cert ∧ wellFormed cert.inst ∧ planOK cert ∧ firstOK cert ∧
  claimedOK cert ∧ optimalOK cert ∧ tieBreakOK cert

instance schemaOKDecidable (cert : Cert) : Decidable (schemaOK cert) := by
  unfold schemaOK; infer_instance

instance planOKDecidable (cert : Cert) : Decidable (planOK cert) := by
  unfold planOK; infer_instance

instance firstOKDecidable (cert : Cert) : Decidable (firstOK cert) := by
  unfold firstOK; infer_instance

instance claimedOKDecidable (cert : Cert) : Decidable (claimedOK cert) := by
  unfold claimedOK; infer_instance

instance optimalOKDecidable (cert : Cert) : Decidable (optimalOK cert) := by
  unfold optimalOK; infer_instance

instance tieBreakOKDecidable (cert : Cert) : Decidable (tieBreakOK cert) := by
  unfold tieBreakOK; infer_instance

instance checkCertDecidable (cert : Cert) : Decidable (CheckCert cert) := by
  unfold CheckCert
  infer_instance

def checkCert (cert : Cert) : Bool := decide (CheckCert cert)

theorem checkCert_sound (cert : Cert) (h : checkCert cert = true) : CheckCert cert :=
  of_decide_eq_true h

/-! ## Enumeration correctness -/

private theorem productAux_length (bs : List Block) :
    ∀ xs ∈ productAux bs, xs.length = bs.length := by
  induction bs with
  | nil =>
      intro xs hx
      simp [productAux] at hx
      subst hx
      rfl
  | cons b bs ih =>
      intro xs hx
      simp [productAux, List.mem_flatMap] at hx
      obtain ⟨a, _, tail, htail, rfl⟩ := hx
      simp [ih tail htail]

private theorem productAux_range (bs : List Block) :
    ∀ xs ∈ productAux bs, planInRange bs xs := by
  induction bs with
  | nil =>
      intro xs hx
      simp [productAux] at hx
      subst hx
      trivial
  | cons b bs ih =>
      intro xs hx
      simp [productAux, List.mem_flatMap] at hx
      obtain ⟨a, ha, tail, htail, rfl⟩ := hx
      have ha' : (a : Int) < (b.n : Int) := by exact_mod_cast ha
      exact ⟨Int.natCast_nonneg a, ha', ih tail htail⟩

private theorem productAux_complete (bs : List Block) :
    ∀ xs : List Int, xs.length = bs.length → planInRange bs xs → xs ∈ productAux bs := by
  induction bs with
  | nil =>
      intro xs hlen hr
      cases xs with
      | nil => simp [productAux]
      | cons _ _ => simp at hlen
  | cons b bs ih =>
      intro xs hlen hr
      cases xs with
      | nil => simp at hlen
      | cons a as =>
          simp only [List.length_cons, Nat.succ.injEq] at hlen
          obtain ⟨ha0, han, hr⟩ := hr
          have hmem : ((a : Int) :: as) ∈ (productAux bs).map (fun t => (a : Int) :: t) :=
            List.mem_map.mpr ⟨as, ih as hlen hr, rfl⟩
          have hcast : ((a.toNat : Int)) = a := Int.toNat_of_nonneg ha0
          have hltInt : (a.toNat : Int) < (b.n : Int) := by rw [hcast]; exact han
          refine List.mem_flatMap.mpr ⟨a.toNat, List.mem_range.mpr ?_, ?_⟩
          · exact_mod_cast hltInt
          · simpa [hcast] using hmem

theorem allPlans_complete (inst : Instance) (xs : List Int)
    (hlen : xs.length = inst.blocks.length) (hrange : planInRange inst.blocks xs) :
    xs ∈ allPlans inst :=
  productAux_complete inst.blocks xs hlen hrange

theorem allPlans_sound (inst : Instance) (xs : List Int) (h : xs ∈ allPlans inst) :
    xs.length = inst.blocks.length ∧ planInRange inst.blocks xs :=
  ⟨productAux_length inst.blocks xs h, productAux_range inst.blocks xs h⟩

/-! ## Soundness: what a passing certificate proves -/

theorem certificate_cost (cert : Cert) (h : CheckCert cert) :
    cost cert.inst cert.plan = cert.claimed :=
  h.2.2.2.2.1

theorem certificate_first_action (cert : Cert) (h : CheckCert cert) :
    firstCommand cert.inst cert.plan = cert.first :=
  h.2.2.2.1

theorem certificate_optimal (cert : Cert) (h : CheckCert cert) :
    ∀ p ∈ allPlans cert.inst, leLex cert.claimed (cost cert.inst p) = true :=
  h.2.2.2.2.2.1

theorem certificate_tie_break (cert : Cert) (h : CheckCert cert) :
    ∀ p ∈ allPlans cert.inst,
      planIndex cert.inst p < planIndex cert.inst cert.plan →
        leLex (cost cert.inst p) cert.claimed = false :=
  h.2.2.2.2.2.2

/-- The intended combined statement: the certified first command is the command
of a plan that costs no more than every admissible plan of the declared model,
and every plan enumerated before it costs strictly more. -/
theorem certificate_first_action_optimal (cert : Cert) (h : CheckCert cert) :
    firstCommand cert.inst cert.plan = cert.first ∧
    (∀ p ∈ allPlans cert.inst, leLex cert.claimed (cost cert.inst p) = true) ∧
    (∀ p ∈ allPlans cert.inst,
      planIndex cert.inst p < planIndex cert.inst cert.plan →
        ltLex cert.claimed (cost cert.inst p) = true) := by
  refine ⟨certificate_first_action cert h, certificate_optimal cert h, ?_⟩
  intro p hp hlt
  exact ltLex_true (certificate_optimal cert h p hp) (certificate_tie_break cert h p hp hlt)

end Corpus.Timed
