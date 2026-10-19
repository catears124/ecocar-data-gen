/-
Controller-level fixed-point specification for `teams/certline.py`.
The longitudinal model is exact at cap_scale = 1: Python's `int(x*cap_scale)` is
represented by identity (finite binary64 |x| < 2^53 is exact after multiplication by 1).
Lateral constants map certline's margin/hold/ramps/width/kappa_max to `Corpus.Steer`;
the controller may run this arithmetic outside Steer's stricter decision domain.
Clearance, projection, publication and actuation are explicit integer models.
-/
import Corpus.Cert
import Corpus.Steer

namespace Corpus.Controller

/-! Longitudinal caps and the shared certified core. -/
structure LongConsts where
  cert : Cert.Consts
  gap_speed : Bool
  gap_base : Int
  gap_per : Int
  gap_margin : Int

def capFromKappa (c : Cert.Consts) (k : Int) : Int :=
  if k ≤ 0 then c.v_max * c.v_max else min (c.v_max * c.v_max) (c.a_lat * 1000000000 / k)

def gapAt (c : LongConsts) (q : Cert.Q) (ob : List Int) : Int :=
  match Cert.obsTriple ob with
  | some (d, lmin, lmax) =>
    let i := Cert.nearestIndex q.s d
    max (q.wl.getD i 0 - lmax) (lmin + q.wr.getD i 0)
  | none => 0

def gapCap (c : LongConsts) (q : Cert.Q) (ob : List Int) : Int :=
  let g := gapAt c q ob
  c.gap_base + c.gap_per * max (g - c.cert.veh_width - 2 * c.gap_margin) 0

def gapOK (c : LongConsts) (q : Cert.Q) : Bool :=
  if !c.gap_speed then true else q.obstacles.all (fun ob =>
    match Cert.obsTriple ob with
    | some _ => decide (gapAt c q ob ≥ c.cert.veh_width + 2 * c.gap_margin)
    | none => false)

def capAtLong (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (extra : List Int := []) (i : Nat) : Int :=
  let base := min (capFromKappa c.cert (ks.getD i 0)) (extra.getD i (c.cert.v_max * c.cert.v_max))
  q.obstacles.foldl (fun acc ob =>
    match Cert.obsTriple ob with
    | some (d, _, _) =>
      if (q.s.getD i 0 - d).natAbs ≤ c.cert.hold.natAbs then
        min acc (gapCap c q ob * gapCap c q ob)
      else acc
    | none => 0) base

def capsLong (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (extra : List Int := []) : List Int :=
  (List.range q.s.length).map (capAtLong c q ks extra)

def segsOf : List Int → List Int → List (Int × Int)
  | c :: cs, s0 :: s1 :: ss => (c, s1 - s0) :: segsOf cs (s1 :: ss)
  | _, _ => []

def longSegs (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (extra : List Int := []) : List (Int × Int) :=
  segsOf (capsLong c q ks extra) q.s

def longLastCap (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (extra : List Int := []) : Int :=
  let xs := capsLong c q ks extra
  min (xs.getD (xs.length - 1) 0) (c.cert.a_lat * c.cert.r_h)

def decideCaps (c : Cert.Consts) (q : Cert.Q) (segs : List (Int × Int)) (last : Int)
    (gaps : Bool) : Cert.Decision :=
  if !(Cert.wellFormed q && Cert.constsOK c &&
      (segs.all (fun p => decide (0 ≤ p.1)) && decide (0 ≤ last))) then
    ⟨Cert.kindInvalid, -c.a_max, none, none, none⟩
  else if !gaps then
    ⟨Cert.kindNoGap, -c.a_max, none, some (q.v0 * q.v0), none⟩
  else if q.v0 * q.v0 > (Cert.bpass c.a_brk segs last).headD 0 then
    ⟨Cert.kindMinViolation, -c.a_max, none, some (q.v0 * q.v0), some ((Cert.bpass c.a_brk segs last).headD 0)⟩
  else
    ⟨Cert.kindGreatestSafe,
      Cert.accelOf ((Cert.fpass c.a_acc c.a_brk segs last (q.v0 * q.v0)).getD 1 0)
        (q.v0 * q.v0) (q.s.getD 1 0 - q.s.getD 0 0),
      some ((Cert.fpass c.a_acc c.a_brk segs last (q.v0 * q.v0)).getD 1 0),
      some (q.v0 * q.v0), some ((Cert.bpass c.a_brk segs last).headD 0)⟩

def decideLong (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (extra : List Int := []) : Cert.Decision :=
  decideCaps c.cert q (longSegs c q ks extra) (longLastCap c q ks extra) (gapOK c q)

theorem decideLong_common_core (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (extra : List Int := []) :
    decideLong c q ks extra =
      decideCaps c.cert q (longSegs c q ks extra) (longLastCap c q ks extra) (gapOK c q) := by rfl

def capRelation (c : LongConsts) (q : Cert.Q) (ks : List Int) : Prop :=
  longSegs c q ks = Cert.segs c.cert q ∧ longLastCap c q ks = Cert.lastCap c.cert q ∧
  gapOK c q = Cert.gapsFeasible c.cert q ∧ Cert.capsNonneg c.cert q = true

theorem decideLong_eq_cert (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (h : capRelation c q ks) : decideLong c q ks = Cert.decide c.cert q := by
  rcases h with ⟨hs, hl, hg, hnon⟩
  unfold decideLong
  rw [hs, hl, hg]
  unfold decideCaps Cert.decide Cert.capsNonneg
  rfl
/-- Main result: the common core's greatest-safe profile is feasible. -/
theorem decideLong_greatest_safe_feasible (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (h : capRelation c q ks) (hd : (decideLong c q ks).kind = Cert.kindGreatestSafe) :
    Cert.Feasible c.cert.a_acc c.cert.a_brk (Cert.segs c.cert q) (Cert.lastCap c.cert q)
      (Cert.fpass c.cert.a_acc c.cert.a_brk (Cert.segs c.cert q) (Cert.lastCap c.cert q)
        (q.v0 * q.v0)) := by
  rw [decideLong_eq_cert c q ks h] at hd
  exact Cert.decide_greatest_safe_feasible c.cert q hd

/-- Main result: greatest-safe next speed dominates every feasible continuation. -/
theorem decideLong_greatest_safe (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (h : capRelation c q ks) (hd : (decideLong c q ks).kind = Cert.kindGreatestSafe)
    (b1 : Int) (hb : (decideLong c q ks).b1 = some b1) (t : List Int)
    (hf : Cert.Feasible c.cert.a_acc c.cert.a_brk (Cert.segs c.cert q) (Cert.lastCap c.cert q)
      (q.v0 * q.v0 :: t)) : (q.v0 * q.v0 :: t).getD 1 0 ≤ b1 := by
  rw [decideLong_eq_cert c q ks h] at hd hb
  exact Cert.decide_greatest_safe c.cert q b1 hd hb t hf

/- The acceleration theorem retains Cert's exact two-station endpoint shape. -/
theorem decideLong_accel_greatest (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (h : capRelation c q ks) (hd : (decideLong c q ks).kind = Cert.kindGreatestSafe)
    (a' : Int) (t : List Int)
    (ha : Cert.Feasible c.cert.a_acc c.cert.a_brk (Cert.segs c.cert q) (Cert.lastCap c.cert q)
      (q.v0 * q.v0 ::
        max 0 (q.v0 * q.v0 + a' * (2 * (q.s.getD 1 0 - q.s.getD 0 0))) :: t)) :
    a' ≤ (decideLong c q ks).a := by
  rw [decideLong_eq_cert c q ks h] at hd ⊢
  have hh := Cert.decide_accel_greatest c.cert q hd
  exact hh.2.2.2 a' t ha

/-- Main result: the min-violation branch is infeasible from the current speed. -/
theorem decideLong_min_violation_infeasible (c : LongConsts) (q : Cert.Q) (ks : List Int)
    (h : capRelation c q ks) (hd : (decideLong c q ks).kind = Cert.kindMinViolation)
    (t : List Int) : ¬ Cert.Feasible c.cert.a_acc c.cert.a_brk (Cert.segs c.cert q)
      (Cert.lastCap c.cert q) (q.v0 * q.v0 :: t) := by
  rw [decideLong_eq_cert c q ks h] at hd
  exact Cert.decide_min_violation_infeasible c.cert q hd t

theorem curvature_cap_sound (c : Cert.Consts) (k b : Int)
    (hk : 0 < k) (hb : b ≤ min (c.v_max * c.v_max) (c.a_lat * 1000000000 / k)) :
    b * k ≤ c.a_lat * 1000000000 := by
  apply Int.mul_le_of_le_ediv hk
  exact Int.le_trans hb (Int.min_le_right _ _)

/-! Lateral model: exact Steer arithmetic, including signed-latitude tie breaking. -/
def decideLateral (c : Steer.Consts) (q : Steer.Q) : Steer.Decision :=
  { kind := if Steer.wellFormed c q then Steer.kindPlanOk else Steer.kindInvalid
    side := Steer.sides q, amp := Steer.amps q, off := Steer.offs c q
    clear := Steer.clears c q (Steer.offs c q)
    curv := Steer.curvs q (Steer.offs c q)
    steer := Steer.steers c (Steer.curvs q (Steer.offs c q)) }

theorem decideLateral_offsets_eq (c : Steer.Consts) (q : Steer.Q) :
    (decideLateral c q).off = Steer.offs c q := by rfl
theorem decideLateral_curv_eq (c : Steer.Consts) (q : Steer.Q) :
    (decideLateral c q).curv = Steer.curvs q (Steer.offs c q) := by rfl
theorem decideLateral_steer_eq (c : Steer.Consts) (q : Steer.Q) :
    (decideLateral c q).steer = Steer.steers c (Steer.curvs q (Steer.offs c q)) := by rfl

structure MemoryOb where
  d : Int
  lmin : Int
  lmax : Int
  side : Int

def covers (hold si : Int) (o : MemoryOb) : Prop := o.d - hold ≤ si ∧ si ≤ o.d + hold

def spareBox (c : Steer.Consts) (q : Steer.Q) (mem : List MemoryOb) (i : Nat) : Int × Int :=
  match mem with
  | [] => (Steer.lo c q i, Steer.hi c q i)
  | o :: os =>
    let b := spareBox c q os i
    if h : o.d - c.lon.hold ≤ q.base.s.getD i 0 ∧ q.base.s.getD i 0 ≤ o.d + c.lon.hold then
      if o.side > 0 then (max b.1 (o.lmax + Steer.halfw c + c.margin), b.2)
      else if o.side < 0 then (b.1, min b.2 (o.lmin - Steer.halfw c - c.margin)) else b
    else b

private def clearMem (c : Steer.Consts) (q : Steer.Q) (mem : List MemoryOb) (i : Nat) (x : Int) : Prop :=
  ∀ o ∈ mem, covers c.lon.hold (q.base.s.getD i 0) o →
    (o.side > 0 → o.lmax + Steer.halfw c + c.margin ≤ x) ∧
    (o.side < 0 → x ≤ o.lmin - Steer.halfw c - c.margin)

private def inCorridor (c : Steer.Consts) (q : Steer.Q) (i : Nat) (x : Int) : Prop :=
  Steer.lo c q i ≤ x ∧ x ≤ Steer.hi c q i

/-- **Soundness of the certified clearance interval** (`_spare_boxes`), for an
ARBITRARY memory list: any offset inside the interval is inside the corridor and
beyond the edge of EVERY remembered obstacle whose hold window covers the
station, on that obstacle's remembered side. -/
theorem spareBox_sound (c : Steer.Consts) (q : Steer.Q) :
    ∀ (mem : List MemoryOb) (i : Nat) (x : Int),
      (spareBox c q mem i).1 ≤ x → x ≤ (spareBox c q mem i).2 →
      inCorridor c q i x ∧ clearMem c q mem i x := by
  intro mem
  induction mem with
  | nil =>
      intro i x h1 h2
      exact ⟨⟨h1, h2⟩, fun o ho => by cases ho⟩
  | cons o os ih =>
      intro i x h1 h2
      simp only [spareBox] at h1 h2
      split at h1
      · rename_i hcov
        rw [dif_pos hcov] at h2
        by_cases hp : o.side > 0
        · simp only [hp, if_true] at h1 h2
          obtain ⟨hc, hm⟩ := ih i x (by omega) h2
          refine ⟨hc, fun o' ho' hcov' => ?_⟩
          rcases List.mem_cons.mp ho' with rfl | ho'
          · exact ⟨fun _ => by omega, fun h => by omega⟩
          · exact hm o' ho' hcov'
        · simp only [hp, if_false] at h1 h2
          by_cases hn : o.side < 0
          · simp only [hn, if_true] at h1 h2
            obtain ⟨hc, hm⟩ := ih i x h1 (by omega)
            refine ⟨hc, fun o' ho' hcov' => ?_⟩
            rcases List.mem_cons.mp ho' with rfl | ho'
            · exact ⟨fun h => absurd h hp, fun _ => by omega⟩
            · exact hm o' ho' hcov'
          · simp only [hn, if_false] at h1 h2
            obtain ⟨hc, hm⟩ := ih i x h1 h2
            refine ⟨hc, fun o' ho' hcov' => ?_⟩
            rcases List.mem_cons.mp ho' with rfl | ho'
            · exact ⟨fun h => absurd h hp, fun h => absurd h hn⟩
            · exact hm o' ho' hcov'
      · rename_i hcov
        rw [dif_neg hcov] at h2
        obtain ⟨hc, hm⟩ := ih i x h1 h2
        refine ⟨hc, fun o' ho' hcov' => ?_⟩
        rcases List.mem_cons.mp ho' with rfl | ho'
        · exact absurd hcov' hcov
        · exact hm o' ho' hcov'

/-- **Exactness (converse)**: an offset inside the corridor and beyond the edge of
every covering remembered obstacle lies in the interval, so the interval is the
whole clearance set (it is empty exactly when no clear offset exists). -/
theorem spareBox_complete (c : Steer.Consts) (q : Steer.Q) :
    ∀ (mem : List MemoryOb) (i : Nat) (x : Int),
      inCorridor c q i x → clearMem c q mem i x →
      (spareBox c q mem i).1 ≤ x ∧ x ≤ (spareBox c q mem i).2 := by
  intro mem
  induction mem with
  | nil => intro i x hc _; exact hc
  | cons o os ih =>
      intro i x hc hm
      have hm' : clearMem c q os i x := fun o' ho' hcov' => hm o' (List.mem_cons_of_mem o ho') hcov'
      obtain ⟨h1, h2⟩ := ih i x hc hm'
      simp only [spareBox]
      by_cases hcov : o.d - c.lon.hold ≤ q.base.s.getD i 0 ∧ q.base.s.getD i 0 ≤ o.d + c.lon.hold
      · rw [dif_pos hcov]
        have ho := hm o (by simp) hcov
        by_cases hp : o.side > 0
        · simp only [hp, if_true]; have := ho.1 hp; omega
        · by_cases hn : o.side < 0
          · simp only [hp, hn, if_true, if_false]; have := ho.2 hn; omega
          · simp only [hp, hn, if_false]; omega
      · rw [dif_neg hcov]; omega

/-- The interval is empty iff no offset is clear: the controller's
"uncertified station" flag is exactly the absence of a clear offset. -/
theorem spareBox_empty_iff (c : Steer.Consts) (q : Steer.Q) (mem : List MemoryOb) (i : Nat) :
    (spareBox c q mem i).2 < (spareBox c q mem i).1 ↔
      ¬ ∃ x, inCorridor c q i x ∧ clearMem c q mem i x := by
  constructor
  · intro h ⟨x, hc, hm⟩
    have := spareBox_complete c q mem i x hc hm
    omega
  · intro h
    by_cases hle : (spareBox c q mem i).1 ≤ (spareBox c q mem i).2
    · exact absurd ⟨_, spareBox_sound c q mem i _ (Int.le_refl _) hle⟩ h
    · omega
def shrinkBox (shrink : Int) (ab : Int × Int) : Int × Int :=
  if ab.2 - ab.1 > 2 * shrink then (ab.1 + shrink, ab.2 - shrink) else ab

def planBox (c : Steer.Consts) (q : Steer.Q) (mem : List MemoryOb) (shrink : Int) (i : Nat) : Int × Int × Bool :=
  let b := shrinkBox shrink (spareBox c q mem i)
  (b.1, b.2, b.1 > b.2)

theorem planBox_inside (shrink lo hi : Int) (hs : 0 ≤ shrink) (h : lo ≤ hi) :
    lo ≤ (shrinkBox shrink (lo, hi)).1 ∧ (shrinkBox shrink (lo, hi)).2 ≤ hi := by
  simp [shrinkBox]
  split <;> omega

theorem planBox_empty_flag (c : Steer.Consts) (q : Steer.Q) (mem : List MemoryOb)
    (shrink : Int) (i : Nat) (hs : 0 ≤ shrink) :
    (planBox c q mem shrink i).2.2 = true ↔ (spareBox c q mem i).1 > (spareBox c q mem i).2 := by
  unfold planBox
  by_cases h : (spareBox c q mem i).2 - (spareBox c q mem i).1 > 2 * shrink
  · simp [shrinkBox, h]
    omega
  · simp [shrinkBox, h]

/-! Exact integer two-pass projection and a pointwise boxed result. -/
def clamp (x lo hi : Int) : Int := min (max x lo) hi

def forwardGo : Int → List Int → List Int → List Int → Int → List Int
  | _, [], [], [], _ => []
  | prev, x :: xs, lo :: los, hi :: his, r =>
    let y := clamp (min (max x (prev - r)) (prev + r)) lo hi
    y :: forwardGo y xs los his r
  | _, _, _, _, _ => []

def forwardPass : List Int → List Int → List Int → Int → List Int
  | [], _, _, _ => []
  | x :: xs, lo :: los, hi :: his, r => x :: forwardGo x xs los his r
  | _, _, _, _ => []

def backwardPass (xs lo hi : List Int) (r : Int) : List Int :=
  (forwardPass xs.reverse lo.reverse hi.reverse r).reverse

def projectPass (xs lo hi : List Int) (r : Int) : List Int :=
  backwardPass (forwardPass xs lo hi r) lo hi r

def projectPlan (xs lo hi : List Int) (r : Int) (iters : Nat) : List Int :=
  match iters with
  | 0 => xs
  | n + 1 => projectPlan (projectPass xs lo hi r) lo hi r n

def Boxed : List Int → List Int → List Int → Prop
  | [], [], [] => True
  | x :: xs, lo :: los, hi :: his => lo ≤ x ∧ x ≤ hi ∧ Boxed xs los his
  | _, _, _ => False

theorem Boxed_append : ∀ (a b c a' b' c' : List Int),
    Boxed a b c → Boxed a' b' c' → Boxed (a ++ a') (b ++ b') (c ++ c')
  | [], [], [], _, _, _, _, h => h
  | x :: xs, l :: ls, h :: hs, a', b', c', ⟨h1, h2, h3⟩, h' =>
      ⟨h1, h2, Boxed_append xs ls hs a' b' c' h3 h'⟩
  | [], _ :: _, _, _, _, _, h, _ => h.elim
  | [], [], _ :: _, _, _, _, h, _ => h.elim
  | _ :: _, [], _, _, _, _, h, _ => h.elim
  | _ :: _, _ :: _, [], _, _, _, h, _ => h.elim

theorem Boxed_reverse : ∀ (a b c : List Int), Boxed a b c → Boxed a.reverse b.reverse c.reverse
  | [], [], [], h => h
  | x :: xs, l :: ls, h :: hs, ⟨h1, h2, h3⟩ => by
      simp only [List.reverse_cons]
      exact Boxed_append _ _ _ _ _ _ (Boxed_reverse xs ls hs h3) ⟨h1, h2, trivial⟩
  | [], _ :: _, _, h => h.elim
  | [], [], _ :: _, h => h.elim
  | _ :: _, [], _, h => h.elim
  | _ :: _, _ :: _, [], h => h.elim

theorem Boxed_length : ∀ (a b c : List Int), Boxed a b c → a.length = b.length ∧ b.length = c.length
  | [], [], [], _ => ⟨rfl, rfl⟩
  | _ :: xs, _ :: ls, _ :: hs, ⟨_, _, h3⟩ => by
      have := Boxed_length xs ls hs h3; simp only [List.length_cons]; omega
  | [], _ :: _, _, h => h.elim
  | [], [], _ :: _, h => h.elim
  | _ :: _, [], _, h => h.elim
  | _ :: _, _ :: _, [], h => h.elim

/-- Pointwise `lo ≤ hi` for equal-length bound lists. -/
def Ordered : List Int → List Int → Prop
  | [], [] => True
  | l :: ls, h :: hs => l ≤ h ∧ Ordered ls hs
  | _, _ => False

theorem Ordered_reverse : ∀ (a b : List Int), Ordered a b → Ordered a.reverse b.reverse
  | [], [], h => h
  | l :: ls, h :: hs, ⟨h1, h2⟩ => by
      simp only [List.reverse_cons]
      have r := Ordered_reverse ls hs h2
      have hl := (List.length_reverse (as := ls))
      -- append of ordered lists
      have app : ∀ (a b a' b' : List Int), Ordered a b → Ordered a' b' → Ordered (a ++ a') (b ++ b') := by
        intro a
        induction a with
        | nil => intro b a' b' h h'; cases b with
          | nil => exact h'
          | cons _ _ => exact h.elim
        | cons x xs ih => intro b a' b' h h'; cases b with
          | nil => exact h.elim
          | cons y ys => exact ⟨h.1, ih ys a' b' h.2 h'⟩
      exact app _ _ _ _ r ⟨h1, trivial⟩
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

theorem clamp_in (x lo hi : Int) (h : lo ≤ hi) : lo ≤ clamp x lo hi ∧ clamp x lo hi ≤ hi := by
  unfold clamp; omega

/-- Every entry the forward pass produces after the first is clamped. -/
theorem forwardGo_boxed (r : Int) : ∀ (xs los his : List Int) (prev : Int),
    xs.length = los.length → Ordered los his → Boxed (forwardGo prev xs los his r) los his
  | [], [], [], _, _, _ => trivial
  | x :: xs, l :: ls, h :: hs, prev, hlen, ⟨hlh, ho⟩ => by
      simp only [forwardGo]
      have := clamp_in (min (max x (prev - r)) (prev + r)) l h hlh
      exact ⟨this.1, this.2, forwardGo_boxed r xs ls hs _ (by simpa using hlen) ho⟩
  | [], _ :: _, _, _, hlen, _ => by simp at hlen
  | _ :: _, [], _, _, hlen, _ => by simp at hlen
  | [], [], _ :: _, _, _, ho => ho.elim
  | _ :: _, _ :: _, [], _, _, ho => ho.elim

/-- A forward pass whose first entry is already in its interval is fully boxed. -/
theorem forwardPass_boxed (r : Int) (x : Int) (xs l : List Int) (lo hi : Int) (ls hs : List Int)
    (hlen : xs.length = ls.length) (ho : Ordered ls hs) (hx : lo ≤ x ∧ x ≤ hi) :
    Boxed (forwardPass (x :: xs) (lo :: ls) (hi :: hs) r) (lo :: ls) (hi :: hs) :=
  ⟨hx.1, hx.2, forwardGo_boxed r xs ls hs x hlen ho⟩

theorem forwardGo_length (r : Int) : ∀ (xs los his : List Int) (prev : Int),
    xs.length = los.length → los.length = his.length → (forwardGo prev xs los his r).length = xs.length
  | [], [], [], _, _, _ => rfl
  | x :: xs, l :: ls, h :: hs, prev, h1, h2 => by
      simp only [forwardGo, List.length_cons]
      rw [forwardGo_length r xs ls hs _ (by simpa using h1) (by simpa using h2)]
  | [], _ :: _, _, _, h1, _ => by simp at h1
  | _ :: _, [], _, _, h1, _ => by simp at h1
  | [], [], _ :: _, _, _, h2 => by simp at h2
  | _ :: _, _ :: _, [], _, _, h2 => by simp at h2

theorem Ordered_length : ∀ (a b : List Int), Ordered a b → a.length = b.length
  | [], [], _ => rfl
  | _ :: ls, _ :: hs, ⟨_, h⟩ => by simp [Ordered_length ls hs h]
  | [], _ :: _, h => h.elim
  | _ :: _, [], h => h.elim

/-- **One projection pass is boxed** (lists of length ≥ 2, as in the controller,
which only projects previews of ≥ 6 stations): the forward pass clamps entries
`1..n-1`, the backward pass re-clamps `0..n-2` and starts from the already
clamped last entry. -/
theorem projectPass_boxed (xs lo hi : List Int) (r : Int)
    (hlen : xs.length = lo.length) (ho : Ordered lo hi) (h2 : 2 ≤ xs.length) :
    Boxed (projectPass xs lo hi r) lo hi ∧ (projectPass xs lo hi r).length = xs.length := by
  have hlh := Ordered_length lo hi ho
  match xs, lo, hi, hlen, ho with
  | x :: xt, l :: lt, h :: ht, hlen, ⟨hl0, hot⟩ =>
    have hlt : xt.length = lt.length := by simpa using hlen
    have hlt2 : lt.length = ht.length := by simpa using hlh
    -- forward output: x :: T with T boxed by (lt, ht)
    have hT := forwardGo_boxed r xt lt ht x hlt hot
    have hTl := forwardGo_length r xt lt ht x hlt hlt2
    generalize hTdef : forwardGo x xt lt ht r = T at hT hTl
    have hfwd : forwardPass (x :: xt) (l :: lt) (h :: ht) r = x :: T := by
      simp only [forwardPass, hTdef]
    -- split off the last entry of T (nonempty because the list has ≥ 2 entries)
    have hTne : T ≠ [] := by
      intro hT0; rw [hT0] at hTl; simp at hTl h2; omega
    obtain ⟨T', t, hTsplit⟩ := List.eq_nil_or_concat T |>.resolve_left hTne
    rw [List.concat_eq_append] at hTsplit
    have hltne : lt ≠ [] := by intro h0; rw [h0] at hlt; simp at hlt; rw [hlt] at h2; simp at h2
    obtain ⟨lt', ll, hlsplit⟩ := List.eq_nil_or_concat lt |>.resolve_left hltne
    rw [List.concat_eq_append] at hlsplit
    have hhtne : ht ≠ [] := by intro h0; rw [h0, hlsplit] at hlt2; simp at hlt2
    obtain ⟨ht', hh, hhsplit⟩ := List.eq_nil_or_concat ht |>.resolve_left hhtne
    rw [List.concat_eq_append] at hhsplit
    subst hTsplit hlsplit hhsplit
    have hlens : T'.length = lt'.length ∧ lt'.length = ht'.length := by
      have a := Boxed_length _ _ _ hT; simp at a; omega
    -- decompose the boxed tail
    have dec : ∀ (a b c : List Int) (u v w : Int), a.length = b.length → b.length = c.length →
        Boxed (a ++ [u]) (b ++ [v]) (c ++ [w]) → Boxed a b c ∧ v ≤ u ∧ u ≤ w := by
      intro a
      induction a with
      | nil => intro b c u v w h1 h2 hb; cases b with
        | nil => cases c with
          | nil => exact ⟨trivial, hb.1, hb.2.1⟩
          | cons _ _ => simp at h2
        | cons _ _ => simp at h1
      | cons y ys ih => intro b c u v w h1 h2 hb; cases b with
        | nil => simp at h1
        | cons bl bs => cases c with
          | nil => simp at h2
          | cons cl cs =>
              obtain ⟨p1, p2, p3⟩ := hb
              obtain ⟨q1, q2, q3⟩ := ih bs cs u v w (by simpa using h1) (by simpa using h2) p3
              exact ⟨⟨p1, p2, q1⟩, q2, q3⟩
    obtain ⟨hT'b, htl, hth⟩ := dec T' lt' ht' t ll hh hlens.1 hlens.2 hT
    -- the backward pass on the reversed lists starts from the boxed last entry
    have hrevx : (x :: (T' ++ [t])).reverse = t :: (T'.reverse ++ [x]) := by simp
    have hrevl : (l :: (lt' ++ [ll])).reverse = ll :: (lt'.reverse ++ [l]) := by simp
    have hrevh : (h :: (ht' ++ [hh])).reverse = hh :: (ht'.reverse ++ [h]) := by simp
    have hord : Ordered (lt'.reverse ++ [l]) (ht'.reverse ++ [h]) := by
      have := Ordered_reverse (l :: (lt' ++ [ll])) (h :: (ht' ++ [hh])) ⟨hl0, hot⟩
      rw [hrevl, hrevh] at this
      exact this.2
    have hbk := forwardPass_boxed r t (T'.reverse ++ [x]) [] ll hh (lt'.reverse ++ [l]) (ht'.reverse ++ [h])
      (by simp; omega) hord ⟨htl, hth⟩
    have hbklen := forwardGo_length r (T'.reverse ++ [x]) (lt'.reverse ++ [l]) (ht'.reverse ++ [h]) t
      (by simp; omega) (by simp; omega)
    unfold projectPass backwardPass
    rw [hfwd, hrevx, hrevl, hrevh]
    constructor
    · have := Boxed_reverse _ _ _ hbk
      have e1 : (ll :: (lt'.reverse ++ [l])).reverse = l :: (lt' ++ [ll]) := by simp
      have e2 : (hh :: (ht'.reverse ++ [h])).reverse = h :: (ht' ++ [hh]) := by simp
      rw [e1, e2] at this
      exact this
    · simp only [forwardPass, List.length_reverse, List.length_cons]
      rw [hbklen]; simp only [List.length_append, List.length_reverse, List.length_cons, List.length_nil] at hTl ⊢; omega

/-- **The controller's projection (`_project_plan`) always returns an offset
inside the certified interval at every station**, for any number `≥ 1` of
iterations and any input, whenever every interval is nonempty and the preview has
at least two stations. -/
theorem projectPlan_boxed (lo hi : List Int) (r : Int) (ho : Ordered lo hi) :
    ∀ (n : Nat) (xs : List Int), xs.length = lo.length → 2 ≤ xs.length →
      Boxed (projectPlan xs lo hi r (n + 1)) lo hi := by
  intro n
  induction n with
  | zero =>
      intro xs hlen h2
      exact (projectPass_boxed xs lo hi r hlen ho h2).1
  | succ n ih =>
      intro xs hlen h2
      have hp := projectPass_boxed xs lo hi r hlen ho h2
      show Boxed (projectPlan (projectPass xs lo hi r) lo hi r (n + 1)) lo hi
      exact ih _ (by rw [hp.2]; exact hlen) (by rw [hp.2]; exact h2)

/-! Final publication and integer actuation. -/
def publishOffset (x lo hi maxOffset : Int) : Int :=
  clamp (min (max x (-maxOffset)) maxOffset) lo hi

theorem publishOffset_inside (x lo hi maxOffset : Int) (h : lo ≤ hi) :
    lo ≤ publishOffset x lo hi maxOffset ∧ publishOffset x lo hi maxOffset ≤ hi := by
  simp [publishOffset, clamp]
  omega

structure ActInput where
  desired : Int
  prev : Int
  rate : Int
  invalid : Bool

/-- Integer (milli-unit) model of the controller's final steering pipeline, in
the controller's order: saturate to `[-1000, 1000]` (`np.clip(..., -1, 1)`, applied
before and after the bias subtraction), then the rate limit
`clip(steer, prev - rate, prev + rate)` as the LAST operation. -/
def steerAct (a : ActInput) : Int :=
  if a.invalid then 0 else
    let d := min 1000 (max (-1000) a.desired)
    min (a.prev + a.rate) (max (a.prev - a.rate) d)

def pedalAct (a : Int) : Int × Int :=
  if a < 0 then (0, min 1000 (-a)) else (min 1000 a, 0)

def gripBudget (budget muHat mu : Int) : Int :=
  min budget (budget * max 0 (min muHat mu) / max mu 1)

/-- Saturation holds although the rate limit is applied last, provided the
previous command was saturated (true initially, `prev = 0`, and inductively). -/
theorem steerAct_saturated (a : ActInput) (hr : 0 ≤ a.rate) (hp : -1000 ≤ a.prev ∧ a.prev ≤ 1000) :
    -1000 ≤ steerAct a ∧ steerAct a ≤ 1000 := by
  simp only [steerAct]
  split <;> omega

theorem steerAct_rate (a : ActInput) (hi : a.invalid = false) (hr : 0 ≤ a.rate) :
    -a.rate ≤ steerAct a - a.prev ∧ steerAct a - a.prev ≤ a.rate := by
  have e : steerAct a = min (a.prev + a.rate) (max (a.prev - a.rate) (min 1000 (max (-1000) a.desired))) := by
    unfold steerAct; rw [if_neg (by simp [hi])]
  rw [e]
  omega

/-- Along any run (any desired commands), every command is saturated and
rate-limited relative to its predecessor. -/
def steerRun (rate : Int) (ds : Nat → Int) : Nat → Int
  | 0 => 0
  | n + 1 => steerAct ⟨ds n, steerRun rate ds n, rate, false⟩

theorem steerRun_invariant (rate : Int) (hr : 0 ≤ rate) (ds : Nat → Int) :
    ∀ n, (-1000 ≤ steerRun rate ds n ∧ steerRun rate ds n ≤ 1000) ∧
      (-rate ≤ steerRun rate ds (n + 1) - steerRun rate ds n ∧
        steerRun rate ds (n + 1) - steerRun rate ds n ≤ rate) := by
  intro n
  induction n with
  | zero =>
      refine ⟨by simp [steerRun], ?_⟩
      exact steerAct_rate ⟨ds 0, 0, rate, false⟩ rfl hr
  | succ n ih =>
      have hs := steerAct_saturated ⟨ds n, steerRun rate ds n, rate, false⟩ hr ih.1
      exact ⟨hs, steerAct_rate ⟨ds (n + 1), steerRun rate ds (n + 1), rate, false⟩ rfl hr⟩

theorem pedalAct_nonoverlap (a : Int) : (pedalAct a).1 * (pedalAct a).2 = 0 := by
  simp [pedalAct]
  split <;> omega

theorem invalid_full_brake : (pedalAct (-1000)).1 = 0 ∧ (pedalAct (-1000)).2 = 1000 := by
  simp [pedalAct]

theorem gripBudget_not_above (budget muHat mu : Int) (hb : 0 ≤ budget) (hm : 0 < mu)
    (hh : 0 ≤ muHat) (hmu : muHat ≤ mu) : gripBudget budget muHat mu ≤ budget := by
  unfold gripBudget
  exact Int.min_le_left _ _

end Corpus.Controller
