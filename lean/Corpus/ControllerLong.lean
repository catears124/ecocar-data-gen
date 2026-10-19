/-
  Longitudinal guarantees for the controller's OWN caps (not only when they
  coincide with `Cert`'s): `Controller.decideCaps` on an arbitrary explicit cap
  list, hence `Controller.decideLong` on the caps the controller builds from the
  published path's curvature, the gap-speed rule and the sidestep caps.

  * `decideCaps_gs_feasible`: in the greatest-safe branch the forward profile is
    `Cert.Feasible` (every cap, acceleration and braking budget, terminal cap).
  * `decideCaps_gs_greatest`: it is pointwise ≥ every feasible profile from the
    current speed (so optimal for every speed-antitone objective).
  * `decideCaps_accel`: the emitted integer acceleration lies in
    `[-a_brk, a_acc]`, its end point starts a feasible profile, and no larger
    integer acceleration does.
  * `decideCaps_mv_infeasible`: the min-violation branch has no feasible profile.
  * `decideLong_*`: the same for `decideLong` (any curvature list, any extra
    caps), using only `Cert.strictlyIncreasing` from the input guard.
-/
import Corpus.Controller

namespace Corpus.Controller
open Corpus

theorem decideCaps_gs_facts (c : Cert.Consts) (q : Cert.Q) (segs : List (Int × Int)) (last : Int)
    (gaps : Bool) (hd : (decideCaps c q segs last gaps).kind = Cert.kindGreatestSafe) :
    Cert.wellFormed q = true ∧ Cert.constsOK c = true ∧ (∀ p ∈ segs, 0 ≤ p.1) ∧ 0 ≤ last ∧
    q.v0 * q.v0 ≤ (Cert.bpass c.a_brk segs last).headD 0 ∧
    decideCaps c q segs last gaps =
      ⟨Cert.kindGreatestSafe,
        Cert.accelOf ((Cert.fpass c.a_acc c.a_brk segs last (q.v0 * q.v0)).getD 1 0)
          (q.v0 * q.v0) (q.s.getD 1 0 - q.s.getD 0 0),
        some ((Cert.fpass c.a_acc c.a_brk segs last (q.v0 * q.v0)).getD 1 0),
        some (q.v0 * q.v0), some ((Cert.bpass c.a_brk segs last).headD 0)⟩ := by
  unfold decideCaps at hd ⊢
  split at hd
  · simp [Cert.kindInvalid, Cert.kindGreatestSafe] at hd
  rename_i h1
  split at hd
  · simp [Cert.kindNoGap, Cert.kindGreatestSafe] at hd
  rename_i h2
  split at hd
  · simp [Cert.kindMinViolation, Cert.kindGreatestSafe] at hd
  rename_i h3
  simp only [Bool.not_eq_true', Bool.not_eq_false] at h1
  have h1' : (Cert.wellFormed q && Cert.constsOK c &&
      (segs.all (fun p => decide (0 ≤ p.1)) && decide (0 ≤ last))) = true := by
    simpa using h1
  simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h1'
  obtain ⟨⟨hw, hc⟩, hs, hl⟩ := h1'
  refine ⟨hw, hc, hs, hl, by omega, ?_⟩
  rw [if_neg (by simpa using h1), if_neg h2, if_neg h3]

theorem decideCaps_mv_facts (c : Cert.Consts) (q : Cert.Q) (segs : List (Int × Int)) (last : Int)
    (gaps : Bool) (hd : (decideCaps c q segs last gaps).kind = Cert.kindMinViolation) :
    (Cert.bpass c.a_brk segs last).headD 0 < q.v0 * q.v0 := by
  unfold decideCaps at hd
  split at hd
  · simp [Cert.kindInvalid, Cert.kindMinViolation] at hd
  split at hd
  · simp [Cert.kindNoGap, Cert.kindMinViolation] at hd
  split at hd
  · rename_i h; omega
  · simp [Cert.kindGreatestSafe, Cert.kindMinViolation] at hd

theorem constsOK_facts {c : Cert.Consts} (h : Cert.constsOK c = true) :
    0 ≤ c.a_acc ∧ 0 ≤ c.a_brk ∧ 0 < c.a_max := by
  simp only [Cert.constsOK, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨h.1.1.1, h.1.1.2, h.1.2⟩

theorem v0sq_nonneg {q : Cert.Q} (hw : Cert.wellFormed q = true) : 0 ≤ q.v0 * q.v0 := by
  simp only [Cert.wellFormed, Bool.and_eq_true, decide_eq_true_eq] at hw
  exact Int.mul_nonneg hw.2 hw.2

/-- Greatest-safe profile is feasible for the caps actually used. -/
theorem decideCaps_gs_feasible (c : Cert.Consts) (q : Cert.Q) (segs : List (Int × Int)) (last : Int)
    (gaps : Bool) (hgap : ∀ p ∈ segs, 0 ≤ p.2)
    (hd : (decideCaps c q segs last gaps).kind = Cert.kindGreatestSafe) :
    Cert.Feasible c.a_acc c.a_brk segs last (Cert.fpass c.a_acc c.a_brk segs last (q.v0 * q.v0)) := by
  obtain ⟨hw, hc, hs, hl, hb, _⟩ := decideCaps_gs_facts c q segs last gaps hd
  obtain ⟨hA, hB, _⟩ := constsOK_facts hc
  exact Cert.fpass_feasible _ _ hA hB segs last _ ⟨fun p hp => ⟨hs p hp, hgap p hp⟩, hl⟩
    (v0sq_nonneg hw) hb

/-- ... and pointwise greatest among all feasible profiles from the current speed. -/
theorem decideCaps_gs_greatest (c : Cert.Consts) (q : Cert.Q) (segs : List (Int × Int)) (last : Int)
    (t : List Int) (hf : Cert.Feasible c.a_acc c.a_brk segs last (q.v0 * q.v0 :: t)) :
    Cert.LeAll (q.v0 * q.v0 :: t) (Cert.fpass c.a_acc c.a_brk segs last (q.v0 * q.v0)) :=
  Cert.fpass_greatest _ _ _ _ _ _ t hf (Int.le_refl _)

/-- Min-violation is exact: no feasible profile exists from the current speed. -/
theorem decideCaps_mv_infeasible (c : Cert.Consts) (q : Cert.Q) (segs : List (Int × Int)) (last : Int)
    (gaps : Bool) (hd : (decideCaps c q segs last gaps).kind = Cert.kindMinViolation)
    (t : List Int) : ¬ Cert.Feasible c.a_acc c.a_brk segs last (q.v0 * q.v0 :: t) := by
  intro hf
  have := Cert.LeAll_head (Cert.le_bpass _ _ _ _ _ hf)
  have := decideCaps_mv_facts c q segs last gaps hd
  omega

/-- Integer-acceleration optimality for the controller's caps. -/
theorem decideCaps_accel (c : Cert.Consts) (q : Cert.Q) (c0 d : Int) (rest : List (Int × Int))
    (last : Int) (gaps : Bool) (hd0 : d = q.s.getD 1 0 - q.s.getD 0 0) (hdpos : 0 < d)
    (hgap : ∀ p ∈ (c0, d) :: rest, 0 ≤ p.2)
    (hd : (decideCaps c q ((c0, d) :: rest) last gaps).kind = Cert.kindGreatestSafe) :
    let a := (decideCaps c q ((c0, d) :: rest) last gaps).a
    (-c.a_brk) ≤ a ∧ a ≤ c.a_acc ∧
    (∀ (a' : Int) (t : List Int),
      Cert.Feasible c.a_acc c.a_brk ((c0, d) :: rest) last
        (q.v0 * q.v0 :: max 0 (q.v0 * q.v0 + a' * (2 * d)) :: t) → a' ≤ a) := by
  intro a
  obtain ⟨hw, hc, hs, hl, hb, he⟩ := decideCaps_gs_facts c q _ last gaps hd
  obtain ⟨hA, hB, _⟩ := constsOK_facts hc
  have ha : a = Cert.accelOf ((Cert.fpass c.a_acc c.a_brk ((c0, d) :: rest) last (q.v0 * q.v0)).getD 1 0)
      (q.v0 * q.v0) d := by
    show (decideCaps c q ((c0, d) :: rest) last gaps).a = _
    rw [he, hd0]
  have := Cert.accel_greatest c.a_acc c.a_brk c0 d last (q.v0 * q.v0) rest hA hB hdpos
    ⟨fun p hp => ⟨hs p hp, hgap p hp⟩, hl⟩ (v0sq_nonneg hw) hb
  simp only at this
  rw [← ha] at this
  exact ⟨this.1, this.2.1, this.2.2.2⟩

/-! ### Instantiation for `decideLong` -/

theorem segsOf_gaps_pos : ∀ (cs s : List Int), Cert.strictlyIncreasing s = true →
    ∀ p ∈ segsOf cs s, 0 < p.2
  | c :: cs, s0 :: s1 :: ss, h, p, hp => by
      simp only [segsOf, List.mem_cons] at hp
      simp only [Cert.strictlyIncreasing, Bool.and_eq_true, decide_eq_true_eq] at h
      rcases hp with rfl | hp
      · show 0 < s1 - s0; omega
      · exact segsOf_gaps_pos cs (s1 :: ss) h.2 p hp
  | [], _, _, p, hp => by simp [segsOf] at hp
  | _ :: _, [], _, p, hp => by simp [segsOf] at hp
  | _ :: _, [_], _, p, hp => by simp [segsOf] at hp

theorem wellFormed_inc {q : Cert.Q} (hw : Cert.wellFormed q = true) : Cert.strictlyIncreasing q.s = true := by
  simp only [Cert.wellFormed, Bool.and_eq_true] at hw
  exact hw.1.1.1.1.1.2

theorem decideLong_wf (c : LongConsts) (q : Cert.Q) (ks extra : List Int)
    (hd : (decideLong c q ks extra).kind = Cert.kindGreatestSafe) : Cert.wellFormed q = true :=
  (decideCaps_gs_facts _ _ _ _ _ hd).1

/-- **Controller speed plan, greatest-safe branch**: for the caps the controller
actually builds (any curvature list `ks`, any extra caps), the forward profile is
feasible, and dominates every feasible profile from the current speed. -/
theorem decideLong_gs (c : LongConsts) (q : Cert.Q) (ks extra : List Int)
    (hd : (decideLong c q ks extra).kind = Cert.kindGreatestSafe) :
    Cert.Feasible c.cert.a_acc c.cert.a_brk (longSegs c q ks extra) (longLastCap c q ks extra)
      (Cert.fpass c.cert.a_acc c.cert.a_brk (longSegs c q ks extra) (longLastCap c q ks extra) (q.v0 * q.v0)) ∧
    (∀ t, Cert.Feasible c.cert.a_acc c.cert.a_brk (longSegs c q ks extra) (longLastCap c q ks extra)
        (q.v0 * q.v0 :: t) →
      Cert.LeAll (q.v0 * q.v0 :: t)
        (Cert.fpass c.cert.a_acc c.cert.a_brk (longSegs c q ks extra) (longLastCap c q ks extra) (q.v0 * q.v0))) := by
  have hw := decideLong_wf c q ks extra hd
  refine ⟨decideCaps_gs_feasible _ _ _ _ _
    (fun p hp => Int.le_of_lt (segsOf_gaps_pos _ _ (wellFormed_inc hw) p hp)) hd, ?_⟩
  intro t hf
  exact decideCaps_gs_greatest _ _ _ _ t hf

/-- **Controller speed plan, braking branch**: min-violation is reported exactly
when no feasible profile exists for the controller's caps. -/
theorem decideLong_mv (c : LongConsts) (q : Cert.Q) (ks extra : List Int)
    (hd : (decideLong c q ks extra).kind = Cert.kindMinViolation) (t : List Int) :
    ¬ Cert.Feasible c.cert.a_acc c.cert.a_brk (longSegs c q ks extra) (longLastCap c q ks extra)
      (q.v0 * q.v0 :: t) :=
  decideCaps_mv_infeasible _ _ _ _ _ hd t

end Corpus.Controller
