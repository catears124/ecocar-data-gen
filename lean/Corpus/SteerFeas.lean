/-
  Corridor and obstacle-clearance algebra for the steer-1 plan.
  These facts deliberately do not use the smooth ramp bounds.
-/
import Corpus.Steer

namespace Corpus.SteerFeas
open Corpus

/-- When the lateral corridor is nonempty, clamping stays in that corridor. -/
theorem clamp_in_corridor (c : Steer.Consts) (q : Steer.Q) (i : Nat) (x : Int)
    (hlo : Steer.lo c q i ≤ Steer.hi c q i) :
    Steer.lo c q i ≤ Steer.clampAt c q i x ∧
      Steer.clampAt c q i x ≤ Steer.hi c q i := by
  simp [Steer.clampAt, hlo]
  omega

/-- The absolute value cannot increase when a value is clamped to an interval
containing zero. -/
theorem clamp_abs_le_gen (c : Steer.Consts) (q : Steer.Q) (i : Nat) (x : Int)
    (hlo : Steer.lo c q i ≤ 0) (hhi : 0 ≤ Steer.hi c q i) :
    Cert.iabs (Steer.clampAt c q i x) ≤ Cert.iabs x := by
  have hle : Steer.lo c q i ≤ Steer.hi c q i := by omega
  simp [Steer.clampAt, hle]
  exact Steer.clamp_abs_le _ _ _ hlo hhi

/-- Midpoint and corridor bounds for the left branch of `sideAmp`. -/
theorem target_in_corridor_left
    (c : Steer.Consts) (wl lmax amp : Int)
    (hgap : c.lon.veh_width + 2 * c.margin ≤ wl - lmax)
    (hsign : 0 ≤ lmax + wl)
    (heven : 2 * Steer.halfw c = c.lon.veh_width)
    (hamp : amp = Cert.cdiv (Cert.iabs (lmax + wl)) 2) :
    lmax + Steer.halfw c + c.margin ≤ amp ∧
      amp ≤ wl - Steer.halfw c - c.margin ∧ 0 ≤ amp := by
  rw [hamp]
  simp [Cert.cdiv, Cert.iabs]
  omega

/-- Midpoint and corridor bounds for the right branch of `sideAmp`. -/
theorem target_in_corridor_right
    (c : Steer.Consts) (wr lmin amp : Int)
    (hgap : c.lon.veh_width + 2 * c.margin ≤ lmin + wr)
    (hsign : lmin - wr ≤ 0)
    (heven : 2 * Steer.halfw c = c.lon.veh_width)
    (hamp : amp = Cert.cdiv (Cert.iabs (lmin - wr)) 2) :
    0 ≤ amp ∧ -(wr - Steer.halfw c - c.margin) ≤ -amp ∧
      -amp ≤ lmin - Steer.halfw c - c.margin := by
  rw [hamp]
  by_cases hz : lmin - wr = 0
  · simp [Cert.cdiv, Cert.iabs, hz]
    omega
  · have hn : lmin - wr < 0 := by omega
    simp [Cert.cdiv, Cert.iabs, hn]
    omega

/-- Clearance in a hold window, once the selected `sideAmp` branch and its
ordinary lateral-placement signs are supplied.

`hbranch` spells out the strict left/right and ties-resolved tie branches of
`sideAmp`; `hchosen` records the frozen selector output.  `hplan` is the
explicit plan-selection fact at this station, rather than an unsound claim
about `rawOffAt` at a boundary tie.  The equality `heven` is needed because
integer vehicle widths need not be even in the frozen input domain.
-/
theorem off_clearance_hold
    (c : Steer.Consts) (q : Steer.Q) (i : Nat)
    (d lmin lmax side amp si : Int)
    (hidx : i = Cert.nearestIndex q.base.s d)
    (hchosen : (side, amp) = Steer.sideAmp q d lmin lmax)
    (hbranch :
      (lmin + q.base.wr.getD i 0 < q.base.wl.getD i 0 - lmax ∧
          side = 1 ∧ amp = Cert.cdiv (Cert.iabs (lmax + q.base.wl.getD i 0)) 2) ∨
      (q.base.wl.getD i 0 - lmax < lmin + q.base.wr.getD i 0 ∧
          side = -1 ∧ amp = Cert.cdiv (Cert.iabs (lmin - q.base.wr.getD i 0)) 2) ∨
      (lmin + q.base.wr.getD i 0 = q.base.wl.getD i 0 - lmax ∧ 0 ≤ q.lat ∧
          side = 1 ∧ amp = Cert.cdiv (Cert.iabs (lmax + q.base.wl.getD i 0)) 2) ∨
      (lmin + q.base.wr.getD i 0 = q.base.wl.getD i 0 - lmax ∧ ¬ 0 ≤ q.lat ∧
          side = -1 ∧ amp = Cert.cdiv (Cert.iabs (lmin - q.base.wr.getD i 0)) 2))
    (hgap : c.lon.veh_width + 2 * c.margin ≤
      max (Steer.gapLeft q d lmax) (Steer.gapRight q d lmin))
    (hleftsign : 0 ≤ lmax + q.base.wl.getD i 0)
    (hrightsign : lmin - q.base.wr.getD i 0 ≤ 0)
    (heven : 2 * Steer.halfw c = c.lon.veh_width)
    (hlo0 : Steer.lo c q i ≤ 0)
    (hhi0 : 0 ≤ Steer.hi c q i)
    (hholdIn : d - c.lon.hold - c.lon.ramp_in ≤ si)
    (hhold0 : d - c.lon.hold ≤ si)
    (hhold1 : si ≤ d + c.lon.hold)
    (hplan : Steer.offAt c q si i = Steer.clampAt c q i (side * amp)) :
    (side = 1 ∧ lmax + Steer.halfw c + c.margin ≤ Steer.offAt c q si i) ∨
      (side = -1 ∧ Steer.offAt c q si i ≤ lmin - Steer.halfw c - c.margin) := by
  have hleft : Steer.gapLeft q d lmax = q.base.wl.getD i 0 - lmax := by
    simp [Steer.gapLeft, hidx]
  have hright : Steer.gapRight q d lmin = lmin + q.base.wr.getD i 0 := by
    simp [Steer.gapRight, hidx]
  have hgap' : c.lon.veh_width + 2 * c.margin ≤
      max (q.base.wl.getD i 0 - lmax) (lmin + q.base.wr.getD i 0) := by
    rw [hleft, hright] at hgap
    exact hgap
  rcases hbranch with hL | hR | hTL | hTR
  · have hg : c.lon.veh_width + 2 * c.margin ≤
        q.base.wl.getD i 0 - lmax := by
      have hm : max (q.base.wl.getD i 0 - lmax) (lmin + q.base.wr.getD i 0) =
          q.base.wl.getD i 0 - lmax := by omega
      rw [hm] at hgap'
      exact hgap'
    have ht := target_in_corridor_left c (q.base.wl.getD i 0) lmax amp hg
      hleftsign heven hL.2.2
    have htarget : Steer.lo c q i ≤ side * amp ∧
        side * amp ≤ Steer.hi c q i := by
      constructor
      · rw [hL.2.1]
        exact Int.le_trans hlo0 (by omega)
      · rw [hL.2.1]
        simpa [Steer.hi] using ht.2.1
    have hoff := Steer.off_eq_amp_in_hold c d side amp si hholdIn hhold0 hhold1
    have hplan' : Steer.offAt c q si i =
        Steer.clampAt c q i (Steer.offOb c d side amp si) := by
      rw [hoff]
      exact hplan
    have hn : Steer.clampAt c q i (side * amp) = side * amp := by
      rw [Steer.clampAt]
      simp [show Steer.lo c q i ≤ Steer.hi c q i by omega]
      exact Steer.clamp_noop_in_hold _ _ _ htarget.1 htarget.2
    rw [hplan', hoff, hn]
    exact Or.inl ⟨hL.2.1, by simpa [hL.2.1] using ht.1⟩
  · have hg : c.lon.veh_width + 2 * c.margin ≤
        lmin + q.base.wr.getD i 0 := by
      have hm : max (q.base.wl.getD i 0 - lmax) (lmin + q.base.wr.getD i 0) =
          lmin + q.base.wr.getD i 0 := by omega
      rw [hm] at hgap'
      exact hgap'
    have ht := target_in_corridor_right c (q.base.wr.getD i 0) lmin amp hg
      hrightsign heven hR.2.2
    have htarget : Steer.lo c q i ≤ side * amp ∧
        side * amp ≤ Steer.hi c q i := by
      constructor
      · rw [hR.2.1]
        unfold Steer.lo
        have hmul : (-1 : Int) * amp = -amp := by omega
        rw [hmul]
        exact ht.2.1
      · rw [hR.2.1]
        simp
        exact Int.le_trans (by omega) hhi0
    have hn : Steer.clampAt c q i (side * amp) = side * amp := by
      rw [Steer.clampAt]
      simp [show Steer.lo c q i ≤ Steer.hi c q i by omega]
      exact Steer.clamp_noop_in_hold _ _ _ htarget.1 htarget.2
    rw [hplan, hn]
    exact Or.inr ⟨hR.2.1, by simpa [hR.2.1] using ht.2.2⟩
  · have hg : c.lon.veh_width + 2 * c.margin ≤
        q.base.wl.getD i 0 - lmax := by
      have hm : max (q.base.wl.getD i 0 - lmax) (lmin + q.base.wr.getD i 0) =
          q.base.wl.getD i 0 - lmax := by omega
      rw [hm] at hgap'
      exact hgap'
    have ht := target_in_corridor_left c (q.base.wl.getD i 0) lmax amp hg
      hleftsign heven hTL.2.2.2
    have htarget : Steer.lo c q i ≤ side * amp ∧ side * amp ≤ Steer.hi c q i := by
      constructor
      · rw [hTL.2.2.1]
        exact Int.le_trans hlo0 (by omega)
      · rw [hTL.2.2.1]
        simpa [Steer.hi] using ht.2.1
    have hn : Steer.clampAt c q i (side * amp) = side * amp := by
      rw [Steer.clampAt]
      simp [show Steer.lo c q i ≤ Steer.hi c q i by omega]
      exact Steer.clamp_noop_in_hold _ _ _ htarget.1 htarget.2
    rw [hplan, hn]
    exact Or.inl ⟨hTL.2.2.1, by simpa [hTL.2.2.1] using ht.1⟩
  · have hg : c.lon.veh_width + 2 * c.margin ≤
        lmin + q.base.wr.getD i 0 := by
      have hm : max (q.base.wl.getD i 0 - lmax) (lmin + q.base.wr.getD i 0) =
          lmin + q.base.wr.getD i 0 := by omega
      rw [hm] at hgap'
      exact hgap'
    have ht := target_in_corridor_right c (q.base.wr.getD i 0) lmin amp hg
      hrightsign heven hTR.2.2.2
    have htarget : Steer.lo c q i ≤ side * amp ∧ side * amp ≤ Steer.hi c q i := by
      constructor
      · rw [hTR.2.2.1]
        unfold Steer.lo
        have hmul : (-1 : Int) * amp = -amp := by omega
        rw [hmul]
        exact ht.2.1
      · rw [hTR.2.2.1]
        simp
        exact Int.le_trans (by omega) hhi0
    have hn : Steer.clampAt c q i (side * amp) = side * amp := by
      rw [Steer.clampAt]
      simp [show Steer.lo c q i ≤ Steer.hi c q i by omega]
      exact Steer.clamp_noop_in_hold _ _ _ htarget.1 htarget.2
    rw [hplan, hn]
    exact Or.inr ⟨hTR.2.2.1, by simpa [hTR.2.2.1] using ht.2.2⟩

/-- Outside an obstacle hold window we make no obstacle-clearance claim:
clamping still keeps the offset in the available lateral corridor. -/
theorem off_clearance_ramps_outside
    (c : Steer.Consts) (q : Steer.Q) (i : Nat) (si : Int)
    (hlo : Steer.lo c q i ≤ Steer.hi c q i) :
    Steer.lo c q i ≤ Steer.offAt c q si i ∧
      Steer.offAt c q si i ≤ Steer.hi c q i := by
  exact clamp_in_corridor c q i (Steer.rawOffAt c q si) hlo

end Corpus.SteerFeas
