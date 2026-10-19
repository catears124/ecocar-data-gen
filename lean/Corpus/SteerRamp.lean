import Corpus.Steer

namespace Corpus.SteerRamp
open Corpus Steer

def n (x L : Int) : Int := 3*x*x*L - 2*x*x*x

theorem n_full (L : Int) : n L L = L * L * L := by
  unfold n
  simp only [Int.mul_assoc]
  omega

theorem ramp_full (A L : Int) (hL : L ≠ 0) : ramp A L L = A := by
  unfold ramp
  have he : 3 * L * L * L - 2 * L * L * L = L * L * L := by
    simp only [Int.mul_assoc]
    omega
  rw [he]
  have hd : L * L * L ≠ 0 := by
    exact Int.mul_ne_zero (Int.mul_ne_zero hL hL) hL
  rw [show A * (L * L * L) = (L * L * L) * A by ac_rfl]
  exact Int.mul_ediv_cancel_left A hd


theorem n_shift (x L : Int) : n x L = x*x*(3*L - 2*x) := by
  unfold n
  have h2 : (2:Int) = 1+1 := by omega
  have h3 : (3:Int) = 1+1+1 := by omega
  simp only [h2, h3, Int.sub_eq_add_neg, Int.mul_add, Int.add_mul,
    Int.neg_mul, Int.mul_neg, Int.neg_add, Int.neg_neg]
  simp only [Int.mul_assoc, Int.mul_comm, Int.mul_left_comm,
    Int.add_assoc, Int.add_comm, Int.add_left_comm, Int.mul_one, Int.one_mul]


/-- The cubic numerator is non-negative on the interpolation interval. -/
theorem n_nonneg (x L : Int) (hx : 0 ≤ x) (hxl : x ≤ L) :
    0 ≤ n x L := by
  rw [n_shift]
  have hfac : 0 ≤ 3 * L - 2 * x := by omega
  exact Int.mul_nonneg (Int.mul_nonneg hx hx) hfac

/-- The cubic numerator is bounded by its endpoint value. -/
theorem n_le_full (x L : Int) (hx : 0 ≤ x) (hxl : x ≤ L) :
    n x L ≤ L * L * L := by
  have hid : L * L * L - n x L = (L-x)*(L-x)*(L+2*x) := by
    unfold n
    grind
  have hp : 0 ≤ (L-x)*(L-x)*(L+2*x) := by
    exact Int.mul_nonneg (Int.mul_nonneg (by omega) (by omega)) (by omega)
  omega

/-- Smoothstep has its exact endpoint and amplitude bounds under positive length. -/
theorem ramp_bounds (A x L : Int) (hA : 0 ≤ A) (hx : 0 ≤ x) (hxl : x ≤ L)
    (hL : 0 < L) :
    0 ≤ ramp A x L ∧ ramp A x L ≤ A := by
  have hD : 0 < L * L * L := Int.mul_pos (Int.mul_pos hL hL) hL
  have hn0 : 0 ≤ n x L := n_nonneg x L hx hxl
  have hn1 : n x L ≤ L * L * L := n_le_full x L hx hxl
  have hp0 : 0 ≤ A * n x L := Int.mul_nonneg hA hn0
  have hp1 : A * n x L ≤ A * (L * L * L) :=
    Int.mul_le_mul_of_nonneg_left hn1 hA
  have hlo : 0 ≤ A * n x L / (L * L * L) := Int.ediv_nonneg hp0 (by omega)
  have hhi : A * n x L / (L * L * L) ≤ A :=
    Int.ediv_le_of_le_mul hD (by simpa [Int.mul_assoc, Int.mul_left_comm, Int.mul_comm] using hp1)
  exact ⟨by simpa [ramp, n], by simpa [ramp, n] using hhi⟩

/-- Strictly before the endpoint, the unit smoothstep is below one thousand. -/
theorem ramp_1000_lt (x L : Int) (hx : 0 ≤ x) (hxl : x < L) (hL : 0 < L) :
    ramp 1000 x L ≤ 999 := by
  have hn : n x L < L * L * L := by
    have hid : L * L * L - n x L = (L-x)*(L-x)*(L+2*x) := by
      unfold n
      grind
    have hp : 0 < (L-x)*(L-x)*(L+2*x) := by
      have h1 : 0 < L - x := by omega
      have h2 : 0 < L + 2*x := by omega
      exact Int.mul_pos (Int.mul_pos h1 h1) h2
    omega
  have hh : 1000 * n x L < 1000 * (L * L * L) :=
    Int.mul_lt_mul_of_pos_left hn (by omega)
  have hD : 0 < L * L * L := Int.mul_pos (Int.mul_pos hL hL) hL
  have hq : 1000 * n x L / (L * L * L) < 1000 :=
    Int.ediv_lt_of_lt_mul hD (by simpa [Int.mul_assoc] using hh)
  have hr : ramp 1000 x L < 1000 := by
    unfold ramp
    exact hq
  omega

/-- Monotonicity of the cubic smoothstep on its interpolation interval. -/
theorem n_mono (x y L : Int) (hx : 0 ≤ x) (hxy : x ≤ y) (hyL : y ≤ L) :
    n x L ≤ n y L := by
  have hid : n y L - n x L =
      (y-x) * (3*L*(x+y) - 2*(x*x + x*y + y*y)) := by
    unfold n
    grind
  have hfac : 0 ≤ 3*L*(x+y) - 2*(x*x + x*y + y*y) := by
    have hu : 0 ≤ L-y := by omega
    have ht : 0 ≤ y-x := by omega
    have hsq : 0 ≤ (y-x)*(y-x) := Int.mul_nonneg ht ht
    have h1 : 0 ≤ 3*(y-x)*(y-x) :=
      Int.mul_nonneg (Int.mul_nonneg (by omega) ht) ht
    have h2 : 0 ≤ (y-x)*(y-x) := hsq
    have h3 : 0 ≤ 3*(L-y)*(2*x + (y-x)) := by
      exact Int.mul_nonneg (Int.mul_nonneg (by omega) hu) (by omega)
    have he : 3*L*(x+y) - 2*(x*x + x*y + y*y) =
        3*x*(y-x) + (y-x)*(y-x) + 3*(L-y)*(2*x + (y-x)) := by
      grind
    rw [he]
    have hxterm : 0 ≤ 3*x*(y-x) :=
      Int.mul_nonneg (Int.mul_nonneg (by omega) hx) ht
    omega
  have hp : 0 ≤ (y-x) * (3*L*(x+y) - 2*(x*x + x*y + y*y)) :=
    Int.mul_nonneg (by omega) hfac
  have hd : 0 ≤ n y L - n x L := by
    rw [hid]
    exact hp
  omega

theorem ramp_mono (A x y L : Int) (hA : 0 ≤ A) (hx : 0 ≤ x) (hxy : x ≤ y)
    (hyL : y ≤ L) (hL : 0 < L) :
    ramp A x L ≤ ramp A y L := by
  have hn := n_mono x y L hx hxy hyL
  have hp := Int.mul_le_mul_of_nonneg_left hn hA
  have hD : 0 < L * L * L := Int.mul_pos (Int.mul_pos hL hL) hL
  unfold ramp
  exact Int.ediv_le_ediv hD (by simpa [n] using hp)
end Corpus.SteerRamp
