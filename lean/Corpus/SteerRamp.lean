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

end Corpus.SteerRamp
