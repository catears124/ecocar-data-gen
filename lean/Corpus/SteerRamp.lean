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

end Corpus.SteerRamp
