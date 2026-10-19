import Lean.Data.Json

/-!
Exact decimal quantisation used by the compiled corpus checker.  For a JSON number
`m / 10^e`, `qfix` computes floor(m*scale/10^e + 1/2) using only integer
arithmetic; the theorems below record its floor, monotonicity, integer exactness,
and half-unit error bounds.
-/

namespace Corpus.Quant

open Lean

def qfix (n : JsonNumber) (scale : Nat) : Int :=
  let den : Int := ((10 : Nat) ^ n.exponent : Nat)
  (2 * n.mantissa * scale + den) / (2 * den)

def den (n : JsonNumber) : Int := ((10 : Nat) ^ n.exponent : Nat)

theorem den_pos (n : JsonNumber) : 0 < den n := by
  unfold den
  have hp : ∀ e : Nat, 0 < ((10 : Nat) ^ e : Int) := by
    intro e
    induction e with
    | zero => decide
    | succ e ih =>
      rw [show e + 1 = Nat.succ e by omega, Int.pow_succ]
      exact Int.mul_pos ih (by decide)
  exact hp n.exponent


theorem qfix_floor_bounds (n : JsonNumber) (scale : Nat) :
    2 * den n * qfix n scale ≤ 2 * n.mantissa * scale + den n ∧
      2 * n.mantissa * scale + den n < 2 * den n * (qfix n scale + 1) := by
  change 2 * den n * ((2 * n.mantissa * scale + den n) / (2 * den n)) ≤
      2 * n.mantissa * scale + den n ∧
    2 * n.mantissa * scale + den n <
      2 * den n * ((2 * n.mantissa * scale + den n) / (2 * den n) + 1)
  have hd : 0 < 2 * den n := by
    have := den_pos n
    omega
  have hlo := Int.ediv_mul_le (2 * n.mantissa * scale + den n)
    (by omega : 2 * den n ≠ 0)
  have hup := Int.lt_ediv_add_one_mul_self (2 * n.mantissa * scale + den n) hd
  simpa [Int.mul_comm] using And.intro hlo hup

theorem qfix_eq_floor (n : JsonNumber) (scale : Nat) :
    qfix n scale = (2 * n.mantissa * scale + den n) / (2 * den n) := by rfl

theorem qfix_monotone_mantissa {n₁ n₂ : JsonNumber} (h : n₁.exponent = n₂.exponent)
    (hm : n₁.mantissa ≤ n₂.mantissa) (scale : Nat) :
    qfix n₁ scale ≤ qfix n₂ scale := by
  rw [qfix_eq_floor, qfix_eq_floor, den, den, h]
  have hd : 0 < 2 * ((10 : Nat) ^ n₂.exponent : Int) := by
    have hp := den_pos n₂
    simpa [den] using (show 0 < 2 * den n₂ by omega)
  have hmul : n₁.mantissa * (scale : Int) ≤ n₂.mantissa * scale :=
    Int.mul_le_mul_of_nonneg_right hm (by omega)
  have hnum : 2 * n₁.mantissa * (scale : Int) ≤ 2 * n₂.mantissa * scale := by
    have h := Int.mul_le_mul_of_nonneg_left hmul (by decide : (0 : Int) ≤ 2)
    simpa [Int.mul_assoc] using h
  have hnum' : 2 * n₁.mantissa * (scale : Int) + ((10 : Nat) ^ n₂.exponent : Int) ≤
      2 * n₂.mantissa * scale + ((10 : Nat) ^ n₂.exponent : Int) := by omega
  apply Int.ediv_le_ediv hd
  exact hnum'
 
theorem qfix_int_exact {n : JsonNumber} (he : n.exponent = 0) (scale : Nat) :
    qfix n scale = n.mantissa * scale := by
  rw [qfix_eq_floor, den]
  simp only [he, Nat.pow_zero]
  have hmul : 2 * n.mantissa * (scale : Int) = 2 * (n.mantissa * scale) := by grind
  rw [hmul]
  omega


theorem qfix_error_bound (n : JsonNumber) (scale : Nat) :
    -(den n) ≤ 2 * den n * qfix n scale - 2 * n.mantissa * scale ∧
      2 * den n * qfix n scale - 2 * n.mantissa * scale ≤ den n := by
  obtain ⟨hlo, hup⟩ := qfix_floor_bounds n scale
  generalize den n = D at hlo hup ⊢
  generalize qfix n scale = Q at hlo hup ⊢
  generalize 2 * n.mantissa * (scale : Int) = M at hlo hup ⊢
  have hexpand : 2 * D * (Q + 1) = 2 * D * Q + 2 * D := by grind
  rw [hexpand] at hup
  omega

end Corpus.Quant
