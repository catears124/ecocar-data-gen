/-
  Option-valued finite-horizon Bellman optimization.  Infeasible terminals and
  empty action lists remain `none`; actions carry their successor and cost.
-/
import Corpus.Bellman
namespace Corpus.BellmanOpt
open Corpus.Bellman
abbrev Stage (σ : Type) := σ → List (Action σ)

def bestO {σ : Type} (f : Action σ → Option Int) : List (Action σ) → Option Int
  | [] => none
  | a :: as => match f a, bestO f as with
    | none, r => r
    | r, none => r
    | some x, some y => some (min x y)

def valueO {σ : Type} (terminal : σ → Option Int) : List (Stage σ) → σ → Option Int
  | [], s => terminal s
  | st :: rs, s => bestO (fun a => (valueO terminal rs a.1).map (fun v => a.2 + v)) (st s)

def TrajOK {σ : Type} (terminal : σ → Option Int) : List (Stage σ) → σ → List (Action σ) → Prop
  | [], s, [] => ∃ v, terminal s = some v
  | st :: rs, s, a :: xs => a ∈ st s ∧ TrajOK terminal rs a.1 xs
  | _, _, _ => False

def trajCost {σ : Type} (terminal : σ → Option Int) : List (Stage σ) → σ → List (Action σ) → Option Int
  | [], s, [] => terminal s
  | _ :: rs, s, a :: xs => (trajCost terminal rs a.1 xs).map (fun v => a.2 + v)
  | _, _, _ => none

def HasTraj {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ)) (s : σ) : Prop :=
  ∃ xs, TrajOK terminal ss s xs
def pick {σ : Type} (terminal : σ → Option Int) (rs : List (Stage σ)) (target : Int) :
    List (Action σ) → Option (Action σ)
  | [] => none
  | a :: as => match valueO terminal rs a.1 with
    | some z => if a.2 + z = target then some a else pick terminal rs target as
    | none => pick terminal rs target as

def extract {σ : Type} (terminal : σ → Option Int) : List (Stage σ) → σ → Option (List (Action σ))
  | [], s => (terminal s).map (fun _ => [])
  | st :: rs, s => match valueO terminal (st :: rs) s with
    | none => none
    | some v => match pick terminal rs v (st s) with
      | none => none
      | some a => (extract terminal rs a.1).map (fun ys => a :: ys)


 theorem bestO_none_iff {σ : Type} (f : Action σ → Option Int) (xs : List (Action σ))
    (h : bestO f xs = none) : ∀ a ∈ xs, f a = none := by
  induction xs with
  | nil => intro a ha; cases ha
  | cons a xs ih =>
      intro b hb
      rcases List.mem_cons.mp hb with hba | hb
      · subst b
        cases hf : f a <;> cases ht : bestO f xs <;> simp [bestO, hf, ht] at h ⊢
      · have htail : bestO f xs = none := by
          cases hf : f a <;> cases ht : bestO f xs <;> simp [bestO, hf, ht] at h ⊢
        exact ih htail b hb


theorem bestO_attains {σ : Type} (f : Action σ → Option Int) :
    ∀ xs v, bestO f xs = some v → ∃ a ∈ xs, f a = some v := by
  intro xs; induction xs with
  | nil => intro v h; simp [bestO] at h
  | cons a xs ih =>
      intro v h
      cases ha : f a with
      | none =>
          cases hb : bestO f xs with
          | none => simp [bestO, ha, hb] at h
          | some w =>
              have hw : w = v := by simpa [bestO, ha, hb] using h
              obtain ⟨b, hbmem, hbval⟩ := ih w hb
              exact ⟨b, List.mem_cons_of_mem a hbmem, hw ▸ hbval⟩
      | some z =>
          cases hb : bestO f xs with
          | none =>
              have hz : z = v := by simpa [bestO, ha, hb] using h
              exact ⟨a, by simp, hz ▸ ha⟩
          | some w =>
              by_cases hzw : z ≤ w
              · have hz : min z w = z := by omega
                have hv : z = v := by simpa [bestO, ha, hb, hz] using h
                exact ⟨a, by simp, hv ▸ ha⟩
              · have hwz : w ≤ z := by omega
                have hz : min z w = w := by omega
                have hv : w = v := by simpa [bestO, ha, hb, hz] using h
                obtain ⟨b, hbmem, hbval⟩ := ih w hb
                exact ⟨b, List.mem_cons_of_mem a hbmem, hv ▸ hbval⟩

/-- **Dead ends cannot masquerade as feasible**: the value is `none` exactly
when no admissible trajectory (one legal action per stage, feasible terminal)
exists from `s`. -/
theorem valueO_none_iff {σ : Type} (terminal : σ → Option Int) :
    ∀ ss s, valueO terminal ss s = none ↔ ¬ HasTraj terminal ss s := by
  intro ss; induction ss with
  | nil =>
      intro s
      cases h : terminal s with
      | none =>
          constructor
          · intro _ h'
            rcases h' with ⟨xs, hx⟩
            cases xs with
            | nil => obtain ⟨v, hv⟩ := hx; rw [h] at hv; cases hv
            | cons _ _ => exact hx
          · intro _; simp [valueO, h]
      | some v =>
          constructor
          · intro hv; simp [valueO, h] at hv
          · intro hn; exfalso; exact hn ⟨[], ⟨v, h⟩⟩
  | cons st rs ih =>
      intro s
      constructor
      · intro hv htr
        rcases htr with ⟨xs, hx⟩
        cases xs with
        | nil => exact hx
        | cons a ys =>
            have ha : a ∈ st s := hx.1
            have ht : TrajOK terminal rs a.1 ys := hx.2
            have hall := bestO_none_iff
              (fun b => (valueO terminal rs b.1).map (fun z => b.2 + z)) (st s)
              (by simpa [valueO] using hv) a ha
            cases hz : valueO terminal rs a.1 with
            | none => exact ((ih a.1).mp hz) ⟨ys, ht⟩
            | some z => simp [hz] at hall
      · intro hn
        cases hval : valueO terminal (st :: rs) s with
        | none => rfl
        | some v =>
            exfalso
            obtain ⟨a, ha, hfa⟩ := bestO_attains
              (fun b => (valueO terminal rs b.1).map (fun z => b.2 + z)) (st s) v
              (by simpa [valueO] using hval)
            cases hc : valueO terminal rs a.1 with
            | none => simp [hc] at hfa
            | some z =>
                have hfeas : HasTraj terminal rs a.1 := by
                  rcases Classical.em (HasTraj terminal rs a.1) with hh | hn'
                  · exact hh
                  · rw [(ih a.1).mpr hn'] at hc; cases hc
                obtain ⟨ys, hys⟩ := hfeas
                exact hn ⟨a :: ys, ha, hys⟩
 theorem trajCost_some {σ : Type} (terminal : σ → Option Int)
    (ss : List (Stage σ)) (s : σ) (xs : List (Action σ)) (h : TrajOK terminal ss s xs) :
    ∃ v, trajCost terminal ss s xs = some v := by
  induction ss generalizing s xs with
  | nil =>
      cases xs with
      | nil => exact h
      | cons a ys => cases h
  | cons st rs ih =>
      cases xs with
      | nil => simp [TrajOK] at h
      | cons a ys =>
          simp only [TrajOK] at h
          obtain ⟨ha, ht⟩ := h
          obtain ⟨v, hv⟩ := ih a.1 ys ht
          exact ⟨a.2 + v, by simp [trajCost, hv]⟩

/-- `bestO` is a lower bound for every feasible offered action. -/
theorem bestO_le {σ : Type} (f : Action σ → Option Int) :
    ∀ (xs : List (Action σ)) (m : Int), bestO f xs = some m →
      ∀ a ∈ xs, ∀ v, f a = some v → m ≤ v := by
  intro xs
  induction xs with
  | nil => intro m h; simp [bestO] at h
  | cons b bs ih =>
      intro m h a ha v hv
      cases hb : f b with
      | none =>
          have ht : bestO f bs = some m := by
            cases hr : bestO f bs <;> simp [bestO, hb, hr] at h ⊢ <;> exact h
          rcases List.mem_cons.mp ha with rfl | ha'
          · rw [hb] at hv; cases hv
          · exact ih m ht a ha' v hv
      | some z =>
          cases hr : bestO f bs with
          | none =>
              have hz : z = m := by simpa [bestO, hb, hr] using h
              rcases List.mem_cons.mp ha with rfl | ha'
              · rw [hb] at hv; cases hv; omega
              · have := bestO_none_iff f bs hr a ha'
                rw [this] at hv; cases hv
          | some y =>
              have hm : min z y = m := by simpa [bestO, hb, hr] using h
              rcases List.mem_cons.mp ha with rfl | ha'
              · rw [hb] at hv; cases hv; omega
              · have := ih y hr a ha' v hv; omega

/-- A feasible offered action makes `bestO` feasible. -/
theorem bestO_some_of_mem {σ : Type} (f : Action σ → Option Int) :
    ∀ (xs : List (Action σ)) (a : Action σ) (v : Int), a ∈ xs → f a = some v →
      ∃ m, bestO f xs = some m := by
  intro xs a v ha hv
  cases h : bestO f xs with
  | some m => exact ⟨m, rfl⟩
  | none =>
      have := bestO_none_iff f xs h a ha
      rw [this] at hv; cases hv

/-- **Lower bound.**  No admissible trajectory is cheaper than the value. -/
theorem valueO_le_cost {σ : Type} (terminal : σ → Option Int) :
    ∀ (ss : List (Stage σ)) (s : σ) (xs : List (Action σ)) (v w : Int),
      TrajOK terminal ss s xs → trajCost terminal ss s xs = some w →
      valueO terminal ss s = some v → v ≤ w := by
  intro ss
  induction ss with
  | nil =>
      intro s xs v w hok hc hv
      cases xs with
      | nil =>
          simp only [trajCost] at hc
          simp only [valueO] at hv
          rw [hc] at hv; cases hv; omega
      | cons a ys => cases hok
  | cons st rs ih =>
      intro s xs v w hok hc hv
      cases xs with
      | nil => cases hok
      | cons a ys =>
          obtain ⟨ha, ht⟩ := hok
          simp only [trajCost] at hc
          cases hw' : trajCost terminal rs a.1 ys with
          | none => rw [hw'] at hc; cases hc
          | some w' =>
              rw [hw'] at hc
              simp only [Option.map_some, Option.some.injEq] at hc
              cases hz : valueO terminal rs a.1 with
              | none =>
                  exact absurd ⟨ys, ht⟩ ((valueO_none_iff terminal rs a.1).mp hz)
              | some z =>
                  have hzw := ih a.1 ys z w' ht hw' hz
                  have hle := bestO_le (fun b => (valueO terminal rs b.1).map (fun u => b.2 + u))
                    (st s) v (by simpa [valueO] using hv) a ha (a.2 + z) (by simp [hz])
                  omega

/-- **Attainment.**  A finite value is the cost of an admissible trajectory. -/
theorem valueO_attained {σ : Type} (terminal : σ → Option Int) :
    ∀ (ss : List (Stage σ)) (s : σ) (v : Int), valueO terminal ss s = some v →
      ∃ xs, TrajOK terminal ss s xs ∧ trajCost terminal ss s xs = some v := by
  intro ss
  induction ss with
  | nil =>
      intro s v hv
      exact ⟨[], ⟨v, by simpa [valueO] using hv⟩, by simpa [trajCost, valueO] using hv⟩
  | cons st rs ih =>
      intro s v hv
      obtain ⟨a, ha, hfa⟩ := bestO_attains
        (fun b => (valueO terminal rs b.1).map (fun u => b.2 + u)) (st s) v
        (by simpa [valueO] using hv)
      cases hz : valueO terminal rs a.1 with
      | none => simp [hz] at hfa
      | some z =>
          have hv' : a.2 + z = v := by simpa [hz] using hfa
          obtain ⟨ys, hok, hc⟩ := ih a.1 z hz
          exact ⟨a :: ys, ⟨ha, hok⟩, by simp [trajCost, hc, hv']⟩

/-- `pick` returns an offered action whose continuation value meets the target. -/
theorem pick_spec {σ : Type} (terminal : σ → Option Int) (rs : List (Stage σ)) (t : Int) :
    ∀ (l : List (Action σ)) (a : Action σ), pick terminal rs t l = some a →
      a ∈ l ∧ ∃ z, valueO terminal rs a.1 = some z ∧ a.2 + z = t := by
  intro l
  induction l with
  | nil => intro a h; simp [pick] at h
  | cons b bs ih =>
      intro a h
      simp only [pick] at h
      cases hz : valueO terminal rs b.1 with
      | none =>
          rw [hz] at h
          obtain ⟨hm, hr⟩ := ih a h
          exact ⟨List.mem_cons_of_mem b hm, hr⟩
      | some z =>
          rw [hz] at h
          by_cases hbt : b.2 + z = t
          · simp only [hbt, if_true, Option.some.injEq] at h
            subst h
            exact ⟨by simp, z, hz, hbt⟩
          · simp only [hbt, if_false] at h
            obtain ⟨hm, hr⟩ := ih a h
            exact ⟨List.mem_cons_of_mem b hm, hr⟩

/-- `pick` succeeds whenever some offered action meets the target. -/
theorem pick_some {σ : Type} (terminal : σ → Option Int) (rs : List (Stage σ)) (t : Int) :
    ∀ (l : List (Action σ)) (a : Action σ) (z : Int), a ∈ l →
      valueO terminal rs a.1 = some z → a.2 + z = t →
      ∃ b, pick terminal rs t l = some b := by
  intro l
  induction l with
  | nil => intro a z h; cases h
  | cons b bs ih =>
      intro a z ha hz hat
      simp only [pick]
      cases hb : valueO terminal rs b.1 with
      | none =>
          rcases List.mem_cons.mp ha with rfl | ha'
          · rw [hb] at hz; cases hz
          · exact ih a z ha' hz hat
      | some y =>
          by_cases hbt : b.2 + y = t
          · exact ⟨b, by simp [hbt]⟩
          · simp only [hbt, if_false]
            rcases List.mem_cons.mp ha with rfl | ha'
            · rw [hb] at hz; cases hz; exact absurd hat hbt
            · exact ih a z ha' hz hat

/-- **Extraction.**  When the value is finite, the executable greedy extraction
returns a trajectory that is admissible at every transition, has exactly one
action per stage, and attains the value. -/
theorem extract_spec {σ : Type} (terminal : σ → Option Int) :
    ∀ (ss : List (Stage σ)) (s : σ) (v : Int), valueO terminal ss s = some v →
      ∃ xs, extract terminal ss s = some xs ∧ TrajOK terminal ss s xs ∧
        trajCost terminal ss s xs = some v ∧ xs.length = ss.length := by
  intro ss
  induction ss with
  | nil =>
      intro s v hv
      have ht : terminal s = some v := by simpa [valueO] using hv
      exact ⟨[], by simp [extract, ht], ⟨v, ht⟩, by simp [trajCost, ht], rfl⟩
  | cons st rs ih =>
      intro s v hv
      obtain ⟨a0, ha0, hfa0⟩ := bestO_attains
        (fun b => (valueO terminal rs b.1).map (fun u => b.2 + u)) (st s) v
        (by simpa [valueO] using hv)
      have hpick : ∃ b, pick terminal rs v (st s) = some b := by
        cases hz : valueO terminal rs a0.1 with
        | none => simp [hz] at hfa0
        | some z => exact pick_some terminal rs v (st s) a0 z ha0 hz (by simpa [hz] using hfa0)
      obtain ⟨a, hpa⟩ := hpick
      obtain ⟨ham, z, hz, haz⟩ := pick_spec terminal rs v (st s) a hpa
      obtain ⟨ys, hex, hok, hc, hlen⟩ := ih a.1 z hz
      refine ⟨a :: ys, ?_, ⟨ham, hok⟩, by simp [trajCost, hc, haz], by simp [hlen]⟩
      simp only [extract, hv, hpa, hex, Option.map_some]

/-- Extraction fails exactly on infeasible states. -/
theorem extract_none_iff {σ : Type} (terminal : σ → Option Int) :
    ∀ (ss : List (Stage σ)) (s : σ), extract terminal ss s = none ↔ valueO terminal ss s = none := by
  intro ss s
  constructor
  · intro he
    cases hv : valueO terminal ss s with
    | none => rfl
    | some v =>
        obtain ⟨xs, hx, _⟩ := extract_spec terminal ss s v hv
        rw [hx] at he; cases he
  · intro hv
    cases ss with
    | nil => simp [extract]; simpa [valueO] using hv
    | cons st rs => simp [extract, hv]

/-- Every admissible trajectory has a defined (finite) cost. -/
theorem trajCost_of_ok {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ)) (s : σ)
    (xs : List (Action σ)) (h : TrajOK terminal ss s xs) :
    ∃ w, trajCost terminal ss s xs = some w := trajCost_some terminal ss s xs h

/-- Admissible trajectories have exactly one action per stage (termination and
station progression at the abstract level). -/
theorem trajOK_length {σ : Type} (terminal : σ → Option Int) :
    ∀ (ss : List (Stage σ)) (s : σ) (xs : List (Action σ)),
      TrajOK terminal ss s xs → xs.length = ss.length := by
  intro ss
  induction ss with
  | nil => intro s xs h; cases xs with
    | nil => rfl
    | cons _ _ => cases h
  | cons st rs ih => intro s xs h; cases xs with
    | nil => cases h
    | cons a ys => simp [ih a.1 ys h.2]

/-- **Optimal value is uniquely determined** by the trajectory set: it is the
least admissible cost, and two values that are both lower bounds attained by
admissible trajectories coincide. -/
theorem optimal_value_unique {σ : Type} (terminal : σ → Option Int)
    (ss : List (Stage σ)) (s : σ) (u : Int)
    (hlow : ∀ xs w, TrajOK terminal ss s xs → trajCost terminal ss s xs = some w → u ≤ w)
    (hatt : ∃ xs, TrajOK terminal ss s xs ∧ trajCost terminal ss s xs = some u) :
    valueO terminal ss s = some u := by
  obtain ⟨xs, hok, hc⟩ := hatt
  cases hv : valueO terminal ss s with
  | none => exact absurd ⟨xs, hok⟩ ((valueO_none_iff terminal ss s).mp hv)
  | some v =>
      have h1 := valueO_le_cost terminal ss s xs v u hok hc hv
      obtain ⟨ys, hok', hc'⟩ := valueO_attained terminal ss s v hv
      have h2 := hlow ys v hok' hc'
      have : v = u := by omega
      rw [this]

/-- Pointwise congruence of `bestO`. -/
theorem bestO_congr {σ : Type} {f g : Action σ → Option Int} (h : ∀ a, f a = g a) :
    ∀ xs, bestO f xs = bestO g xs := by
  intro xs
  induction xs with
  | nil => rfl
  | cons a as ih => simp [bestO, h a, ih]

/-- A list of value functions solving the option-valued Bellman equations. -/
def IsBellmanO {σ : Type} (terminal : σ → Option Int) : List (Stage σ) → List (σ → Option Int) → Prop
  | [], [f] => ∀ s, f s = terminal s
  | st :: rs, f :: g :: gs =>
      (∀ s, f s = bestO (fun a => (g a.1).map (fun v => a.2 + v)) (st s)) ∧ IsBellmanO terminal rs (g :: gs)
  | _, _ => False

/-- **Uniqueness of the Bellman solution**: any solution of the option-valued
Bellman equations is the backward-induction value. -/
theorem bellmanO_unique {σ : Type} (terminal : σ → Option Int) :
    ∀ (ss : List (Stage σ)) (fs : List (σ → Option Int)), IsBellmanO terminal ss fs →
      ∀ s, fs.headD terminal s = valueO terminal ss s := by
  intro ss
  induction ss with
  | nil =>
      intro fs h s
      match fs, h with
      | [f], h => simpa [valueO] using h s
  | cons st rs ih =>
      intro fs h s
      match fs, h with
      | f :: g :: gs, ⟨hf, ht⟩ =>
          have hg : ∀ x, g x = valueO terminal rs x := fun x => by
            simpa using ih (g :: gs) ht x
          show f s = valueO terminal (st :: rs) s
          rw [hf s]
          simp only [valueO]
          exact bestO_congr (fun a => by rw [hg a.1]) (st s)

/-- The backward-induction value functions themselves solve the equations
(existence half of uniqueness). -/
def valueFns {σ : Type} (terminal : σ → Option Int) : List (Stage σ) → List (σ → Option Int)
  | [] => [terminal]
  | st :: rs => valueO terminal (st :: rs) :: valueFns terminal rs

theorem valueFns_head {σ : Type} (terminal : σ → Option Int) :
    ∀ ss : List (Stage σ), (valueFns terminal ss).headD terminal = valueO terminal ss := by
  intro ss
  cases ss with
  | nil => funext s; rfl
  | cons st rs => rfl

theorem valueFns_bellman {σ : Type} (terminal : σ → Option Int) :
    ∀ ss : List (Stage σ), IsBellmanO terminal ss (valueFns terminal ss) := by
  intro ss
  induction ss with
  | nil => intro s; rfl
  | cons st rs ih =>
      cases rs with
      | nil => exact ⟨fun s => rfl, ih⟩
      | cons st' rs' => exact ⟨fun s => rfl, ih⟩


abbrev Policy (σ : Type) := List (σ → Action σ)

/-- The trajectory a policy generates from `s` (one action per stage). -/
def policyActs {σ : Type} : List (Stage σ) → Policy σ → σ → List (Action σ)
  | [], _, _ => []
  | _ :: _, [], _ => []
  | _ :: rs, p :: ps, s => p s :: policyActs rs ps (p s).1

/-- A policy is admissible from `s` when the trajectory it generates is. -/
def PolicyOK {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ))
    (ps : Policy σ) (s : σ) : Prop := TrajOK terminal ss s (policyActs ss ps s)

/-- The open-loop policy that replays an action list. -/
def replay {σ : Type} (xs : List (Action σ)) : Policy σ := xs.map (fun a _ => a)

theorem replay_acts {σ : Type} :
    ∀ (ss : List (Stage σ)) (xs : List (Action σ)) (s : σ), xs.length = ss.length →
      policyActs ss (replay xs) s = xs := by
  intro ss
  induction ss with
  | nil => intro xs s h; cases xs with
    | nil => rfl
    | cons _ _ => simp at h
  | cons st rs ih => intro xs s h; cases xs with
    | nil => simp at h
    | cons a ys =>
        simp only [replay, List.map_cons, policyActs]
        exact congrArg (a :: ·) (ih ys a.1 (by simpa using h))

/-- **Policy optimality**: no admissible policy beats the value. -/
theorem policy_lower_bound {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ))
    (ps : Policy σ) (s : σ) (v w : Int) (h : PolicyOK terminal ss ps s)
    (hc : trajCost terminal ss s (policyActs ss ps s) = some w)
    (hv : valueO terminal ss s = some v) : v ≤ w :=
  valueO_le_cost terminal ss s _ v w h hc hv

/-- **The extracted plan, replayed as a policy, is an admissible optimal policy.** -/
theorem extracted_policy_optimal {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ))
    (s : σ) (v : Int) (hv : valueO terminal ss s = some v) :
    ∃ xs, extract terminal ss s = some xs ∧ PolicyOK terminal ss (replay xs) s ∧
      trajCost terminal ss s (policyActs ss (replay xs) s) = some v := by
  obtain ⟨xs, hx, hok, hc, hlen⟩ := extract_spec terminal ss s v hv
  refine ⟨xs, hx, ?_, ?_⟩
  · unfold PolicyOK; rw [replay_acts ss xs s hlen]; exact hok
  · rw [replay_acts ss xs s hlen]; exact hc

/-! ## Relation to the old `Corpus.Bellman.value` -/

/-- The old model with the same stages, successor `a.1`, and the terminal values
(infeasible terminals read as `0`, which is exactly the old convention). -/
def oldModel {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ)) : Bellman.Model σ where
  states := []
  stages := ss
  successor := fun _ a => a.1
  terminal := fun s => (terminal s).getD 0

/-- The old value is a lower bound on every admissible trajectory's cost... -/
theorem old_value_le_cost {σ : Type} (terminal : σ → Option Int) (ss0 : List (Stage σ)) :
    ∀ (ss : List (Stage σ)) (s : σ) (xs : List (Action σ)) (w : Int),
      TrajOK terminal ss s xs → trajCost terminal ss s xs = some w →
      Bellman.value (oldModel terminal ss0) ss s ≤ w := by
  intro ss
  induction ss with
  | nil =>
      intro s xs w hok hc
      cases xs with
      | nil =>
          simp only [trajCost] at hc
          simp [Bellman.value, oldModel, hc]
      | cons _ _ => cases hok
  | cons st rs ih =>
      intro s xs w hok hc
      cases xs with
      | nil => cases hok
      | cons a ys =>
          obtain ⟨ha, ht⟩ := hok
          simp only [trajCost] at hc
          cases hw' : trajCost terminal rs a.1 ys with
          | none => rw [hw'] at hc; cases hc
          | some w' =>
              rw [hw'] at hc
              simp only [Option.map_some, Option.some.injEq] at hc
              have h1 := ih a.1 ys w' ht hw'
              have h2 := Bellman.bestCost_le
                (fun q => Bellman.value (oldModel terminal ss0) rs ((oldModel terminal ss0).successor s q))
                (st s) a ha
              simp only [Bellman.value]
              have : (oldModel terminal ss0).successor s a = a.1 := rfl
              rw [this] at h2
              omega

/-- ... hence it never exceeds the true optimum (it may be strictly below it:
see `dead_mid_old_below`). -/
theorem old_value_le_valueO {σ : Type} (terminal : σ → Option Int) (ss : List (Stage σ))
    (s : σ) (v : Int) (hv : valueO terminal ss s = some v) :
    Bellman.value (oldModel terminal ss) ss s ≤ v := by
  obtain ⟨xs, hok, hc⟩ := valueO_attained terminal ss s v hv
  exact old_value_le_cost terminal ss ss s xs v hok hc

/-- `bestO` of a total next-cost agrees with the old `bestCost` on nonempty lists. -/
theorem bestO_some_eq_bestCost {σ : Type} (g : Action σ → Int) :
    ∀ (x : Action σ) (xs : List (Action σ)),
      bestO (fun a => some (a.2 + g a)) (x :: xs) = some (Bellman.bestCost g (x :: xs)) := by
  intro x xs
  let f : Action σ → Option Int := fun a => some (a.2 + g a)
  obtain ⟨m, hm⟩ := bestO_some_of_mem f (x :: xs) x (x.2 + g x) (by simp) rfl
  obtain ⟨a, ha, hfa⟩ := bestO_attains f (x :: xs) m hm
  have hma : a.2 + g a = m := by simpa [f] using hfa
  have h1 := Bellman.bestCost_le g (x :: xs) a ha
  obtain ⟨b, hb⟩ := Bellman.greedy_exists g (x :: xs) (by simp)
  have hbm := Bellman.greedy_mem' g (x :: xs) b hb
  have hbc := Bellman.greedy_cost g (x :: xs) b hb
  have h2 := bestO_le f (x :: xs) m hm b hbm (b.2 + g b) rfl
  show bestO f (x :: xs) = _
  rw [hm]
  congr 1
  omega

/-- No dead ends anywhere: every stage offers an action at every state. -/
def NoDeadEnds {σ : Type} (ss : List (Stage σ)) : Prop := ∀ st ∈ ss, ∀ s, st s ≠ []

/-- **The old formulation is correct exactly in the dead-end-free case**: with
feasible terminals and no empty action lists, the option value is the old value. -/
theorem valueO_eq_old {σ : Type} (T : σ → Int) (ss0 : List (Stage σ)) :
    ∀ ss : List (Stage σ), NoDeadEnds ss → ∀ s,
      valueO (fun x => some (T x)) ss s =
        some (Bellman.value (oldModel (fun x => some (T x)) ss0) ss s) := by
  intro ss
  induction ss with
  | nil => intro _ s; rfl
  | cons st rs ih =>
      intro hnd s
      have hrs : NoDeadEnds rs := fun st' h' => hnd st' (List.mem_cons_of_mem st h')
      simp only [valueO, Bellman.value]
      have hcongr : (fun a : Action σ => (valueO (fun x => some (T x)) rs a.1).map (fun v => a.2 + v)) =
          (fun a => some (a.2 + Bellman.value (oldModel (fun x => some (T x)) ss0) rs a.1)) := by
        funext a; rw [ih hrs a.1]; rfl
      rw [hcongr]
      cases hst : st s with
      | nil => exact absurd hst (hnd st (by simp) s)
      | cons x xs =>
          exact bestO_some_eq_bestCost
            (fun a => Bellman.value (oldModel (fun x => some (T x)) ss0) rs a.1) x xs

/-! ## Counterexamples to the old `bestCost [] = 0` formulation (kernel-checked) -/

/-- Infeasible terminal read as 0 by the old model. -/
def deadStages : List (Stage Nat) := [fun s => if s = 0 then [(1, 5), (2, 1)] else []]
def deadTerminal : Nat → Option Int := fun s => if s = 2 then some 10 else none
theorem dead_old_value : Bellman.value (oldModel deadTerminal deadStages) deadStages 0 = 5 := by decide
theorem dead_valueO : valueO deadTerminal deadStages 0 = some 11 := by decide
theorem dead_old_below_optimum :
    Bellman.value (oldModel deadTerminal deadStages) deadStages 0 < 11 := by decide

/-- Empty action list in the MIDDLE of the horizon, all terminals feasible and
all costs positive.  State 1 is a dead end at stage 2. -/
def midStages : List (Stage Nat) :=
  [fun s => if s = 0 then [(1, 1), (2, 5)] else [],
   fun s => if s = 2 then [(3, 1)] else []]
def midTerminal : Nat → Option Int := fun _ => some 0

/-- The old value is 1: it routes through the dead end 1 whose `bestCost [] = 0`. -/
theorem dead_mid_old_value : Bellman.value (oldModel midTerminal midStages) midStages 0 = 1 := by decide
/-- The old greedy extraction picks the dead-end action `(1, 1)`. -/
theorem dead_mid_old_greedy_dead_end :
    (Bellman.greedy? (fun a => Bellman.value (oldModel midTerminal midStages) (midStages.drop 1) a.1)
      ((midStages.headD (fun _ => [])) 0)).map (fun a => ((a.1, a.2) : Nat × Int)) = some (1, 1) := by
  decide
/-- ... and that action has no continuation at all. -/
theorem dead_mid_no_continuation : ((midStages.getD 1 (fun _ => [])) 1).length = 0 := by decide
/-- The option value is the true optimum 6 and extraction returns the feasible plan. -/
theorem dead_mid_valueO : valueO midTerminal midStages 0 = some 6 := by decide
theorem dead_mid_extract :
    (extract midTerminal midStages 0).map (List.map fun a => ((a.1, a.2) : Nat × Int)) =
      some [(2, 5), (3, 1)] := by decide
theorem dead_mid_old_below : Bellman.value (oldModel midTerminal midStages) midStages 0 < 6 := by decide
/-- Infeasible start: from state 1 nothing is feasible, and the option value says so. -/
theorem dead_mid_infeasible_start : valueO midTerminal midStages 1 = none ∧
    (extract midTerminal midStages 1).isNone = true := by decide
end Corpus.BellmanOpt
