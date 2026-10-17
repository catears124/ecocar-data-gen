/-
  Core-only finite-horizon Bellman machinery.

  A stage offers a finite list of `(action, immediate cost)` pairs.  The action
  component is abstract, while the model's successor function determines the
  next state.
-/
namespace Corpus.Bellman

/-- An action value together with its immediate integer cost. -/
def Action (σ : Type) := σ × Int

/-- One stage's actions, indexed by the current state. -/
def Stage (σ : Type) := σ → List (Action σ)

/-- A finite state/action dynamic-programming model. -/
structure Model (σ : Type) where
  states : List σ
  stages : List (Stage σ)
  successor : σ → Action σ → σ
  terminal : σ → Int

/-- Minimum cost in a finite action list.  The zero empty-list branch is not
used by the optimality theorems, whose hypotheses require offered actions. -/
def bestCost {σ : Type} (next : Action σ → Int) : List (Action σ) → Int
  | [] => 0
  | [a] => a.2 + next a
  | a :: b :: as => min (a.2 + next a) (bestCost next (b :: as))

/-- The first minimising action, with `none` exactly for an empty action list. -/
def greedy? {σ : Type} (next : Action σ → Int) : List (Action σ) → Option (Action σ)
  | [] => none
  | a :: as =>
      match greedy? next as with
      | none => some a
      | some b => if a.2 + next a ≤ b.2 + next b then some a else some b

/-- Backward induction over a list of stages. -/
def value {σ : Type} (m : Model σ) : List (Stage σ) → σ → Int
  | [], s => m.terminal s
  | st :: rest, s =>
      bestCost (fun a => value m rest (m.successor s a)) (st s)

/-- A policy is one action-valued function per stage. -/
def Policy (σ : Type) := List (σ → Action σ)

/-- The accumulated cost of a policy, including the terminal value. -/
def policyCost {σ : Type} (m : Model σ) : List (Stage σ) → Policy σ → σ → Int
  | [], [], s => m.terminal s
  | st :: rest, p :: ps, s =>
      let a := p s
      a.2 + policyCost m rest ps (m.successor s a)
  | _, _, s => m.terminal s

/-- Every selected policy action is offered at every state. -/
inductive Admissible {σ : Type} : List (Stage σ) → Policy σ → Prop
  | nil : Admissible [] []
  | cons {st : Stage σ} {rest : List (Stage σ)} {p : σ → Action σ} {ps : Policy σ}
      (head : ∀ s, p s ∈ st s) (tail : Admissible rest ps) :
      Admissible (st :: rest) (p :: ps)

/-- `bestCost` is no greater than every listed action's cost. -/
theorem bestCost_le {σ : Type} (next : Action σ → Int) :
    ∀ (as : List (Action σ)) (a : Action σ), a ∈ as →
      bestCost next as ≤ a.2 + next a := by
  intro as
  induction as with
  | nil => intro a h; cases h
  | cons b bs ih =>
    intro a h
    rcases List.mem_cons.mp h with rfl | h
    · cases bs with
      | nil => simp [bestCost]
      | cons c cs => exact Int.min_le_left _ _
    · cases bs with
      | nil => simp at h
      | cons c cs =>
        exact Int.le_trans (Int.min_le_right _ _) (ih a h)

/-- Pointwise equal next-cost functions give equal list minima. -/
theorem bestCost_congr {σ : Type} {f g : Action σ → Int}
    (h : ∀ a, f a = g a) : ∀ as, bestCost f as = bestCost g as := by
  intro as
  induction as with
  | nil => rfl
  | cons a as ih =>
    cases as with
    | nil => simp [bestCost, h a]
    | cons b bs => simp [bestCost, h a, ih]

/- A nonempty action list has a selected greedy action. -/
theorem greedy_exists {σ : Type} (next : Action σ → Int) :
    ∀ as : List (Action σ), as ≠ [] → ∃ a, greedy? next as = some a := by
  intro as hne
  cases as with
  | nil => exact (hne rfl).elim
  | cons a as =>
    cases htail : greedy? next as with
    | none => exact ⟨a, by simp [greedy?, htail]⟩
    | some b =>
      by_cases hle : a.2 + next a ≤ b.2 + next b
      · exact ⟨a, by simp [greedy?, htail, hle]⟩
      · exact ⟨b, by simp [greedy?, htail, hle]⟩

/- The greedy action is present whenever it is selected. -/
theorem greedy_mem' {σ : Type} (next : Action σ → Int) :
    ∀ (as : List (Action σ)) (a : Action σ),
      greedy? next as = some a → a ∈ as := by
  intro as
  induction as with
  | nil => intro a h; simp [greedy?] at h
  | cons b bs ih =>
    intro a h
    cases htail : greedy? next bs with
    | none =>
      have hab : b = a := by simpa [greedy?, htail] using h
      subst a
      simp
    | some c =>
      by_cases hkeep : b.2 + next b ≤ c.2 + next c
      · have hab : b = a := by simpa [greedy?, htail, hkeep] using h
        subst a
        simp
      · have hac : c = a := by simpa [greedy?, htail, hkeep] using h
        subst a
        exact List.mem_cons_of_mem _ (ih c htail)

/-- The greedy action attains the minimum cost. -/
theorem greedy_cost {σ : Type} (next : Action σ → Int) :
    ∀ (as : List (Action σ)) (a : Action σ),
      greedy? next as = some a → bestCost next as = a.2 + next a := by
  intro as
  induction as with
  | nil => intro a h; simp [greedy?] at h
  | cons b bs ih =>
    intro a h
    cases htail : greedy? next bs with
    | none =>
      have hbs : bs = [] := by
        cases bs with
        | nil => rfl
        | cons x xs =>
          obtain ⟨y, hy⟩ := greedy_exists next (x :: xs) (by simp)
          rw [hy] at htail
          simp at htail
      have hab : b = a := by simpa [greedy?, htail] using h
      subst a
      simp [bestCost, hbs]
    | some c =>
      cases bs with
      | nil => simp [greedy?] at htail
      | cons x xs =>
        change (match greedy? next (x :: xs) with
          | none => some b
          | some d => if b.2 + next b ≤ d.2 + next d then some b else some d) = some a at h
        rw [htail] at h
        have htailCost : bestCost next (x :: xs) = c.2 + next c := ih c htail
        by_cases hkeep : b.2 + next b ≤ c.2 + next c
        · have hab : b = a := by simpa [greedy?, htail, hkeep] using h
          subst a
          change min (b.2 + next b) (bestCost next (x :: xs)) = b.2 + next b
          rw [htailCost]
          omega
        · have hac : c = a := by simpa [greedy?, htail, hkeep] using h
          subst a
          change min (b.2 + next b) (bestCost next (x :: xs)) = c.2 + next c
          rw [htailCost]
          omega

/-- The greedy action satisfies the Bellman equation at the current stage. -/
theorem value_attains {σ : Type} (m : Model σ) (st : Stage σ) (rest : List (Stage σ))
    (hact : ∀ s, (st s).length > 0) :
    ∀ s, ∃ a, a ∈ st s ∧
      value m (st :: rest) s = a.2 + value m rest (m.successor s a) := by
  intro s
  let next : Action σ → Int := fun a => value m rest (m.successor s a)
  obtain ⟨a, ha⟩ := greedy_exists next (st s) (by
    intro hz
    have hzero : (st s).length = 0 := by simp [hz]
    have hpos := hact s
    omega)
  refine ⟨a, greedy_mem' next (st s) a ha, ?_⟩
  change bestCost next (st s) = a.2 + next a
  exact greedy_cost next (st s) a ha

/-- No admissible policy beats the backward value. -/
theorem no_policy_beats {σ : Type} (m : Model σ) :
    ∀ (stages : List (Stage σ)) (policy : Policy σ),
      Admissible stages policy → ∀ s, policyCost m stages policy s ≥ value m stages s := by
  intro stages
  induction stages with
  | nil =>
    intro policy hp s
    cases hp
    simp [policyCost, value]
  | cons st rest ih =>
    intro policy hp s
    cases policy with
    | nil => cases hp
    | cons p ps =>
      cases hp with
      | cons hhead htail =>
        let a := p s
        have hle := bestCost_le (fun q => value m rest (m.successor s q)) (st s) a (hhead s)
        have hrec := ih ps htail (m.successor s a)
        change (p s).2 + policyCost m rest ps (m.successor s (p s)) ≥
          bestCost (fun q => value m rest (m.successor s q)) (st s)
        change a.2 + policyCost m rest ps (m.successor s a) ≥
          bestCost (fun q => value m rest (m.successor s q)) (st s)
        omega

/-- A list of functions is a Bellman solution when its head is the current
value function and its tail solves the remaining horizon. -/
inductive BellmanSolution {σ : Type} (m : Model σ) :
    List (Stage σ) → List (σ → Int) → Prop
  | nil (f : σ → Int) (hterminal : ∀ s, f s = m.terminal s) :
      BellmanSolution m [] [f]
  | cons (st : Stage σ) (rest : List (Stage σ))
      (f : σ → Int) (tail : List (σ → Int))
      (head : ∀ s, f s = bestCost (fun a => tail.headD m.terminal (m.successor s a)) (st s))
      (htail : BellmanSolution m rest tail) :
      BellmanSolution m (st :: rest) (f :: tail)

/-- Uniqueness of a Bellman solution on every state; `[DecidableEq σ]` makes
state equality available to finite-state clients even though the proof itself
uses function extensionality rather than enumeration. -/
theorem value_unique {σ : Type} [DecidableEq σ] (m : Model σ) :
    ∀ (stages : List (Stage σ)) (sol : List (σ → Int)),
      BellmanSolution m stages sol →
      ∀ s, sol.headD m.terminal s = value m stages s := by
  intro stages
  induction stages with
  | nil =>
    intro sol hs s
    cases hs with
    | nil f hterminal =>
      simp [hterminal, value]
  | cons st rest ih =>
    intro sol hs s
    cases hs with
    | cons st rest f tail hhead htail =>
      have htail_eq : ∀ a : Action σ,
          tail.headD m.terminal (m.successor s a) = value m rest (m.successor s a) := by
        intro a
        exact ih tail htail (m.successor s a)
      change f s = value m (st :: rest) s
      rw [hhead s, value]
      exact bestCost_congr htail_eq (st s)

end Corpus.Bellman
