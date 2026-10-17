/-
Properties of the exported manifest / corpus index, stated as propositions and decided by
their `Decidable` instances (so each check is correct by construction), plus the lemmas that
justify the efficient checks and the meaning of the reports.
-/
import Corpus.Spec

namespace Corpus

/-- Parameter value (numeric or categorical) of a scenario. -/
def Scenario.val (s : Scenario) (n : String) : Option (Int ⊕ String) :=
  match s.nums.lookup n with
  | some v => some (.inl v)
  | none => (s.cats.lookup n).map .inr

def accepted (d : List Scenario) : List Scenario := d.filter (·.accepted)

/-! ### Scenario validity and drift -/

/-- The generator's formal verdict equals Lean's for every exported draw. -/
def FormalVerdictsAgree (sp : Spec) (ss : List Scenario) : Prop :=
  ∀ s ∈ ss, (s.formalRejected = true ↔ ¬ Valid sp s)

instance (sp : Spec) (ss : List Scenario) : Decidable (FormalVerdictsAgree sp ss) := by
  unfold FormalVerdictsAgree; infer_instance

def AcceptedValid (sp : Spec) (ss : List Scenario) : Prop :=
  ∀ s ∈ accepted ss, Valid sp s ∧ BinsCorrect sp s

instance (sp : Spec) (ss : List Scenario) : Decidable (AcceptedValid sp ss) := by
  unfold AcceptedValid; infer_instance

def UniqueIds (ss : List Scenario) : Prop := ((accepted ss).map (·.id)).Nodup
def UniqueHashes (ss : List Scenario) : Prop := ((accepted ss).map (·.hash)).Nodup

instance (ss : List Scenario) : Decidable (UniqueIds ss) := by unfold UniqueIds; infer_instance
instance (ss : List Scenario) : Decidable (UniqueHashes ss) := by unfold UniqueHashes; infer_instance

/-! ### Contrast pairs -/

/-- `a` and `b` differ in factor `f` (value and bin) and agree on every other declared parameter. -/
def DiffersExactlyIn (sp : Spec) (a b : Scenario) (f : String) : Prop :=
  (∀ n ∈ sp.paramNames, n ≠ f → a.val n = b.val n) ∧ a.val f ≠ b.val f ∧ a.bins.lookup f ≠ b.bins.lookup f

instance (sp : Spec) (a b : Scenario) (f : String) : Decidable (DiffersExactlyIn sp a b f) := by
  unfold DiffersExactlyIn; infer_instance

def PairOK (sp : Spec) (acc : List Scenario) (p : Pair) : Prop :=
  p.factor ∈ sp.factors.map Factor.name ∧
  ∃ a ∈ acc, a.id = p.base ∧ ∃ b ∈ acc, b.id = p.variant ∧
    a.family = b.family ∧ a.group = b.group ∧ DiffersExactlyIn sp a b p.factor

instance (sp : Spec) (acc : List Scenario) (p : Pair) : Decidable (PairOK sp acc p) := by
  unfold PairOK; infer_instance

def PairsOK (sp : Spec) (ss : List Scenario) (ps : List Pair) : Prop := ∀ p ∈ ps, PairOK sp (accepted ss) p

instance (sp : Spec) (ss : List Scenario) (ps : List Pair) : Decidable (PairsOK sp ss ps) := by
  unfold PairsOK; infer_instance

/-! ### Splits -/

def SplitRule (pol : SplitPolicy) (ss : List Scenario) : Prop :=
  ∀ s ∈ accepted ss, s.split ∈ splitNames ∧ s.split = regionSplit pol s.bins ∧ s.region = regionKey pol s.bins

instance (pol : SplitPolicy) (ss : List Scenario) : Decidable (SplitRule pol ss) := by
  unfold SplitRule; infer_instance

def GroupSplitConsistent (ss : List Scenario) : Prop :=
  ∀ a ∈ accepted ss, ∀ b ∈ accepted ss, a.group = b.group → a.split = b.split

instance (ss : List Scenario) : Decidable (GroupSplitConsistent ss) := by
  unfold GroupSplitConsistent; infer_instance

def HashSplitConsistent (ss : List Scenario) : Prop :=
  ∀ a ∈ accepted ss, ∀ b ∈ accepted ss, a.hash = b.hash → a.split = b.split

instance (ss : List Scenario) : Decidable (HashSplitConsistent ss) := by
  unfold HashSplitConsistent; infer_instance

/-- Contrast pairs never cross splits: follows from group consistency and pair integrity. -/
theorem pairs_within_split (sp : Spec) (ss : List Scenario) (ps : List Pair)
    (hg : GroupSplitConsistent ss) (hp : PairsOK sp ss ps) :
    ∀ p ∈ ps, ∃ a ∈ accepted ss, a.id = p.base ∧ ∃ b ∈ accepted ss, b.id = p.variant ∧ a.split = b.split := by
  intro p hpm
  obtain ⟨_, a, ha, hid, b, hb, hbid, _, hgrp, _⟩ := hp p hpm
  exact ⟨a, ha, hid, b, hb, hbid, hg a ha b hb hgrp⟩

/-! ### Coverage -/

def Covered (ss : List Scenario) (c : List (String × String)) : Prop := ∃ s ∈ ss, cellHolds s.bins c = true

/-- A coverage report is exact: a required cell is listed missing iff no scenario covers it, and
every listed cell is a required cell. -/
def CoverageExact (ss : List Scenario) (cells : List (List (String × String))) (missing : List String) : Prop :=
  (∀ c ∈ cells, (Covered ss c ↔ cellId c ∉ missing)) ∧ (∀ m ∈ missing, ∃ c ∈ cells, cellId c = m)

instance (ss : List Scenario) (cells : List (List (String × String))) (missing : List String) :
    Decidable (CoverageExact ss cells missing) := by
  unfold CoverageExact Covered; infer_instance

theorem full_coverage_of_exact {ss : List Scenario} {cells : List (List (String × String))}
    (h : CoverageExact ss cells []) : ∀ c ∈ cells, Covered ss c := by
  intro c hc
  exact (h.1 c hc).2 (by simp)

/-! ### Clean samples: no identical content in two splits -/

/-- Adjacent relation of a hash-sorted list: strictly increasing hash, or equal hash and split. -/
def AdjOK (x y : Nat × Nat) : Prop := x.1 < y.1 ∨ (x.1 = y.1 ∧ x.2 = y.2)

instance (x y : Nat × Nat) : Decidable (AdjOK x y) := by unfold AdjOK; infer_instance

def adjCheck : List (Nat × Nat) → Bool
  | x :: y :: t => decide (AdjOK x y) && adjCheck (y :: t)
  | _ => true

def NoCrossSplitDuplicates (l : List (Nat × Nat)) : Prop := ∀ x ∈ l, ∀ y ∈ l, x.1 = y.1 → x.2 = y.2

theorem AdjOK_trans {x y z : Nat × Nat} (h1 : AdjOK x y) (h2 : AdjOK y z) : AdjOK x z := by
  unfold AdjOK at *; omega

theorem adjCheck_tail {x : Nat × Nat} {l : List (Nat × Nat)} (h : adjCheck (x :: l) = true) : adjCheck l = true := by
  cases l with
  | nil => rfl
  | cons y t => simp [adjCheck] at h; exact h.2

theorem adjCheck_head : ∀ (l : List (Nat × Nat)) (x : Nat × Nat), adjCheck (x :: l) = true → ∀ y ∈ l, AdjOK x y := by
  intro l
  induction l with
  | nil => intro x _ y hy; simp at hy
  | cons z t ih =>
    intro x h y hy
    simp only [adjCheck, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases List.mem_cons.mp hy with rfl | hy'
    · exact h.1
    · exact AdjOK_trans h.1 (ih z h.2 y hy')

/-- Soundness of the linear-time check: a hash-sorted list passing `adjCheck` never has the
same content hash in two splits. -/
theorem adjCheck_sound : ∀ l : List (Nat × Nat), adjCheck l = true → NoCrossSplitDuplicates l := by
  intro l
  induction l with
  | nil => intro _ x hx; simp at hx
  | cons a t ih =>
    intro h x hx y hy hxy
    have hhead := adjCheck_head t a h
    rcases List.mem_cons.mp hx with rfl | hx'
    · rcases List.mem_cons.mp hy with rfl | hy'
      · rfl
      · have := hhead y hy'; unfold AdjOK at this; omega
    · rcases List.mem_cons.mp hy with rfl | hy'
      · have := hhead x hx'; unfold AdjOK at this; omega
      · exact ih (adjCheck_tail h) x hx' y hy' hxy

end Corpus
