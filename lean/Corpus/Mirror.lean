/-
Mirror (y ↦ −y) transform of a sample, and what it preserves. The checker additionally
verifies that the generator's `mirror_observation` / `mirror_action` agree with these
definitions on exported samples.
-/
import Corpus.Model

namespace Corpus

def flipY (p : Int × Int) : Int × Int := (p.1, -p.2)

def mirrorObstacle (b : ObstacleM) : ObstacleM := { b with latMin := -b.latMax, latMax := -b.latMin }

def mirrorObs (o : ObsM) : ObsM where
  lateral := -o.lateral
  heading := -o.heading
  curvature := -o.curvature
  widthLeft := o.widthRight
  widthRight := o.widthLeft
  distLeft := o.distRight
  distRight := o.distLeft
  vy := -o.vy
  yawRate := -o.yawRate
  ay := -o.ay
  steer := -o.steer
  previewCurvature := o.previewCurvature.map (- ·)
  previewWidthLeft := o.previewWidthRight
  previewWidthRight := o.previewWidthLeft
  centerline := o.centerline.map flipY
  leftBoundary := o.rightBoundary.map flipY
  rightBoundary := o.leftBoundary.map flipY
  obstacles := o.obstacles.map mirrorObstacle

def mirrorAct (a : ActM) : ActM := { a with steer := -a.steer, path := a.path.map flipY }

theorem flipY_flipY (p : Int × Int) : flipY (flipY p) = p := by
  simp [flipY]

theorem map_flipY_flipY (l : List (Int × Int)) : (l.map flipY).map flipY = l := by
  simp [List.map_map, Function.comp_def, flipY_flipY]

theorem map_neg_neg (l : List Int) : (l.map (- ·)).map (- ·) = l := by
  simp [List.map_map, Function.comp_def]

theorem mirrorObstacle_mirrorObstacle (b : ObstacleM) : mirrorObstacle (mirrorObstacle b) = b := by
  cases b; simp [mirrorObstacle]

theorem mirrorObs_involutive (o : ObsM) : mirrorObs (mirrorObs o) = o := by
  cases o
  simp [mirrorObs, List.map_map, Function.comp_def, flipY_flipY, mirrorObstacle_mirrorObstacle]

theorem mirrorAct_involutive (a : ActM) : mirrorAct (mirrorAct a) = a := by
  cases a; simp [mirrorAct, List.map_map, Function.comp_def, flipY_flipY]

/-- Pedal labels are unchanged; the steering label changes sign (so its bound is preserved). -/
theorem mirrorAct_labels (a : ActM) :
    (mirrorAct a).throttle = a.throttle ∧ (mirrorAct a).brake = a.brake ∧ (mirrorAct a).steer = -a.steer :=
  ⟨rfl, rfl, rfl⟩

theorem mirror_steer_bound (a : ActM) (S : Int) (h : -S ≤ a.steer ∧ a.steer ≤ S) :
    -S ≤ (mirrorAct a).steer ∧ (mirrorAct a).steer ≤ S := by
  simp [mirrorAct]; omega

/-- Obstacle extents stay well-formed intervals. -/
theorem mirror_interval (b : ObstacleM) (h : b.latMin ≤ b.latMax) :
    (mirrorObstacle b).latMin ≤ (mirrorObstacle b).latMax := by
  simp [mirrorObstacle]; omega

def gapLeft (wl : Int) (b : ObstacleM) : Int := wl - b.latMax
def gapRight (wr : Int) (b : ObstacleM) : Int := b.latMin + wr

/-- After mirroring (widths swap, extents reflect) the left gap is the old right gap and vice versa. -/
theorem mirror_gap_swap (wl wr : Int) (b : ObstacleM) :
    gapLeft wr (mirrorObstacle b) = gapRight wr b ∧ gapRight wl (mirrorObstacle b) = gapLeft wl b := by
  simp [gapLeft, gapRight, mirrorObstacle]; omega

/-- The gap-side rule of the expert (strict cases; ties are resolved by an asymmetric tie-break
in the expert and are excluded here) flips under mirroring. -/
def chooseSide (wl wr : Int) (b : ObstacleM) : Int :=
  if gapLeft wl b > gapRight wr b then 1 else if gapRight wr b > gapLeft wl b then -1 else 0

theorem chooseSide_mirror (wl wr : Int) (b : ObstacleM) :
    chooseSide wr wl (mirrorObstacle b) = - chooseSide wl wr b := by
  unfold chooseSide gapLeft gapRight mirrorObstacle
  simp only
  by_cases h1 : wl - b.latMax > b.latMin + wr <;> by_cases h2 : b.latMin + wr > wl - b.latMax <;>
    by_cases h3 : wr - -b.latMin > -b.latMax + wl <;> by_cases h4 : -b.latMax + wl > wr - -b.latMin <;>
    simp only [h1, h2, h3, h4, ite_true, ite_false] <;> omega

end Corpus
