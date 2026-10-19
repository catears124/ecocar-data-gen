# Analysis

## Paired condition differences

Comparisons are paired by (course, seed), then averaged within course before bootstrap resampling. Bootstrap confidence intervals therefore resample courses, not frames or JSONL examples.

### Budget 100
- optimal-vs-approx: mean time difference (s) = -14.547539325698118; 95% CI = [-15.968580548133087, -13.126498103263149]; courses = 4
- optimal-vs-heuristic: mean time difference (s) = 21.392726436818787; 95% CI = [17.114556948166783, 25.358186558195882]; courses = 3
- optimal-vs-cert_control: mean time difference (s) = 0.0; 95% CI = [0.0, 0.0]; courses = 4

### Budget 300
- optimal-vs-approx: mean time difference (s) = -13.05792639205312; 95% CI = [-14.163308153217045, -11.952544630889196]; courses = 4
- optimal-vs-heuristic: mean time difference (s) = 22.934217229795085; 95% CI = [19.23180478229803, 26.415516066529918]; courses = 3
- optimal-vs-cert_control: mean time difference (s) = 0.0; 95% CI = [0.0, 0.0]; courses = 4

### Budget 720
- optimal-vs-approx: mean time difference (s) = -13.032511339345216; 95% CI = [-13.872242363932994, -12.192780314757439]; courses = 4
- optimal-vs-heuristic: mean time difference (s) = 22.56872221541094; 95% CI = [19.04828436459889, 25.42609450917764]; courses = 3
- optimal-vs-cert_control: mean time difference (s) = 0.0; 95% CI = [0.0, 0.0]; courses = 4

## Failure analysis

Recorded source failures: 0. Controller exceptions, DNFs, cone hits, and off-course counts remain explicit in the per-condition metrics; no failed run is converted into a successful metric.

## Are optimal labels harder to imitate?

This is not assumed. Compare the reported held-out offset and acceleration MAE by condition and budget; only the observed values can support that conclusion.
