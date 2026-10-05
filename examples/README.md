# Artificial GPS demo input

This 200-row trajectory is constructed analytically; it contains no observed human GPS records.
For row i=0..199: timestamp=2020-01-01 +5i seconds, latitude=0.001 sin(0.01i), longitude=0.001 cos(0.01i), altitude=0.
The origin and identity are artificial. No Final cohort or prior-user trajectory was copied.
This input demonstrates validation/features/frozen inference/notification output, not labelled accuracy.
It is separate from the frozen research synthetic generator and never enters any evaluation dataset.
