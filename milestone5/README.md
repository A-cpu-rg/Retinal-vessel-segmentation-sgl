# Milestone 5 · Failure analysis and limitations

[Project overview](../README.md)

The [IEEE paper](../report/paper/retinal_sgl_ieee.pdf) includes a completed
failure analysis for the reduced CPU pilot. The full eight-group DRIVE
comparison is pending. Use the saved evaluation outputs to
carry out this review after both model runs finish.
[Output locations](../milestone4/README.md#review-generated-outputs)

## Review baseline and SGL predictions

1. Match the same test image IDs across both models' metric files and
   prediction folders. Record the field-of-view Dice, sensitivity, and
   specificity for each model on each reviewed image.
2. Compare the corresponding binary masks and overlays against the first
   manual annotation inside the field of view. The error maps show correctly
   detected vessels in white, false vessel predictions in red, and missed
   vessels in blue. [Visualization code](../src/visualize.py)
3. Record where SGL recovers a vessel missed by the baseline, misses a vessel
   detected by the baseline, adds a false vessel, or shares the same error.
   Examine thin branches, vessel boundaries, and low-contrast areas.
4. Interpret those examples alongside the aggregate sensitivity and
   specificity changes in `comparison.json`. Record image IDs and the
   affected regions so the examples can be checked independently.

## Limitations

- The implementation has passed synthetic end-to-end checks and real DRIVE
  data-loading checks. Those checks establish pipeline execution and dataset
  compatibility. Full GPU training and the DRIVE benchmark remain pending.
- The paper's controlled label-erasure experiments and CHASE_DB1 benchmark
  are outside the scope of this reproduction.
- The learning rate, epoch count, batch size, and vessel threshold follow
  conventions from the reference implementation. The primary field-of-view
  scoring protocol can differ from the paper's reported values.
- Error maps measure agreement with the first manual DRIVE annotation.
  Ambiguous thin structures require visual review before assigning an error
  cause.
