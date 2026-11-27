# Milestone 3 · Hypothesis and experiment plan

[Project overview](../README.md)

## Hypothesis

Out-of-fold pseudo-labels may identify vessel pixels that are difficult to
trace by hand. The project expects the SGL model to reach higher sensitivity
than the manual-label baseline while retaining similar specificity.
Specificity measures how often background pixels are correctly rejected, so
it helps distinguish recovered vessels from increased false positives.

## Comparison protocol

1. Use the official DRIVE training and test partitions for both models.
2. Hold the CON architecture, final training settings, and inference procedure
   constant. Change the supervision from manual labels alone to manual labels
   plus out-of-fold pseudo-labels.
3. Select each checkpoint using the same rule. The documented Colab run uses
   the last fixed epoch. When using validation IDs, hold out the same training
   images from both models and from every fold teacher, then select checkpoints
   using validation Dice or ROC-AUC.
4. Compare mean per-image sensitivity and specificity on the 20 DRIVE test
   images inside the field of view. Report the signed SGL-minus-baseline
   differences alongside accuracy, ROC-AUC, Dice, and vessel IoU.
5. Review prediction maps to see where SGL recovers missed vessels and where
   it introduces extra vessel predictions. [Failure analysis](../milestone5/README.md)

The hypothesis is directional. The project has no predefined numerical
specificity margin. Report the magnitude of both changes before interpreting
any sensitivity gain. [Reproducible comparison](../milestone4/README.md)
