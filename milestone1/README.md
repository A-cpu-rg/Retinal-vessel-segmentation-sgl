# Milestone 1 · Background and base paper

[Project overview](../README.md)

Retinal vessel segmentation is a pixel-level task. Thin and low-contrast
vessels are difficult to annotate, which motivates testing whether model
predictions can complement manual labels. This project focuses on the DRIVE
experiment in Zhou, Yu, and Shi’s Study Group Learning paper.

The shared CON architecture first uses EnhanceNet to produce a learned
enhancement of the fundus image. SegNet converts that representation into a
vessel probability map. Both the manual-label baseline and SGL use this
architecture. [Architecture diagram](../assets/architecture.svg)

## Published DRIVE benchmark

The paper reports the following DRIVE results at the original training label
ratio in Appendix Table 3. These are published reference values.

| Metric | Published baseline (K=1) | Published SGL (K=8) |
|---|---:|---:|
| Accuracy | 0.9695 | 0.9705 |
| Sensitivity | 0.8341 | 0.8380 |
| Specificity | 0.9827 | 0.9834 |
| ROC-AUC | 0.9867 | 0.9886 |
| Dice / F1 | 0.8262 | 0.8316 |
| Vessel IoU | 0.7041 | 0.7120 |

## References

- Y. Zhou, H. Yu, and H. Shi. *Study Group Learning: Improving Retinal Vessel
  Segmentation Trained with Noisy Labels.* MICCAI 2021. [Paper](https://arxiv.org/html/2103.03451#S6.T3)
- [Official SGL implementation](https://github.com/SHI-Labs/SGL-Retinal-Vessel-Segmentation)
- J. Staal et al. *Ridge-based vessel segmentation in color images of the
  retina.* IEEE Transactions on Medical Imaging, 2004. [DRIVE](https://drive.grand-challenge.org/)
