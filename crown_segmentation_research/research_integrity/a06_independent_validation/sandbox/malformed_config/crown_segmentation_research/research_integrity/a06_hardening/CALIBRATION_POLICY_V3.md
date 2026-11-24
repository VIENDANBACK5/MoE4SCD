# TreeFlowNet Calibration & Score Control Policy V3

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Governing Standard:** Decoupling of Mask Geometry from Confidence Scores and Prevention of Calibration Contamination.

---

## 1. Score–Geometry Decoupling

A central finding of the historical TreeFlowNet audits is that altering confidence scores can artificially shift COCO AP by several points without modifying a single pixel of predicted mask geometry:
- Score shifts alter ranking order in the precision–recall curve.
- Score thresholds alter the set of accepted proposals.

### Invariant Controls
1. **Separation of Representations**: Predictions must store mask geometry ($m_k$) and scalar score ($s_k$) in decoupled fields.
2. **Structural Invariance Invariant**:
   If mask set $\mathcal{M} = \{m_1, \dots, m_N\}$ is held constant and scores $\mathcal{S} = \{s_1, \dots, s_N\}$ are modified, **structural grouping metrics MUST remain exactly identical**:
   $$\Delta \text{StructuralGrouping}(\mathcal{M}, \mathcal{S}) = 0$$
   Any violation indicates an accidental leak of scores into structural geometry evaluation.
3. **Stale Score Protection**: Any re-ranking or calibration update must regenerate the evaluation manifest and update the SHA256 score checksum.

---

## 2. Event-Driven Confidence Calibration

Confidence scores $s_k \in [0, 1]$ must be calibrated against an explicit, observable verification event:
$$E_k = \mathbf{1}_{\exists g \in \mathcal{G} \text{ s.t. } \text{IoU}(m_k, g) \ge 0.5}$$
- **Target Event:** Probability that accepted proposal $m_k$ achieves a true-positive 1-to-1 match against ground truth.
- **Calibration Methods:** Temperature scaling, Platt scaling, or Isotonic regression fitted exclusively on the `CALIBRATION` split.
- **Reporting:**
  - Expected Calibration Error (ECE) across 10 equal-width confidence bins.
  - Reliability diagrams (plotting empirical precision vs. mean predicted confidence).
  - Accuracy–Coverage curves across varying acceptance thresholds.

---

## 3. Spatial Calibration Independence Standards

1. **Rejection of Unbuffered OOF Folds**:
   - Historical within-validation OOF scoring partitioned tiles without spatial buffering, creating 378 overlapping tile pairs and 370 shared crowns.
   - These historical OOF scores are classified as `SPATIALLY_CONTAMINATED` and cannot be reported as independent generalization estimates.
2. **Mandatory Buffered Spatial Folds**:
   - Future calibration MUST use spatial block K-fold cross-validation with a minimum geographic buffer $B_{min} = 30.0\text{ m}$ between the fold used for model training and the fold used for calibration fitting.
3. **No Test-Set Threshold Fitting**:
   - Optimal threshold search ($\arg\max_\tau \text{AP}(\tau)$ or $\arg\max_\tau F_1(\tau)$) must occur strictly on `DEVELOPMENT` or `CALIBRATION` splits. Applying threshold search to `CONFIRMATORY_TEST` constitutes invalid test-label optimization.
