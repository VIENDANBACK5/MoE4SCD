# TreeFlowNet Spatial Split Policy V3

**Version:** 3.0.0  
**Effective Date:** 2026-10-15  
**Governing Standard:** Prevention of Data Leakage, Spatial Autocorrelation Bias, and Contaminated Generalization Claims.

---

## 1. Dataset Roles & Access Control

Every dataset or subset in TreeFlowNet MUST be assigned exactly one of four mutually exclusive roles:

| Role | Permitted Usages | Prohibited Usages |
|---|---|---|
| `TRAIN` | Gradient updates, supervised loss calculation, data augmentation exploration, feature learning. | Validation metric reporting, hyperparameter tuning, model checkpoint selection. |
| `DEVELOPMENT` | Hyperparameter selection, checkpoint early stopping, architecture comparison, exploratory diagnostic audits. | Model training / backpropagation, threshold overfitting, claiming unbiased generalization. |
| `CALIBRATION` | Fitting temperature scaling, Platt scaling, confidence threshold grids, objectness calibration. | Training feature extractors, fitting on the same fold used to evaluate test performance. |
| `CONFIRMATORY_TEST` | One-shot final evaluation of frozen model checkpoints under preregistered protocols. | Any gradient updates, architecture search, threshold tuning, error-driven diagnostic redesign. |

### 1.1. Access Isolation Rule
- Test annotations and imagery MUST NOT be accessible during model training, feature preprocessing, hyperparameter search, or calibration.
- Re-use of a dataset for model selection permanently revokes its `CONFIRMATORY_TEST` eligibility and downgrades it to `DEVELOPMENT`.
- Historical test set status: Because the 168 historical test tiles were evaluated during EXP003, EXP015, and EXP020 development cycles, they are permanently classified as `DEVELOPMENT (HISTORICAL_TEST)`.
- For any future publication claims of generalization: `EXTERNAL_HOLDOUT_REQUIRED`.

---

## 2. Spatial Independence & Buffering Standards

Contiguous raster slicing creates touching tile borders that share forest canopy crowns, tree root zones, soil conditions, and illumination angles.

### 2.1. Invariants for Confirmatory Splits
A proposed split $\mathcal{S}_{conf} = \{\mathcal{D}_{train}, \mathcal{D}_{val}, \mathcal{D}_{test}\}$ passes confirmatory validation IF AND ONLY IF it satisfies all three criteria:
1. **Zero Source Pixel Overlap**:
   $$\forall T_a \in \mathcal{D}_i, T_b \in \mathcal{D}_j \ (i \ne j): \text{Pixels}(T_a) \cap \text{Pixels}(T_b) = \emptyset$$
2. **Mandatory Geographic Buffer Zone**:
   $$\text{dist}_{meters}(\text{Footprint}(T_a), \text{Footprint}(T_b)) \ge B_{min}$$
   where $B_{min} = 30.0\text{ meters}$ (1,000 pixels at $0.03\text{ m}$ GSD), exceeding maximum tree crown diameter and spatial correlation range.
3. **No Cross-Split Connected Components**:
   The spatial adjacency graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$, where edges represent distance $< B_{min}$, must have zero edges connecting different split partitions.

### 2.2. Exploratory Protocol Exception
- For rapid algorithmic prototyping (Level 2 Readiness), development on Z1 is permitted despite the 1-pixel boundary seam, provided:
  - All metrics are explicitly tagged as `EXPLORATORY_PROTOTYPE`.
  - No claims of statistical or spatial independence are made.
  - Valid boundary pixels are controlled via 1-pixel erosion sensitivity tests.

---

## 3. Spatial Cross-Validation and Calibration

1. **Buffered Block Spatial Folds**:
   - Random tile splitting is strictly prohibited for cross-validation and calibration.
   - Cross-validation must partition tiles into contiguous geographic blocks separated by a buffer zone $\ge B_{min}$.
2. **Out-of-Fold (OOF) Scoring**:
   - Historical within-validation OOF scores (which had 378 overlapping tile pairs) are marked as non-independent.
   - Any future OOF calibration must enforce the spatial buffer policy between training and prediction folds.

---

## 4. Waiver Policy

Under extraordinary circumstances where an exploratory protocol requires relaxing a non-critical spatial constraint, a formal waiver must be recorded.
A valid waiver must include:
- `waiver_id`: Unique tracking ID (e.g., `WVR-2026-001`).
- `requirement_id`: The specific requirement waived (e.g., `REQ-BUF-30M`).
- `justification`: Detailed scientific rationale for the exception.
- `scope`: Exact files, models, and experiments covered.
- `approver`: Senior scientific auditor.
- `effective_date` & `expiry_date`: Time-bounded validity.
- `scientific_claim_restriction`: Explicit statement of claims forbidden due to the waiver (e.g., "Cannot claim spatial generalization").

*Note: Waivers CANNOT be granted to bypass confirmatory holdout requirements.*
