# Research Log: MoE4SCD + SECOND-OC

Nhật ký nghiên cứu theo session. Mỗi session ghi: việc đã làm, phát hiện quan trọng, kết quả, bước tiếp theo.

---

## 2026-09-28 — Session 6: Rò rỉ test, lỗi đa giác, và benchmark sạch trên CanopyRS/QuebecTrees

### Việc đã làm
- **Phát hiện rò rỉ test:** V1 train trên Quebec 600 patch = Z1+Z2+Z3, và mọi checkpoint sau đều warm-start từ V1 (`tree_flow_exhaustive_v1/train_resume.log`). Mọi số Z2/Z3 của TreeFlowNet đều không phải held-out.
- **Sửa lỗi lọc đa giác** trong `large_scale_v2` (bỏ nhầm tán có đường viền thắt eo). Dùng `make_valid`, có test, chạy lại mọi cấu hình (hậu tố `_pf`). Tài liệu trả lời Thầy đã cập nhật số `_pf`; hình 24/25 vẽ lại.
- **Thí nghiệm decoder không cần train lại** trên Z2: tường từ độ phân kỳ flow (`decode_flowseam.decode_flowwall`), gán theo điểm hút (`decode_flowassign`).
- **Benchmark sạch** `crown_segmentation_research/clean_benchmark/`: chuẩn bị CanopyRS/QuebecTrees (train 148 tile/53.289 tán, val 139/15.155, test 168/22.018); target nhanh v1 (khớp chính xác target gốc, có test) và v2; trainer TreeFlowNet + Mask R-CNN; `evaluate.py` (COCO AP maxDets 400, F1 theo ngưỡng điểm, số đếm theo tile cho bootstrap); `select_on_val.py` (chọn trên val, test chạy một lần); `render_preview.py`.
- Kế hoạch có giả thuyết đăng ký trước (H1–H5): `meeting/week1/KE_HOACH_NGHIEN_CUU_TIEP_THEO.md`.
- Sửa code train cũ (29/09): chỉ dùng Z1, cửa sổ ngẫu nhiên mỗi lần với nhãn trên lưới nới lề, chặn khởi tạo từ checkpoint rò rỉ (`crown_segmentation_research/lineage.py`), ImageNet thay cho khởi tạo ngẫu nhiên, snapshot để chọn trên validation, seed, SimMIM chỉ dùng Z1. Test: `tests/test_legacy_training_fixes.py`.
- Kết quả test sạch (seed 42): Mask R-CNN 34,29; C2 26,47; C1 25,86; C0 24,92 AP.
- 3 seed (29/09): C0 24,81±0,58; C1 25,78±0,16; C2 25,81±0,58; Mask R-CNN 34,82±0,67. Target v2 có tác dụng (+0,96, dương ở cả 3 seed); lề target thì không (+0,04). Ablation đầu centroid (seed 42): C3 26,61; C4 26,18 (H8 bị bác bỏ); **C5 không có đầu centroid, tâm lấy từ SDT: 28,81 (+2,3)**. D (decoder dùng thứ hạng) làm validation giảm, H6 bị bác bỏ.
- Checkpoint đầy đủ + `--resume` cho hai trainer (`clean_benchmark/checkpointing.py`: optimizer, scheduler, RNG, generator của DataLoader, lịch sử); có test train liền mạch = train–lưu–resume. Ba run seed 42 đang chạy vẫn dùng định dạng cũ (chỉ trọng số): nếu phải dừng, resume sẽ là *gần đúng* và được ghi vào `resume.log`.

### Phát hiện quan trọng
- Sau khi sửa lỗi đa giác, Z3 R6: F1@50 **0,577** (trước 0,530), F1@75 0,255; đúng 1 tán 49%, gộp **27%**, sót **15%** (trước 29%). Guided filter **trung tính** (0,590 so với 0,589), không có hại như ghi ở Session 5.
- Oracle trên Z2 (sau khi sửa): bề mặt ngập là thành phần có tác động lớn nhất (F1@50 0,606 → 0,755; F1@75 0,239 → 0,581). Gộp chỉ hết khi cả seed và bề mặt ngập đều là oracle.
- Tường từ flow / gán theo điểm hút: F1@50 0,615 / 0,599 so với 0,607, tỉ lệ gộp vẫn 14–17%, tức flow hiện tại không mang tín hiệu ranh giới.
- Hai lỗi trong code đánh giá bắt được nhờ chạy thử trên checkpoint epoch 1: pycocotools `stats[0]` dùng maxDets = 100 cố định (trả về −1 khi maxDets = [1, 10, 400]); `loadRes` chèn kiểu numpy vào dict kết quả. Đã sửa và có test.

### Trạng thái cuối session
- C0 (`runs/c0_v1_s42`) train xong 30 epoch; đang chọn cấu hình trên val rồi chạy test một lần (`run_clean_eval.sh`).
- C1 (dừng ở epoch 8) và Mask R-CNN (dừng ở epoch 14) **bị kill theo yêu cầu** để train lại sau với checkpoint đầy đủ. Hai run dở dang lưu ở `runs/_killed_20260928/` (không dùng cho đánh giá).
- Train lại: `crown_segmentation_research/clean_benchmark/run_clean_training.sh` (bỏ qua run đã xong, tự resume run bị ngắt), sau đó `run_clean_eval.sh`.

---

## 2026-09-27 — Session 5: Trả lời phản biện của Thầy (Tuần 1) + sửa pipeline large-scale TreeFlowNet

### Việc đã làm
- Đọc và kiểm chứng tài liệu cho từng câu hỏi của Thầy: Cellpose, Omnipose, HoVer-Net, Freudenberg 2022, FG-TreeSeg 2026, TreeCoG 2026, Detectree2, SelvaBox, Teng et al. 2025, Allen et al. 2025, Huang et al. 2018, nnU-Net, U-Net, Hao et al. 2023 và các paper khác (danh sách trong tài liệu trả lời).
- Rà code và slide. Phát hiện: lỗi sigmoid hai lần trong script large-scale; decoder của ảnh 17 không phải decoder đã benchmark; số baseline Mask2Former/RSPrompter/YOLO chỉ là hằng số gõ tay; "Sobolev loss" trong slide không có trong code; "1 km²" thực ra là 0,042 km² và nằm trên vùng train.
- Viết pipeline `crown_segmentation_research/experiments/large_scale_v2/`: mỗi nguyên nhân là một flag, chấm GT ở mức raster, ghép tile theo luật sở hữu, cache đầu ra mạng. Kèm test `tests/test_large_scale_v2.py`.
- Chọn cấu hình trên Z2 (val), sau đó chạy test một lần trên Z3. *(Đính chính Session 6: Z2/Z3 nằm trong dữ liệu train của V1.)*
- Tài liệu trả lời Thầy: `meeting/week1/TRA_LOI_CAU_HOI_THAY_SAU_HOP_TUAN1.md`.

### Phát hiện quan trọng
- Ảnh 17 so với GT: F1@50 = 0,17; chỉ 20,7% số cây được khoanh đúng 1 tán; 82% dự đoán sai nằm trong tán đã có nhãn.
- Z3 test: F1@50 tăng từ 0,188 lên 0,530, F1@75 từ 0,052 lên 0,236; tỉ lệ cây bị xé giảm từ 32% còn 1%. Lỗi còn lại chủ yếu là gộp (22%) và sót (29%). *(Đính chính Session 6: số trước bản sửa lỗi đa giác; sau khi sửa là 0,577, gộp 27%, sót 15%.)*
- Với 30 bước Euler, 58% số tán chưa hội tụ. Ở 250 bước thì tán nào cũng hội tụ, nhưng 43% số tán bị flow chia thành ≥ 2 điểm hút, tức gốc rễ nằm ở head chứ không chỉ ở decoder.
- Khoảng cách seed 2,0 m (chọn trên Z2) cho kết quả *kém hơn* 0,83 m trên Z3, vì tán Z3 nhỏ hơn. Hằng số decoder không chuyển được giữa các lâm phần khác nhau.
- Độ nhạy theo tỉ lệ ảnh (Z3): F1 = 0,585 ở 1,0×; 0,335 ở 0,5×; 0,216 ở 1,5×. Model không bất biến theo tỉ lệ.
- Guided filter làm F1 giảm từ 0,538 xuống 0,457 trên Z2, nên đã bỏ. *(Đính chính Session 6: do lỗi đa giác; sau khi sửa thì trung tính.)*

### Bước tiếp theo
- Train lại với target theo đơn vị vật lý (SDT không chặn ở 15 px, σ centroid theo kích thước tán), patch ≥ 1024 px, augmentation đổi tỉ lệ; bán kính seed thích ứng. (Đính chính 28/09: DSM không giúp trên SBL theo Teng et al. 2025, nên không ưu tiên.)
- Train baseline Mask R-CNN trên cùng split (dùng AOI của Ramesh/Teng) để so sánh công bằng.

---

## 2026-09-17 — Session 4: Neuro-Algorithmic Reasoning & Track 2 Tree Crown Segmentation Focus

### Việc đã làm & Đột phá Lý thuyết
- Phân tích và tiếp thu toàn diện bài giảng & nghiên cứu của **Giáo sư Yusu Wang** (UCSD) về *Algorithmic Alignment, Size Generalization, $\varepsilon$-Kernels (NeurIPS 2025), và Spectral Graph Modularity*.
- Ứng dụng skill `/creative-thinking-for-research` (8 cognitive frameworks) để giải quyết các nút thắt cổ chai đã đo lường được ở v6/v7:
  - **Vấn đề v7**: Cắt giảm 51% `split_rate` nhờ embedding head nhưng tăng +15% `miss_rate`; chẩn đoán chỉ ra **70% ca miss là "blind spots" (prob < 0.01)** do sự sụp đổ của việc dò tâm đỉnh (peak finding) trong các vùng tán cây rậm rạp (`val:203`).
- Khóa ưu tiên chiến lược: **Tập trung toàn lực vào Track 2: Segmentation trên bộ dữ liệu DeadTrees** trước, hoàn thiện bài toán phân đoạn cây chết / standing deadwood / fallen logs trước khi tiến hành downstream classification.
- Phân tích thách thức hình học đặc thù của DeadTrees:
  - Thân cây đổ (fallen logs) có tỷ lệ chiều dài/rộng cực lớn ($L/W > 10:1$, chỉ dày 2-5 pixel) khiến biểu diễn tia tròn StarDist bị lãng phí $>85\%$ tia và gãy vụn.
  - Cây chết đứng (standing snags) là bộ xương cành trơ trụi không lồi (non-convex), vi phạm giả định star-convex.
  - Khối gỗ đổ đan xen ("pick-up sticks") làm sụp đổ hoàn toàn tâm điểm hình học và phép lọc NMS truyền thống.
- Soạn thảo và lưu trữ 2 bản thiết kế phương pháp v8 & v9:
  - `crown_segmentation_research/design_docs/method_design_neuro_algorithmic_crown_segmentation_v8.md` (Khung lý thuyết Neuro-Algorithmic tổng quát)
  - `crown_segmentation_research/design_docs/deadtrees_neuro_algorithmic_segmentation_v9.md` (Thiết kế chuyên sâu cho **DeadTrees**: $\varepsilon$-Kernel định hướng bất đẳng hướng, Bellman-Ford Medial Axis Tracing, và Spectral Modularity Cut cho thân cây đan xen).

---

## 2026-07-01 — Session 3: SECOND-OC Benchmark + Git Setup

### Việc đã làm
- Literature review 24 paper về object-level SCD, change captioning (2023-2025)
  - Đọc full-text: AnyChange (NeurIPS 2024), SECOND-CC (2025), RCD (2025), SCanNet
  - Kết quả: chốt novelty "exhaustive per-object enumeration trên 1.694 test pairs"
- Viết `plan/Second oc plan v2.md` (Plan v2) và `plan/ Second oc agent plan.md` (Agent Plan với runnable code)
- Phát hiện blocker: `tokens_T1_test/*.pt` không có key "masks", chỉ có features/centroids/areas
- Viết Phase -1 → Phase 1 pipeline scripts:
  - `generate_sam2_masks_test.py` — chạy SAM2 gen masks, lưu `SECOND/sam2_masks_T1_test/*.npz`
  - `SECOND-OC/config.py` — RGB→class mapping (7 màu SECOND), shared config
  - `SECOND-OC/sam2_loader.py` — load masks từ .npz
  - `SECOND-OC/phase0_verify.py` — verify 1694 pairs, check GT label format
  - `SECOND-OC/phase1_extract_instances.py` — SAM2 mask ∩ GT label → instance JSON
  - `SECOND-OC/phase1_visualize.py` — spot-check visualization

### Phát hiện quan trọng
- GT labels của SECOND là **RGB không phải grayscale**: 7 màu: tree=[0,128,0], buildings=[128,0,0], water=[0,0,255], non_veg=[128,128,128], playground=[255,255,255], low_veg=[0,255,0], other=[255,0,0]
- `_rgb_to_class()` cần exact lookup + nearest-color fallback (có trong code `run_multiscale_token_graph_reasoning.py:87`)
- SAM2 masks tiêu tốn ~1.5-2.2s/pair trên RTX 5880 Ada; ~60-90 phút cho 1694 pairs

### Trạng thái cuối session
- SAM2 mask generation: **đang chạy** (PID 2599468), ~97/1694 khi kết thúc session
- Monitor: `ls SECOND/sam2_masks_T1_test/ | wc -l`
- Log: `/tmp/sam2_masks_gen.log` (mất sau reboot)

### Bước tiếp theo
```bash
# Khi generation xong:
python3.11 SECOND-OC/phase0_verify.py
python3.11 SECOND-OC/phase1_extract_instances.py
python3.11 SECOND-OC/phase1_visualize.py   # view phase1_viz_*.png
```

---

## 2026-04-10 — Session 2: MOB-GCN + Phuong Dao Integration

### Việc đã làm
- Đọc và tích hợp 2 bài báo vào pipeline:
  - **MOB-GCN (2025)**: Multiresolution Graph Network với Gumbel-Softmax pooling
  - **Phuong D. Dao (ISPRS 2021)**: Inverse noise weighting (CV) + MAD outlier removal
- Viết `mob_gcn_modules.py`: `GumbelSoftmaxPool`, `MGNLayer`, `compute_smoothness_loss`
- Viết `token_hierarchical_reasoner.py`: `HierarchicalGraphReasoner`, `HierarchicalChangeReasoner`
- Chạy thực nghiệm multiscale graph vs baseline:
  - `run_multiscale_token_graph_reasoning.py`
  - Kết quả: Baseline F1=0.4763 → Multiscale F1=0.4880 (ΔIoU=+0.0101)

### Kết quả
- MOB-GCN thêm ΔIoU=+0.01 so với baseline graph — cải thiện nhỏ nhưng có ý nghĩa
- Expert specialization analysis: experts chuyên biệt theo **task axis** (change/stability), không phải class

### Bước tiếp theo ghi lại lúc đó
- Run longer training (30+ epochs) cho MOB-GCN
- Fix contrastive crash (disk full at step 1300/2968)
- Implement Rich Token (shape descriptor + spectral signature)

---

## 2026-04-10 — Session 1: Initial Setup + Training + Test Evaluation

### Việc đã làm
- Setup pipeline: SAM2 features → Tokenization → Hungarian Matching → Transformer+GNN+MoE
- Train model `stage5_6_semantic` (best checkpoint)
- Chạy test set evaluation lần đầu với `eval_test_set.py`

### Kết quả chính
```
Test set: 1694 pairs (official SECOND split)
F1     = 0.542
IoU    = 0.372
P      = 0.414   ← quá nhiều FP
R      = 0.784
TP=41,552  FP=58,765  FN=11,452  TN=199,221
```

### Phát hiện
- Token coverage: SAM2 chỉ cover 68.5% pixels, overlap 20%, có 25 giant masks
- Tokenization Projection Loss = 0.178 F1 (gap giữa token F1=0.542 → pixel F1=0.364)
- GNN > MoE routing — spatial context quan trọng hơn expert routing

### Bước tiếp theo ghi lại lúc đó
- Tìm hiểu SECOND GT labels để eval thật (không phải proxy labels)
- Oracle tokenizer experiment (dùng GT masks thay vì SAM2)

---

## Model Checkpoint Inventory

| Checkpoint | F1 (val) | F1 (test proxy) | Notes |
|---|---|---|---|
| `SECOND/stage5_6_semantic/best_model.pt` | 0.547 | 0.542 | Best overall, semantic routing |
| `SECOND/stage4/best_model.pt` | ~0.48 | — | Baseline transformer |
| `SECOND/stage4B/best_model.pt` | ~0.50 | — | +GNN |

## Key Files

| File | Vai trò |
|---|---|
| `tokenize_regions.py` | SAM2 tokenization (Stage 2) |
| `token_matching.py` | Hungarian matching (Stage 3) |
| `token_change_reasoner_moe.py` | MoE model architecture |
| `train_reasoner.py` | Training script |
| `eval_test_set.py` | Test set evaluation |
| `generate_sam2_masks_test.py` | Phase -1: gen masks for SECOND-OC |
| `SECOND-OC/` | Object-centric benchmark pipeline |
| `plan/` | Research plans (v2 + agent-executable) |


## 2026-11-02 — ITC independent identity–boundary review decision

[OBSERVED/INHERITED] Researcher keeps EVIDENCE_INSUFFICIENT_FOR_METHOD_SELECTION and pauses new model training. Established tested failure: hardownershipprojection discards useful overlapping hypotheses; no RGB-only solutionmechanism established. Small positive reference-conditionedRGBmembershipsignal is not biologicalidentity or downstreamsegmentationproof.

[OBSERVED] Specificreviewreadinesscheck: original12randompackets contain2allNoDatacases(case00/case03), bothA/Binvalid;10remaininghavevalidRGBandA/Bsupport. Prior wording thatall12were readyforreview was too broad. Originalsealedartifacts retained. Newversionedreviewpackage: research/RGB_ITC_REVIEW_PILOT_V1_20261102/START_HERE.md. Eligibility is not a humanassessmentofhardness or boundaryclarity.

[BLOCKED/UNVERIFIED] Nextdependency is twoindependent RGBreviews thenseparate expertadjudication, withuncertainty/abstention andStage1submissionslockedbeforemodel/referenceunblinding.0reviewers/0adjudications; no newtraining, no contacttoothers, noLiDARexperiment. No CCBR/backbone/jointmodel or privilegedtargetselected. Narrowedquestion: Which ambiguous crown identity/boundary decisions are resolvable from RGB, and how can competing hypotheses be retained until sufficient evidence exists?
