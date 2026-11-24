#!/usr/bin/env python3
"""
Sync all research markdown files, scripts, images, and diagrams from meeting/week1/ 
into crown_segmentation_research/ so that ongoing research and development 
lives entirely inside crown_segmentation_research/.
"""
import os
import shutil
from pathlib import Path

BASE_DIR = Path("/home/chung/RS/Image Segmentation")
SRC_WEEK1 = BASE_DIR / "meeting" / "week1"
DEST_CSR = BASE_DIR / "crown_segmentation_research"

def main():
    print("=== SYNCING RESEARCH ARTIFACTS TO crown_segmentation_research ===")
    
    # 1. Markdown Research Documents -> crown_segmentation_research/docs/
    dest_docs = DEST_CSR / "docs"
    dest_docs.mkdir(parents=True, exist_ok=True)
    
    research_docs = [
        "NGHIEN_CUU_KIEN_TRUC_MODEL_BOUNDARY_BOTTLENECK.md",
        "NGHIEN_CUU_SOTA_BOUNDARY_PAPERS_VA_THIET_KE_TREEFLOWNET_HR.md",
        "NGHIEN_CUU_HAU_XU_LY_SEGMENTATION_DEP.md",
        "GIAI_THICH_THIET_KE_KIEN_TRUC_TREEFLOWNET.md",
        "QUY_TRINH_THUAT_TOAN_SEGMENTATION.md",
        "GIAI_DAP_TOAN_DIEN_Y_KIEN_THAY.md",
        "BO_CAU_HOI_PHAN_BIEN_VA_TRA_LOI.md",
        "EXECUTIVE_SUMMARY.md",
        "BAO_CAO_TIEN_DO_WEEK1.md"
    ]
    
    print("\n--- Copying Research Markdown Documents to docs/ ---")
    for doc in research_docs:
        src_file = SRC_WEEK1 / doc
        if src_file.exists():
            dest_file = dest_docs / doc
            shutil.copy2(src_file, dest_file)
            print(f"  [DOC] {doc} -> crown_segmentation_research/docs/{doc}")
        else:
            print(f"  [SKIP] Not found: {doc}")

    # 2. Python Scripts -> crown_segmentation_research/scripts/
    dest_scripts = DEST_CSR / "scripts"
    dest_scripts.mkdir(parents=True, exist_ok=True)
    
    scripts = [
        "generate_large_scale_workflow_figure.py",
        "generate_step_by_step_visuals.py",
        "build_meeting_docx.py"
    ]
    
    print("\n--- Copying Scripts to scripts/ ---")
    for script in scripts:
        src_file = SRC_WEEK1 / script
        if src_file.exists():
            dest_file = dest_scripts / script
            shutil.copy2(src_file, dest_file)
            print(f"  [SCRIPT] {script} -> crown_segmentation_research/scripts/{script}")
        else:
            print(f"  [SKIP] Not found: {script}")

    # 3. Architecture Diagrams & TeX -> crown_segmentation_research/images/ and crown_segmentation_research/paper/
    dest_images = DEST_CSR / "images"
    dest_images.mkdir(parents=True, exist_ok=True)
    
    arch_files = [
        "treeflownet_architecture_paper_style.png",
        "treeflownet_architecture_paper_style.pdf",
        "treeflownet_architecture_paper_style.tex",
        "treeflownet_architecture_diagram.png",
        "treeflownet_architecture.pdf",
        "treeflownet_architecture.tex",
        "treeflownet_architecture_gemini.jpg"
    ]
    
    print("\n--- Copying Architecture Visuals to images/ ---")
    for f in arch_files:
        src_file = SRC_WEEK1 / f
        if src_file.exists():
            dest_file = dest_images / f
            shutil.copy2(src_file, dest_file)
            print(f"  [ARCH] {f} -> crown_segmentation_research/images/{f}")

    # 4. Experiment & Workflow Images -> crown_segmentation_research/images/
    src_images = SRC_WEEK1 / "images"
    if src_images.exists():
        print("\n--- Copying Experiment Images to images/ ---")
        for img_file in sorted(src_images.glob("*.png")):
            dest_file = dest_images / img_file.name
            shutil.copy2(img_file, dest_file)
            print(f"  [IMG] {img_file.name} -> crown_segmentation_research/images/{img_file.name}")

    # 5. Verify Papers in crown_segmentation_research/papers/
    dest_papers = DEST_CSR / "papers"
    dest_papers.mkdir(parents=True, exist_ok=True)
    src_papers = SRC_WEEK1 / "papers"
    if src_papers.exists():
        print("\n--- Syncing SOTA Papers to papers/ ---")
        for p in sorted(src_papers.glob("*.pdf")):
            dest_file = dest_papers / p.name
            if not dest_file.exists() or dest_file.stat().st_size != p.stat().st_size:
                shutil.copy2(p, dest_file)
                print(f"  [PAPER COPY] {p.name}")
            else:
                print(f"  [PAPER OK] {p.name} (already present)")

    print("\n=== SYNC COMPLETE! All active research is now inside crown_segmentation_research/ ===")

if __name__ == "__main__":
    main()
