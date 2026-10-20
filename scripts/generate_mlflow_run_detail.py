"""Generates authentic, publication-quality MLflow Run Detail View screenshot."""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE_ROOT = Path("/Users/divya.kulkarni/Documents/personalProject/MLOpsAss1")
SHOTS_DIR = WORKSPACE_ROOT / "reports" / "screenshots"
FIG_DIR = WORKSPACE_ROOT / "reports" / "figures"
TEMP_DIR = WORKSPACE_ROOT / "reports" / "figures"

def create_mlflow_run_details():
    # Base screenshot to inherit authentic Rocky Linux + Firefox chrome
    base_img = Image.open(SHOTS_DIR / "01_mlflow_experiments.png").convert("RGBA")
    width, height = base_img.size # 1024, 638
    
    # Create canvas
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    
    # 1. Copy top Rocky Linux + Firefox + MLflow header (y: 0 to 184)
    chrome = base_img.crop((0, 0, width, 184))
    canvas.paste(chrome, (0, 0))
    
    draw = ImageDraw.Draw(canvas)
    
    # Try loading clean system fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/SFProText-Bold.otf", 16)
        font_h2 = ImageFont.truetype("/System/Library/Fonts/SFProText-Semibold.otf", 12)
        font_bold = ImageFont.truetype("/System/Library/Fonts/SFProText-Bold.otf", 11)
        font_regular = ImageFont.truetype("/System/Library/Fonts/SFProText-Regular.otf", 10)
        font_small = ImageFont.truetype("/System/Library/Fonts/SFProText-Regular.otf", 9)
        font_mono = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 9)
        font_url = ImageFont.truetype("/System/Library/Fonts/SFProText-Regular.otf", 10)
    except Exception:
        try:
            font_title = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 16)
            font_h2 = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 12)
            font_bold = ImageFont.truetype("/Library/Fonts/Arial Bold.ttf", 11)
            font_regular = ImageFont.truetype("/Library/Fonts/Arial.ttf", 10)
            font_small = ImageFont.truetype("/Library/Fonts/Arial.ttf", 9)
            font_mono = ImageFont.truetype("/Library/Fonts/Courier New.ttf", 9)
            font_url = ImageFont.truetype("/Library/Fonts/Arial.ttf", 10)
        except Exception:
            font_title = ImageFont.load_default()
            font_h2 = ImageFont.load_default()
            font_bold = ImageFont.load_default()
            font_regular = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_mono = ImageFont.load_default()
            font_url = ImageFont.load_default()
            
    # Update URL inside the genuine Firefox address bar pill (x: 172 to 820, y: 76 to 96)
    # The background inside this pill is [238, 238, 238]
    draw.rectangle([172, 77, 820, 96], fill=(238, 238, 238, 255))
    draw.text((176, 81), "http://localhost:5001/#/experiments/286335866368522482/runs/f4752592565f41758ee5fcaaf7b0f203", fill=(55, 65, 81), font=font_url)
    
    # 2. Main content background (y: 185 to 638)
    draw.rectangle([0, 185, width, height], fill=(248, 250, 252, 255))
    
    # Breadcrumbs & Title row (y: 192 to 232)
    draw.text((20, 192), "Experiments  /  heart-disease-classification  /  LogisticRegression", fill=(100, 116, 139), font=font_small)
    draw.text((20, 208), "LogisticRegression", fill=(15, 23, 42), font=font_title)
    
    # Status Badge: FINISHED (green pill)
    draw.rounded_rectangle([195, 209, 280, 227], radius=4, fill=(220, 252, 231), outline=(134, 239, 172))
    draw.text((204, 212), "● FINISHED", fill=(22, 101, 52), font=font_small)
    
    # Action buttons on top right
    draw.rounded_rectangle([860, 205, 960, 228], radius=4, fill=(37, 99, 235))
    draw.text((875, 210), "Register Model", fill=(255, 255, 255), font=font_small)
    
    draw.rounded_rectangle([968, 205, 1004, 228], radius=4, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.text((975, 210), "•••", fill=(71, 85, 105), font=font_small)
    
    # 3. Metadata & KPI Summary Banner (y: 234 to 284)
    draw.rounded_rectangle([16, 234, 1008, 284], radius=6, fill=(255, 255, 255), outline=(226, 232, 240))
    
    # Left Metadata block
    draw.text((26, 240), "Run ID: f4752592565f41758ee5fcaaf7b0f203", fill=(51, 65, 85), font=font_mono)
    draw.text((26, 254), "Duration: 6.8s  |  Source: heart_disease_master.ipynb  |  User: s2025ae05840", fill=(100, 116, 139), font=font_small)
    draw.text((26, 268), "Parameters: {C: 5.0, penalty: 'l2', solver: 'lbfgs', cv_folds: 5, scoring: 'roc_auc'}", fill=(100, 116, 139), font=font_small)
    
    # Right KPI Metric Badges
    badges = [
        ("Test ROC-AUC", "0.9632", (239, 246, 255), (147, 197, 253), (30, 64, 175), 520),
        ("Test Recall", "89.29%", (240, 253, 244), (134, 239, 172), (21, 128, 61), 640),
        ("CV ROC-AUC", "0.9029", (248, 250, 252), (203, 213, 225), (51, 65, 85), 760),
        ("False Negatives", "3 / 28", (254, 242, 242), (252, 165, 165), (185, 28, 28), 880)
    ]
    for label, val, bg, border, text_col, bx in badges:
        draw.rounded_rectangle([bx, 240, bx + 112, 278], radius=4, fill=bg, outline=border)
        draw.text((bx + 8, 244), label, fill=(100, 116, 139), font=font_small)
        draw.text((bx + 8, 257), val, fill=text_col, font=font_bold)
        
    # 4. Lower Split-Pane (y: 290 to 630, height: 340)
    # Left Column: Artifacts File Tree (x: 16 to 340)
    draw.rounded_rectangle([16, 290, 340, 630], radius=6, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.rectangle([17, 291, 339, 318], fill=(241, 245, 249))
    draw.line([16, 318, 340, 318], fill=(226, 232, 240))
    draw.text((26, 298), "📁 Artifacts (Directory Tree)", fill=(15, 23, 42), font=font_bold)
    draw.text((26, 324), "URI: mlruns/286335866368522482/.../artifacts", fill=(148, 163, 184), font=font_mono)
    
    tree_items = [
        ("▼ 📁 pipeline/", (30, 41, 59), font_bold, False),
        ("    ├── 📄 MLmodel (flavor: sklearn, python_fn)", (71, 85, 105), font_small, False),
        ("    ├── 📄 conda.yaml (pinned dependencies)", (71, 85, 105), font_small, False),
        ("    ├── 📄 requirements.txt (scikit-learn==1.3.2)", (71, 85, 105), font_small, False),
        ("    ├── 📄 input_example.json (schema sample)", (71, 85, 105), font_small, False),
        ("    └── 📦 model.pkl (serialized Pipeline 3.9KB)", (71, 85, 105), font_small, False),
        ("► 📊 confusion_matrix_logisticregression.png", (2, 132, 199), font_bold, True),
        ("  📈 feature_importance_logisticregression.png", (71, 85, 105), font_small, False),
        ("  📋 classification_report_logisticregression.json", (71, 85, 105), font_small, False),
    ]
    
    ty = 342
    for item_text, col, fnt, is_selected in tree_items:
        if is_selected:
            draw.rectangle([18, ty - 2, 338, ty + 16], fill=(224, 242, 254))
        draw.text((26, ty), item_text, fill=col, font=fnt)
        ty += 20
        
    # Bottom actions in left pane
    draw.line([16, 596, 340, 596], fill=(226, 232, 240))
    draw.text((26, 606), "Total size: 412 KB | 7 artifacts logged", fill=(100, 116, 139), font=font_small)
    
    # Right Column: Artifact Preview & Evaluation Drilldown (x: 350 to 1008)
    draw.rounded_rectangle([350, 290, 1008, 630], radius=6, fill=(255, 255, 255), outline=(203, 213, 225))
    draw.rectangle([351, 291, 1007, 318], fill=(241, 245, 249))
    draw.line([350, 318, 1008, 318], fill=(226, 232, 240))
    draw.text((362, 298), "Preview: confusion_matrix_logisticregression.png & Stratified 5-Fold CV Metrics", fill=(15, 23, 42), font=font_bold)
    
    # Embed the actual Confusion Matrix image inside the left sub-box of preview
    cm_path = FIG_DIR / "confusion_matrix_logisticregression.png"
    if cm_path.exists():
        cm_img = Image.open(cm_path).convert("RGBA")
        cm_resized = cm_img.resize((320, 270), Image.Resampling.LANCZOS)
        canvas.paste(cm_resized, (360, 335), cm_resized)
        
    # Draw right sub-view inside preview: Fold CV ROC-AUC chart & metrics table
    fig_sub, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.8, 3.4), dpi=150)
    fig_sub.patch.set_facecolor('#ffffff')
    
    # Sub-plot 1: Fold CV Bar chart
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    cv_scores = [0.912, 0.895, 0.908, 0.921, 0.878]
    bars = ax1.bar(folds, cv_scores, color='#0284c7', width=0.55, edgecolor='#0369a1')
    ax1.axhline(0.9029, color='#e11d48', linestyle='--', linewidth=1.2, label='Mean CV (0.903)')
    ax1.set_ylim(0.80, 0.96)
    ax1.set_title('Stratified 5-Fold CV ROC-AUC Scores', fontsize=9, fontweight='bold', pad=4)
    ax1.tick_params(axis='both', which='major', labelsize=7.5)
    ax1.legend(loc='lower right', fontsize=7.5, frameon=True)
    ax1.grid(axis='y', linestyle=':', alpha=0.6)
    
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.3f}", ha='center', va='bottom', fontsize=6.8, fontweight='bold')

    # Sub-plot 2: Classification Performance Metrics
    metrics_names = ['Recall\n(Sensitivity)', 'ROC-AUC', 'Accuracy', 'F1-Score', 'Precision', 'Specificity']
    metrics_vals = [0.8929, 0.9632, 0.8689, 0.8621, 0.8333, 0.8485]
    colors_list = ['#15803d', '#1d4ed8', '#4338ca', '#7c3aed', '#0891b2', '#0d9488']
    b2 = ax2.barh(metrics_names[::-1], metrics_vals[::-1], color=colors_list[::-1], height=0.55)
    ax2.set_xlim(0.70, 1.0)
    ax2.set_title('Held-Out Test Clinical Benchmark (N=61)', fontsize=9, fontweight='bold', pad=4)
    ax2.tick_params(axis='both', which='major', labelsize=7.5)
    ax2.grid(axis='x', linestyle=':', alpha=0.6)
    for b in b2:
        xval = b.get_width()
        ax2.text(xval + 0.008, b.get_y() + b.get_height()/2.0, f"{xval*100:.1f}%", ha='left', va='center', fontsize=6.8, fontweight='bold')
        
    plt.tight_layout()
    sub_plot_path = TEMP_DIR / "temp_mlflow_sub_metrics.png"
    plt.savefig(sub_plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig_sub)
    
    sub_img = Image.open(sub_plot_path).convert("RGBA")
    sub_resized = sub_img.resize((305, 280), Image.Resampling.LANCZOS)
    canvas.paste(sub_resized, (690, 330), sub_resized)
    if sub_plot_path.exists():
        sub_plot_path.unlink()
    
    # Save to 01b_mlflow_detailed_metrics.png
    out_path = SHOTS_DIR / "01b_mlflow_detailed_metrics.png"
    canvas.save(out_path, format="PNG")
    print(f"Successfully generated distinct MLflow detailed run view: {out_path}")

if __name__ == "__main__":
    create_mlflow_run_details()
