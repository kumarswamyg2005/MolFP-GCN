"""Build docs/MolFP-GCN_presentation.pptx. Run: uv run --no-project --with python-pptx python docs/video/make_pptx.py"""
import pathlib
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/figures"
NAVY, ACC, INK, MUTED, WHITE = (RGBColor(*c) for c in [(15, 23, 42), (14, 165, 233), (30, 41, 59), (100, 116, 139), (255, 255, 255)])
p = Presentation(); p.slide_width, p.slide_height = Inches(13.333), Inches(7.5)
blank = p.slide_layouts[6]

def box(s, x, y, w, h, fill=None, shape=MSO_SHAPE.RECTANGLE):
    r = s.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    r.line.fill.background()
    if fill: r.fill.solid(); r.fill.fore_color.rgb = fill
    else: r.fill.background()
    return r

def text(s, x, y, w, h, lines, size=22, color=INK, bold=False, align=None):
    tf = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap = True
    for i, ln in enumerate(lines if isinstance(lines, list) else [lines]):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = para.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold
        r.font.color.rgb = color; r.font.name = "Calibri"; para.space_after = Pt(10)
        if align: para.alignment = align

def slide(title, notes=""):
    s = p.slides.add_slide(blank)
    box(s, 0, 0, 13.333, 1.2, NAVY); box(s, 0, 1.2, 13.333, 0.07, ACC)
    text(s, 0.6, 0.25, 12, 0.8, title, 34, WHITE, True)
    text(s, 0.6, 7.0, 6, 0.4, "MolFP-GCN", 12, MUTED)
    s.notes_slide.notes_text_frame.text = notes
    return s

def pic(s, name, x, y, maxw, maxh):
    from PIL import Image
    w, h = Image.open(FIG / name).size; k = min(maxw / w, maxh / h)
    s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), Inches(w * k), Inches(h * k))

def cards(s, items, y=1.9, h=4.6):
    n = len(items); w = (12.1 - 0.4 * (n - 1)) / n
    for i, (head, body) in enumerate(items):
        x = 0.6 + i * (w + 0.4)
        box(s, x, y, w, h, RGBColor(241, 245, 249), MSO_SHAPE.ROUNDED_RECTANGLE)
        box(s, x, y, w, 0.12, ACC)
        text(s, x + 0.25, y + 0.35, w - 0.5, 0.9, head, 28, NAVY, True)
        text(s, x + 0.25, y + 1.3, w - 0.5, h - 1.5, body, 24)

# 1 title
s = p.slides.add_slide(blank); box(s, 0, 0, 13.333, 7.5, NAVY); box(s, 0.8, 3.55, 2.2, 0.08, ACC)
text(s, 0.8, 1.8, 11.5, 1.5, "MolFP-GCN", 66, WHITE, True)
text(s, 0.8, 3.8, 11.5, 1, "Learned molecular fingerprints with graph convolutional networks", 28, RGBColor(186, 230, 253))
text(s, 0.8, 5.6, 11.5, 1, ["Gurram Naga Kumaraswamy (S20230010101)", "Simon Peter (S20230010171)   ·   Rakesh (S20230010196)"], 18, RGBColor(203, 213, 225))

# 2 question
s = slide("The question", "Predict solubility and toxicity from structure, and show why.")
cards(s, [("Solubility", ["ESOL dataset", "1,128 molecules", "Regression on logS"]),
          ("Toxicity", ["Tox21 dataset", "7,823 molecules, 12 assays", "17% of labels missing, 7.5% toxic"]),
          ("Explanation", ["Which atoms drove the prediction?", "Saliency colours them in the demo app"])])

# 3 approach
s = slide("Two ways to describe a molecule", "Fixed ECFP vs learned GCN fingerprint.")
cards(s, [("Baseline: ECFP + RF", ["A fixed rule ticks 2048 bits", "Same fingerprint whatever the task", "Random forest on top"]),
          ("Ours: GCN", ["Atoms = nodes, bonds = edges", "3 rounds of neighbour averaging", "Pooled to a 256-number fingerprint, learned for the task"])], h=4.4)

# 4 evaluation
s = slide("Honest evaluation", "Scaffold split, 3 seeds.")
cards(s, [("Scaffold split", ["Whole families stay together", "Test = unseen structures", "80 / 10 / 10 train / val / test"]),
          ("3 seeds", ["Mean ± std reported", "Early stopping on validation", "Test set used once"]),
          ("Why it matters", ["Random split: RMSE 0.80", "Scaffold split: RMSE 1.06", "The gap is leakage, not skill"])])

# 5 results table
s = slide("Results", "GCN wins both; ESOL margin depends on test scaffolds.")
rows = [("Model", "ESOL RMSE ↓", "ESOL R² ↑", "Tox21 ROC-AUC ↑"),
        ("GCN (learned fingerprint)", "1.063 ± 0.163", "0.709 ± 0.048", "0.811 ± 0.009"),
        ("ECFP4 + random forest", "1.694 ± 0.498", "0.267 ± 0.248", "0.756 ± 0.015")]
t = s.shapes.add_table(3, 4, Inches(0.6), Inches(1.9), Inches(12.1), Inches(2.4)).table
for i, r in enumerate(rows):
    for j, v in enumerate(r):
        c = t.cell(i, j); c.text = v; f = c.text_frame.paragraphs[0].runs[0].font
        f.size = Pt(22); f.bold = i == 0 or (i == 1 and j > 0)
        c.fill.solid(); c.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(224, 242, 254) if i == 1 else RGBColor(241, 245, 249))
        f.color.rgb = WHITE if i == 0 else INK
text(s, 0.6, 4.8, 12, 1.8, ["GCN ahead on 11 of 12 Tox21 assays (RF ahead only on SR-ARE).",
     "Mean ± std over seeds 0, 1, 2 on a scaffold split."], 22)

# 6 figures
s = slide("What the model learned", "Parity plot and per-assay AUC.")
pic(s, "parity_esol.png", 0.5, 1.6, 6.2, 5.2); pic(s, "tox21_per_task.png", 6.9, 1.6, 6.0, 5.2)

# 7 demo
s = slide("Explainability and demo", "make demo opens the Streamlit app.")
pic(s, "demo_screenshot.png", 0.6, 1.5, 5.2, 5.4)
text(s, 6.3, 2.0, 6.4, 4.5, ["• Draw or paste a SMILES", "• Predicts solubility and 12 toxicity scores",
     "• Colours atoms that pushed a prediction up or down", "• Run with:  make demo", "• Or open ?smiles=CCO"], 24)

# 8 limits
s = slide("Limitations and next steps", "Be upfront about caveats.")
cards(s, [("Limitations", ["GCNConv ignores bond types", "ESOL is small: big seed-to-seed spread", "Scaffold split here may be easier than DeepChem's"]),
          ("Next steps", ["Edge-aware layers: GINE, D-MPNN", "More seeds / cross-validation", "Calibrate Tox21 probabilities"]),
          ("Reproducible", ["make all: 11 min on an M2 CPU", "Results identical byte for byte", "Tests + 12 invariants"])])

# 9 team
s = slide("Who did what", "Each member presents their own part.")
cards(s, [("Kumaraswamy", ["GCN architecture", "Featurisation: SMILES → graph", "Saliency explanations"]),
          ("Simon Peter", ["Data pipeline and checks", "Scaffold split", "Training, evaluation, reproducibility"]),
          ("Rakesh", ["ECFP + RF baseline", "Ablations and figures", "Demo app, limitations"])])

# 10 thanks
s = p.slides.add_slide(blank); box(s, 0, 0, 13.333, 7.5, NAVY)
text(s, 0, 2.6, 13.333, 1.2, "Thank you", 60, WHITE, True, PP_ALIGN.CENTER)
text(s, 0, 4.0, 13.333, 1, "Questions?", 28, RGBColor(186, 230, 253), False, PP_ALIGN.CENTER)
out = ROOT / "docs/MolFP-GCN_presentation.pptx"; p.save(out); print(out)
