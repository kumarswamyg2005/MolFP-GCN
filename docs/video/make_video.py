"""Build docs/video/MolFP-GCN_explained.mp4: PIL slides + macOS `say` voice + ffmpeg."""
import subprocess, pathlib
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/figures"
OUT = pathlib.Path(__file__).parent
TMP = OUT / "_build"; TMP.mkdir(exist_ok=True)
W, H = 1920, 1080
BG, FG, ACC, MUTED = (15, 23, 42), (241, 245, 249), (56, 189, 248), (148, 163, 184)
F = "/System/Library/Fonts/Helvetica.ttc"
font = lambda s, b=False: ImageFont.truetype(F, s, index=1 if b else 0)

# (title, bullets, image or None, narration)
SLIDES = [
 ("MolFP-GCN", ["Learned molecular fingerprints with graph convolutional networks",
   "Gurram Naga Kumaraswamy  ·  Simon Peter  ·  Rakesh"], None,
  "Hello. This is MolFP-GCN: learned molecular fingerprints with graph convolutional networks. "
  "We are Kumaraswamy, Simon Peter and Rakesh, and in the next few minutes we will explain what we built and what we found."),
 ("The question", ["Can a computer predict a molecule's properties from its structure?",
   "1. Solubility in water (ESOL)", "2. Toxicity in 12 lab tests (Tox21)",
   "And can it show WHICH atoms drove the prediction?"], None,
  "The question is simple. Given only the structure of a molecule, can a computer predict how well it dissolves in water, "
  "and whether it is toxic in twelve different lab tests? And when it makes a guess, can it show us which atoms it was looking at?"),
 ("Two ways to turn a molecule into numbers", ["OLD: ECFP fingerprint — a fixed rule ticks 2048 boxes, same for every task. Fed to a random forest. This is our baseline.",
   "OURS: a GCN — atoms are nodes, bonds are edges. Each atom repeatedly averages its neighbours.",
   "After 3 rounds, each atom knows its 3-bond neighbourhood.",
   "Pooled into 256 numbers: a fingerprint learned for the task."], None,
  "A computer cannot learn from text, so a molecule must become numbers. The old way is the ECFP fingerprint. A fixed rule ticks boxes in a 2048 long checklist, "
  "and it is the same whatever you want to predict. We feed it to a random forest, and that is our baseline. "
  "Our way treats the molecule as a graph. Atoms are nodes, bonds are edges. In each round every atom averages information from its neighbours. After three rounds, each atom knows about everything within three bonds. "
  "We pool the atoms into a vector of 256 numbers. That is a learned fingerprint, shaped by the task."),
 ("Honest evaluation", ["Scaffold split: whole families of similar molecules stay together",
   "Train 80%  ·  Validation 10%  ·  Test 10% (used once)",
   "3 seeds, mean ± std reported",
   "Random split gives RMSE 0.80 — that gain is leakage, not skill",
   "Tox21: missing labels masked, rare toxic labels up-weighted"], None,
  "Evaluation has to be fair. We use a scaffold split, so that families of near-identical molecules stay together, and the test set contains structures the model has never seen. "
  "We run three seeds and report the mean and spread. When we tried a random split, the error dropped from one point oh six to zero point eight. That looks better, but it is leakage. "
  "For Tox21 we ignore missing labels instead of treating them as safe, and we give extra weight to the rare toxic labels."),
 ("Results", ["GCN: ESOL RMSE 1.063, R² 0.709, Tox21 ROC-AUC 0.811",
   "ECFP + RF: ESOL RMSE 1.694, R² 0.267, Tox21 ROC-AUC 0.756",
   "GCN wins both tasks, and 11 of 12 Tox21 assays"], "parity_esol.png",
  "Here are the results. On ESOL the GCN reaches an RMSE of one point oh six three against one point six nine for the random forest. "
  "On Tox21 the mean ROC-AUC is zero point eight one one against zero point seven five six, and the GCN is ahead on eleven of the twelve assays. "
  "The ESOL margin varies between seeds, because it depends on which scaffolds land in the test set, and we say so openly in the report."),
 ("Explainability and the demo", ["Saliency colours the atoms that pushed a prediction up or down",
   "Streamlit app: draw a molecule, get both predictions",
   "Run it with: make demo"], "demo_screenshot.png",
  "Finally, explainability. Using saliency, the app colours the atoms that pushed a prediction up or down. "
  "It is a small Streamlit app. You draw a molecule, it predicts solubility and toxicity, and highlights the responsible atoms. Run it with make demo."),
 ("Limitations and next steps", ["GCNConv ignores bond types — next: GINE or D-MPNN",
   "ESOL is small (~1,100 molecules): large seed-to-seed spread",
   "Everything reproduces with: make all (11 min on an M2 CPU)"], None,
  "There are limits. The GCN layer ignores bond types, so an edge-aware layer is the natural next step. ESOL is small, so results vary between seeds. "
  "Everything is reproducible with a single make all command, which takes about eleven minutes on a laptop."),
 ("Who did what", ["Kumaraswamy — GCN architecture, featurisation, saliency",
   "Simon Peter — data pipeline, scaffold split, training, evaluation",
   "Rakesh — ECFP + RF baseline, ablations, demo app, limitations"], None,
  "The work was divided in three. Kumaraswamy built the GCN model, the featurisation from SMILES to graphs, and the saliency explanations. "
  "Simon Peter built the data pipeline, the scaffold split, the training loop and the evaluation. "
  "Rakesh built the random forest baseline, ran the ablations, and made the demo app. Thank you."),
]

def wrap(d, text, f, width):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width: cur = t
        else: lines.append(cur); cur = w
    return lines + [cur]

def render(i, title, bullets, img):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 16, H], fill=ACC)
    d.text((90, 70), title, font=font(78, True), fill=FG)
    tw = 1000 if img else 1740
    y = 230 if i else 330
    for b in bullets:
        for j, ln in enumerate(wrap(d, b, font(44), tw)):
            d.text((90 + (0 if j == 0 else 30), y), ("•  " if j == 0 and i else "") + ln, font=font(44), fill=FG if i else ACC)
            y += 62
        y += 24
    if img:
        p = Image.open(FIG / img).convert("RGBA"); p.thumbnail((760, 800))
        bg = Image.new("RGBA", p.size, (255, 255, 255, 255)); bg.alpha_composite(p)
        im.paste(bg.convert("RGB"), (1100, 230))
    d.text((90, H - 60), "MolFP-GCN  ·  slide %d/%d" % (i + 1, len(SLIDES)), font=font(26), fill=MUTED)
    im.save(TMP / f"s{i}.png")

clips = []
for i, (t, b, img, narr) in enumerate(SLIDES):
    render(i, t, b, img)
    aiff, mp4 = TMP / f"s{i}.aiff", TMP / f"s{i}.mp4"
    subprocess.run(["say", "-v", "Samantha", "-r", "165", "-o", str(aiff), narr], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(TMP / f"s{i}.png"), "-i", str(aiff),
                    "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p", "-r", "24",
                    "-c:a", "aac", "-b:a", "160k", "-af", "apad=pad_dur=0.8", "-shortest", str(mp4)], check=True)
    clips.append(mp4)
lst = TMP / "list.txt"; lst.write_text("".join(f"file '{c}'\n" for c in clips))
out = OUT / "MolFP-GCN_explained.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
print(out)
