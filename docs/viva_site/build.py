"""Build index.html from ../viva_scripts/*.md. Run: python build.py (needs `markdown`)."""
import html, re
from pathlib import Path
import markdown

SRC = Path(__file__).resolve().parent.parent / "viva_scripts"
# tab id, label, file, member
TABS = [
    ("overview", "Overview (all 3)", "00_OVERVIEW.md"),
    ("kumaraswamy", "Kumaraswamy", "01_KUMARASWAMY.md"),
    ("simon", "Simon Peter", "02_SIMON_PETER.md"),
    ("rakesh", "Rakesh", "03_RAKESH.md"),
    ("cheat", "Cheat sheet", "04_CHEAT_SHEET.md"),
]
# (tab, heading prefix) -> [(image, caption)], figure goes right after that heading
FIGS = {
    ("kumaraswamy", "1.1"): [("tsne_pair.png", "t-SNE of the learned 256-number fingerprints: molecules with similar properties cluster together.")],
    ("kumaraswamy", "3.3"): [("saliency_examples.png", "Saliency maps: red atoms push the prediction up, blue push it down.")],
    ("simon", "2.3"): [("ablations_esol.png", "Random split looks better than scaffold split. That gap is leakage, not skill.")],
    ("simon", "3.2"): [("training_curves.png", "Training and validation curves; early stopping keeps the best epoch.")],
    ("simon", "4.2"): [("parity_esol.png", "ESOL predicted vs measured logS on the test set (GCN vs random forest).")],
    ("simon", "4.4"): [("tox21_per_task.png", "Tox21 ROC-AUC per assay: GCN is ahead on 11 of 12.")],
    ("rakesh", "2.2"): [("depth_ablation.png", "Depth ablation: 1 to 4 layers are all within noise.")],
    ("rakesh", "3.1"): [
        ("demo_screenshot.png", "The demo app: draw a molecule, read both predictions, see atoms coloured by saliency."),
        ("1_empty.png", "Screen 1: empty start state."),
        ("2_aspirin.png", "Screen 2: aspirin, ESOL prediction plus saliency."),
        ("3_tox21_positive.png", "Screen 3: a Tox21-positive molecule with per-assay scores."),
        ("4_invalid.png", "Screen 4: an invalid SMILES gives a friendly error, not a crash."),
    ],
}

def fig(items):
    return "".join(
        f'<figure><a href="img/{i}" target="_blank"><img src="img/{i}" alt="{html.escape(c)}" loading="lazy"></a>'
        f"<figcaption>{html.escape(c)}</figcaption></figure>" for i, c in items)

def render(tab, text):
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: f'<div class="eq">{html.escape(m.group(1).strip())}</div>', text, flags=re.S)
    out = markdown.markdown(text, extensions=["fenced_code", "tables", "sane_lists"])
    out = re.sub(r"(\.md)\)?", r"\1", out)  # keep names readable
    out = re.sub(r'<a href="[^"]*\.md">', "<a href=\"#\">", out)
    for (t, pre), items in FIGS.items():
        if t == tab:
            out, n = re.subn(rf"(<h2>\s*{re.escape(pre)}\s.*?</h2>)", lambda m: m.group(1) + f'<div class="figs">{fig(items)}</div>', out, count=1)
            assert n == 1, (tab, pre)
    return out

nav = "".join(f'<a href="#{t}" data-tab="{t}">{l}</a>' for t, l, _ in TABS)
secs = "".join(f'<section id="{t}">{render(t, (SRC / f).read_text())}</section>' for t, l, f in TABS)
Path(__file__).with_name("index.html").write_text(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>MolFP-GCN Viva Guide</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d2327;--mut:#5d6b73;--card:#fff;--line:#dfe3e6;--acc:#0b6e69;--code:#f0f2f3}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14181a;--fg:#e6eaec;--mut:#9aa8b0;--card:#1c2225;--line:#2d373c;--acc:#4fd1c5;--code:#232b2f}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 system-ui,sans-serif}}
header{{padding:28px 16px 12px;max-width:900px;margin:auto}}header h1{{margin:0;font-size:1.6rem}}header p{{margin:.3rem 0;color:var(--mut)}}
nav{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);z-index:5}}
nav div{{max-width:900px;margin:auto;padding:8px 16px;display:flex;gap:6px;overflow-x:auto}}
nav a{{white-space:nowrap;padding:6px 12px;border-radius:99px;border:1px solid var(--line);color:var(--fg);text-decoration:none;font-size:.9rem}}
nav a.on{{background:var(--acc);color:var(--bg);border-color:var(--acc)}}
main{{max-width:900px;margin:auto;padding:8px 16px 80px}}section{{display:none}}section.on{{display:block}}
h1{{margin-top:1.6em}}h2{{margin-top:2em;border-bottom:1px solid var(--line);padding-bottom:.2em}}
pre{{background:var(--code);padding:12px;border-radius:8px;overflow-x:auto;font-size:.85rem;line-height:1.45}}
code{{background:var(--code);padding:1px 5px;border-radius:4px;font-size:.88em}}pre code{{padding:0;background:none}}
table{{border-collapse:collapse;display:block;overflow-x:auto;margin:1em 0}}th,td{{border:1px solid var(--line);padding:6px 10px;text-align:left}}th{{background:var(--code)}}
blockquote{{margin:1em 0;padding:.4em 1em;border-left:4px solid var(--acc);background:var(--card);color:var(--fg)}}
.eq{{font:1.05rem ui-monospace,monospace;background:var(--card);border:1px solid var(--line);padding:14px;border-radius:8px;text-align:center;overflow-x:auto}}
.figs{{display:grid;gap:14px;margin:14px 0}}figure{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px}}
figure img{{width:100%;height:auto;border-radius:6px;display:block}}figcaption{{color:var(--mut);font-size:.88rem;margin-top:6px}}
</style></head><body>
<header><h1>MolFP-GCN: viva guide</h1>
<p>Kumaraswamy (S20230010101) · Simon Peter (S20230010171) · Rakesh (S20230010196)</p>
<p>Learned graph fingerprints vs ECFP + random forest on ESOL and Tox21. Read Overview first, then your own tab.</p></header>
<nav><div>{nav}</div></nav><main>{secs}</main>
<script>
const tabs=[...document.querySelectorAll('nav a')],secs=[...document.querySelectorAll('section')];
function show(){{const t=(location.hash||'#overview').slice(1);const ok=secs.some(s=>s.id===t)?t:'overview';
secs.forEach(s=>s.classList.toggle('on',s.id===ok));tabs.forEach(a=>a.classList.toggle('on',a.dataset.tab===ok));scrollTo(0,0)}}
addEventListener('hashchange',show);show();
</script></body></html>""")
