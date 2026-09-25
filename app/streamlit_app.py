"""MolFP-GCN demo: SMILES -> drawing, solubility, 12 Tox21 probabilities, atom saliency,
nearest training molecules. Layout and copy follow docs/DESIGN.md.

    make demo   (= streamlit run app/streamlit_app.py)

No training, no uploads, no network calls. SVG shown on the page comes only from RDKit;
user text is always HTML-escaped.
"""

import base64
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # `streamlit run app/...` does not put the repo root on sys.path

import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from molfp.data import TOX21_TASKS, load_dataset, load_graphs  # noqa: E402
from molfp.explain import draw_svg, saliency  # noqa: E402
from molfp.featurize import mol_to_data, parse_smiles  # noqa: E402
from molfp.invariants import MAX_HEAVY_ATOMS, MAX_SMILES_CHARS  # noqa: E402
from molfp.split import get_split  # noqa: E402
from molfp.train import DEVICE, RUNS, load_model  # noqa: E402

PRESETS = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Benzene": "c1ccccc1",
    "Ethanol": "CCO",
    "Nicotine": "CN1CCC[C@H]1c1cccnc1",
    # 4-nitrosodiphenylamine: Tox21 seed-0 TEST molecule, active in 8 of its 10 labelled
    # assays (python -m scripts.pick_tox21_preset); the demo model never trained on it
    "Tox21 positive (test set)": "O=Nc1ccc(Nc2ccccc2)cc1",
}

COPY = {
    "invalid": "Couldn't read that SMILES. Check brackets and ring numbers, e.g. c1ccccc1.",
    "too_large": "Molecule too large for the demo (limit 100 heavy atoms).",
    "too_long": "SMILES too long for the demo (limit 200 characters).",
    "empty": "Enter a SMILES string or pick a molecule to start.",
    "no_model": "No trained model found. Run `make train` first.",
    "footer": "Research demo. Not for safety or medical decisions.",
}


def validate(smiles):
    """Return (mol, None) or (None, message). Length is checked before parsing."""
    smiles = (smiles or "").strip()
    if not smiles:
        return None, COPY["empty"]
    if len(smiles) > MAX_SMILES_CHARS:
        return None, COPY["too_long"]
    mol, _ = parse_smiles(smiles)
    if mol is None:
        return None, COPY["invalid"]
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return None, COPY["too_large"]
    return mol, None


def solubility_band(logs):
    if logs >= -2:
        return "soluble"
    return "moderately soluble" if logs >= -4 else "poorly soluble"


@st.cache_resource
def load_models():
    paths = {ds: RUNS / ds / "gcn" / "seed0" / "best.pt" for ds in ("esol", "tox21")}
    if not all(p.exists() for p in paths.values()):
        return None
    models = {ds: load_model(p) for ds, p in paths.items()}

    # learned fingerprints of the ESOL seed-0 training molecules, for nearest neighbours
    smiles, y, _ = load_dataset("esol")
    split = get_split("esol", 0)
    if models["esol"][1]["split_hash"] != split["hash"]:
        raise RuntimeError("ESOL checkpoint was trained on a different split; rerun `make train`")
    tr = split["train"]
    graphs = load_graphs("esol")
    model = models["esol"][0]
    with torch.no_grad():
        emb = torch.cat(
            [
                model.embed(b.to(DEVICE)).cpu()
                for b in DataLoader([graphs[i] for i in tr], batch_size=256)
            ]
        )
    emb = torch.nn.functional.normalize(emb, dim=1).numpy()
    return models, emb, [smiles[i] for i in tr], y[tr, 0]


@st.cache_data
def fonts_css():
    fonts = ROOT / "app" / "static" / "fonts"
    faces = [
        ("Atkinson Hyperlegible", 400, "atkinson-400"),
        ("Atkinson Hyperlegible", 700, "atkinson-700"),
        ("IBM Plex Mono", 400, "plexmono-400"),
    ]
    css = ""
    for family, weight, name in faces:
        b64 = base64.b64encode((fonts / f"{name}.woff2").read_bytes()).decode()
        css += (
            f"@font-face{{font-family:'{family}';font-weight:{weight};font-display:swap;"
            f"src:url(data:font/woff2;base64,{b64}) format('woff2');}}\n"
        )
    return css + (ROOT / "app" / "style.css").read_text()


def svg_html(svg):
    return svg[svg.find("<svg") :]  # drop the XML prolog; content is RDKit-generated


def predict(mol, smiles, models):
    data = mol_to_data(mol, smiles)
    batch = Batch.from_data_list([data]).to(DEVICE)
    (m_esol, ck_esol), (m_tox, _) = models["esol"], models["tox21"]
    with torch.no_grad():
        logs = m_esol(batch).item() * ck_esol["norm"]["std"] + ck_esol["norm"]["mean"]
        probs = torch.sigmoid(m_tox(batch))[0].cpu().numpy()
        fp = torch.nn.functional.normalize(m_esol.embed(batch), dim=1)[0].cpu().numpy()
    return data, logs, probs, fp


def render_drawing(smiles, scores):
    plain = svg_html(draw_svg(smiles))
    tinted = svg_html(draw_svg(smiles, scores))
    st.markdown(
        f'<div class="mf-mol">{plain}<div class="mf-tint">{tinted}</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="mf-cbar"><span>lowers prediction</span><div></div>'
        "<span>raises prediction</span></div>",
        unsafe_allow_html=True,
    )


def tox_table(probs, selected):
    rows = "".join(
        f'<tr class="{"sel" if TOX21_TASKS[i] == selected else ""}"><td>{TOX21_TASKS[i]}</td>'
        f'<td class="p">{probs[i]:.2f}</td></tr>'
        for i in np.argsort(-probs)
    )
    return (
        f'<table class="mf-tox"><thead><tr><th>Assay</th><th class="p">model probability'
        f"</th></tr></thead><tbody>{rows}</tbody></table>"
    )


def neighbours_html(fp, emb, train_smiles, train_y):
    sims = emb @ fp  # cosine similarity: both sides are unit-normalized
    cells = ""
    for i in np.argsort(-sims)[:3]:
        svg = svg_html(draw_svg(train_smiles[i], size=(240, 180), padding=0.04))
        cells += (
            f'<figure><div class="pic">{svg}</div><figcaption>logS {train_y[i]:.2f}'
            f"<br>similarity {sims[i]:.2f}</figcaption></figure>"
        )
    return f'<div class="mf-nn">{cells}</div>'


HOW_IT_WORKS = """
Each atom becomes a node with 35 numbers describing it (element, bonds, charge, hydrogens,
ring membership), and each bond becomes a pair of edges. A graph convolution layer lets every
atom average its own numbers with its bonded neighbours', then transforms the result. After
three layers each atom has seen everything up to three bonds away. Averaging and taking the
maximum over all atoms gives a 256-number learned fingerprint, which a small network maps to
solubility or to the 12 toxicity probabilities. The colours show which atoms pushed the
selected prediction up (red) or down (blue), estimated from gradients.

```
   H   H                node = atom  (C, C, O + features)
   |   |                edge = bond  (each bond stored both ways)
 H-C - C - O-H          C1 ⇄ C2 ⇄ O
   |   |
   H   H                hydrogens are a feature of each atom, not nodes
```
"""


def on_preset():
    name = st.session_state["preset"]
    if name is not None:
        st.session_state["smiles"] = PRESETS[name]
        st.session_state["query"] = PRESETS[name]


def on_predict():
    st.session_state["query"] = st.session_state["smiles"]


def main():
    st.set_page_config(page_title="MolFP-GCN", layout="wide")
    st.markdown(f"<style>{fonts_css()}</style>", unsafe_allow_html=True)
    # a div, not <h1>: Streamlit restyles headings and adds anchor links to them
    st.markdown(
        '<div class="mf-head"><div class="mf-title">MolFP-GCN</div><span>Learned '
        "molecular fingerprints</span></div>",
        unsafe_allow_html=True,
    )

    loaded = load_models()
    if loaded is None:
        st.markdown(COPY["no_model"])
        return
    models, emb, train_smiles, train_y = loaded
    if "query" not in st.session_state and st.query_params.get("smiles"):
        # ?smiles=... opens the app on a molecule; validated like typed input below
        st.session_state["smiles"] = st.session_state["query"] = st.query_params["smiles"]

    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.text_input(
            "SMILES", key="smiles", max_chars=MAX_SMILES_CHARS, placeholder="CC(=O)Oc1ccccc1C(=O)O"
        )
        st.selectbox(
            "or pick",
            list(PRESETS),
            index=None,
            placeholder="Choose a molecule",
            key="preset",
            on_change=on_preset,
        )
        st.button("Predict", type="primary", on_click=on_predict)

    query = st.session_state.get("query", "")
    mol, error = validate(query)
    if mol is None:
        with left:
            css = "mf-empty" if error == COPY["empty"] else "mf-error"
            st.markdown(f'<div class="{css}">{html.escape(error)}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mf-foot">{COPY["footer"]}</div>', unsafe_allow_html=True)
        return

    data, logs, probs, fp = predict(mol, query, models)
    # the Explain controls sit under the drawing (DESIGN.md) but decide what it shows,
    # so read their current values from session state first
    target = st.session_state.get("explain", "solubility")
    task = st.session_state.get("task") or TOX21_TASKS[int(np.argmax(probs))]
    with left:
        if target == "solubility":
            scores = saliency(models["esol"][0], data)
        else:
            scores = saliency(models["tox21"][0], data, target=TOX21_TASKS.index(task))
        render_drawing(query, scores)
        c1, c2 = st.columns([1, 1.3])
        c1.radio("Explain", ["solubility", "toxicity"], key="explain")
        c2.selectbox(
            "Toxicity assay",
            TOX21_TASKS,
            key="task",
            index=TOX21_TASKS.index(task),
            disabled=target != "toxicity",
        )

    with right:
        st.markdown(
            '<p class="mf-label">Solubility</p><div class="mf-logs"><span class="k">logS</span>'
            f'<span class="v">{logs:.2f}</span><span class="band">{solubility_band(logs)}</span>'
            '</div><p class="mf-sub">log10 of solubility in mol/L, predicted</p>'
            '<hr class="mf-rule"><p class="mf-label">Toxicity (12 assays)</p>'
            '<p class="mf-sub" style="margin-bottom:8px">trained with rare positives up-weighted, '
            "so compare molecules rather than read values as absolute</p>"
            + tox_table(probs, task if target == "toxicity" else None)
            + '<hr class="mf-rule"><p class="mf-label">Closest training molecules</p>'
            '<p class="mf-sub" style="margin-bottom:8px">by cosine similarity of the learned '
            "fingerprint (ESOL model)</p>" + neighbours_html(fp, emb, train_smiles, train_y),
            unsafe_allow_html=True,
        )
        with st.expander("How it works"):
            st.markdown(HOW_IT_WORKS)

    st.markdown(f'<div class="mf-foot">{COPY["footer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
