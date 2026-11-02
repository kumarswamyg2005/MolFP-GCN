# Project division: MolFP-GCN

Three members, three areas. Each person can explain their own part end to end and the shared
overview. Detailed viva scripts for each are in [viva_scripts/](viva_scripts/).

| Member | Roll no. | Area | Code | Viva script |
|---|---|---|---|---|
| Gurram Naga Kumaraswamy | S20230010101 | **Model**: GCN architecture, featurisation (SMILES → graph), saliency explanations | `molfp/featurize.py`, `model.py`, `explain.py` | [01](viva_scripts/01_KUMARASWAMY.md) |
| Simon Peter | S20230010171 | **Pipeline**: data download and checks, scaffold split, training loop, evaluation, reproducibility | `molfp/data.py`, `split.py`, `train.py`, `evaluate.py`, `metrics.py`, `scripts/`, `Makefile` | [02](viva_scripts/02_SIMON_PETER.md) |
| Rakesh | S20230010196 | **Comparison and demo**: ECFP4 + random forest baseline, ablations, figures, Streamlit app, limitations | `molfp/baseline.py`, `figures.py`, `app/`, `configs/esol_ablations.yaml` | [03](viva_scripts/03_RAKESH.md) |

## Shared by everyone
One-sentence summary, datasets, results table, what a scaffold split is and why:
[viva_scripts/00_OVERVIEW.md](viva_scripts/00_OVERVIEW.md). Plain-language guide:
[PROJECT_IN_SIMPLE_WORDS.md](PROJECT_IN_SIMPLE_WORDS.md).

## Hand-offs (who depends on whom)
Simon's splits and training loop → feed Kumaraswamy's model and Rakesh's baseline →
both write `results/*.json` → Rakesh's figures and app read them.

## Explainer video
`docs/video/MolFP-GCN_explained.mp4` (about 3 min, regenerate with
`python docs/video/make_video.py`).
