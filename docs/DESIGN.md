# MolFP-GCN demo — DESIGN.md

## Subject
A bench tool for a chemist: type a molecule, see its structure, what the model predicts, and which atoms it looked at. Vernacular: structure diagrams on lab paper, CPK atom colours, a colorbar next to a figure. Audience: a viva panel watching on a projector. Primary job: make the "which atoms mattered" picture legible from the back of the room.

## Palette (5 tokens)
--paper      #F6F8F7   cool lab-paper white, page background
--ink        #1C2321   text, bonds, rules
--graphite   #5E6B67   secondary text, axis labels
--grid       #DCE3E0   hairlines, table rules, input borders
--signal     #2F6FB0   the one accent: primary button, focus ring, selected task. Nothing else.
Saliency uses its own diverging scale (blue #2F6FB0 → white → #B8412C), shown ONLY on atoms and in the colorbar. It is data, not decoration.

## Type
- Atkinson Hyperlegible (Google Fonts) — all UI text. Chosen for projector legibility.
- IBM Plex Mono — SMILES strings only (they are code and must show exact characters). Not used for labels or numbers.
Scale: 14 / 16 / 20 / 28 px. Predicted logS number at 28px, weight 600. Sentence case everywhere.

## Layout
Two columns on desktop, stacked on mobile. Left-aligned throughout.

```
┌───────────────────────────────────────────────────────────────┐
│ MolFP-GCN        Learned molecular fingerprints               │
├───────────────────────────────┬───────────────────────────────┤
│ SMILES [CC(=O)Oc1ccccc1C(=O)O]│ Solubility                    │
│ or pick  [Aspirin ▾]          │ logS  -1.72   soluble         │
│ [Predict]                     │ ───────────────────────────── │
│                               │ Toxicity (12 assays)          │
│   ┌───────────────────────┐   │ SR-ARE        0.31            │
│   │  molecule drawing     │   │ NR-AhR        0.18            │
│   │  atoms tinted by      │   │ ...                           │
│   │  saliency             │   │ ───────────────────────────── │
│   └───────────────────────┘   │ Closest training molecules    │
│  low ▮▮▮▮▮▮▮▮▮ high  colorbar │ [mol] [mol] [mol]             │
│  Explain: (•) solubility      │                               │
│           ( ) toxicity: [▾]   │ ▸ How it works                │
└───────────────────────────────┴───────────────────────────────┘
```

The molecule drawing is the memorable element: large (min 420px), white background, generous margin. Everything else stays quiet.

## Motion
None on load. When Predict is pressed, the saliency tint fades in over 250ms on the drawing only. Respect prefers-reduced-motion (no fade).

## Copy
- Button: "Predict". Error: "Couldn't read that SMILES. Check brackets and ring numbers, e.g. c1ccccc1." Too large: "Molecule too large for the demo (limit 100 heavy atoms)."
- Empty state: "Enter a SMILES string or pick a molecule to start."
- Solubility bands in plain words: soluble / moderately soluble / poorly soluble.
- Toxicity probabilities labelled "model probability", never "risk".
- Footer line: "Research demo. Not for safety or medical decisions."

## Forbidden
Uniform rounded cards with shadows, gradient headers, all-caps labels, emoji, arrows in button text, Streamlit default red accent, rainbow colormaps (jet), monospace for numbers or labels, sidebar navigation.

## Streamlit implementation notes
Hide default menu/footer via CSS. Inject app/style.css with st.markdown. Set theme in .streamlit/config.toml: primaryColor #2F6FB0, backgroundColor #F6F8F7, textColor #1C2321, font sans serif. Draw molecules with RDKit MolDraw2DSVG and render via st.markdown(svg, unsafe_allow_html=True) — SVG only from RDKit, never from user text.
