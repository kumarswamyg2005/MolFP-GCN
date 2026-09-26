# MolFP-GCN in simple words

This page explains the whole project for someone with no machine-learning or chemistry
background. If you can follow this page, you can follow the viva. For the maths, see
[EXPLAIN.md](EXPLAIN.md); for the full write-up, see [REPORT.md](../reports/REPORT.md).

## 1. The question we asked

Can a computer look at the structure of a molecule and guess two things about it?

1. **How well it dissolves in water** (solubility). This matters for medicines: a drug that
   doesn't dissolve can't get into the blood.
2. **Whether it might be toxic** in 12 different lab tests.

And when it makes a guess, can it show us **which part of the molecule** it was looking at?

## 2. How a computer "sees" a molecule

Chemists write molecules as short text strings called **SMILES**. Ethanol (drinking alcohol)
is `CCO`: carbon, carbon, oxygen. Aspirin is `CC(=O)Oc1ccccc1C(=O)O`.

A computer can't learn from text like that directly. It needs numbers. There are two ways
to turn a molecule into numbers, and comparing them is the heart of this project.

### The old way: a fixed fingerprint (ECFP)

Picture a checklist of 2048 boxes. For every atom, a fixed rule looks at the atom and its
neighbours two bonds away and ticks one box. The ticked boxes are the molecule's
**fingerprint**.

The rule never changes. It doesn't matter whether you care about solubility or toxicity;
aspirin always ticks the same boxes. We then give this checklist to a **random forest**, a
standard model that makes predictions by combining hundreds of simple yes/no decision trees.
This is our **baseline**, the thing to beat.

### Our way: a learned fingerprint (GCN)

We treat the molecule as a **graph**, the way you might draw it on paper:

```
   atoms  = dots   (each dot carries 35 numbers: what element, how many bonds,
                    its charge, how many hydrogens, is it in a ring, ...)
   bonds  = lines between dots
```

A **graph convolutional network (GCN)** then works in rounds. In each round, every atom
looks at its bonded neighbours and updates its own numbers by averaging theirs with its
own. It's like people in a room each asking their neighbours what they know and updating
their opinion.

- After round 1, each atom knows about atoms 1 bond away.
- After round 3, each atom knows about atoms up to 3 bonds away.

At the end we combine all the atoms into one list of 256 numbers. That list is the
**learned fingerprint**. The difference from the old way is that the rule for building it
is *learned during training*. If the task is solubility, the fingerprint shifts to capture
what matters for solubility. If the task is toxicity, it captures something else.

Finally, a small extra network turns those 256 numbers into the answer: one solubility
number, or 12 toxicity probabilities.

## 3. The data

| Dataset | What it has | Size |
|---|---|---|
| **ESOL** | measured water solubility of real molecules | {{esol_n}} molecules |
| **Tox21** | results of 12 toxicity lab tests | {{tox21_n}} molecules |

Solubility is given as **logS**, a log scale. −1 means ten times more soluble than −2, which
is ten times more soluble than −3, and so on. Higher (closer to 0) means dissolves better.

Tox21 is messy. Not every molecule was run through every test, so {{tox_missing_pct}} of the
answers are simply missing. And toxic results are rare: only {{tox_pos_pct}} of the known
answers say "toxic". We handle both carefully (see section 5).

## 4. Keeping the test fair

Any model can look good if it gets tested on things it has already seen. So we split the
molecules into three groups:

- **Training set (80%)**: the model learns from these.
- **Validation set (10%)**: used during training to decide when to stop.
- **Test set (10%)**: locked away and used exactly **once** at the very end to get the final
  score.

There's a catch. Chemistry datasets contain "families" of near-identical molecules: the same
core ring structure with a small change on the side. If one family member is in training and
its twin is in test, the test is too easy.

So we use a **scaffold split**. We find each molecule's core ring structure (its
**scaffold**) and keep each whole family together in one group. The test set then contains
molecule families the model has never seen, which is what real use looks like.

To show why this matters, we also tried an ordinary random split. The model's error dropped
from {{abl_split_scaffold}} to {{abl_split_random}}. That looks better, but it's partly
cheating, because the model had seen close relatives of the test molecules.

We repeat everything **3 times** with different random seeds (different family shuffles and
different starting weights) and report the average ± the spread. One lucky run proves
nothing.

## 5. Training, in plain steps

1. Show the model a batch of 64 training molecules.
2. It guesses; we measure how wrong it was (the **loss**).
3. Nudge all its internal numbers slightly in the direction that makes it less wrong.
4. Repeat for all batches (one pass through all of them is an **epoch**).
5. After each epoch, check the score on the validation set. If it hasn't improved for 30
   epochs, stop and keep the best version (**early stopping**). This stops the model from
   memorising the training set.

Two details for Tox21:
- **Missing answers are ignored**, not counted as "not toxic". Guessing that an untested
  molecule is safe would teach the model something false.
- **Rare "toxic" answers get extra weight**, so the model can't score well by always saying
  "not toxic".

## 6. What we found

### How to read the scores

- **RMSE (solubility)**: the typical size of the error, in logS units. Lower is better.
  An RMSE of 1 means the guess is usually off by about one step on the log scale (a factor
  of 10 in how much dissolves).
- **R²**: how much of the variation between molecules the model explains. 1 is perfect,
  0 is no better than always guessing the average.
- **ROC-AUC (toxicity)**: if you pick one toxic and one non-toxic molecule at random, how
  often does the model rank the toxic one as more toxic? 0.5 is a coin flip, 1.0 is perfect.

### The results

{{table_main}}

**The learned fingerprint (GCN) beat the fixed fingerprint on both tasks.** On toxicity it
won on {{tox_gcn_task_wins}} of the 12 tests.

Two honest caveats:

- **The solubility win is partly luck of the split.** In one of the three runs, a family of
  extremely insoluble chlorine-heavy molecules (PCBs) all landed in the test set. The random
  forest can't guess values more extreme than anything it saw in training, so it did badly
  there. On the other two runs the GCN still won, but by less.
- **The toxicity score is a bit higher than published papers.** Our way of picking test
  families is probably a little easier than the standard one, so we don't compare our
  numbers directly with papers.

We also tested variations: 1 to 4 rounds of message passing, and different ways of
combining atoms (average, maximum, both). **None of these made a real difference**; the
changes were smaller than the run-to-run noise. We report that as it is rather than
picking the best-looking one.

## 7. Seeing what the model looked at

For any prediction we can ask: "if this atom were slightly different, how much would the
answer change?" Maths gives us that for free (it's called a **gradient**). Combining it with
the atom's own features gives each atom a score:

- **red** atom: pushed the prediction **up** (more soluble, or more likely toxic)
- **blue** atom: pushed the prediction **down**
- white: didn't matter much

For solubility the colours usually make chemical sense. Oxygen atoms (which like water) show
up red, and big carbon rings and chlorine groups (which avoid water) show up blue.

This is a hint, not proof. It measures what the *model* relied on, which may not be the true
chemistry, and neighbouring atoms share credit because they have been mixing information.

## 8. The demo app

Run `make demo` and a web page opens at `http://localhost:8501`. You can:

1. Type a SMILES string, or pick one of 8 ready-made molecules (aspirin, caffeine,
   paracetamol, ibuprofen, benzene, ethanol, nicotine, and one known toxic molecule the model
   never saw during training).
2. Press **Predict**.
3. See the molecule drawn with atoms coloured by importance, the predicted solubility in
   plain words (soluble / moderately soluble / poorly soluble), the 12 toxicity probabilities,
   and the 3 training molecules whose learned fingerprints are most similar.
4. Switch the colouring between solubility and any of the 12 toxicity tests.

It refuses bad input politely (a typo in the SMILES, more than 200 characters, or more than
100 heavy atoms) instead of crashing. It can't train models, doesn't accept uploads and
never goes online.

Two things to know when demoing:
- The toxicity numbers run high on purpose (because of the extra weight on rare toxic
  results), so compare molecules with each other rather than reading 0.8 as "80% risk".
- It isn't perfect even on molecules it trained on (caffeine, for example), because early
  stopping keeps it from memorising.

## 9. What the project doesn't do (limitations)

- **It ignores bond types.** To our GCN a single bond and a double bond look the same. We
  already store bond types in the data, so a smarter layer (GAT or MPNN) could use them.
  That is the obvious next step.
- **The solubility dataset is small** (about a thousand molecules), so results swing between
  runs.
- **Settings weren't tuned** for either model. We fixed them in advance to keep the
  comparison fair.
- **Three runs** give a rough error bar, not a formal statistical test.
- **It's a research demo**, not a safety tool.

## 10. How to run everything

You need [uv](https://docs.astral.sh/uv/) installed. From the project folder:

```
make all        # set up, download data, train everything, evaluate, make figures, run tests
make demo       # open the app
make ablations  # the extra experiments (then run `make figures` again)
```

`make all` takes about 11 minutes on a laptop CPU (no GPU needed) and reproduces every number
exactly.

## 11. Where things live

| Folder | What's inside |
|---|---|
| `molfp/` | all the code: turning SMILES into graphs, the model, training, testing, figures |
| `app/` | the demo web page |
| `configs/` | the settings for each experiment |
| `data/` | downloaded datasets (not in git) and the saved train/validation/test splits |
| `results/` | the final scores, as JSON files |
| `reports/` | the full report and all figures |
| `tests/` | automatic checks that the code does what we claim |
| `docs/` | design decisions, viva notes, and this page |

Every number on this page, in the report and in the README is filled in automatically from
`results/` by `make figures`. None are typed by hand, so they can't go out of date.

## 12. Word list

| Word | Meaning |
|---|---|
| SMILES | a text way of writing a molecule, e.g. `CCO` for ethanol |
| graph | dots (atoms) connected by lines (bonds) |
| fingerprint | a fixed-length list of numbers that describes a molecule |
| ECFP | the standard fixed fingerprint; same result for every task |
| GCN | graph convolutional network; atoms learn from their neighbours in rounds |
| learned fingerprint | the 256 numbers the GCN builds, shaped by the task |
| random forest | a classic model made of many decision trees; our baseline |
| scaffold | the core ring structure of a molecule; we split data by it |
| training / validation / test set | learn from / decide when to stop / final exam, used once |
| epoch | one pass through all the training molecules |
| early stopping | stop training once validation stops improving |
| logS | solubility on a log scale; closer to 0 dissolves better |
| RMSE | typical error size; lower is better |
| ROC-AUC | ranking quality for yes/no answers; 0.5 = coin flip, 1 = perfect |
| saliency | the red/blue colouring showing which atoms drove a prediction |
| seed | a number that fixes the randomness so a run can be repeated exactly |
| ablation | an experiment that changes one part to see if it matters |
