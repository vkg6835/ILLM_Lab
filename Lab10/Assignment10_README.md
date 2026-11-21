# Assignment 10

Open [Assignment10_Midsem_BERT.ipynb](Assignment10_Midsem_BERT.ipynb).
The [Python version](Assignment10_Midsem_BERT.py) contains the same solution.

Part A solves questions 2, 4, and 5 from the supplied midsem paper:

- Beam search with all expanded cumulative scores and the two retained beams.
- Two frequency-weighted BPE merges, including the tie convention and OOV handling.
- TF-IDF, class centroids, and a drawn Doc2Vec PV-DM architecture.

The beam-search question ends after three steps. Its highest-scoring hypothesis
is `cat sat mat` (0.24), which is not EOS-terminated. The best retained completed
sequence is `sat mat </s>` (0.189). Both interpretations are explained.
TF-IDF uses raw TF and unsmoothed natural-log IDF; alternative normalized TF
divides every value by four in this example.

Part B fine-tunes all parameters of the real 12-layer multilingual BERT base
model. It directly reuses Lab 6's 36,800/4,600/4,600 JSONL splits and verifies
their hashes. This is the saved reconstruction of Assignment 1's 23-language
dataset, which also underlies all seven earlier classifier results.

The default is one complete training epoch, batch size 16, maximum 64 WordPiece
tokens, AdamW learning rate 3e-5, 10% warmup, and validation-only checkpoint
selection. The comparison includes logistic regression, all three MLP depths,
LSTM, GRU, and supervised FastText. Duplicate-text overlap, unknown tokens and
truncations are reported. The shared older splits contain some exact-text
overlap, so the comparison is fair to those saved baselines but is not a
duplicate-free generalization estimate.

From the parent folder:

```powershell
python -m pip install -r Lab10/requirements-assignment10.txt
python Lab10/Assignment10_Midsem_BERT.py
```

Set `LAB10_EPOCHS` before running for a longer experiment. CUDA is used when
available; the completed local experiment uses the RTX 3050 with mixed precision.
Select the installed `Labs GPU (Python 3.13)` notebook kernel on this computer,
or run `.venv-labs-gpu/Scripts/python.exe Lab10/Assignment10_Midsem_BERT.py` from
the parent folder. Repeated runs reuse a
matching saved experiment. Keep the `assignment6_artifacts/data/` files,
`labels.json`, and classifier-comparison CSV in the parent folder.

Outputs are in `assignment10_artifacts/`: worked-question CSV tables, Doc2Vec
diagram, BERT checkpoint, validation/test predictions, confusion matrices,
per-class metrics, training history, tokenization audit, eight-model comparison,
comparison plot, and the generated report. The model and external pretraining,
tokenization/context limits and training budgets differ across classifiers.
