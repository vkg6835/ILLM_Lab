# Assignment 9

Open [Assignment9_Pretrained_Embeddings_MT.ipynb](Assignment9_Pretrained_Embeddings_MT.ipynb).
The [Python version](Assignment9_Pretrained_Embeddings_MT.py) contains the same code and explanations.

The solution trains two English-to-Hindi OpenNMT Transformers, one initialized
with official FastText Wiki word vectors and one with official BPEmb subword
vectors. It reuses the exact Lab 8 train/validation/test pairs, checks their
hashes, and keeps the same four English-length bins and SacreBLEU settings.
Both variants use two encoder and two decoder layers, hidden size 128, four
heads, and 8,000 updates by default. Both use Noam factor 0.2, 500 warmup
updates, and gradient norm clipping at 1. Checkpoints are selected on validation
perplexity. Every test prediction is saved before overall and bin-wise scoring.

FastText uses a 12,000-word training vocabulary per language. The default scans
the first 250,000 English vector rows and all Hindi rows. Remaining vocabulary
entries receive seeded random trainable embeddings. The `.vec` files do not
supply the subword synthesis available in FastText `.bin` models. BPEmb uses
10,000 pretrained pieces and encodes text before vector lookup. It uses the
library's standard lowercasing/digit replacement and canonical piece IDs, so
unknown surface forms map consistently to `<unk>`. References stay unchanged.
Both 300D
embedding types receive a training-only PCA projection to 128D and equal row
scaling. Coverage and these limitations are documented in the notebook/report.

On this computer, the notebook runs in the existing Python environment and
OpenNMT uses the isolated CUDA Python 3.12 runtime under `Lab9/runtime/`.
It shares Lab 8's existing non-PyTorch packages through its local path file.
The notebook falls back to Lab 8's CPU runtime when that CUDA runtime is absent.
From the parent folder:

```powershell
python Lab9/Assignment9_Pretrained_Embeddings_MT.py
```

Install [requirements-assignment9.txt](requirements-assignment9.txt) for the
notebook on another computer. Install [requirements-onmt.txt](requirements-onmt.txt)
in a separate Python 3.12 environment and set `ONMT_PYTHON` to its executable.
OpenNMT 3.5.1 needs PyTorch 2.2; Labs 10/11 need a newer PyTorch, so separate
environments are required. Keep `prepare_onmt_embeddings.py` beside the notebook.
Also keep Lab 8's saved plain-text splits and pair CSVs available.

Settings can be changed before running:

```powershell
$env:LAB9_TRAIN_STEPS = '12000'
$env:LAB9_EN_FASTTEXT_ROWS = '0' # Scan the entire English vector file
$env:ONMT_PYTHON = 'C:/path/to/python312/python.exe'
python Lab9/Assignment9_Pretrained_Embeddings_MT.py
```

Downloads and training take time on CPU. Matching completed experiments and
decoded predictions are cached. A changed configuration invalidates the cache.
Generated results are in `assignment9_artifacts/`: report, `bin_wise_bleu.csv`,
embedding coverage, comparison plot, and per-variant predictions, checkpoints,
PCA matrices, validation history, and full OpenNMT logs. Lab 8's 5,000-update
model is reference context, not an equal-budget comparison.
