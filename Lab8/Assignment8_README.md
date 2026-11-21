# Assignment 8

Open [Assignment8_English_Hindi_MT.ipynb](Assignment8_English_Hindi_MT.ipynb)
and run the cells from top to bottom. The notebook contains the full solution,
including explanations, data preparation, real OpenNMT training and translation,
SacreBLEU evaluation, and observed bin-wise analysis.

The [Python version](Assignment8_English_Hindi_MT.py) has the same code and
explanations in notebook-style cells. On this computer, from the parent folder:

```powershell
python Lab8/Assignment8_English_Hindi_MT.py
```

The existing Python 3.14 kernel can run the notebook. OpenNMT commands use the
separate Python 3.12 runtime in `Lab8/runtime/python312/`. This avoids changing
the environment used by your earlier assignments. OpenNMT-py 3.5.1 requires
PyTorch 2.1-2.2, which is unavailable for Python 3.14.

## Required dataset access

1. Sign in at https://huggingface.co/datasets/ai4bharat/BPCC and accept access.
2. Run `hf auth login` in your own terminal using that account.
3. Run the notebook. It downloads only `ilci/hin_Deva.tsv` at the pinned revision.

If you already have the prescribed file, place it at
`Lab8/assignment8_artifacts/raw/ilci/hin_Deva.tsv`. Local input is hashed for the
data manifest. Do not put your Hugging Face token into the notebook or report.

The official TSV contains `src_lang`, `tgt_lang`, `src`, and `tgt` columns. The
reader checks the English/Hindi language codes and extracts the aligned text.
It also accepts a local two-column English/Hindi TSV. Its cleaning audit records
malformed, empty, wrong-language, and duplicate rows.

## Design and connections to previous assignments

- English source and Hindi target, as confirmed for this lab and used in the
  earlier Hindi assignments.
- Assignment 3's length-aware sampling and training-only BPE approach.
- Assignment 4's source word counts and validation/test separation.
- Labs 6 and 7's reproducible settings, saved predictions, metrics, and plots.
- Full cleaned ILCI corpus, with an 80/10/10 split within each source-length bin.
- Four bins: 1-10, 11-19, 20-29, and 30+ English whitespace words before BPE.
  The PDF omits exactly 20; this solution explicitly puts it in bin 3.
- Repeated source and target texts are removed before splitting to prevent
  exact-text leakage. The cleaning audit records every removed row category.
- Separate English/Hindi SentencePiece BPE tokenizers trained only on training
  data; vocabularies of up to 4,000 pieces.
- OpenNMT count vocabularies are normalized to LF line endings to avoid its
  Windows CRLF parsing bug. OpenNMT's own loader must confirm 100% coverage of
  the training BPE tokens before training can start.
- Two-layer encoder/two-layer decoder Transformer, size 128, four heads,
  feed-forward size 512, trained from scratch using OpenNMT-py.
- Default budget: 5,000 updates; checkpoint selected by validation perplexity.
- Beam size 5; detokenized corpus BLEU from SacreBLEU, overall and per bin.
- Case-sensitive 13a tokenization, exponential smoothing, one reference,
  standard four n-gram orders, with the full scoring signature saved.

Prior monolingual Hindi data is not a replacement for the required ILCI bitext.
The analysis is generated from actual predictions after training, so the
notebook does not invent numerical BLEU scores while dataset access is pending.

## Completed CPU experiment

The executed notebook uses 128,605 training pairs, 16,075 validation pairs, and
16,075 test pairs. The selected checkpoint is step 5,000, with validation
perplexity 207.697. These are actual SacreBLEU results on a 0-100 scale:

| Bin | Source word length | Test pairs | BLEU |
|:---|:---|---:|---:|
| bin1 | 1-10 | 3,232 | 0.739 |
| bin2 | 11-19 | 6,729 | 0.306 |
| bin3 | 20-29 | 4,165 | 0.044 |
| bin4 | 30+ | 1,949 | 0.002 |
| Overall | All lengths | 16,075 | 0.130 |

The scores are very low. Predictions contain only 27.8% as many scoring tokens
as the references overall, indicating severe undertranslation. This is a
working educational baseline, not a useful translator. The
[completed report](assignment8_artifacts/assignment8_report.md) analyzes that
limitation with brevity penalties, bin sizes, and actual Hindi examples.

The full pipeline was checked on synthetic data separately, and the real
notebook executed all cells. OpenNMT's own vocabulary loader verifies training
token coverage before optimization. The synthetic checks are not lab results.

## Outputs after a successful run

All outputs go into `assignment8_artifacts/`:

- `assignment8_report.md`: completed experimental report and bin-wise analysis.
- `bin_wise_bleu.csv`: four bin scores plus overall corpus BLEU.
- `test_predictions.csv`: every source, Hindi reference, prediction, and bin.
- `split_length_summary.csv`, `data_manifest.json`, `cleaning_audit.json`.
- `validation_history.csv`, `selected_model.json`, and full OpenNMT logs.
- `plots/assignment8_evaluation.png`: split counts, BLEU, and validation curve.
- `data/`: aligned original and BPE train/validation/test files.
- `models/`: BPE models, OpenNMT vocabularies, and training checkpoints.

Matching completed artifacts are reused on later runs. Set `FORCE_RETRAIN = True`
in the setup cell to retrain. Change the training budget before running:

```powershell
$env:LAB8_TRAIN_STEPS = '10000'
python Lab8/Assignment8_English_Hindi_MT.py
```

## Running on another computer

Use Python 3.12 and install [requirements-assignment8.txt](requirements-assignment8.txt)
in a virtual environment. Run the notebook in that environment or set
`ONMT_PYTHON` to its Python executable. The provided configuration automatically
uses one CUDA GPU if the selected OpenNMT interpreter has a CUDA-enabled PyTorch
installation; otherwise it trains on CPU.

```powershell
python -m pip install -r Lab8/requirements-assignment8.txt
python Lab8/Assignment8_English_Hindi_MT.py
```

The local runtime is a convenience for this computer. Submit the notebook and
any requested reports/results, rather than the installed runtime directory.
