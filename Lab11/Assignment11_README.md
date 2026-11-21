# Assignment 11

Open [Assignment11_BERT_Layer_Probing.ipynb](Assignment11_BERT_Layer_Probing.ipynb).
The [Python version](Assignment11_BERT_Layer_Probing.py) has the same code and explanations.

The executed experiment samples 50,000 unique, consistently labelled SentEval
examples for each of sentence length, bigram shift, and subject number. It
creates new balanced 45,000/5,000 train/test splits. Binary classes are exactly
balanced; six-class length counts differ by at most one because 50,000 is not
divisible by six. Exact duplicate sentences cannot cross the new splits.

The pretrained compact BERT has 12 encoder layers, hidden size 128 and two
heads. It remains frozen. Mean pooling excludes special tokens and padding,
and the embedding output is not counted as an encoder layer. No sentence was
truncated. Each of the 36 independent probes is a training-fitted scaler and
L2 logistic regression with fixed C=1. Scores describe linear recoverability;
the test-layer ranking does not tune the model.

Observed highest accuracies:

| Task | Encoder layer | Test accuracy |
| --- | ---: | ---: |
| Sentence length | 1 | 96.94% |
| Bigram shift | 9 | 79.06% |
| Subject number | 12 | 83.70% |

The assignment's arXiv link is the SentEval toolkit paper, 1803.05449. The
dataset README identifies 1805.01070 as the probing-task paper. The notebook
cites both and distinguishes qualitative expectations from measured scores.

From the parent folder:

```powershell
python -m pip install -r Lab11/requirements-assignment11.txt
python Lab11/Assignment11_BERT_Layer_Probing.py
```

The notebook automatically downloads the public data and the pinned BERT
checkpoint. CUDA is used when available. Feature extraction is cached by
model/split/settings fingerprints and file hashes. All probes converged.

Results are in `assignment11_artifacts/`: `layer_probe_results.csv`, report,
layer-comparison plot, 36 saved probes, every layer's test predictions,
confusion matrices and metrics, class-balance audit, source/split hashes,
and cached representations. Feature arrays are rebuildable and need not be
submitted unless requested.
