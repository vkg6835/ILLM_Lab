# Assignment 6 - Language Identification

Vishal Gautam, U23AI127

The experiment compares all seven required classifiers on the 23-language
Assignment 1 dataset. It uses 36,800 training, 4,600 validation, and 4,600 test
sentences, with 1,600/200/200 examples per language. Reconstructed with Assignment 1's sampling and split procedure.
The reconstructed source revision, split hashes, labels, and hyperparameters are
saved with the results. The original notebook used 2,000 sentences per language,
so those counts are preserved despite Assignment 1's PDF specifying 1,000.

| Classifier | Validation macro-F1 | Test macro-F1 |
| --- | ---: | ---: |
| FastText classification | 0.9479 | 0.9417 |
| LSTM | 0.8938 | 0.8877 |
| GRU | 0.8937 | 0.8851 |
| MLP (n=1) | 0.8842 | 0.8696 |
| MLP (n=2) | 0.8833 | 0.8707 |
| MLP (n=0) | 0.8758 | 0.8606 |
| Logistic regression | 0.8570 | 0.8360 |

The model selected by validation macro-F1 is **FastText classification**, with
validation macro-F1 **0.9479** and test macro-F1
**0.9417**. Test scores were not used for checkpoint or
model selection. Macro-F1 is implemented from a 23-by-23 confusion matrix using
NumPy; no sklearn evaluation classes or functions are used.

Logistic regression and MLP with zero hidden layers have the same linear-softmax
form. Their separate scores reflect different optimizers and validation-selected
checkpoints. The deeper MLPs introduce nonlinear features. LSTM and GRU also use
word order, while supervised FastText learns word and character n-grams and can
learn representations for training words missing from the Wiki lookup.

All six NumPy/PyTorch classifiers use one label-independent Wiki vector table.
The same word always has the same input vector regardless of its sentence's true
language. FastText receives the merged training-word vectors as initialization.
The full Wiki vector files for 18 languages were scanned; no row limit was used.
Their monolingual spaces are unaligned, and the fixed averaging rule should be
interpreted as a classification feature construction.

The official Wiki release lacks models for Bodo, Dogri, Manipuri, Khasi, and
Santali. These classes remain in all training and evaluation splits. Overall
test token coverage is **68.48%**; language-level coverage and
all-OOV sentence counts are saved in `embedding_coverage.csv`. Entirely uncovered
sentences cannot be distinguished by the averaged zero-vector features, which
limits the embedding-only classifiers. Recurrent inputs are capped at
96 words; truncation counts are also saved.

The unchanged Assignment 1 sampling procedure produces **155**
validation sentences and **110** test sentences whose exact text also
appears in training. They are retained for consistency with that procedure; this
overlap can increase measured scores and should be considered when interpreting
generalization. The language-level counts are in `duplicate_audit.csv`.

Artifacts include the comparison CSV, per-class precision/recall/F1, confusion
matrices, predictions, training histories, model checkpoints, cached vectors,
dataset splits, and plots. Set `FORCE_RETRAIN = True` to repeat model training.
Cached checkpoints are accepted only when their data/vector/configuration
signature matches the current experiment.
