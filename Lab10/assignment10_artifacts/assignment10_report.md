# Assignment 10 - Midsem and BERT Language Identification

Vishal Gautam, U23AI127

## Midsem Question 2

The beams are `cat`/`sat` at t=1, `cat sat`/`sat mat` at t=2, and
`cat sat mat` (0.24)/`sat mat </s>` (0.189) at t=3. The highest three-step
sequence is `cat sat mat`. It has not ended; `sat mat </s>` is the best completed
candidate. No t=4 probabilities are supplied. P(cat|mat) and P(sat|mat)
are irrelevant because the prefix mat was pruned at t=1.

|   t | hypothesis     |   prefix_probability |   conditional_probability |   cumulative_probability | retained   |
|----:|:---------------|---------------------:|--------------------------:|-------------------------:|:-----------|
|   1 | `cat`          |                 1    |                       0.5 |                    0.5   | True       |
|   1 | `sat`          |                 1    |                       0.3 |                    0.3   | True       |
|   1 | `mat`          |                 1    |                       0.2 |                    0.2   | False      |
|   1 | `</s>`         |                 1    |                       0   |                    0     | False      |
|   2 | `cat sat`      |                 0.5  |                       0.6 |                    0.3   | True       |
|   2 | `sat mat`      |                 0.3  |                       0.7 |                    0.21  | True       |
|   2 | `cat mat`      |                 0.5  |                       0.3 |                    0.15  | False      |
|   2 | `sat cat`      |                 0.3  |                       0.2 |                    0.06  | False      |
|   2 | `cat </s>`     |                 0.5  |                       0.1 |                    0.05  | False      |
|   2 | `sat </s>`     |                 0.3  |                       0.1 |                    0.03  | False      |
|   2 | `cat cat`      |                 0.5  |                       0   |                    0     | False      |
|   2 | `sat sat`      |                 0.3  |                       0   |                    0     | False      |
|   3 | `cat sat mat`  |                 0.3  |                       0.8 |                    0.24  | True       |
|   3 | `sat mat </s>` |                 0.21 |                       0.9 |                    0.189 | True       |
|   3 | `cat sat </s>` |                 0.3  |                       0.2 |                    0.06  | False      |
|   3 | `sat mat cat`  |                 0.21 |                       0.1 |                    0.021 | False      |
|   3 | `cat sat cat`  |                 0.3  |                       0   |                    0     | False      |
|   3 | `cat sat sat`  |                 0.3  |                       0   |                    0     | False      |
|   3 | `sat mat mat`  |                 0.21 |                       0   |                    0     | False      |
|   3 | `sat mat sat`  |                 0.21 |                       0   |                    0     | False      |

## Midsem Question 4

Word frequencies: low=5, lower=2, lowest=3, new=2, newest=1, widest=3.
Two frequency-weighted merges, with lexicographic tie-breaking:
(l,o) -> lo (count 10), followed by (lo,w) -> low (count 10).
Initial character vocabulary: `['</w>', 'd', 'e', 'i', 'l', 'n', 'o', 'r', 's', 't', 'w']`. Add `lo` and `low`.
Unseen `lowernew` becomes `low e r n e w </w>`. Unseen characters need
an unknown token or byte fallback.

## Midsem Question 5

TF is raw count; IDF is ln(4/df), without smoothing or normalization.

| term       |   document_frequency |      idf |
|:-----------|---------------------:|---------:|
| machine    |                    2 | 0.693147 |
| learning   |                    3 | 0.287682 |
| is         |                    4 | 0.000000 |
| useful     |                    3 | 0.287682 |
| powerful   |                    1 | 1.386294 |
| language   |                    2 | 0.693147 |
| processing |                    1 | 1.386294 |

|    |   machine |   learning |       is |   useful |   powerful |   language |   processing |
|:---|----------:|-----------:|---------:|---------:|-----------:|-----------:|-------------:|
| D1 |  0.693147 |   0.287682 | 0.000000 | 0.287682 |   0.000000 |   0.000000 |     0.000000 |
| D2 |  0.693147 |   0.287682 | 0.000000 | 0.000000 |   1.386294 |   0.000000 |     0.000000 |
| D3 |  0.000000 |   0.287682 | 0.000000 | 0.287682 |   0.000000 |   0.693147 |     0.000000 |
| D4 |  0.000000 |   0.000000 | 0.000000 | 0.287682 |   0.000000 |   0.693147 |     1.386294 |

Class vectors are means of their document TF-IDF vectors:

|            |   machine |   learning |       is |   useful |   powerful |   language |   processing |
|:-----------|----------:|-----------:|---------:|---------:|-----------:|-----------:|-------------:|
| Technology |  0.693147 |   0.287682 | 0.000000 | 0.143841 |   0.693147 |   0.000000 |     0.000000 |
| Language   |  0.000000 |   0.143841 | 0.000000 | 0.287682 |   0.000000 |   0.693147 |     0.693147 |

Doc2Vec PV-DM jointly learns a document-ID vector and context-word vectors
to predict a target word using softmax. See plots/q5_doc2vec_pvdm.png.
New document inference optimizes its vector while fixing the word and output weights.

## BERT experiment

Pretrained model: google-bert/bert-base-multilingual-cased, revision 3f076fdb1ab68d5b2880cb87a0886f315b8146f8. Original 12-layer base
architecture, all parameters fine-tuned. Exactly 36,800 training, 4,600
validation and 4,600 test examples from the saved Lab 6 JSONL files.
Split hashes were verified against Lab 6. All 23 labels are retained.
Training: 1 complete epoch(s), batch 16, learning rate 3e-5,
AdamW weight decay 0.01, 10% warmup, linear decay, gradient clipping 1.0.
Device: cuda; CUDA mixed precision: True.
Best validation epoch: 1.
Maximum context: 64 WordPiece tokens. Test truncations:
699/4,600.

| Model | Validation macro-F1 | Test macro-F1 | Test accuracy |
| --- | ---: | ---: | ---: |
| FastText classification | 0.9479 | 0.9417 | 0.9417 |
| Multilingual BERT base (fine-tuned) | 0.9062 | 0.8950 | 0.8963 |
| LSTM | 0.8938 | 0.8877 | 0.8876 |
| GRU | 0.8937 | 0.8851 | 0.8852 |
| MLP (n=1) | 0.8842 | 0.8696 | 0.8776 |
| MLP (n=2) | 0.8833 | 0.8707 | 0.8791 |
| MLP (n=0) | 0.8758 | 0.8606 | 0.8698 |
| Logistic regression | 0.8570 | 0.8360 | 0.8511 |

The BERT-minus-FastText test macro-F1 difference is
-0.0467. This reports the observed run;
it does not establish which model wins under larger training budgets.
The short training schedule, different context limits, external pretraining,
and uneven script coverage affect interpretation.

The retained prior splits contain 155
validation rows and 110 test rows whose exact
texts occur in training. Earlier baselines share this limitation; these scores
should not be interpreted as a fully duplicate-free generalization estimate.
Per-language unknown-token and truncation rates are in tokenization_audit.csv.

## References

- Supplied midsem question paper and Lab 10 assignment PDF in this folder.
- Lab 6: ../Assignment6_Language_Identification.ipynb and saved artifacts.
- Multilingual BERT: https://huggingface.co/google-bert/bert-base-multilingual-cased
- BERT: https://arxiv.org/abs/1810.04805
