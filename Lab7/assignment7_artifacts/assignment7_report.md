# Assignment 7 - Noun Phrase Identification using an HMM

**Name:** Vishal Gautam  
**Roll number:** U23AI127

## Objective

Train a supervised HMM on the supplied NP-tagged data, calculate every test sentence's likelihood with the forward algorithm, and predict B-NP/I-NP/O tags using Viterbi decoding.

## Dataset

| Split | Sentences | Tokens | Unique surface words |
| --- | ---: | ---: | ---: |
| train | 823 | 19172 | 4570 |
| test | 77 | 1896 | 744 |

Files are read at blank-line sentence boundaries with case and punctuation preserved. All supplied labels satisfy IOB2. The training and test sets stay as supplied; no test labels are used to estimate or tune parameters.

## Method

The NumPy HMM estimates initial-state, tag-transition, and word-emission probabilities from training counts. Add-K smoothing uses **K = 0.1**, following Assignment 4. Impossible IOB2 events (initial I-NP and O -> I-NP) stay zero. Counts do not cross sentence boundaries, and parameter distributions sum to one.

Training words with frequency below 2 map to `<UNK>`, as do unseen test words. The resulting observation vocabulary has **1760** symbols, including `<UNK>`; **2811** training tokens contribute to its emissions. Capitalization is preserved. No POS tags or pretrained embeddings are used.

The forward algorithm sums all valid tag paths in log space using log-sum-exp. Viterbi uses max and backpointers in log space to find the highest-probability complete tag path. Both require O(T S^2) time, where S = 3.

The likelihood is **P(mapped observations | supplied length)** under a fixed-length HMM without an END state. It does not assign a separate probability to length or distinguish surface words pooled into `<UNK>`. The Viterbi joint probability is never greater than the forward likelihood.

## Test results

| Measure | Score |
| --- | ---: |
| Exact NP-span precision | 0.683983 |
| Exact NP-span recall | 0.665263 |
| Exact NP-span F1 | 0.674493 |
| Token accuracy | 0.852848 |
| Token macro-F1 over B-NP, I-NP, O | 0.847328 |

There are **475 gold NP spans**, **462 predicted NP spans**, and **316 exact matches**. Seqeval evaluates strict IOB2 spans. O is predicted and included in token metrics, but is excluded from entity-span F1. Independent span counts agree with seqeval.

### Token metrics

| Tag | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| B-NP | 0.8290 | 0.8063 | 0.8175 | 475 |
| I-NP | 0.8325 | 0.8670 | 0.8494 | 579 |
| O | 0.8809 | 0.8694 | 0.8751 | 842 |

## Observations

- Forward and Viterbi completed for all **77** test sentences and **1896** tokens. All log likelihoods are finite; **0** exponentiated probabilities underflowed.
- **370** test tokens (**19.51%**) never occur in training. **507** tokens (**26.74%**) map to `<UNK>`, including rare words seen once in training.
- Token accuracy is **88.60%** for seen training words and **71.62%** for unseen words.
- **0** test sentences exactly match a training sentence.
- Exact-span evaluation requires correct starts and ends. A phrase with one incorrect boundary is counted as an extra prediction and a missed gold span, even if several token labels match.
- This small first-order word HMM does not represent POS, word shape, or longer context. Unseen words share a single emission distribution. These are limits of the chosen model; improving them would require a separate experiment.

### Example prediction

Sentence: Confidence in the pound is widely expected to take another sharp dive if trade figures for September , due for release tomorrow , fail to show a substantial improvement from July and August 's near-record deficits .

Gold noun phrases: Confidence; the pound; another sharp dive; trade figures; September; release; tomorrow; a substantial improvement; July and August; 's near-record deficits

Predicted noun phrases: Confidence; the pound; another sharp dive; trade figures; September; release tomorrow; a substantial improvement; July and August; 's near-record deficits

Forward log probability: -173.68238110; probability: 3.72134798e-76.

## Verification

Exhaustive enumeration on a tiny corpus agrees with the forward sum and Viterbi maximum/backtracking. Checks cover empty, one-word, and unknown-word inputs, normalized parameters, complete token alignment, valid BIO paths, forward >= Viterbi, and independent exact-span agreement with seqeval. Input SHA-256 hashes are saved with the evaluation metrics.

## Saved outputs

All paths below are relative to `Lab7/assignment7_artifacts/`.

- `test_sentence_probabilities.csv`: all sentence texts, lengths, forward probabilities/log probabilities, Viterbi log joint probabilities, and tags.
- `test_token_predictions.csv`, `test_predictions.conll`, and `test_predictions.jsonl`: aligned gold/predicted tags and NP spans.
- `assignment7_results.csv`, `evaluation_metrics.json`, `seqeval_report.txt`, `test_per_tag_metrics.csv`, and `test_confusion_matrix.csv`: evaluated scores.
- `initial_probabilities.csv`, `transition_probabilities.csv`, `emission_probabilities.csv`, and `models/`: parameters, counts, and vocabulary.
- `noun_phrase_errors.csv` and `plots/test_evaluation.png`: error analysis and visualization.
- `dataset_summary.csv` and `tag_counts.csv`: input statistics.

## References

1. Supplied `ILLM-Lab7-Assignment.pdf` and `Data/train_NP.txt`, `Data/test_NP.txt`.
2. `Assign4_NGram.ipynb`: count estimation and Add-K smoothing with K = 0.1.
3. `Assignment6_Language_Identification.ipynb`: notebook structure and saved evaluation artifacts.
4. Seqeval: https://pypi.org/project/seqeval/
