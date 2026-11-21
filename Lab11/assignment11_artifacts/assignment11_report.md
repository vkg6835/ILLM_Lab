# Assignment 11 - BERT Layer Probing

Vishal Gautam, U23AI127

Pretrained encoder: `google/bert_uncased_L-12_H-128_A-2` at `769ccb240a0acd026d94f84a96e8a754e4b8eff1`. Twelve encoder layers,
hidden size 128; frozen content-token mean representations.

All three tasks use 50,000 unique sentences, 45,000 training and 5,000 test.
Binary classes are exactly balanced. Six-class counts differ by at most one.
Scaler fitting and classifier learning use only training data.

| Task | Layer | Test accuracy | Test macro-F1 |
| --- | ---: | ---: | ---: |
| sentence_length | 1 | 0.9694 | 0.9694 |
| sentence_length | 2 | 0.9668 | 0.9668 |
| sentence_length | 3 | 0.9630 | 0.9630 |
| sentence_length | 4 | 0.9616 | 0.9616 |
| sentence_length | 5 | 0.9670 | 0.9670 |
| sentence_length | 6 | 0.9626 | 0.9626 |
| sentence_length | 7 | 0.9466 | 0.9465 |
| sentence_length | 8 | 0.9458 | 0.9457 |
| sentence_length | 9 | 0.9310 | 0.9310 |
| sentence_length | 10 | 0.9182 | 0.9182 |
| sentence_length | 11 | 0.9040 | 0.9039 |
| sentence_length | 12 | 0.8906 | 0.8904 |
| bigram_shift | 1 | 0.5074 | 0.5074 |
| bigram_shift | 2 | 0.5054 | 0.5054 |
| bigram_shift | 3 | 0.5022 | 0.5022 |
| bigram_shift | 4 | 0.5060 | 0.5060 |
| bigram_shift | 5 | 0.5170 | 0.5169 |
| bigram_shift | 6 | 0.6616 | 0.6616 |
| bigram_shift | 7 | 0.7038 | 0.7038 |
| bigram_shift | 8 | 0.7838 | 0.7838 |
| bigram_shift | 9 | 0.7906 | 0.7906 |
| bigram_shift | 10 | 0.7848 | 0.7848 |
| bigram_shift | 11 | 0.7828 | 0.7828 |
| bigram_shift | 12 | 0.7772 | 0.7772 |
| subj_number | 1 | 0.8102 | 0.8102 |
| subj_number | 2 | 0.8136 | 0.8136 |
| subj_number | 3 | 0.8096 | 0.8096 |
| subj_number | 4 | 0.8126 | 0.8126 |
| subj_number | 5 | 0.8136 | 0.8135 |
| subj_number | 6 | 0.8184 | 0.8183 |
| subj_number | 7 | 0.8124 | 0.8123 |
| subj_number | 8 | 0.8152 | 0.8151 |
| subj_number | 9 | 0.8172 | 0.8171 |
| subj_number | 10 | 0.8202 | 0.8201 |
| subj_number | 11 | 0.8314 | 0.8313 |
| subj_number | 12 | 0.8370 | 0.8370 |

## Observations

sentence_length: the highest observed test accuracy is 0.9694 at layer 1. Layer 1 scores 0.9694, and layer 12 scores 0.8906. The final-minus-first difference is -0.0788. This is a comparison of fixed probes, not test-based hyperparameter tuning.

bigram_shift: the highest observed test accuracy is 0.7906 at layer 9. Layer 1 scores 0.5074, and layer 12 scores 0.7772. The final-minus-first difference is +0.2698. This is a comparison of fixed probes, not test-based hyperparameter tuning.

subj_number: the highest observed test accuracy is 0.8370 at layer 12. Layer 1 scores 0.8102, and layer 12 scores 0.8370. The final-minus-first difference is +0.0268. This is a comparison of fixed probes, not test-based hyperparameter tuning.

A linear probe measures recoverability of a property. Scores alone do not
establish that a particular layer performs grammatical reasoning. The compact
BERT, chosen pooling, new sampled splits, and one seed limit generalization.
Sentence-length chance is 0.1667; binary-task chance is 0.5000.

## References

- Dataset descriptions: https://github.com/facebookresearch/SentEval/tree/main/data/probing
- Assignment reference (toolkit): https://arxiv.org/abs/1803.05449
- Probing-task paper identified by the dataset: https://arxiv.org/abs/1805.01070
- Compact BERT: https://huggingface.co/google/bert_uncased_L-12_H-128_A-2
