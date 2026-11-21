# Assignment 8 - English-to-Hindi Machine Translation

Name: Vishal Gautam  
Roll number: U23AI127

## Experiment

OpenNMT-py 3.5.1 trains a two-layer encoder/two-layer decoder Transformer from
scratch using BPCC's ILCI English-Hindi pairs. Embedding/hidden size is 128,
four attention heads, feed-forward size 512, dropout 0.1, label smoothing 0.1,
Adam with Noam scheduling, batch size 1,024 tokens, and 5,000 updates.
The complete cleaned dataset is used. Separate English and Hindi SentencePiece
BPE tokenizers, up to 4,000 pieces each, use training data only.

## Data and split

Dataset: https://huggingface.co/datasets/ai4bharat/BPCC  
Requested revision: `5bde309374f21c39e7c3d505f28ae11f58f2a220`  
Subset: `ilci/hin_Deva.tsv`  
Raw SHA-256: `d4f768bcc12f58e648c3d5105744b3e61ba6fd29e563394e4c4ff26788a43bce`

Length is English whitespace word count before BPE. Within each of four bins,
80/10/10 train/validation/test allocation is used with seed 42, subject to
rounding. The omitted boundary L=20 is assigned to bin3. Exact pair, source, and
target duplicates are removed before splitting, preventing exact-text leakage.
Alternative translations for repeated sources are discarded by this policy.

Cleaning audit:

|   input_rows |   header_rows |   malformed_rows |   wrong_language_rows |   empty_rows |   duplicate_pairs |   repeated_sources |   repeated_targets |   retained_pairs |
|-------------:|--------------:|-----------------:|----------------------:|-------------:|------------------:|-------------------:|-------------------:|-----------------:|
|       165650 |             1 |                1 |                     0 |            0 |                 0 |               4190 |                703 |           160755 |

Split distribution:

| split      | bin   | length_range   |   sentences |   source_mean_words |   source_min_words |   source_max_words |   target_mean_words |
|:-----------|:------|:---------------|------------:|--------------------:|-------------------:|-------------------:|--------------------:|
| train      | bin1  | L <= 10        |       25856 |                7.56 |                  1 |                 10 |                9.17 |
| train      | bin2  | 11 <= L <= 19  |       53836 |               14.85 |                 11 |                 19 |               16.67 |
| train      | bin3  | 20 <= L <= 29  |       33320 |               23.72 |                 20 |                 29 |               25.61 |
| train      | bin4  | L >= 30        |       15593 |               37.61 |                 30 |                269 |               40.17 |
| validation | bin1  | L <= 10        |        3232 |                7.58 |                  1 |                 10 |                9.13 |
| validation | bin2  | 11 <= L <= 19  |        6729 |               14.88 |                 11 |                 19 |               16.61 |
| validation | bin3  | 20 <= L <= 29  |        4165 |               23.63 |                 20 |                 29 |               25.59 |
| validation | bin4  | L >= 30        |        1949 |               37.93 |                 30 |                206 |               40.41 |
| test       | bin1  | L <= 10        |        3232 |                7.58 |                  1 |                 10 |                9.19 |
| test       | bin2  | 11 <= L <= 19  |        6729 |               14.81 |                 11 |                 19 |               16.62 |
| test       | bin3  | 20 <= L <= 29  |        4165 |               23.71 |                 20 |                 29 |               25.74 |
| test       | bin4  | L >= 30        |        1949 |               37.48 |                 30 |                257 |               39.84 |

## Model selection and evaluation

Selected checkpoint: `mt_step_5000.pt` at step 5,000, validation
perplexity 207.6970. Selection uses minimum validation
perplexity. Test scores do not select the checkpoint. Training time:
21.23 minutes. Decoding uses beam size 5,
Wu length penalty alpha 0.6, maximum output length 728 BPE pieces
(determined from training targets). OpenNMT receives no test references to decode.

Predictions are detokenized Hindi. The score is case-sensitive corpus BLEU with
13a tokenization, exponential smoothing, standard four n-gram orders, one
reference, and a 0-100 scale. Overall BLEU is computed on the full test corpus.

Signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`

| bin     | length_range   |   sentences |   BLEU |   brevity_penalty |   length_ratio |
|:--------|:---------------|------------:|-------:|------------------:|---------------:|
| bin1    | L <= 10        |        3232 |  0.739 |             0.525 |          0.608 |
| bin2    | 11 <= L <= 19  |        6729 |  0.306 |             0.175 |          0.365 |
| bin3    | 20 <= L <= 29  |        4165 |  0.044 |             0.024 |          0.211 |
| bin4    | L >= 30        |        1949 |  0.002 |             0.001 |          0.121 |
| overall | all lengths    |       16075 |  0.13  |             0.074 |          0.278 |

## Bin-wise result analysis

Overall test BLEU is **0.130**. The strongest observed bin is
**bin1** (0.739); the weakest is
**bin4** (0.002).

BLEU is very low across this experiment. This small model and training budget provide an executable baseline, but do not achieve useful translation accuracy. The strongest bin is only strongest relative to the other bins, not a high-quality translation result.

The complete hypotheses contain 96,377 SacreBLEU tokens against 347,288 reference tokens, a ratio of 0.278. The overall brevity penalty is 0.074.

The large aggregate length deficit is evidence of severe undertranslation. The displayed examples include generic Hindi phrases that omit most source information. Teacher-forced validation improvement therefore did not translate into adequate sentence-level generation. A larger training budget, validation-based checks of decoded translations, and a stronger model are justified follow-up experiments; their settings should be chosen on validation data.

In this run, BLEU decreases with each successive source-length bin. Read the brevity penalties and length ratios beside the scores: the output-length deficit also becomes more severe for long inputs.

- **bin1 (L <= 10):** BLEU 0.739 on 3232 test pairs, with 25856 training pairs. Output/reference token ratio 0.608; brevity penalty 0.525. Short outputs suggest inspecting omitted phrases and premature endings.
- **bin2 (11 <= L <= 19):** BLEU 0.306 on 6729 test pairs, with 53836 training pairs. Output/reference token ratio 0.365; brevity penalty 0.175. Short outputs suggest inspecting omitted phrases and premature endings.
- **bin3 (20 <= L <= 29):** BLEU 0.044 on 4165 test pairs, with 33320 training pairs. Output/reference token ratio 0.211; brevity penalty 0.024. Short outputs suggest inspecting omitted phrases and premature endings.
- **bin4 (L >= 30):** BLEU 0.002 on 1949 test pairs, with 15593 training pairs. Output/reference token ratio 0.121; brevity penalty 0.001. Short outputs suggest inspecting omitted phrases and premature endings.

Scores should be assessed alongside the unequal training/test counts. With a
small model and finite training budget, long sequences can accumulate decoding
errors, but a length-related decline is not guaranteed. Bin membership itself
does not prove the cause of an error. A low brevity penalty identifies short
translations as one possible contributor. Single-reference BLEU can undercount
valid paraphrases. Qualitative checks should examine omissions, repetition,
Hindi word order, agreement, and domain-specific terminology. Improved models
can be compared on these same saved splits; select their settings using validation.

## One actual test translation per bin

### bin1

English: A Frobenius norm how many of you get that now?

Reference Hindi: एक फ्रोबेनियस मानदंड है कि आप में से कितनो को अब वह मिलता है?

Predicted Hindi: तो, यह है।

### bin2

English: Colourful birds capture the heart of tourists by their sweet songs .

Reference Hindi: रंग - बिरंगे पक्षी अपने मधुर गीत - संगीत से भ्रमणकारियों का मन मोह लेते हैं ।

Predicted Hindi: इस प्रकार के लिए एक बार-स के रूप से अधिक है ।

### bin3

English: National Institution of Nutrition , Hyderabad has researched that the seeds of fenugreek are panacea for the diabetes patients .

Reference Hindi: नेशनल इंस्टीट्‍यूट ऑफ न्यूटीशन हैदराबाद के वैज्ञानिकों का शोध है कि मेंथी के दाने मधुमेह के रोगियों के लिए रामबाण हैं

Predicted Hindi: इस तरह से कम हो जाता है ।

### bin4

English: The celebrated Chinese traveler Huein Tsang visited the state in the middle of 7th century A.D. Nearly half (44%) of the state is forested and offers a unique panorama of flora and fauna.

Reference Hindi: प्रसिद्ध चीनी यात्री ह्युनसांग ने 7वीं शताब्दी ई. के मध्य राज्य की यात्रा की।

Predicted Hindi: यह है।


## References to earlier assignments

- Assignment 3: source-word length-aware held-out sampling and training subword
  models from scratch. This lab uses the four explicit ranges instead of quantiles.
- Assignment 4: word counts and separate validation/test evaluation.
- Assignments 6 and 7: reproducible settings, checkpoints, per-group metrics,
  prediction files, plots, and an observed-result report.
- Prior IndicCorpV2 data is monolingual, so it is not substituted for ILCI bitext.

## References

- Supplied ILLM-Lab8-Assignment.pdf.
- OpenNMT-py: https://github.com/OpenNMT/OpenNMT-py
- BPCC: https://huggingface.co/datasets/ai4bharat/BPCC
- SacreBLEU: https://github.com/mjpost/sacrebleu
- SentencePiece: https://github.com/google/sentencepiece
