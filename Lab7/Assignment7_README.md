# Assignment 7

Open [Assignment7_HMM_Noun_Phrase.ipynb](Assignment7_HMM_Noun_Phrase.ipynb) and run the cells from top to bottom. The notebook is self-contained and reads the supplied files in `Data/`. It works from `Lab7` or the parent assignment folder.

The [Python version](Assignment7_HMM_Noun_Phrase.py) has the same code and explanations in notebook-style cells. To run it from the parent folder:

```powershell
python -m pip install -r Lab7/requirements-assignment7.txt
python Lab7/Assignment7_HMM_Noun_Phrase.py
```

The solution follows Assignment 4's count-based probabilities and Add-K smoothing with K = 0.1, plus Assignment 6's notebook and result structure. It implements the HMM, log-space forward algorithm, and Viterbi decoding from scratch using NumPy. Seqeval evaluates exact noun-phrase spans in strict IOB2 mode.

The vocabulary uses training words occurring at least twice. Rarer training words and unseen test words share `<UNK>`. Sentence likelihoods describe these mapped observations under a fixed-length HMM. All settings are fixed before test evaluation.

Results are saved in `assignment7_artifacts/`:

- [Completed report](assignment7_artifacts/assignment7_report.md)
- [Evaluation summary](assignment7_artifacts/assignment7_results.csv)
- [Every test sentence's probability and tags](assignment7_artifacts/test_sentence_probabilities.csv)
- [Token predictions](assignment7_artifacts/test_token_predictions.csv)
- [Seqeval report](assignment7_artifacts/seqeval_report.txt)
- [Evaluation plot](assignment7_artifacts/plots/test_evaluation.png)

The same folder contains NP spans, error analysis, input summaries, parameter tables, and model arrays/metadata. Exhaustive short-sequence checks verify the forward sum and Viterbi maximum independently of evaluation scores.
