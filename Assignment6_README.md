# Assignment 6

Open [Assignment6_Language_Identification.ipynb](Assignment6_Language_Identification.ipynb) and run the cells from top to bottom from this folder. The [Python version](Assignment6_Language_Identification.py) contains the same code and explanations in numbered notebook cells.

The solution follows `assign1&assign2.ipynb`: 23 languages, 2,000 sentences per language, the same paragraph tokenizer, and a 1,600/200/200 split for each language. The original notebook did not save its sentence lists. The new notebook can reuse those lists if they are still available in its kernel; otherwise it reconstructs that sampling procedure from a pinned IndicCorpV2 revision and saves the splits.

It trains logistic regression, MLP with 0/1/2 hidden layers, LSTM, GRU, and supervised FastText. The classifiers use official FastText Wiki vectors, and macro-F1 is calculated from scratch over all 23 languages. The report explains the five unavailable Wiki models, word-vector coverage, and exact-text overlaps between the splits.

The existing Python 3.14 environment can run the notebook on this computer. FastText uses the isolated Python 3.12 runtime already prepared in `assignment6_artifacts/runtime/fasttext_python`. For a separate Windows installation, use Python 3.12 in a virtual environment and install [requirements-assignment6.txt](requirements-assignment6.txt).

The first execution scans the complete Wiki vector files for the 18 available languages, reading their compressed entries from the official archives. Only matching vectors are stored, and each completed download chunk is checkpointed. Extraction checks the file size, CRC, and vocabulary row count. Later executions reuse the saved data, vectors, and trained models. Set `FORCE_RETRAIN = True` in the setup cell to retrain the classifiers.

The main outputs are [the comparison CSV](assignment6_artifacts/assignment6_classifier_comparison.csv) and [the completed report](assignment6_artifacts/assignment6_report.md). The same artifact folder contains predictions, per-language precision/recall/F1, confusion matrices, training histories, model checkpoints, the dataset manifest, and comparison plots.
