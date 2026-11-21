# Labs 9, 10 and 11

Solutions follow the earlier notebooks' name/roll-number header, numbered
explanations, fixed settings, saved predictions, tables, plots and reports.

| Lab | Notebook | Main reference |
| --- | --- | --- |
| 9 | [Pretrained embeddings for MT](Lab9/Assignment9_Pretrained_Embeddings_MT.ipynb) | Lab 8's exact English-Hindi splits; Labs 3 and 5 |
| 10 | [Midsem solutions and multilingual BERT](Lab10/Assignment10_Midsem_BERT.ipynb) | Midsem paper; Labs 1 and 6's language-identification data/results |
| 11 | [Twelve-layer BERT probing](Lab11/Assignment11_BERT_Layer_Probing.ipynb) | SentEval probing data; earlier stratification/evaluation conventions |

Each folder has a matching Python version, requirements, and a README with
run instructions. Reports and measured outputs are in that lab's
`assignmentN_artifacts/` folder. Earlier notebooks and their results are kept
as references. Lab 9 uses a separate CUDA Python 3.12 OpenNMT runtime, sharing
Lab 8's other packages. Lab 10 uses the installed `Labs GPU (Python 3.13)`
kernel in `.venv-labs-gpu/`. Lab 11's completed CPU experiment uses the existing
main Python environment. Each notebook contains its actual experiment settings.

For submission, use the executed notebooks and any requested reports/results.
The installed runtimes, downloaded pretrained checkpoints, and feature caches
are computational resources, so include them only if your instructor asks.
