# ILLM Lab Assignments

Coursework by **Vishal Gautam (U23AI127)**, B.Tech Artificial Intelligence, SVNIT Surat.

This repository contains the executed notebooks for assignments 1 through 11,
assignment PDFs, Python scripts where available, requirements, reports, tables,
plots, and saved predictions.

| Assignment | Topic | Notebook | Instructions |
| --- | --- | --- | --- |
| 1 and 2 | Initial language-model labs | [Assignments 1 and 2](assign1%26assign2.ipynb) | See the assignment PDFs |
| 3 | BPE and WordPiece tokenization | [Assignment 3](Assignment3_BPE_WordPiece.ipynb) | Notebook instructions |
| 4 | N-gram language models | [Assignment 4](Assign4_NGram.ipynb) | Notebook instructions |
| 5 | Embeddings and clustering | [Assignment 5](Assignment5_Embeddings_Clustering.ipynb) | Notebook instructions |
| 6 | Language identification | [Assignment 6](Assignment6_Language_Identification.ipynb) | [README](Assignment6_README.md) |
| 7 | HMM noun-phrase tagging | [Assignment 7](Lab7/Assignment7_HMM_Noun_Phrase.ipynb) | [README](Lab7/Assignment7_README.md) |
| 8 | English-to-Hindi machine translation | [Assignment 8](Lab8/Assignment8_English_Hindi_MT.ipynb) | [README](Lab8/Assignment8_README.md) |
| 9 | Pretrained embeddings for machine translation | [Assignment 9](Lab9/Assignment9_Pretrained_Embeddings_MT.ipynb) | [README](Lab9/Assignment9_README.md) |
| 10 | Midsem solutions and multilingual BERT | [Assignment 10](Lab10/Assignment10_Midsem_BERT.ipynb) | [README](Lab10/Assignment10_README.md) |
| 11 | BERT layer probing | [Assignment 11](Lab11/Assignment11_BERT_Layer_Probing.ipynb) | [README](Lab11/Assignment11_README.md) |

The `Lab7` through `Lab11` folders contain the current notebooks and their
supporting files. Existing top-level notebook copies are retained for compatibility.
Results are saved in `assignmentN_artifacts/`, either at the repository root
or inside the relevant lab folder.

## Running the assignments

Start with the notebook or the linked README. For labs with a requirements file,
install dependencies in a suitable Python environment before running the notebook
or matching Python script. Labs that use OpenNMT and BERT may need separate
environments, as explained in their READMEs.

Python environments, installed runtimes, temporary files, downloaded datasets,
trained model checkpoints, embedding caches, and extracted feature caches are
excluded from this upload. Recreate the required data and models using the
notebook instructions before rerunning experiments. Lab 7's supplied input files
are included in `Lab7/Data/`. The saved outputs remain available for reviewing
the completed experiments.
