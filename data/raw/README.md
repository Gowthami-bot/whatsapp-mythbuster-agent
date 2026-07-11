# IFND Dataset Directory

This directory is the expected location for the **Indian Fake News Dataset (IFND)** CSV file.

## Setup

1. Download the IFND dataset (e.g., from Kaggle or the original research source).
2. Place the CSV file in this directory as `IFND.csv`.

## Expected CSV Format

The corpus loader (`data/load_corpus.py`) expects the following columns:

| Column      | Description                                      |
|-------------|--------------------------------------------------|
| `statement` | The claim text (alternatives: `text`, `claim`)   |
| `label`     | Verdict label: `fake`/`real` or `0`/`1`          |
| `source`    | *(optional)* Original source of the claim        |
| `date`      | *(optional)* Publication date                    |

The loader will gracefully skip this source if the file is missing, but having
it indexed significantly improves the agent's accuracy on common Indian myths.
