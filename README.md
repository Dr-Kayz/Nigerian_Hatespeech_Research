# Cross-lingual Hate Speech Detection

Cross-lingual hate speech classification on English (high-resource) and three low-resource African languages: Yoruba, Igbo, Hausa.

3-class task: **Hate / Not Hate / Neutral**.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Verify the GPU (Apple Silicon MPS):

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Layout

```
data/             raw + cleaned + aligned + processed CSVs, train/val/test splits
src/
  data/           pipeline: clean, align_translations, preprocess, splits, eda
  features/       Phase 3: TF-IDF, LaBSE/LASER embeddings
  models/         Phase 3-4: ML baselines, transformer fine-tuning
  eval/           Phase 5: metrics, error analysis
  utils/          seed.py, io.py (paths + constants)
configs/          YAML per experiment
outputs/          tables, figures, checkpoints, predictions, logs
reports/          write-up
```

## Data pipeline

Run in order; each stage reads the previous stage's output:

```bash
python -m src.data.clean                # raw  -> cleaned
python -m src.data.align_translations   # cleaned -> aligned (adds source_id)
python -m src.data.preprocess           # aligned -> processed (artifacts, LID flags)
python -m src.data.splits               # processed -> data/splits/*.csv
python -m src.data.eda                  # writes tables + figures
```

## Class encoding

| class | label    |
|-------|----------|
| 0     | Not Hate |
| 1     | Neutral  |
| 2     | Hate     |

Source: Kaggle "Nigerian Multilingual Hate Speech" data card (NaijaHate-derived, NLLB-translated to Yoruba/Igbo/Hausa).
