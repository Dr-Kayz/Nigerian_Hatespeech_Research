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
data/         raw + cleaned CSVs, splits/
notebooks/    EDA + per-scenario notebooks (numbered 01..10)
src/          reusable modules (data, features, models, eval, utils)
configs/      YAML per experiment
outputs/      tables, figures, checkpoints, predictions, logs
reports/      write-up
```

## Class encoding

| class | label    |
|-------|----------|
| 0     | Not Hate |
| 1     | Neutral  |
| 2     | Hate     |

Source: Kaggle "Nigerian Multilingual Hate Speech" data card (NaijaHate-derived, NLLB-translated to Yoruba/Igbo/Hausa).
