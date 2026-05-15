from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SPLITS_DIR = DATA_DIR / "splits"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
LOGS_DIR = OUTPUTS_DIR / "logs"
CONFIGS_DIR = PROJECT_ROOT / "configs"

RAW_CSV = DATA_DIR / "multilingual_hate_speech_dataset.csv"
CLEAN_CSV = DATA_DIR / "cleaned_multilingual_hate_speech_dataset.csv"

LABELS = ["Not Hate", "Neutral", "Hate"]
LABEL_TO_ID = {"Not Hate": 0, "Neutral": 1, "Hate": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

LANGUAGES = ["English", "Yoruba", "Igbo", "Hausa"]
