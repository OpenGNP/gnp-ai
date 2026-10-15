"""Every tunable the pipeline reads, from gnp-ai/.env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DECOMPOSE_MODEL = os.environ.get("DECOMPOSE_MODEL", "mistral:7b-instruct")
DECOMPOSE_PROMPT = os.environ.get("DECOMPOSE_PROMPT", "v2")

# Same value on every call: Ollama reloads the model when num_ctx changes.
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))
OLLAMA_CONNECT_TIMEOUT = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5"))
# Max gap between streamed tokens, not a limit on the whole answer.
OLLAMA_READ_TIMEOUT = float(os.environ.get("OLLAMA_READ_TIMEOUT", "60"))
# Failures on one answer before it is kept whole as a single point.
DECOMPOSE_MAX_ATTEMPTS = int(os.environ.get("DECOMPOSE_MAX_ATTEMPTS", "3"))

# Answers this short with no clause break skip the model (decompose.needs_splitting).
SHORT_ANSWER_WORDS = int(os.environ.get("SHORT_ANSWER_WORDS", "6"))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SENTIMENT_MODEL = os.environ.get(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
# Legacy toxic-bert backend, only used by scripts/test_violation.py.
VIOLATION_MODEL = os.environ.get("VIOLATION_MODEL", "unitary/toxic-bert")
VIOLATION_THRESHOLD = float(os.environ.get("VIOLATION_THRESHOLD", "0.5"))
VIOLATION_ZEROSHOT_MODEL = os.environ.get(
    "VIOLATION_ZEROSHOT_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
)
# Swept on data/violation_testset.csv: 83% correct, 3 false positives.
VIOLATION_ZEROSHOT_THRESHOLD = float(os.environ.get("VIOLATION_ZEROSHOT_THRESHOLD", "0.75"))
CLASSIFIER_BATCH_SIZE = int(os.environ.get("CLASSIFIER_BATCH_SIZE", "16"))

# min_cluster_size is the larger of the two. A percentage alone grows with the
# corpus and drops real themes as noise (at 2.0% / 1109 points it reached 22).
MIN_CLUSTER_POINTS = int(os.environ.get("MIN_CLUSTER_POINTS", "8"))
MIN_CLUSTER_SIZE_PCT = float(os.environ.get("MIN_CLUSTER_SIZE_PCT", "1.0"))
OUTLIER_THRESHOLD = float(os.environ.get("OUTLIER_THRESHOLD", "0.30"))

# Share of an old topic's points a new group needs to keep its id / to count as a split.
TOPIC_CARRY_MIN = float(os.environ.get("TOPIC_CARRY_MIN", "0.50"))
TOPIC_SPLIT_MIN = float(os.environ.get("TOPIC_SPLIT_MIN", "0.20"))

# Workers sleep this long only when they find no work.
INGEST_IDLE_SEC = float(os.environ.get("INGEST_IDLE_SEC", "30"))
CLUSTER_IDLE_SEC = float(os.environ.get("CLUSTER_IDLE_SEC", "60"))
# Re-cluster a form once ingest has caught up and it has been quiet this long,
# or once its oldest new point has waited MAX_WAIT.
CLUSTER_QUIET_MIN = float(os.environ.get("CLUSTER_QUIET_MIN", "30"))
CLUSTER_MAX_WAIT_MIN = float(os.environ.get("CLUSTER_MAX_WAIT_MIN", "360"))


def require_database_url() -> str:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return DATABASE_URL
