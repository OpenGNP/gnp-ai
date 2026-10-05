"""Every tunable the pipeline reads, in one place.

Values live in gnp-ai/.env (gitignored) so the same code runs on a laptop and
on the faculty VM without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# The server moved its tables into the "test" Postgres schema (see
# gnp-server/src/db/schema.ts: pgSchema("test")); "public" is a stale copy.
DB_SCHEMA = os.environ.get("DB_SCHEMA", "test")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DECOMPOSE_MODEL = os.environ.get("DECOMPOSE_MODEL", "mistral:7b-instruct")
DECOMPOSE_PROMPT = os.environ.get("DECOMPOSE_PROMPT", "v1")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
SENTIMENT_MODEL = os.environ.get(
    "SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
VIOLATION_MODEL = os.environ.get("VIOLATION_MODEL", "unitary/toxic-bert")
# Zero-shot NLI alternative to asking the 7B LLM: same task, ~370MB and ~10x faster.
# Swap for MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 to cover Thai.
VIOLATION_ZEROSHOT_MODEL = os.environ.get(
    "VIOLATION_ZEROSHOT_MODEL", "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
)
# Swept over the 34-case labelled set in data/violation_testset.csv: 0.65 gives
# 88% correct with zero false positives, against 85% and 4 false positives when
# the raw argmax is taken. Genuine reports score 0.72-0.88, ordinary complaints
# that the model leans on score 0.53-0.62, so the gap sits here.
VIOLATION_ZEROSHOT_THRESHOLD = float(os.environ.get("VIOLATION_ZEROSHOT_THRESHOLD", "0.65"))
VIOLATION_THRESHOLD = float(os.environ.get("VIOLATION_THRESHOLD", "0.5"))
CLASSIFIER_BATCH_SIZE = int(os.environ.get("CLASSIFIER_BATCH_SIZE", "16"))

# Calibrated on the 145-point corpus (see data/reports/cluster_all-MiniLM-L6-v2_
# n145_mcs4_reduce0.3_split13_*.txt). Kept as percentages so they still hold as
# the corpus grows, which is why cluster_points.py takes --mcs-pct at all.
MIN_CLUSTER_SIZE_PCT = float(os.environ.get("MIN_CLUSTER_SIZE_PCT", "2.8"))
SPLIT_LARGE_PCT = float(os.environ.get("SPLIT_LARGE_PCT", "9.0"))
OUTLIER_THRESHOLD = float(os.environ.get("OUTLIER_THRESHOLD", "0.30"))

# Topic identity across runs. A new group inherits an existing topic's id when
# it holds at least TOPIC_CARRY_MIN of that topic's surviving points, which is
# what keeps topic_trends continuous. TOPIC_SPLIT_MIN is the share a second
# group needs before the run is reported as a split rather than noise.
# Both are starting points, not calibrated numbers — see docs/pipeline.md.
TOPIC_CARRY_MIN = float(os.environ.get("TOPIC_CARRY_MIN", "0.50"))
TOPIC_SPLIT_MIN = float(os.environ.get("TOPIC_SPLIT_MIN", "0.20"))

# Written into topic_versions.model_version so a later run can tell which
# canonical_topics rows this pipeline owns and may replace. Rows without this
# marker (the seeded demo topics) are never touched.
PIPELINE_TAG = "pipeline:gnp-ai"


def require_database_url() -> str:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return DATABASE_URL
