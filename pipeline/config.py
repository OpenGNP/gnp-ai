"""Every tunable the pipeline reads, from gnp-ai/.env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# The server's tables live in "public" — gnp-server/src/db/schema.ts declares
# them with pgTable and no schema. A "test" schema still exists in the
# database from an earlier move, holding stale copies of all twelve tables, so
# the default matters: pointed there the pipeline would read and write real
# rows and report nothing wrong.
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
# The score is the entailment probability of the strongest violation label (see
# violation.classify_zeroshot). Swept over data/violation_testset.csv: 0.08 gives
# F1 0.90 (P 1.00, R 0.81), and 0.05-0.10 all stay within 0.02 of that. It looks
# low because each label is judged alone and ordinary complaints sit near 0.00.
# Between the two thresholds a point is violation_status 'possible' (needs a
# human look); at or above the strong one it is 'violation'. The median genuine
# report scores 0.93.
VIOLATION_ZEROSHOT_THRESHOLD = float(os.environ.get("VIOLATION_ZEROSHOT_THRESHOLD", "0.08"))
VIOLATION_ZEROSHOT_STRONG_THRESHOLD = float(
    os.environ.get("VIOLATION_ZEROSHOT_STRONG_THRESHOLD", "0.5")
)
VIOLATION_THRESHOLD = float(os.environ.get("VIOLATION_THRESHOLD", "0.5"))
CLASSIFIER_BATCH_SIZE = int(os.environ.get("CLASSIFIER_BATCH_SIZE", "16"))

# min_cluster_size is the larger of the two. A percentage alone grows with the
# corpus and drops real themes as noise (at 2.0% / 1109 points it reached 22).
MIN_CLUSTER_POINTS = int(os.environ.get("MIN_CLUSTER_POINTS", "8"))
MIN_CLUSTER_SIZE_PCT = float(os.environ.get("MIN_CLUSTER_SIZE_PCT", "1.0"))
OUTLIER_THRESHOLD = float(os.environ.get("OUTLIER_THRESHOLD", "0.30"))

# Cosine above which two points are treated as saying the same thing, and so
# are forced into one topic however the density landscape fell (see
# cluster.py:_merge_near_duplicates). Swept on form 3's 1109 points, counting
# how many points the pass moves and how large a transitive group it can build:
#
#   0.90   4 moved    0.85  15 moved    0.82  22 moved
#   0.80  24 moved    0.75  45 moved
#
# Group size stays at 15-16 across the whole range, so there is no cascade to
# guard against; the threshold only decides how much is caught. 0.80 takes the
# families that read as one complaint — three phrasings of the lunch queue,
# three of the signatures-per-request — and stops short of 0.75, where pairs
# start differing by more than a slot value.
NEAR_DUPLICATE_SIM = float(os.environ.get("NEAR_DUPLICATE_SIM", "0.80"))

# Topic identity across runs. A new group inherits an existing topic's id when
# it holds at least TOPIC_CARRY_MIN of that topic's surviving points, which is
# what keeps a topic's trend line continuous. TOPIC_SPLIT_MIN is the share a second
# group needs before the run is reported as a split rather than noise.
# Both are starting points, not calibrated numbers — see docs/pipeline.md.
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
