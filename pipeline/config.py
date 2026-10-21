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

# The server's tables live in "public" — gnp-server/src/db/schema.ts declares
# them with pgTable and no schema. A "test" schema still exists in the
# database from an earlier move, holding stale copies of all twelve tables, so
# the default matters: pointed there the pipeline would read and write real
# rows and report nothing wrong.
DB_SCHEMA = os.environ.get("DB_SCHEMA", "public")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
DECOMPOSE_MODEL = os.environ.get("DECOMPOSE_MODEL", "mistral:7b-instruct")
DECOMPOSE_PROMPT = os.environ.get("DECOMPOSE_PROMPT", "v1")

# Longest answer that bypasses decomposition when it holds no clause break.
# Measured on the ingested corpus: no answer of two to six words ever split
# into more than one point, while answers of seven to ten did — eleven of
# fifty-two. See pipeline/decompose.py:needs_splitting for why the model must
# not see these at all.
SHORT_ANSWER_WORDS = int(os.environ.get("SHORT_ANSWER_WORDS", "6"))
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

# The smallest group worth calling a topic. Two bars, because they answer
# different questions, and min_cluster_size is whichever is higher:
#
#   MIN_CLUSTER_POINTS  below this a group is too small to act on, however
#                       small the corpus. A floor of 2 — HDBSCAN's own minimum
#                       — said nothing about the work.
#   MIN_CLUSTER_SIZE_PCT  on a large corpus, do not surface what is
#                       statistically trivial.
#
# A percentage alone scales the wrong way: it coarsens topics as feedback
# accumulates, when what should stay constant is how fine a topic is. At 2.0%
# the 1109-point corpus reached min_cluster_size 22, and themes raised eight
# or fifteen times — orientation missing visa information, announcements lost
# in a Teams scroll — were dropped as noise while coherence (NPMI) fell to
# -0.197, meaning a topic's own words co-occurred less than chance. These
# values give 8 at 398 points and 11 at 1109, both measured directly.
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

# Recorded as ai_model_runs.model_version so a run can be traced back to the
# code that produced it. It is not an ownership marker: canonical_topics.form_id
# already scopes a run to the topics it is responsible for.
PIPELINE_TAG = "pipeline:gnp-ai"


def require_database_url() -> str:
    if not DATABASE_URL:
        raise SystemExit("DATABASE_URL is not set. Copy .env.example to .env and fill it in.")
    return DATABASE_URL
