# Pipeline: from form answers to dashboard topics

The AI pipeline is a batch job. It connects to the same Postgres the server
uses, reads answers that have not been processed, and writes the tables the
server already serves. Nothing in `gnp-server` or `gnp-client` changes.

```
answers ──► decompose ──► embed ──► sentiment ──► points
                                                    │
                                                    ▼
                                        cluster (whole corpus)
                                                    │
                    canonical_topics · topic_trends · topic_versions
                    unassigned_points · ai_model_runs
                                                    │
                                                    ▼
              /api/topics · /api/analytics/summary · /api/analytics/trends
```

## Models

| Stage | Model |
|---|---|
| Decompose | `mistral:7b-instruct` via Ollama (local) |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Violation / toxicity | `unitary/toxic-bert` |
| Embedding | `all-MiniLM-L6-v2` (384d) |
| Clustering | BERTopic + HDBSCAN |
| Priority | pure formula, computed at read time (see below) |

Decomposition is one local-LLM call per answer and dominates runtime. The two
classifiers are small CPU models and run batched over the whole pass. After
changing either classifier or `VIOLATION_THRESHOLD`, re-run
`python -m pipeline.run --relabel`.

## Setup

Python **3.10–3.13**. Not 3.14: `numba`, which `umap-learn` needs for BERTopic,
refuses to build there.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -e .                 # add [experiments] for scripts/build_dataset.py
cp .env.example .env             # then fill in DATABASE_URL
```

## Running it

```bash
python -m pipeline.run --ingest-only --dry-run   # what is waiting, writes nothing
python -m pipeline.run --once                    # ingest new answers, then re-cluster
python -m pipeline.run --cluster-only            # re-cluster what is already stored
python -m pipeline.run --watch 60                # poll every 60s (demo stand-in for cron)
python -m pipeline.run --backfill                # embed pre-existing points (see below)
```

On the VM, `--watch` is replaced by cron:

```
0 2 * * *  cd ~/gnp-ai && .venv/bin/python -m pipeline.run --once >> ~/pipeline.log 2>&1
```

## Why the pipeline is split in two

Decomposition, embedding and sentiment are per-answer: they can run the moment
a response arrives. Clustering is corpus-level — HDBSCAN and BERTopic need
every point at once — so it cannot run per answer. The schema already assumes
this split: `points.processing_status` is per row, `topic_trends` is per
period, `ai_model_runs` is per pipeline run.

## Safety on a shared database

The server owns this database; the pipeline is a second writer. Three rules keep
that safe:

1. **Ingest is append-only.** It only inserts points for answers that have none,
   so re-running it is a no-op rather than a duplicate.
2. **Clustering only sees points with an embedding.** The seeded demo points have
   none, so they and their topic assignments are never disturbed. `--backfill`
   opts them in — after which clustering will reassign them.
3. **Clustering only deletes topics it created.** Each topic the pipeline writes
   gets a `topic_versions` row tagged `pipeline:gnp-ai`; the next run replaces
   exactly those. Seeded topics carry no marker and survive.

`--dry-run` prints every count and writes nothing.

## Where the modelling lives

`pipeline/` contains no modelling of its own. The prompt, the JSON recovery, the
BERTopic/HDBSCAN/UMAP configuration, the 0.30 outlier threshold and the
oversized-topic re-clustering are imported from `scripts/decompose_ollama.py` and
`scripts/cluster_points.py`, which remain the experiment CLIs that produced the
reports in `data/reports/`. One implementation, two entry points.

Defaults reproduce the tuned run at n=145 (`mcs=4`, `split-large=13`,
outlier threshold `0.30`), expressed as percentages of corpus size so they still
hold as the corpus grows. Override any of them in `.env`.

## Priority is not stored

Priority is a pure function of counts the database already holds, so the
pipeline does not compute or store it — whatever renders the dashboard computes
it from `topic_trends`. That keeps it in sync with the counts automatically and
needs no schema change.

The formula, per topic, over its most recent trend row:

```
reach     = feedback_count / max(feedback_count across topics)   # 0..1, relative
negative  = negative_count / feedback_count                      # 0..1
severity  = severe_count   / feedback_count                      # 0..1

priority  = 0.4 * reach + 0.4 * negative + 0.2 * severity
```

The weights are a judgement call, not a measurement — state them as such. Reach
is normalised against the largest topic rather than the corpus so a single
dominant topic cannot flatten everything else to near zero.

Reference SQL, so server and client cannot drift apart:

```sql
SELECT t.canonical_topic_id,
       0.4 * t.feedback_count::real / MAX(t.feedback_count) OVER ()
     + 0.4 * t.negative_count::real / NULLIF(t.feedback_count, 0)
     + 0.2 * t.severe_count::real   / NULLIF(t.feedback_count, 0) AS priority
FROM topic_trends t
ORDER BY priority DESC;
```

## What is_severe actually means

`is_severe` carries the `unitary/toxic-bert` flag: abusive or offensive
*language*, not the seriousness of the *problem being reported*. "The lab has no
working power outlets" is a serious complaint that scores near zero. The
`severeIssues` tile served by `/api/analytics/summary` therefore counts posts
needing moderation, not urgent issues. Problem-severity would need a second
column.

## Column vocabulary enforced by the database

- `points.processing_status` — `pending` (embedded, awaiting clustering),
  `assigned` (in a topic), `unknown` (outlier), `clustered`. Anything else
  violates `chk_processing`.
- `points.sentiment_label` — `positive`, `neutral`, `negative`, or NULL.
  A NULL is written when the model's answer cannot be parsed; a guessed label
  would be worse.

## Known gaps

- **Topic summaries are NULL.** `canonical_summary` and auto-generated titles are
  still an open decision; topics are currently named from their top c-TF-IDF
  terms.
- **Topic identity is not stable across runs.** A full re-cluster produces new
  topic ids, so `topic_trends` history does not carry over. Matching new
  centroids against existing topics is future work.
- **The English-only stack.** `all-MiniLM-L6-v2`,
  `twitter-roberta-base-sentiment-latest` and `toxic-bert` are all
  English-trained. Thai feedback will embed and classify poorly. The
  multilingual counterparts (`paraphrase-multilingual-MiniLM-L12-v2`,
  `twitter-xlm-roberta-base-sentiment`) are untested.
- **Sentiment is judged per atomic point, not per response.** A response split
  into three points contributes three sentiment votes, so over-segmentation
  inflates whichever sentiment dominates that response.
