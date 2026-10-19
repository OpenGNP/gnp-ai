# Pipeline: from form answers to dashboard topics

The AI pipeline connects to the same Postgres the server uses, reads answers
that have not been processed, and writes the tables the server already serves.
`gnp-server` never calls it: the database is the whole interface.

```
answers (decomposed_at NULL, field marked analyze_with_ai)
   │
   │  gnp-ai@ingest     decompose ─► embed ─► sentiment + violation
   ▼                    one transaction per chunk: INSERT points, SET decomposed_at
points (processing_status = 'pending')
   │
   │  gnp-ai@cluster    per form, once it is due (see below)
   ▼                    BERTopic + HDBSCAN ─► topic identity ─► LLM naming
canonical_topics · points.canonical_topic_id · ai_model_runs
   │
   ▼
gnp-server reads these → dashboard
```

## Models

| Stage | Model |
|---|---|
| Decompose | `mistral:7b-instruct` via Ollama |
| Embedding | `all-MiniLM-L6-v2` (384d) |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Violation | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` (zero-shot NLI) |
| Clustering | BERTopic + HDBSCAN + UMAP |
| Topic naming | the same Mistral, same Ollama |

Decomposition is one LLM call per answer and dominates runtime; answers of six
words or fewer with no clause break skip the model entirely
(`decompose.needs_splitting`). The classifiers are small CPU models run once
per chunk. After changing either classifier or `VIOLATION_ZEROSHOT_THRESHOLD`,
run `python -m pipeline.run --relabel`.

## Setup

Python **3.10–3.13**. Not 3.14: `numba`, which `umap-learn` needs for BERTopic,
refuses to build there.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .                 # add [experiments] for scripts/build_dataset.py
cp .env.example .env             # then fill in DATABASE_URL and OLLAMA_URL
```

Ollama runs as its own service, normally on another host:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral:7b-instruct
```

## Running it

In production two workers run continuously (next section). By hand:

```bash
python -m pipeline.run --ingest-only --dry-run   # what is waiting, writes nothing
python -m pipeline.run --once                    # ingest new answers, then re-cluster
python -m pipeline.run --cluster-only            # re-cluster every form (after a parameter change)
python -m pipeline.run --report                  # what every stage produced, read-only
python -m pipeline.worker --stage ingest         # one worker in the foreground
```

A run that writes takes the same lock as the worker for its stage, so it
refuses to start while that worker is running. Stop the worker first
(`sudo systemctl stop gnp-ai@cluster`) or add `--dry-run`.

## The two workers

`pipeline/worker.py`, one process per stage. Each loops for as long as it finds
work and sleeps only when it finds none (`INGEST_IDLE_SEC`, `CLUSTER_IDLE_SEC`),
so a round that takes minutes delays the next check rather than overlapping it.
The queue is the data itself — there is no jobs table.

**ingest** takes one chunk of 20 answers from each field in turn, so one form's
backlog cannot hold up another's. It writes the chunk's points and marks the
answers in a single transaction. An answer Ollama times out or errors on is
retried up to `DECOMPOSE_MAX_ATTEMPTS`, then kept whole as one point. If Ollama
cannot be reached at all the round stops and retries with backoff, counting no
attempts against the answers.

**cluster** re-clusters a form that has pending points once either

- ingest has caught up on that form and no new point has arrived for
  `CLUSTER_QUIET_MIN` (default 30) — the responses have stopped; or
- its oldest pending point has waited `CLUSTER_MAX_WAIT_MIN` (default 360) — a
  form that never goes quiet still gets refreshed.

A form with nothing pending is never re-clustered. One that fails is retried
with backoff while the others carry on. If Ollama is down while naming, new
topics keep their c-TF-IDF keyword names rather than the run failing.

Why two and not one: clustering must not wait behind an ingest backlog. Both
share one Ollama and one loaded model; a naming call queues behind the
decomposition in flight, which costs seconds.

**One process per stage.** Each stage holds a Postgres advisory lock, on a
connection of its own. A second worker — most likely a laptop pointed at the
live database — waits for it instead of decomposing the same answers twice.

**Stopping is safe at any point.** SIGTERM finishes the answer or naming call in
progress and writes what is done. Even SIGKILL loses nothing but time: every
write is one transaction, and unfinished work is still in the queue.

### Ollama calls

`llm.chat` streams. `OLLAMA_READ_TIMEOUT` is the gap allowed between tokens, so
a long answer that keeps producing tokens is never cut off, while a hung
server is noticed within a minute. `OLLAMA_NUM_CTX` (8192) is sent on every
call: Ollama's default is much smaller and silently truncates the prompt, and
using one value everywhere stops Ollama reloading the model between
decomposition and naming. The decomposition output budget grows with the
answer so long answers are not cut off mid-JSON.

## Deploying

The workers run from a git checkout under systemd on the self-hosted runner's
machine (`ontario2`). Pushing to `main` runs `.github/workflows/deploy.yml`:
reset the checkout to `origin/main`, `pip install -e .` only if
`pyproject.toml` changed, a dry-run smoke test against the database and
Ollama, then restart both workers.

One-time setup on the host:

1. Find the user the runner runs as: `ps -o user= -p $(pgrep -f Runner.Listener)`.
   The checkout, the unit's `User=` and the workflow's `APP_DIR` (repository
   variable `GNP_AI_DIR`, default `/home/deploy/gnp-ai`) must all agree.
2. As that user: clone the repo to `APP_DIR`, create `.venv`, `pip install -e .`,
   and write `.env`. If the repo is private, add a read-only deploy key.
3. Install the unit, after editing `User=` and the paths in it:

   ```bash
   sudo cp deploy/gnp-ai@.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now gnp-ai@ingest gnp-ai@cluster
   ```

4. Let the runner restart them without a password. The command must match the
   workflow's exactly:

   ```bash
   echo '<runner-user> ALL=(root) NOPASSWD: /usr/bin/systemctl restart gnp-ai@ingest.service gnp-ai@cluster.service' \
     | sudo tee /etc/sudoers.d/gnp-ai && sudo visudo -cf /etc/sudoers.d/gnp-ai
   ```

5. Ollama has no authentication. Allow port 11434 only from the worker host
   (`ufw allow from <ip> to any port 11434 proto tcp`), or keep it on
   `127.0.0.1` when both run on one machine.

Changes to `.env` are not deployed — edit it on the host and restart the
workers.

Watching it:

```bash
journalctl -u 'gnp-ai@*' -f              # both workers' logs
systemctl status gnp-ai@ingest gnp-ai@cluster
```

```sql
-- what ingest still has to do, per form
SELECT s.form_id, count(*) FROM answers a
JOIN submissions s ON s.id = a.submission_id
JOIN form_fields f ON f.id = a.field_id
WHERE f.analyze_with_ai AND a.decomposed_at IS NULL AND btrim(a.answer_text) <> ''
GROUP BY 1;

-- when each form was last clustered
SELECT form_id, max(completed_at) FROM ai_model_runs GROUP BY 1;
```

## Safety on a shared database

The server owns this database; the pipeline is a second writer.

1. **Ingest is append-only.** It reads only answers with `decomposed_at IS NULL`
   and sets it in the same transaction as their points, so re-running it never
   duplicates.
2. **Clustering only sees points with an embedding.** Seeded demo points have
   none, so they are never disturbed. `--backfill` opts them in.
3. **Clustering only touches its own form.** `canonical_topics.form_id` scopes a
   run; a topic left holding no points at the end of a run is deleted.

## Where the modelling lives

`pipeline/` imports the prompt, the JSON recovery and the
BERTopic/HDBSCAN/UMAP configuration from `scripts/decompose_ollama.py` and
`scripts/cluster_points.py`, which remain the experiment CLIs that produced the
reports in `data/reports/`. One implementation, two entry points. Settings that
only make sense for an experiment run (single-threaded OpenMP, silenced
warnings) apply when those scripts are run directly, not when imported.

## What is_violation means

`is_violation` is the zero-shot flag in `pipeline/violation.py`: the feedback
*reports* a legal, ethical or moral breach — bribery, cheating, harassment —
however calmly it is worded. An ordinary complaint about teaching or
facilities is not one, however angrily it is written.

## Column vocabulary enforced by the database

- `points.processing_status` — `pending` (awaiting clustering), `assigned` (in a
  topic), `unknown` (outlier), `clustered`. `chk_status_topic` requires
  `assigned` exactly when a topic is set.
- `points.sentiment_label` and `is_violation` — set together or both NULL
  (`chk_labels_together`).

## Known gaps

- **The English-only stack.** The embedding, sentiment and violation models are
  English-trained; Thai feedback embeds and classifies poorly. The
  multilingual counterparts named in `.env.example` are untested.
- **Sentiment is judged per atomic point, not per response.** A response split
  into three points contributes three votes.
- **Edited or deleted answers are not reprocessed.** The server does neither
  today. If it starts to, it must delete the answer's points and clear
  `decomposed_at` so ingest picks it up again.
