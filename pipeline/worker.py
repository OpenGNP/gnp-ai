"""Long-running worker for one pipeline stage.

    python -m pipeline.worker --stage ingest    # answers -> points
    python -m pipeline.worker --stage cluster   # points  -> topics

Runs under systemd (deploy/gnp-ai@.service), one per stage. Each loops while
there is work and sleeps only when there is none, so rounds never overlap.
The queue is the data: answers.decomposed_at IS NULL for ingest,
points.processing_status = 'pending' for cluster. Separate workers so that
clustering never waits behind an ingest backlog.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
import traceback

import httpx
import psycopg

from . import config, db, run

_log = run._log

# Backoff after a failed round, doubling up to the max.
_BACKOFF_START = 30.0
_BACKOFF_MAX = 5 * 60.0


class Worker:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.stopping = threading.Event()
        self.lock = db.StageLock(stage)
        self._backoff = _BACKOFF_START
        # cluster only: form -> (failures, monotonic time of next try)
        self._retry_at: dict[int, tuple[int, float]] = {}

    # ── lifecycle ──────────────────────────────────────────────────────

    def request_stop(self, signum, _frame) -> None:
        _log(f"{self.stage}: {signal.Signals(signum).name} received, finishing the current unit")
        self.stopping.set()

    def sleep(self, seconds: float) -> None:
        self.stopping.wait(seconds)

    def wait_for_lock(self) -> bool:
        """Block until this process holds the stage lock. False on stop."""
        warned = False
        while not self.stopping.is_set():
            if self.lock.held():
                if warned:
                    _log(f"{self.stage}: lock acquired")
                return True
            if not warned:
                _log(f"{self.stage}: another {self.stage} process holds the lock; waiting")
                warned = True
            self.sleep(30)
        return False

    def failed(self, exc: BaseException) -> None:
        _log(f"{self.stage}: round failed ({exc.__class__.__name__}: {exc}); "
             f"retrying in {self._backoff:.0f}s")
        if not isinstance(exc, (httpx.HTTPError, psycopg.OperationalError)):
            # not a dependency being down — probably a bug
            traceback.print_exc()
        self.sleep(self._backoff)
        self._backoff = min(self._backoff * 2, _BACKOFF_MAX)

    def run(self) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)

        _log(f"{self.stage}: starting")
        conn = db.connect()
        try:
            loop = self.ingest_round if self.stage == "ingest" else self.cluster_round
            while self.wait_for_lock():
                try:
                    did_work = loop(conn)
                except Exception as exc:  # noqa: BLE001 — the worker must outlive one bad round
                    self.failed(exc)
                    continue
                self._backoff = _BACKOFF_START
                if not did_work:
                    self.sleep(self.idle_seconds)
        finally:
            conn.close()
            self.lock.release()
            _log(f"{self.stage}: stopped")

    @property
    def idle_seconds(self) -> float:
        return config.INGEST_IDLE_SEC if self.stage == "ingest" else config.CLUSTER_IDLE_SEC

    # ── ingest ─────────────────────────────────────────────────────────

    def ingest_round(self, conn) -> bool:
        """One chunk from each field (the limit is per field)."""
        written = run.ingest(
            conn,
            limit=run.CHUNK,
            dry_run=False,
            skip_sentiment=False,
            stop=self.stopping.is_set,
            log_idle=False,
        )
        return written > 0

    # ── cluster ────────────────────────────────────────────────────────

    def cluster_round(self, conn) -> bool:
        did_work = False
        for form_id, title in db.analysable_forms(conn):
            if self.stopping.is_set():
                break
            state = db.cluster_readiness(conn, form_id)
            if not state.pending:
                continue

            failures, retry_at = self._retry_at.get(form_id, (0, 0.0))
            if time.monotonic() < retry_at:
                continue

            reason = self.due(state)
            if reason is None:
                continue

            _log(f"cluster: form {form_id} due — {reason}")
            try:
                run.cluster_form(
                    conn, form_id, title, dry_run=False, stop=self.stopping.is_set
                )
            except Exception as exc:  # noqa: BLE001 — one form must not starve the rest
                failures += 1
                delay = min(_BACKOFF_START * 2 ** failures, _BACKOFF_MAX * 4)
                self._retry_at[form_id] = (failures, time.monotonic() + delay)
                _log(f"cluster: form {form_id} failed ({exc.__class__.__name__}: {exc}); "
                     f"next try in {delay / 60:.0f} min")
                if not isinstance(exc, (httpx.HTTPError, psycopg.OperationalError)):
                    traceback.print_exc()
                continue

            self._retry_at.pop(form_id, None)
            did_work = True
        return did_work

    @staticmethod
    def due(state: db.Readiness) -> str | None:
        """Why the form is due now, or None. Quiet is measured from the newest
        pending point, so it only counts once ingest has caught up; max-wait
        covers a form that never goes quiet."""
        oldest = state.oldest_age_min or 0.0
        newest = state.newest_age_min or 0.0
        if oldest >= config.CLUSTER_MAX_WAIT_MIN:
            return (f"{state.pending} point(s) waiting {oldest:.0f} min "
                    f"(max {config.CLUSTER_MAX_WAIT_MIN:.0f}), backlog {state.backlog}")
        if state.backlog == 0 and newest >= config.CLUSTER_QUIET_MIN:
            return f"{state.pending} new point(s), quiet for {newest:.0f} min"
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["ingest", "cluster"], required=True)
    args = ap.parse_args()

    Worker(args.stage).run()
    sys.exit(0)


if __name__ == "__main__":
    main()
