#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

# We keep history helpers decoupled: import from webapp where they currently live
# To avoid circular imports at module level, we'll accept the functions to be injected

class CancelledError(Exception):
    pass

class JobManager:
    def __init__(self, max_workers: int = 2, load_history_fn: Optional[Callable[[], list]] = None,
                 save_history_fn: Optional[Callable[[list], None]] = None):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._load_history = load_history_fn
        self._save_history = save_history_fn

    def inject_history_funcs(self, load_fn: Callable[[], list], save_fn: Callable[[list], None]):
        self._load_history = load_fn
        self._save_history = save_fn

    def enqueue(self, job_id: str, fn: Callable[..., str], *, params: Dict[str, Any]) -> None:
        cancel_event = threading.Event()
        state = {
            "status": "queued",
            "progress": 0.0,
            "step": "queued",
            "eta_seconds": None,
            "error": None,
            "result_path": None,
            "cancel_event": cancel_event,
            "started_at": None,
            "finished_at": None,
            "params": params,
        }
        with self._lock:
            self._jobs[job_id] = state

        def progress_cb(step: str, pct: float, eta_seconds: Optional[float]):
            self.update(job_id, {"step": step, "progress": float(pct or 0.0), "eta_seconds": eta_seconds})
            self._persist_to_history(job_id)

        def run():
            self.update(job_id, {"status": "running", "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            self._persist_to_history(job_id)
            try:
                result_path = fn(progress_cb=progress_cb, cancel_event=cancel_event)
                self.update(job_id, {
                    "status": "completed",
                    "progress": 100.0,
                    "result_path": result_path,
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            except CancelledError:
                self.update(job_id, {
                    "status": "canceled",
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            except Exception as e:
                self.update(job_id, {
                    "status": "failed",
                    "error": str(e),
                    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                })
            finally:
                self._persist_to_history(job_id)

        self._executor.submit(run)

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                job["cancel_event"].set()
                return True
        return False

    def status(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            st = self._jobs.get(job_id)
            if not st:
                return None
            # Return copy without cancel_event
            copy = dict(st)
            copy.pop("cancel_event", None)
            return copy

    def update(self, job_id: str, patch: Dict[str, Any]):
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(patch)

    def _persist_to_history(self, job_id: str):
        if not (self._load_history and self._save_history):
            return
        with self._history_lock:
            items = self._load_history()
            for it in items:
                if it.get("id") == job_id:
                    st = self._jobs.get(job_id, {})
                    it.update({
                        "status": st.get("status", it.get("status")),
                        "result_path": st.get("result_path", it.get("result_path")),
                        "finished_at": st.get("finished_at", it.get("finished_at")),
                        "error": st.get("error", it.get("error")),
                        "progress": st.get("progress", it.get("progress")),
                        "step": st.get("step", it.get("step")),
                        "eta_seconds": st.get("eta_seconds", it.get("eta_seconds")),
                    })
                    break
            self._save_history(items)

# Create a default instance; history funcs will be injected by web layer to avoid circular import
job_manager = JobManager(max_workers=int(os.environ.get("MAX_WORKERS", "2")) )
