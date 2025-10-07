#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
import json
import datetime
from typing import Dict, Any, List

from flask import Flask, render_template, request, redirect, url_for, session, send_file, flash
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import the converter
from .video2docs import Video2Docs
from .jobs import job_manager

# Load environment variables
load_dotenv()

# Configuration
ADMIN_USER = os.environ.get("ADMIN_USER", "admin")
ADMIN_PASS = os.environ.get("ADMIN_PASS", "admin123")
# Resolve OUTPUT_DIR to an absolute path based on the repository root (one level up from this file)
_default_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "output"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", _default_output_dir)
TEMP_DIR = os.environ.get("TEMP_DIR")
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), "templates"), static_folder=os.path.join(os.path.dirname(__file__), "static"))
app.secret_key = os.environ.get("SECRET_KEY", "video2docs-secret-key")

# Ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
HISTORY_FILE = os.path.join(OUTPUT_DIR, "conversions.json")


def _load_history() -> List[Dict[str, Any]]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(items: List[Dict[str, Any]]):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

# Inject history helpers into job manager (after functions are defined)
try:
    job_manager.inject_history_funcs(_load_history, _save_history)
except Exception:
    pass


def _add_history_item(item: Dict[str, Any]):
    items = _load_history()
    items.insert(0, item)
    _save_history(items)


def _get_item(item_id: str) -> Dict[str, Any] | None:
    for it in _load_history():
        if it.get("id") == item_id:
            return it
    return None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(fn):
    from functools import wraps

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return fn(*args, **kwargs)

    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == ADMIN_USER and password == ADMIN_PASS:
            session["logged_in"] = True
            session["username"] = username
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        flash("Invalid credentials", "danger")
    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    # New conversion form
    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    items = _load_history()
    return render_template("history.html", items=items)


@app.route("/history/<item_id>")
@login_required
def history_detail(item_id: str):
    item = _get_item(item_id)
    if not item:
        flash("Conversion not found", "warning")
        return redirect(url_for("history"))
    return render_template("detail.html", item=item)


@app.route("/history/<item_id>/delete", methods=["POST"])
@login_required
def history_delete(item_id: str):
    items = _load_history()
    index_to_delete = None
    for idx, it in enumerate(items):
        if it.get("id") == item_id:
            index_to_delete = idx
            break
    if index_to_delete is None:
        flash("Conversion not found", "warning")
        return redirect(url_for("history"))

    # Try to cancel running job if any (safe to call even if not running)
    try:
        job_manager.cancel(item_id)
    except Exception:
        pass

    # Do not delete files on disk to avoid accidental data loss; this only removes from history
    del items[index_to_delete]
    _save_history(items)
    flash("Conversion deleted from history", "success")
    return redirect(url_for("history"))


@app.route("/download/<item_id>")
@login_required
def download(item_id: str):
    item = _get_item(item_id)
    if not item:
        flash("Conversion not found", "warning")
        return redirect(url_for("history"))
    result_path = item.get("result_path")
    # Fallback: if the stored absolute path does not exist (e.g., due to cwd changes),
    # try to resolve by filename in the current OUTPUT_DIR
    candidate_path = None
    if result_path and not os.path.isabs(result_path):
        candidate_path = os.path.join(OUTPUT_DIR, os.path.basename(result_path))
    elif result_path:
        candidate_path = result_path

    if not candidate_path or not os.path.exists(candidate_path):
        # Try another fallback using just the basename in OUTPUT_DIR
        base = os.path.basename(result_path or "")
        if base:
            alt = os.path.join(OUTPUT_DIR, base)
            if os.path.exists(alt):
                candidate_path = alt
                # Update history to fix the path for future requests
                items = _load_history()
                for it in items:
                    if it.get("id") == item_id:
                        it["result_path"] = candidate_path
                        break
                _save_history(items)
        if not candidate_path or not os.path.exists(candidate_path):
            flash("File not found for this conversion", "warning")
            return redirect(url_for("history_detail", item_id=item_id))

    return send_file(candidate_path, as_attachment=True)


@app.route("/run", methods=["POST"])
@login_required
def run_conversion():
    input_source = request.form.get("input", "").strip()
    output_format = request.form.get("format", "docx").strip().lower()
    output_name = request.form.get("output_name", "").strip()

    uploaded_file_path = None
    if "file" in request.files and request.files["file"] and request.files["file"].filename:
        file = request.files["file"]
        if file and allowed_file(file.filename):
            safe_name = secure_filename(file.filename)
            upload_dir = os.path.join(OUTPUT_DIR, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            uploaded_file_path = os.path.join(upload_dir, safe_name)
            file.save(uploaded_file_path)
        else:
            flash("Unsupported file type", "danger")
            return redirect(url_for("index"))

    # Determine the actual input for processing
    actual_input = uploaded_file_path or input_source
    if not actual_input:
        flash("Please provide a URL or upload a video file", "danger")
        return redirect(url_for("index"))

    # Default output name based on input if not provided
    if not output_name:
        if uploaded_file_path:
            base = os.path.splitext(os.path.basename(uploaded_file_path))[0]
        else:
            # derive from URL last path or generic timestamp
            base = os.path.splitext(os.path.basename(input_source.split("?")[0]))[0] or "conversion"
        output_name = base

    # Prepare conversion record
    item_id = uuid.uuid4().hex
    started_at = datetime.datetime.now().isoformat()
    record: Dict[str, Any] = {
        "id": item_id,
        "input": actual_input,
        "input_is_url": uploaded_file_path is None,
        "format": output_format,
        "output_name": output_name,
        "status": "running",
        "result_path": None,
        "started_at": started_at,
        "finished_at": None,
        "error": None,
        # live fields updated by job manager
        "progress": 0.0,
        "step": "queued",
        "eta_seconds": None,
    }
    _add_history_item(record)

    # Prepare per-job temporary directory based on input source to avoid collisions
    temp_root = TEMP_DIR or os.path.join(OUTPUT_DIR, "temp")
    try:
        os.makedirs(temp_root, exist_ok=True)
    except Exception:
        pass
    # Derive a slug from the uploaded file name or URL path
    if uploaded_file_path:
        src_base = os.path.splitext(os.path.basename(uploaded_file_path))[0]
    else:
        # Use last path segment of URL without query, or filename if local path provided
        url_part = os.path.basename(input_source.split("?")[0]) or input_source
        src_base = os.path.splitext(url_part)[0] or "input"
    src_slug = secure_filename(src_base) or "input"
    per_job_temp_dir = os.path.join(temp_root, f"{src_slug}-{item_id[:8]}")
    os.makedirs(per_job_temp_dir, exist_ok=True)

    # Enqueue background job
    def job_fn(progress_cb=None, cancel_event=None):
        converter = Video2Docs(output_dir=OUTPUT_DIR, temp_dir=per_job_temp_dir)
        return converter.process(actual_input, output_format=output_format, output_name=output_name,
                                 progress_callback=progress_cb, cancel_event=cancel_event)

    job_manager.enqueue(item_id, job_fn, params={
        "input": actual_input,
        "format": output_format,
        "output_name": output_name,
    })
    flash("Conversion started in background", "info")
    return redirect(url_for("running", item_id=item_id))


@app.route("/running/<item_id>")
@login_required
def running(item_id: str):
    item = _get_item(item_id)
    if not item:
        flash("Conversion not found", "warning")
        return redirect(url_for("history"))
    return render_template("running.html", item=item)


@app.route("/jobs/<item_id>/status")
@login_required
def job_status(item_id: str):
    st = job_manager.status(item_id)
    if st is None:
        it = _get_item(item_id)
        if not it:
            return {"error": "not found"}, 404
        return {
            "status": it.get("status"),
            "progress": it.get("progress", 100 if it.get("status") == "completed" else 0),
            "step": it.get("step"),
            "eta_seconds": it.get("eta_seconds"),
            "result_path": it.get("result_path"),
            "error": it.get("error"),
        }
    return {
        "status": st.get("status"),
        "progress": st.get("progress"),
        "step": st.get("step"),
        "eta_seconds": st.get("eta_seconds"),
        "result_path": st.get("result_path"),
        "error": st.get("error"),
    }


@app.route("/jobs/<item_id>/cancel", methods=["POST"])
@login_required
def job_cancel(item_id: str):
    ok = job_manager.cancel(item_id)
    if ok:
        flash("Cancellation requested", "warning")
    else:
        flash("Job not found or already finished", "warning")
    return redirect(url_for("running", item_id=item_id))


@app.route("/rerun/<item_id>", methods=["POST"])
@login_required
def rerun(item_id: str):
    # Rerun with potentially modified params
    original = _get_item(item_id)
    if not original:
        flash("Conversion not found", "warning")
        return redirect(url_for("history"))

    input_source = request.form.get("input", original.get("input", "")).strip()
    output_format = request.form.get("format", original.get("format", "docx")).strip().lower()
    output_name = request.form.get("output_name", original.get("output_name", "")).strip()

    # Create a new conversion (do not overwrite original)
    request.form = request.form.copy()
    return run_conversion()


def create_app() -> Flask:
    return app


def main():
    """Console entry point to run the web application."""
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug_env = os.environ.get("FLASK_DEBUG") or os.environ.get("DEBUG") or "0"
    debug = str(debug_env).lower() in ("1", "true", "yes", "on")
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
