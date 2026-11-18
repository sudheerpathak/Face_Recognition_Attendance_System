"""Flask web app implementing the pipeline:
Enroll -> Face Embedding -> Live Camera -> Matching -> Database -> Attendance -> Web
"""
import os
import threading
import time
from datetime import datetime
from functools import wraps

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

import database as db
from face_engine import COSINE_MATCH_THRESHOLD, get_engine

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-change-me")
db.init_db()

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD_HASH = generate_password_hash(os.environ.get("ADMIN_PASSWORD", "admin123"))

SAMPLES_PER_STUDENT = 15
CAPTURE_COOLDOWN = 0.3       # seconds between accepted samples while enrolling
ATTENDANCE_MARK_COOLDOWN = 5  # seconds before re-checking an already-marked student

camera_lock = threading.Lock()

enroll_state = {
    "active": False,
    "roll_no": None,
    "name": None,
    "embeddings": [],
    "count": 0,
    "done": False,
    "student_id": None,
    "error": None,
}

attendance_state = {"active": False}
_last_marked = {}  # student_id -> datetime, resets on process restart


def _open_camera():
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam = cv2.VideoCapture(0)
    return cam


def _encode_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def _mjpeg_chunk(jpeg_bytes):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


# ---------- Auth ----------

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and check_password_hash(ADMIN_PASSWORD_HASH, password):
            session["logged_in"] = True
            return redirect(request.args.get("next") or url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------- Home ----------

@app.route("/")
@login_required
def index():
    students = db.get_all_students()
    today = datetime.now().strftime("%Y-%m-%d")
    today_rows = db.get_attendance(today)
    recent = db.get_attendance()[:5]
    return render_template(
        "index.html",
        student_count=len(students),
        today_count=len(today_rows),
        recent=recent,
    )


# ---------- Enroll ----------

@app.route("/enroll")
@login_required
def enroll_page():
    return render_template("enroll.html", target=SAMPLES_PER_STUDENT)


@app.route("/enroll/start", methods=["POST"])
@login_required
def enroll_start():
    roll_no = request.form.get("roll_no", "").strip()
    name = request.form.get("name", "").strip()
    if not roll_no or not name:
        return jsonify({"error": "Roll number and name are required."}), 400

    enroll_state.update({
        "active": True,
        "roll_no": roll_no,
        "name": name,
        "embeddings": [],
        "count": 0,
        "done": False,
        "student_id": None,
        "error": None,
    })
    return jsonify({"ok": True})


def _finish_enrollment():
    try:
        stacked = np.vstack(enroll_state["embeddings"])
        avg_embedding = np.mean(stacked, axis=0, keepdims=True).astype(np.float32)
        student_id = db.add_student(
            enroll_state["roll_no"], enroll_state["name"], avg_embedding.tobytes()
        )
        enroll_state["student_id"] = student_id
    except Exception as e:
        enroll_state["error"] = str(e)
    finally:
        enroll_state["done"] = True
        enroll_state["active"] = False


@app.route("/enroll_feed")
@login_required
def enroll_feed():
    def generate():
        engine = get_engine()
        with camera_lock:
            cam = _open_camera()
            last_capture = 0.0
            try:
                while enroll_state["active"] and not enroll_state["done"]:
                    ok, frame = cam.read()
                    if not ok:
                        break

                    faces = engine.detect_faces(frame)
                    face = engine.largest_face(faces)

                    if face is not None:
                        x, y, w, h = face[:4].astype(int)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        now = time.time()
                        if (
                            now - last_capture > CAPTURE_COOLDOWN
                            and enroll_state["count"] < SAMPLES_PER_STUDENT
                        ):
                            embedding = engine.embed(frame, face)
                            enroll_state["embeddings"].append(embedding)
                            enroll_state["count"] += 1
                            last_capture = now

                    cv2.putText(
                        frame, f"Samples: {enroll_state['count']}/{SAMPLES_PER_STUDENT}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
                    )

                    if enroll_state["count"] >= SAMPLES_PER_STUDENT:
                        _finish_enrollment()

                    jpeg = _encode_jpeg(frame)
                    if jpeg:
                        yield _mjpeg_chunk(jpeg)
            finally:
                cam.release()

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/enroll_status")
@login_required
def enroll_status():
    return jsonify({
        "count": enroll_state["count"],
        "target": SAMPLES_PER_STUDENT,
        "done": enroll_state["done"],
        "student_id": enroll_state["student_id"],
        "error": enroll_state["error"],
    })


# ---------- Attendance ----------

@app.route("/attendance")
@login_required
def attendance_page():
    return render_template("attendance.html")


@app.route("/attendance/start", methods=["POST"])
@login_required
def attendance_start():
    attendance_state["active"] = True
    return jsonify({"ok": True})


@app.route("/attendance/stop", methods=["POST"])
@login_required
def attendance_stop():
    attendance_state["active"] = False
    return jsonify({"ok": True})


@app.route("/attendance_feed")
@login_required
def attendance_feed():
    def generate():
        engine = get_engine()
        known = [
            (sid, roll_no, name, np.frombuffer(emb, dtype=np.float32).reshape(1, -1))
            for sid, roll_no, name, emb in db.get_all_students_with_embeddings()
        ]

        with camera_lock:
            cam = _open_camera()
            attendance_state["active"] = True
            try:
                while attendance_state["active"]:
                    ok, frame = cam.read()
                    if not ok:
                        break

                    faces = engine.detect_faces(frame)

                    for face in faces:
                        x, y, w, h = face[:4].astype(int)
                        embedding = engine.embed(frame, face)

                        best_score, best_match = -1.0, None
                        for sid, roll_no, name, known_emb in known:
                            score = engine.cosine_score(embedding, known_emb)
                            if score > best_score:
                                best_score, best_match = score, (sid, roll_no, name)

                        if best_match and best_score >= COSINE_MATCH_THRESHOLD:
                            sid, roll_no, name = best_match
                            color, label = (0, 255, 0), f"{name} ({best_score:.2f})"

                            now = datetime.now()
                            last = _last_marked.get(sid)
                            if not last or (now - last).total_seconds() > ATTENDANCE_MARK_COOLDOWN:
                                db.mark_attendance(
                                    sid, roll_no, name,
                                    now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"),
                                )
                                _last_marked[sid] = now
                        else:
                            color, label = (0, 0, 255), "Unknown"

                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(
                            frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2,
                        )

                    jpeg = _encode_jpeg(frame)
                    if jpeg:
                        yield _mjpeg_chunk(jpeg)
            finally:
                cam.release()
                attendance_state["active"] = False

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/attendance/today")
@login_required
def api_attendance_today():
    today = datetime.now().strftime("%Y-%m-%d")
    rows = db.get_attendance(today)
    return jsonify([
        {"roll_no": r[0], "name": r[1], "date": r[2], "time": r[3]} for r in rows
    ])


# ---------- Records ----------

@app.route("/records")
@login_required
def records_page():
    date_filter = request.args.get("date") or None
    rows = db.get_attendance(date_filter)
    return render_template("records.html", rows=rows, date_filter=date_filter or "")


if __name__ == "__main__":
    app.run(debug=True, threaded=True)
