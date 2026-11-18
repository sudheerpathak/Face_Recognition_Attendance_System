# Face Recognition Attendance System

A beginner-level university project that replaces manual roll-call with
webcam-based face recognition, served as a web app.

**Pipeline:** Enroll → Face Embedding → Live Camera → Matching → Database → Attendance → Web

1. **Enroll** — a student's roll number + name are submitted via a web form.
2. **Face Embedding** — while the student looks at the webcam, each captured
   frame is run through a face detector and a recognizer that turns the face
   into a 128-dimensional embedding vector; several samples are averaged
   into one embedding per student.
3. **Live Camera** — the browser streams the server's webcam feed live via
   MJPEG (no native OpenCV windows — everything renders inside the page).
4. **Matching** — every face seen live is embedded the same way and compared
   (cosine similarity) against all enrolled embeddings to find the best match.
5. **Database** — SQLite stores each student's embedding and every
   attendance row.
6. **Attendance** — a match above the threshold marks that student present
   for today (once per day, enforced by a DB constraint).
7. **Web** — the whole pipeline (enroll form, live camera, matching overlay,
   records table) is exposed through a Flask app in the browser.

## Tech stack

- **AI / Computer Vision:** OpenCV's YuNet face detector + SFace face
  recognizer (`cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF`), both ship as
  small ONNX models via `opencv-contrib-python`. No dlib/CMake build and no
  TensorFlow required — just two model files downloaded once.
- **Database:** SQLite (`attendance.db`, created automatically).
- **Web:** Flask, with live video served as an MJPEG stream and vanilla JS
  for polling/refreshing the UI (no frontend build step).

## Project structure

```
face-recognition-attendance/
├── app.py                 # Flask routes: enroll/attendance/records + MJPEG streams
├── face_engine.py         # YuNet detection + SFace embedding/matching wrapper
├── database.py            # SQLite schema + queries (students.embedding, attendance)
├── download_models.py     # fetches the two ONNX models into models/
├── models/                 # created by download_models.py
├── templates/              # index.html, enroll.html, attendance.html, records.html
├── static/style.css
├── requirements.txt
└── attendance.db           # created at runtime
```

## Download

```powershell
git clone https://github.com/gabal-ahmed/face-recognition-attendance.git
cd face-recognition-attendance
```

(No git? Click the green **Code** button on the GitHub page → **Download ZIP**,
then extract it and open the extracted folder instead of cloning.)

Requirements to run it: **Python 3.9+** installed and on your PATH, and a
webcam. Check with `python --version` in a terminal.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
```

`download_models.py` fetches two files from the OpenCV Zoo into `models/`:
`face_detection_yunet_2023mar.onnx` (~230 KB) and
`face_recognition_sface_2021dec.onnx` (~37 MB). Run it once; `app.py` will
refuse to start the recognizer if they're missing.

> Install `opencv-contrib-python` only — do **not** also have plain
> `opencv-python` installed at the same time; they conflict and the
> `cv2.FaceRecognizerSF` / `cv2.FaceDetectorYN` APIs won't be available.

## Running

```powershell
python app.py
```

Open http://127.0.0.1:5000 in a browser. You'll land on a login screen first.

**Default admin login:** username `admin`, password `admin123`. Change them
before a real demo by setting environment variables before running the app:

```powershell
$env:ADMIN_USERNAME = "your_username"
$env:ADMIN_PASSWORD = "your_password"
$env:SECRET_KEY = "any-random-string"
python app.py
```

- **Enroll** — enter roll number + name, then look at the camera until the
  sample counter reaches its target (15 by default).
- **Take Attendance** — click Start; recognized students get a green box +
  name and are marked present (once per day); unrecognized faces get a red
  "Unknown" box. Click Stop to release the camera.
- **Records** — browse/filter all attendance rows by date.

## Notes / known limitations (good material for the report)

- This is a **single-machine, single-camera** demo: only one browser tab
  should drive the camera at a time (enroll and attendance both hold a lock
  around the webcam so they can't run simultaneously). Fine for a class demo,
  not written for multi-room/multi-camera deployment.
- `COSINE_MATCH_THRESHOLD = 0.363` in `face_engine.py` is SFace's officially
  recommended threshold; lower it for stricter matching, raise it if
  legitimate students are being marked "Unknown".
- Enrollment averages several embeddings into one vector per student instead
  of storing every sample — simpler to compare at attendance time, but more
  sensitive to a bad capture angle than a nearest-neighbor-over-all-samples
  approach.

## Suggested report additions

- A short **system architecture diagram** following the pipeline above:
  Browser → Flask → Webcam → YuNet (detect) → SFace (embed) → cosine match
  → SQLite → Browser.
- A brief **evaluation**: test with N students, report recognition accuracy
  and false-accept/false-reject observations at the default threshold.
- A note on why SFace embeddings (deep, 128-d, cosine-comparable) are a
  meaningful upgrade over classical methods like Eigenfaces/LBPH for the
  "AI" component of the abstract.
