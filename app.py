from flask import Flask, render_template, jsonify, request, send_file
import cv2
import face_recognition
import pickle
import numpy as np
import os
import time
import logging
import glob
from datetime import datetime

from database import mark_attendance, get_connection

app = Flask(__name__)

ENCODINGS_PATH = "encodings.pkl"
THRESHOLD = 0.40
COOLDOWN = 5


logging.basicConfig(level=logging.INFO)


if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    known_encodings  = data.get("encodings", [])
    known_names      = data.get("names", [])
    known_ids        = data.get("ids", [])
    known_classes    = data.get("classes", [])
else:
    known_encodings  = []
    known_names      = []
    known_ids        = []
    known_classes    = []


if len(known_encodings) == 0:
    logging.warning("No encodings found — enroll students first.")

last_seen_time = {}


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/bus-scan')
def index():
    return render_template('index.html')


@app.route('/enroll')
def enroll():
    return render_template('enroll.html')


@app.route('/enroll/capture')
def enroll_capture():
    return render_template('enroll_capture.html')


@app.route('/api/enroll', methods=['POST'])
def api_enroll():
    global known_encodings, known_names

    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400

    student_id = request.form.get('student_id', '').strip()
    full_name  = request.form.get('full_name',  '').strip()
    if not student_id or not full_name:
        return jsonify({'success': False, 'error': 'Student ID and name are required'}), 400

    # Save image to datasets/
    os.makedirs('datasets', exist_ok=True)
    safe_name  = full_name.replace(' ', '_')
    img_path   = os.path.join('datasets', f'{safe_name}_{student_id}.jpg')

    file      = request.files['image']
    img_bytes = file.read()
    np_arr    = np.frombuffer(img_bytes, np.uint8)
    frame     = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'success': False, 'error': 'Invalid image data'}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model='hog')
    if not locations:
        return jsonify({'success': False, 'error': 'No face detected in the captured image. Please try again.'})

    encoding = face_recognition.face_encodings(rgb, locations)[0]

    # Check for duplicate face
    if known_encodings:
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        if any(matches):
            matched_name = known_names[matches.index(True)]
            return jsonify({'success': False, 'error': f'This face is already enrolled as "{matched_name}".'})

    # Save image file
    cv2.imwrite(img_path, frame)

    # Update in-memory encodings
    known_encodings.append(encoding)
    known_names.append(full_name)
    known_ids.append(student_id)
    known_classes.append(request.form.get('class_section', '').strip())

    # Persist to encodings.pkl
    with open(ENCODINGS_PATH, 'wb') as f:
        pickle.dump({'encodings': known_encodings, 'names': known_names, 'ids': known_ids, 'classes': known_classes}, f)

    logging.info(f'Enrolled: {full_name} ({student_id})')
    return jsonify({'success': True, 'message': f'{full_name} enrolled and face encoded successfully.'})


@app.route('/api/photo/<path:name>')
def student_photo(name):
    for ext in ('jpg', 'jpeg', 'png'):
        matches = glob.glob(os.path.join('datasets', f'{name}.{ext}'))
        if matches:
            return send_file(matches[0], mimetype=f'image/{ext}')
    return '', 404


@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400

    bus_no = request.form.get("bus_no", "Unknown")

    file = request.files['image']
    img_bytes = file.read()

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

    face_locations = face_recognition.face_locations(small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    if not face_encodings:
        return jsonify({
            "message": "No face detected",
            "results": []
        })

    results = []
    current_time = time.time()
    recognized_names = set()

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        top *= 2
        right *= 2
        bottom *= 2
        left *= 2

        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"
        student_id = ''
        attendance_status = "Face not recognized"
        confidence = 0.0
        beep = False

        if len(distances) > 0:
            best_match = np.argmin(distances)
            best_distance = distances[best_match]
            confidence = float(round(1 - best_distance, 3))

            if best_distance < THRESHOLD:
                name       = known_names[best_match]
                student_id = known_ids[best_match] if best_match < len(known_ids) else ''

                if name not in recognized_names:
                    recognized_names.add(name)

                    if name not in last_seen_time or (current_time - last_seen_time[name]) > COOLDOWN:
                        attendance_status = mark_attendance(name, bus_no)
                        last_seen_time[name] = current_time
                        beep = True
                        logging.info(f"Attendance marked for {name}")
                    else:
                        attendance_status = "Already marked recently"
                        logging.info(f"{name} skipped due to cooldown")

        results.append({
            "name": name,
            "student_id": student_id if name != 'Unknown' else '',
            "status": attendance_status,
            "confidence": confidence,
            "beep": beep,
            "box": {
                "top": int(top),
                "right": int(right),
                "bottom": int(bottom),
                "left": int(left)
            }
        })

    return jsonify({
        "message": f"{len(results)} face(s) processed",
        "total_faces": len(results),
        "results": results
    })


@app.route('/records')
def records():
    return render_template('records.html')


@app.route('/api/records')
def api_records():
    date = request.args.get('date', datetime.now().strftime('%Y-%m-%d'))

    # Build lookup: name -> {id, class}
    lookup = {}
    for i, n in enumerate(known_names):
        lookup[n] = {
            'id':    known_ids[i]    if i < len(known_ids)    else '',
            'class': known_classes[i] if i < len(known_classes) else ''
        }

    try:
        conn   = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, date, time, bus_no FROM attendance WHERE date=? ORDER BY time DESC",
            (date,)
        )
        rows = cursor.fetchall()
        conn.close()

        records_list = []
        for row in rows:
            name = row[0]
            info = lookup.get(name, {'id': '', 'class': ''})
            # Format time to 12h
            try:
                t = datetime.strptime(str(row[2]), '%H:%M:%S').strftime('%I:%M:%S %p')
            except:
                t = str(row[2])
            records_list.append({
                'student_id': info['id'],
                'name':       name,
                'class':      info['class'] or '—',
                'time':       t,
                'bus_no':     row[3] or ''
            })

        total_enrolled = len(set(known_names)) or 1
        boarded        = len(records_list)
        rate           = round(boarded / total_enrolled * 100)

        return jsonify({'records': records_list, 'boarded': boarded, 'rate': rate})
    except Exception as e:
        logging.error(f'Records error: {e}')
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)