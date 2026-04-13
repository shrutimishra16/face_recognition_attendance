from flask import Flask, render_template, Response, jsonify, request
import cv2
import face_recognition
import pickle
import numpy as np
import os
import time

from encode_faces import generate_encodings
from database import mark_attendance

app = Flask(__name__)

ENCODINGS_PATH = "encodings.pkl"

# Generate encodings only if missing
if not os.path.exists(ENCODINGS_PATH):
    generate_encodings()

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

marked_names = set()
attendance_message = ""

# Cooldown to avoid message spam
last_message_time = 0
COOLDOWN = 1   # increase a bit for stability


def generate_frames():
    global attendance_message, last_message_time

    video = cv2.VideoCapture(0)

    while True:
        success, frame = video.read()
        if not success:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            distances = face_recognition.face_distance(known_encodings, face_encoding)

            name = "Unknown"

            if len(distances) > 0:
                best_match = np.argmin(distances)

                if matches[best_match]:
                    name = known_names[best_match]

                    current_time = time.time()

                    if current_time - last_message_time > COOLDOWN:

                        if name not in marked_names:
                            mark_attendance(name)  
                            attendance_message = f"{name} marked present"
                            marked_names.add(name)
                        else:
                            attendance_message = f"{name} already present"

                        last_message_time = current_time

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_message')
def get_message():
    return jsonify({"message": attendance_message})


@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    if 'image' not in request.files:
        return jsonify({"error": "Missing 'image' file in form-data"}), 400

    file = request.files['image']
    img_bytes = file.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "Could not decode image"}), 400

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if not face_encodings:
        return jsonify({"message": "No face detected", "results": []})

    results = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)
        name = "Unknown"
        attendance_status = None

        if len(distances) > 0:
            best_match = np.argmin(distances)
            if matches[best_match]:
                name = known_names[best_match]
                attendance_status = mark_attendance(name)

        results.append({"name": name, "status": attendance_status})

    return jsonify({"message": f"{len(results)} face(s) processed", "results": results})


if __name__ == "__main__":
    app.run(debug=True, port=8000, use_reloader=False)