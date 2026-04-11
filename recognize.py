import cv2
import face_recognition
import pickle
import numpy as np
from database import mark_attendance

# Load saved encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("Starting camera...")

video = cv2.VideoCapture(0)
marked_names = set()

while True:
    ret, frame = video.read()

    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        distances = face_recognition.face_distance(known_encodings, face_encoding)

        name = "Unknown"

        if len(distances) > 0:
            best_match = np.argmin(distances)

            if matches[best_match]:
                name = known_names[best_match]
                if name not in marked_names:
                  mark_attendance(name)
                  marked_names.add(name)

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
