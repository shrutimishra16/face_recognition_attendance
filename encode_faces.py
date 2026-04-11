import face_recognition
import os
import pickle

dataset_path = "datasets"

known_encodings = []
known_names = []

print("Encoding faces...")

for file in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, file)

    image = face_recognition.load_image_file(img_path)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) > 0:
        encoding = encodings[0]

        name = file.split(".")[0]  # filename = name

        known_encodings.append(encoding)
        known_names.append(name)

        print(f"Encoded {name}")

# Save encodings
data = {
    "encodings": known_encodings,
    "names": known_names
}

with open("encodings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Done! Encodings saved.")
