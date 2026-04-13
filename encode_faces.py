import face_recognition
import os
import pickle

def generate_encodings():
    dataset_path = "datasets"

    if os.path.exists("encodings.pkl"):
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
        print("Loaded existing encodings...")
    else:
        known_encodings = []
        known_names = []
        print("No existing encodings found. Creating new...")

    print("Checking for new faces...")

    for file in os.listdir(dataset_path):

        if file.startswith(".") or not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name = file.split(".")[0]

        if name in known_names:
            continue

        img_path = os.path.join(dataset_path, file)
        image = face_recognition.load_image_file(img_path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]

            known_encodings.append(encoding)
            known_names.append(name)

            print(f"New face encoded: {name}")

    data = {
        "encodings": known_encodings,
        "names": known_names
    }

    with open("encodings.pkl", "wb") as f:
        pickle.dump(data, f)

    print("Encodings updated successfully!")


if __name__ == "__main__":
    generate_encodings()