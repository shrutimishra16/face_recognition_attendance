import requests
import os
import re

API_URL = "https://gyanshree.in/School/Bus/BusStdDataApp.ashx"
DATASET_PATH = "datasets"

os.makedirs(DATASET_PATH, exist_ok=True)

def sanitize_name(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name.strip())

def fetch_and_save():
    response = requests.get(API_URL)
    response.raise_for_status()
    students = response.json()

    print(f"Found {len(students)} students.")

    for student in students:
        name = sanitize_name(student["Name"])
        photo_url = student["PhotoUrl"]
        admission_no = student["AdmissionNo"]

        # Use name + admission number to avoid collisions
        filename = f"{name}_{admission_no}.jpg"
        filepath = os.path.join(DATASET_PATH, filename)

        if os.path.exists(filepath):
            print(f"Skipping (already exists): {filename}")
            continue

        try:
            img_response = requests.get(photo_url, timeout=10)
            img_response.raise_for_status()

            with open(filepath, "wb") as f:
                f.write(img_response.content)

            print(f"Saved: {filename}")
        except Exception as e:
            print(f"Failed to download {name} ({admission_no}): {e}")

    print("Done. Run encode_faces.py to generate encodings.")

if __name__ == "__main__":
    fetch_and_save()
