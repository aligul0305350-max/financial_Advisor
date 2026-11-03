import cv2
import os
import numpy as np
from PIL import Image


# =========================================================
# SETTINGS
# =========================================================

FACE_DATA_DIR = "face_data"
MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.npy"

# OpenCV ka built-in face detector
CASCADE_PATH = cv2.data.haarcascades + (
    "haarcascade_frontalface_default.xml"
)

# Face data folder agar nahi hai to bana do
os.makedirs(FACE_DATA_DIR, exist_ok=True)


# =========================================================
# FACE DETECTOR LOAD
# =========================================================

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


# =========================================================
# STREAMLIT IMAGE KO OPENCV IMAGE MEIN CONVERT KARNA
# =========================================================

def image_to_cv2(uploaded_image):

    image = Image.open(uploaded_image).convert("RGB")

    image_np = np.array(image)

    frame = cv2.cvtColor(
        image_np,
        cv2.COLOR_RGB2BGR
    )

    return frame


# =========================================================
# FACE DETECTION
# =========================================================

def detect_faces(frame):

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    return gray, faces


# =========================================================
# FACE REGISTER
# =========================================================

def register_face(
    uploaded_image,
    username,
    image_number
):

    # Image ko OpenCV mein convert karo
    frame = image_to_cv2(
        uploaded_image
    )

    # Face detect karo
    gray, faces = detect_faces(
        frame
    )

    # Face nahi mila
    if len(faces) == 0:

        return (
            False,
            "Face detect nahi hua. "
            "Camera ke samne apna face clear rakhein."
        )

    # Ek se zyada faces
    if len(faces) > 1:

        return (
            False,
            "Image mein sirf aik face hona chahiye."
        )

    # Face ki position
    x, y, w, h = faces[0]

    # Face crop karo
    face = gray[
        y:y + h,
        x:x + w
    ]

    # Username ka folder
    user_folder = os.path.join(
        FACE_DATA_DIR,
        username
    )

    os.makedirs(
        user_folder,
        exist_ok=True
    )

    # Image ka naam
    image_path = os.path.join(
        user_folder,
        f"{image_number}.jpg"
    )

    # Face save karo
    cv2.imwrite(
        image_path,
        face
    )

    return (
        True,
        f"Face image {image_number} save ho gayi."
    )


# =========================================================
# TRAIN FACE RECOGNITION MODEL
# =========================================================

def train_model():

    faces = []
    labels = []

    label_names = {}

    current_label = 0

    # Face data folder check
    if not os.path.exists(
        FACE_DATA_DIR
    ):

        return (
            False,
            "Face data folder nahi mila."
        )

    # Har user ka folder
    for username in os.listdir(
        FACE_DATA_DIR
    ):

        user_folder = os.path.join(
            FACE_DATA_DIR,
            username
        )

        # Sirf folders
        if not os.path.isdir(
            user_folder
        ):
            continue

        # Username ko numeric label
        label_names[
            current_label
        ] = username

        # User ki images
        for image_name in os.listdir(
            user_folder
        ):

            image_path = os.path.join(
                user_folder,
                image_name
            )

            # Grayscale image read
            image = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            # Invalid image skip
            if image is None:
                continue

            faces.append(image)

            labels.append(
                current_label
            )

        current_label += 1

    # Agar images nahi hain
    if len(faces) == 0:

        return (
            False,
            "Koi registered face image nahi mili."
        )

    # Check LBPH available hai
    if not hasattr(
        cv2,
        "face"
    ):

        return (
            False,
            "OpenCV Face module nahi mila. "
            "opencv-contrib-python install karein."
        )

    # LBPH recognizer
    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    # Model train
    recognizer.train(
        faces,
        np.array(labels)
    )

    # Model save
    recognizer.save(
        MODEL_PATH
    )

    # Labels save
    np.save(
        LABELS_PATH,
        label_names
    )

    return (
        True,
        "Face Recognition Model successfully train ho gaya."
    )


# =========================================================
# FACE RECOGNITION
# =========================================================

def recognize_face(
    uploaded_image
):

    # Model check
    if not os.path.exists(
        MODEL_PATH
    ):

        return (
            False,
            "Face model nahi mila. "
            "Pehle face register aur model train karein."
        )

    # Labels check
    if not os.path.exists(
        LABELS_PATH
    ):

        return (
            False,
            "Labels file nahi mili. "
            "Pehle model train karein."
        )

    # OpenCV Face module check
    if not hasattr(
        cv2,
        "face"
    ):

        return (
            False,
            "OpenCV Face module nahi mila. "
            "opencv-contrib-python install karein."
        )

    # Image convert
    frame = image_to_cv2(
        uploaded_image
    )

    # Face detect
    gray, faces = detect_faces(
        frame
    )

    # Face nahi mila
    if len(faces) == 0:

        return (
            False,
            "Face detect nahi hua. "
            "Apna face camera ke samne clear rakhein."
        )

    # Multiple faces
    if len(faces) > 1:

        return (
            False,
            "Sirf aik face camera ke samne hona chahiye."
        )

    # Recognizer
    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    # Trained model load
    recognizer.read(
        MODEL_PATH
    )

    # Labels load
    labels = np.load(
        LABELS_PATH,
        allow_pickle=True
    ).item()

    # Face crop
    x, y, w, h = faces[0]

    face = gray[
        y:y + h,
        x:x + w
    ]

    # Prediction
    label, distance = (
        recognizer.predict(face)
    )

    # Label exist karta hai?
    if label not in labels:

        return (
            False,
            "Unknown Face. Face match nahi hua."
        )

    # Username
    username = labels[label]

    # LBPH distance:
    # kam distance = behtar match
    #
    # 50 se kam = strong match
    # 50-70 = possible match
    # 70 se zyada = weak match

    if distance > 70:

        return (
            False,
            "Face match nahi hua. "
            "Ye registered face nahi lagta."
        )

    if distance <= 50:

        match_level = "Strong Match"

    else:

        match_level = "Possible Match"

    return (
        True,
        f"Face Recognized: {username} | "
        f"{match_level} | "
        f"Distance: {distance:.2f}"
    )