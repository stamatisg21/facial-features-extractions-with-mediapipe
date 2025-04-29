import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import time

# Function to compute the eye aspect ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to extract facial landmarks using Mediapipe
def extract_landmarks(frame, face_mesh):
    results = face_mesh.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmark_coords = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmark_coords.append((x, y))
            return landmark_coords
    return None

# Path to the input video file
video_path = "pat1_aligned.MP4"
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("[INFO] opening video file...")

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
blink_durations = []
blink_amplitudes = []

# Initialize lists to store EAR values
left_ear_values = []
right_ear_values = []
frame_numbers = []

# Initialize Mediapipe face mesh model
mp_face_mesh = mp.solutions.face_mesh
with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = extract_landmarks(frame_rgb, face_mesh)

        if landmarks:
            leftEye = np.array([landmarks[i] for i in [33, 160, 158, 133, 153, 144]])
            rightEye = np.array([landmarks[i] for i in [362, 385, 387, 263, 373, 380]])

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            left_ear_values.append(leftEAR)
            right_ear_values.append(rightEAR)
            frame_numbers.append(frame_number)

            # Check for blink
            if leftEAR < 0.2 and rightEAR < 0.2:
                COUNTER += 1
            else:
                if COUNTER >= 3:  # if the eyes were below the threshold for enough frames, count as blink
                    TOTAL += 1
                    blink_durations.append(COUNTER)
                    blink_amplitudes.append((leftEAR + rightEAR) / 2)  # average EAR during blink
                COUNTER = 0

        frame_number += 1

cap.release()

# Calculate blink statistics
dur = total_frames / fps
blink_freq = TOTAL / dur
blink_amplitude_mean = np.mean(blink_amplitudes) if blink_amplitudes else float('nan')
blink_amplitude_min = np.min(blink_amplitudes) if blink_amplitudes else None
blink_amplitude_max = np.max(blink_amplitudes) if blink_amplitudes else None
blink_duration_stdev = np.std(blink_durations) / fps if blink_durations else float('nan')
blink_frame_percentage = (TOTAL / total_frames) * 100

# Print the results
print("Total blinks:", TOTAL)
print("Blink frequency:", blink_freq)
print("Blink Amplitude Mean:", blink_amplitude_mean)
print("Blink Amplitude Minimum:", blink_amplitude_min)
print("Blink Amplitude Maximum:", blink_amplitude_max)
print("Blink Duration Standard Dev.:", blink_duration_stdev)
print("Percentage of Total Frames with Blinks:", blink_frame_percentage)

# Plot the EAR values over time
plt.figure(figsize=(12, 8))

# Subplot for left eye EAR
plt.subplot(2, 1, 1)
plt.plot(frame_numbers, left_ear_values, label='Left Eye EAR', color='blue')
plt.xlabel('Frame Number')
plt.ylabel('Eye Aspect Ratio (EAR)')
plt.title('Left Eye EAR Over Time')
plt.legend()

# Subplot for right eye EAR
plt.subplot(2, 1, 2)
plt.plot(frame_numbers, right_ear_values, label='Right Eye EAR', color='red')
plt.xlabel('Frame Number')
plt.ylabel('Eye Aspect Ratio (EAR)')
plt.title('Right Eye EAR Over Time')
plt.legend()

# Adjust layout
plt.tight_layout()
plt.show()
