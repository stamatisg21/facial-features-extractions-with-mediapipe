# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

video_path = "pat3_aligned.MP4"
# Path to the input video file
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# initialize the frame counters and the total number of blinks
COUNTER = 0
TOTAL = 0
blink_durations = []
blink_amplitudes = []
blink_duration_stdev = []
blink_frame_count = 0

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

# This is a function provided by the dlib library. The shape_predictor function loads a pre-trained model for detecting 
#facial landmarks on a detected face. Facial landmarks are specific points on a face, such as the corners of the eyes, 
#the tip of the nose, and the edges of the lips.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream
print("[INFO] opening video file...")
vs = FileVideoStream(video_path).start()
time.sleep(1.0)

# Initialize lists to store EAR values
left_ear_values = []
right_ear_values = []
frame_numbers = []

# loop over frames from the video stream
frame_number = 0
while True:
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale
    frame = vs.read()
    if frame is None:
        break
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # loop over the face detections
    for rect in rects:
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the
        # coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # append the EAR values and frame number
        left_ear_values.append(leftEAR)
        right_ear_values.append(rightEAR)
        frame_numbers.append(frame_number)

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0

        # check to see if the eye aspect ratio is below the blink
        # threshold, and if so, increment the blink frame counter
        if ear < 0.2:
            COUNTER += 1

        # otherwise, the eye aspect ratio is not below the blink
        # threshold
        else:
            # if the eyes were closed for a sufficient number of
            # then increment the total number of blinks and store the blink duration
            if COUNTER >= 2:
                TOTAL += 1
                blink_durations.append(COUNTER / fps)
                blink_frame_count = 0

            # reset the eye frame counter
            COUNTER = 0
            blink_frame_count += 1
    
    frame_number += 1

# do a bit of cleanup
vs.stop()

dur = total_frames / fps
blink_freq = TOTAL / dur

blink_amplitude_mean = np.mean(blink_durations)

# Check if blink_durations is not empty before calculating min and max
if blink_durations:
    blink_amplitude_min = min(blink_durations)
    blink_amplitude_max = max(blink_durations)
else:
    blink_amplitude_min = blink_amplitude_max = None

blink_duration_stdev = np.std(blink_durations)
blink_frame_percentage = (TOTAL / total_frames) * 100

print("Total blinks: ", TOTAL)

# Print the results
print("Total blinks:", TOTAL)
print("Blink frequency:", blink_freq)
print("Blink Amplitude Mean:", blink_amplitude_mean)
print("Blink Amplitude Minimum:", blink_amplitude_min)
print("Blink Amplitude Maximum:", blink_amplitude_max)
print("Blink Duration Standard Dev.:", blink_duration_stdev)
print("Percentage of Total Frames with Blinks:", blink_frame_percentage)

# Create a dictionary with the features
features_dict = {
    "Total blinks": TOTAL,
    "Blink frequency": blink_freq,
    "Blink Amplitude Mean": blink_amplitude_mean,
    "Blink Amplitude Minimum": blink_amplitude_min,
    "Blink Amplitude Maximum": blink_amplitude_max,
    "Blink Duration Standard Dev.": blink_duration_stdev,
    "Percentage of Total Frames with Blinks": blink_frame_percentage
}

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
