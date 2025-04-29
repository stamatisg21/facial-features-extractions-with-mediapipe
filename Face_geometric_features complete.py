import pandas as pd
import numpy as np
import cv2
import os


rightEyebrowLower = [35, 124, 46, 53, 52, 65]
rightEyebrowUpper = [156, 70, 63, 105, 66, 107, 55, 193]
leftEyebrowLower = [265, 353, 276, 283, 282, 295]
leftEyebrowUpper = [383, 300, 293, 334, 296, 336, 285, 417]
LineEyesInnerCorners = [133, 243, 244, 245, 122, 6, 351, 465, 464, 463, 362]

inner_mouth_uppper = 14
inner_mouth_lower = 15

right_eyebrow_upper = 105
right_eyebrow_lower = 52

left_eyebrow_upper = 334
left_eyebrow_lower = 282

right_eye_line = 133
left_eye_line = 362

eye_line_center = 6

left_outer_eye_corner = 263
right_outer_eye_corner = 33

left_inner_eye_corner = 362
right_inner_eye_corner = 133

left_upper_eyelid = 386
right_upper_eyelid = 159

left_lower_eyelid = 374
right_lower_eyelid = 145

mouth_upper_lip = 0
mouth_lower_lip = 17

mouth_left_corner = 291
mouth_right_corner = 61

right_eye_upper = 159
right_eye_lower = 145

left_eye_upper = 386
left_eye_lower = 374

nose_tip = 1
upper_lip_center = 0

interstitial_cleft_left = 426
interstitial_cleft_right = 206

left_eyebrow_corner = 285
right_eyebrow_corner = 55
eyebrow_line = 8

left_iris = 468
right_iris = 473

# Function to compute Euclidean distance between two points
def euclidean_distance(frame, index1, index2):
    
    
    x1 = landmarks_df.loc[frame, f'x{index1}']
    y1 = landmarks_df.loc[frame, f'y{index1}']
    
    #print(x1,y1)
    
    x2 = landmarks_df.loc[frame, f'x{index2}']
    y2 = landmarks_df.loc[frame, f'y{index2}']
    
    #print(x2,y2)
    
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) 

def compute_angle(frame, index1_1, index1_2, index2_1, index2_2):
    # Get coordinates of the points
    x1_1, y1_1 = landmarks_df.loc[frame, f'x{index1_1}'], landmarks_df.loc[frame, f'y{index1_1}']
    x1_2, y1_2 = landmarks_df.loc[frame, f'x{index1_2}'], landmarks_df.loc[frame, f'y{index1_2}']
    x2_1, y2_1 = landmarks_df.loc[frame, f'x{index2_1}'], landmarks_df.loc[frame, f'y{index2_1}']
    x2_2, y2_2 = landmarks_df.loc[frame, f'x{index2_2}'], landmarks_df.loc[frame, f'y{index2_2}']

    # Compute vectors
    vector1 = np.array([x1_2 - x1_1, y1_2 - y1_1])
    vector2 = np.array([x2_2 - x2_1, y2_2 - y2_1])

    # Compute angle
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm1 * norm2)
    angle_rad = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)

    return angle_deg
    

def vertical_distance(frame, index1, index2):
    
    return abs(landmarks_df.loc[frame, f'y{index1}'] - landmarks_df.loc[frame, f'y{index2}']) 

def horizontal_distance(frame, index1, index2):
    
    return abs(landmarks_df.loc[frame, f'x{index1}'] - landmarks_df.loc[frame, f'x{index2}']) 

video_path = "pat1_aligned.mp4"

#video_path = 'H_PD001_OFF_open_close_mouth_aligned.MP4'
print("Now working on video: ", video_path)
vidcap = cv2.VideoCapture(video_path)
fps = vidcap.get(cv2.CAP_PROP_FPS)
width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the file path for saving the statistics
statistics_file_path = os.path.splitext(video_path)[0] + '_geometric_features.txt'

# Read the Excel file
landmarks_df = pd.read_csv(video_path[:-4]+"_landmarks.txt", delimiter=",")
print("Now working on landmarks: ", video_path[:-4]+"_landmarks.txt")

# Multiply all elements of columns x0 to x477 by 3840
landmarks_df.iloc[:, :956:2] *= width

# Multiply all elements of columns y0 to y477 by 2160
landmarks_df.iloc[:, 1:957:2] *= height


# Manually set the column names
num_landmarks = 478

x_num = 0
y_num = 0
for i in range(2*num_landmarks):
    
    if i % 2 == 0:
        landmarks_df.rename(columns={landmarks_df.columns[i]: f'x{x_num}'},inplace=True)
        x_num+=1
        
    else : 
        landmarks_df.rename(columns={landmarks_df.columns[i]: f'y{y_num}'},inplace=True)
        y_num+=1

# Separate x and y columns
x_columns = landmarks_df.columns[::2]  # Select every other column starting with the first
y_columns = landmarks_df.columns[1::2]  # Select' every other column starting with the second



mouth_upper_lower_distance = []
angle_left_eyebrow_eye_line = []
angle_right_eyebrow_eye_line = []
left_eyebrow_eye_centerline_distance = []
right_eyebrow_eye_centerline_distance = []
left_outer_eye_corner_upper_eyelid_distance = []
right_outer_eye_corner_upper_eyelid_distance = []
left_inner_eye_corner_upper_eyelid_distance = []
right_inner_eye_corner_upper_eyelid_distance = []
left_outer_eye_corner_lower_eyelid_distance = []
right_outer_eye_corner_lower_eyelid_distance = []
left_inner_eye_corner_lower_eyelid_distance = []
right_inner_eye_corner_lower_eyelid_distance = []
left_eyelids_distance = []
right_eyelids_distance = []
mouth_upper_left_corner_distance = []
mouth_upper_right_corner_distance = []
mouth_lower_left_corner_distance = []
mouth_lower_right_corner_distance = []
mouth_corners_horizontal_distance = []
mouth_lips_vertical_distance = []
right_eye_open_close = []
left_eye_open_close = []
lip_nose_distance = [] # nasolabial
interstitial_cleft_distance = []
eyebrows_corners_vertical = []
eyebrows_corners_horizontal = []
eyebrows_corners_euclidean = []
left_eyebrow_line_distance = []
right_eyebrow_line_distance  = []


for i in range(len(landmarks_df)):


    # distances between the outer eyes’ corner and their upper eyelids
    left_outer_eye_corner_upper_eyelid_distance.append(euclidean_distance(i, left_outer_eye_corner, left_upper_eyelid))
    right_outer_eye_corner_upper_eyelid_distance.append(euclidean_distance(i, right_outer_eye_corner, right_upper_eyelid))

    # vertical distances between the upper eyelids and the lower eyelids
    left_eyelids_distance.append(vertical_distance(i, left_lower_eyelid, left_upper_eyelid))
    right_eyelids_distance.append(vertical_distance(i, right_lower_eyelid, right_upper_eyelid))

     # distances between the inner eyes’ corner and their upper eyelids
    left_inner_eye_corner_upper_eyelid_distance.append(euclidean_distance(i, left_inner_eye_corner, left_upper_eyelid))
    right_inner_eye_corner_upper_eyelid_distance.append(euclidean_distance(i, right_inner_eye_corner, right_upper_eyelid))

    # distances between the outer eyes’ corner and their lower eyelids
    left_outer_eye_corner_lower_eyelid_distance.append(euclidean_distance(i, left_outer_eye_corner, left_lower_eyelid))
    right_outer_eye_corner_lower_eyelid_distance.append(euclidean_distance(i, right_outer_eye_corner, right_lower_eyelid))

    # distances between the inner eyes’ corner and their lower eyelids
    left_inner_eye_corner_lower_eyelid_distance.append(euclidean_distance(i, left_inner_eye_corner, left_lower_eyelid))
    right_inner_eye_corner_lower_eyelid_distance.append(euclidean_distance(i, right_inner_eye_corner, right_lower_eyelid))

    # distances between the upper and lower lips
    mouth_upper_lower_distance.append(vertical_distance(i, mouth_upper_lip, mouth_lower_lip))

    # distances between the left and right corners of the mouth
    mouth_corners_horizontal_distance.append(horizontal_distance(i, mouth_left_corner, mouth_right_corner))

    # distances between the upper lip and the left corner of the mouth
    mouth_upper_left_corner_distance.append(euclidean_distance(i, mouth_upper_lip, mouth_left_corner))

    # distances between the upper lip and the right corner of the mouth
    mouth_upper_right_corner_distance.append(euclidean_distance(i, mouth_upper_lip, mouth_right_corner))

    # distances between the lower lip and the left corner of the mouth
    mouth_lower_left_corner_distance.append(euclidean_distance(i, mouth_lower_lip, mouth_left_corner))

    # distances between the lower lip and the right corner of the mouth
    mouth_lower_right_corner_distance.append(euclidean_distance(i, mouth_lower_lip, mouth_right_corner))

    # distances between the left and right eyes' upper and lower eyelids
    left_eye_open_close.append(vertical_distance(i, left_upper_eyelid, left_lower_eyelid))
    right_eye_open_close.append(vertical_distance(i, right_upper_eyelid, right_lower_eyelid))

    # distances between the nose tip and the upper lip center (nasolabial distance)
    lip_nose_distance.append(vertical_distance(i, nose_tip, upper_lip_center))

    # distances between the left and right interstitial clefts
    interstitial_cleft_distance.append(horizontal_distance(i, interstitial_cleft_left, interstitial_cleft_right))

    # vertical distances between the left and right eyebrow corners
    eyebrows_corners_vertical.append(vertical_distance(i, left_eyebrow_corner, right_eyebrow_corner))

    # horizontal distances between the left and right eyebrow corners
    eyebrows_corners_horizontal.append(horizontal_distance(i, left_eyebrow_corner, right_eyebrow_corner))

    # Euclidean distances between the left and right eyebrow corners
    eyebrows_corners_euclidean.append(euclidean_distance(i, left_eyebrow_corner, right_eyebrow_corner))

    # distances between the left eyebrow corner and the eyebrow line
    left_eyebrow_line_distance.append(vertical_distance(i, left_eyebrow_corner, eyebrow_line))

    # distances between the right eyebrow corner and the eyebrow line
    right_eyebrow_line_distance.append(vertical_distance(i, right_eyebrow_corner, eyebrow_line))

    # angles between the left eyebrow line and the eye line
    angle_left_eyebrow_eye_line.append(compute_angle(i, left_eyebrow_corner, left_eyebrow_lower, left_eye_line, eye_line_center))

    # angles between the right eyebrow line and the eye line
    angle_right_eyebrow_eye_line.append(compute_angle(i, right_eyebrow_corner, right_eyebrow_lower, right_eye_line, eye_line_center))

# Dictionary to store all the feature lists
features_dict = {
    'mouth_upper_lower_distance': mouth_upper_lower_distance,
    'angle_left_eyebrow_eye_line': angle_left_eyebrow_eye_line,
    'angle_right_eyebrow_eye_line': angle_right_eyebrow_eye_line,
    'left_eyebrow_eye_centerline_distance': left_eyebrow_eye_centerline_distance,
    'right_eyebrow_eye_centerline_distance': right_eyebrow_eye_centerline_distance,
    'left_outer_eye_corner_upper_eyelid_distance': left_outer_eye_corner_upper_eyelid_distance,
    'right_outer_eye_corner_upper_eyelid_distance': right_outer_eye_corner_upper_eyelid_distance,
    'left_inner_eye_corner_upper_eyelid_distance': left_inner_eye_corner_upper_eyelid_distance,
    'right_inner_eye_corner_upper_eyelid_distance': right_inner_eye_corner_upper_eyelid_distance,
    'left_outer_eye_corner_lower_eyelid_distance': left_outer_eye_corner_lower_eyelid_distance,
    'right_outer_eye_corner_lower_eyelid_distance': right_outer_eye_corner_lower_eyelid_distance,
    'left_inner_eye_corner_lower_eyelid_distance': left_inner_eye_corner_lower_eyelid_distance,
    'right_inner_eye_corner_lower_eyelid_distance': right_inner_eye_corner_lower_eyelid_distance,
    'left_eyelids_distance': left_eyelids_distance,
    'right_eyelids_distance': right_eyelids_distance,
    'mouth_upper_left_corner_distance': mouth_upper_left_corner_distance,
    'mouth_upper_right_corner_distance': mouth_upper_right_corner_distance,
    'mouth_lower_left_corner_distance': mouth_lower_left_corner_distance,
    'mouth_lower_right_corner_distance': mouth_lower_right_corner_distance,
    'mouth_corners_horizontal_distance': mouth_corners_horizontal_distance,
    'right_eye_open_close': right_eye_open_close,
    'left_eye_open_close': left_eye_open_close,
    'lip_nose_distance': lip_nose_distance,
    'interstitial_cleft_distance': interstitial_cleft_distance,
    'eyebrows_corners_vertical': eyebrows_corners_vertical,
    'eyebrows_corners_horizontal': eyebrows_corners_horizontal,
    'eyebrows_corners_euclidean': eyebrows_corners_euclidean,
    'left_eyebrow_line_distance': left_eyebrow_line_distance,
    'right_eyebrow_line_distance': right_eyebrow_line_distance,
}

# Calculate temporal statistical features
statistics = {}
for feature_name, feature_values in features_dict.items():
    statistics[feature_name] = {
        'mean': np.mean(feature_values),
        'std': np.std(feature_values),
        'min': np.min(feature_values),
        'max': np.max(feature_values),
        'range': np.ptp(feature_values)
    }

# Convert the statistics dictionary to a DataFrame
statistics_df = pd.DataFrame(statistics).T
statistics_df.reset_index(inplace=True)
statistics_df.columns = ['Feature', 'Mean', 'Std', 'Min', 'Max', 'Range']

# Save the statistics DataFrame to a file
statistics_df.to_csv(statistics_file_path, index=False)

print("Temporal statistical features saved to:", statistics_file_path)

