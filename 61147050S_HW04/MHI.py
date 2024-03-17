import cv2
import numpy as np
from PIL import Image

# Initialize video capture
cap = cv2.VideoCapture("./CV_beginner/Keck_Dataset/Keck Dataset/testingfiles/person4_gesture7_com.avi")

# Set start and end times (in seconds)
start_time = 0
end_time = 4

# Convert start and end times to frame numbers
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

# Read the first frame
ret, prev_frame = cap.read()
prev_frame_gray = Image.fromarray(prev_frame).convert('L')

# Initialize motion history image (MHI)
height, width = prev_frame_gray.size
mhi = np.zeros((height, width), np.float32)

# Set constants
thresh = 127  # motion threshold
delta = 0.3  # MHI decrement factor
max_value = 255  # maximum MHI value

# Set frame counter to 0
frame_count = 0

while ret:
    # Read the next frame
    ret, curr_frame = cap.read()
    if ret:
        # Check if we are in the selected time range
        if frame_count >= start_frame and frame_count < end_frame:
            curr_frame_gray = Image.fromarray(curr_frame).convert('L')

            # Compute motion mask (absolute difference)
            motion_mask = np.abs(np.subtract(curr_frame_gray, prev_frame_gray)) > thresh
            motion_mask = np.uint8(motion_mask)

            # Check if motion mask and MHI have the same shape
            if motion_mask.shape != mhi.shape:
                mhi = np.zeros_like(motion_mask, np.float32)

            # Update motion history image (MHI)
            mhi[motion_mask == 1] += 1
            mhi[motion_mask == 0] -= delta
            mhi[mhi < 0] = 0
            mhi[mhi > max_value] = max_value

            # Normalize and convert MHI to 8-bit image
            mhi_norm = cv2.normalize(mhi, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Increment frame counter
        frame_count += 1

        # Set current frame as previous frame
        prev_frame_gray = curr_frame_gray.copy()

        # Stop loop if end frame is reached
        if frame_count == end_frame:
            break


# Convert MHI to PIL Image format
mhi_pil = Image.fromarray(mhi_norm)

# Display the result
mhi_pil.show()

# Release video capture
cap.release()
