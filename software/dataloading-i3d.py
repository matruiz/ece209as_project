import cv2
from PIL import Image
import numpy as np
################################################
videonum = '10'
video_path = '/content/' + videonum + '.mp4'
################################################
cap = cv2.VideoCapture(video_path)
def crop_center(img, cropx, cropy):
    y, x, *_ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty + cropy, startx:startx + cropx, ...]

#Get video length
import subprocess
def get_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

frames = []

while (cap.isOpened()):
    # Get a video frame
    hasFrame, frame = cap.read()


    if hasFrame == True:
        ## Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #print(frame.ravel())


        #resize so that smallst is 256 (assuming height is smallest)
        
        height, width, _ = frame.shape
        ratio = width / height
        new_height = 256
        new_width = int(ratio * new_height)
        #new_height = 224
        #new_width = 224
        frame = cv2.resize(frame, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)

        
        #print(frame.shape)

        frame = crop_center(frame, 224, 224)

        #print(frame.shape)


        frames.append(frame)

        #print(len(frame.ravel()))
        #print(frame[0][1])

    else:
        break

cap.release()

add_dim = []
add_dim.append(frames)
frames_np = np.asarray(add_dim)
frames_np = np.interp(frames_np, (frames_np.min(), frames_np.max()), (-1, +1))
frames_np = (frames_np + 1) / 2

print("Frames per second:", frames_np.shape[1] / get_length(video_path))
print("RGB shape:", frames_np.shape)

test_img = frames_np[0][30]
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

plt.imshow(test_img)

#Save as .npy
save_path = '/content/frames' + videonum + '.npy'
np.save(save_path, frames_np)