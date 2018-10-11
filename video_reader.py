from minimal_object_detection_lib import MinimalObjectDetector
import cv2
import shutil
import os
import time
import numpy as np

def rescale_by_height(image, target_height, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv2.resize(image, (w, target_height), interpolation=method)

def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)

def AutoResize(frame):
    height, width, _ = frame.shape

    if height > 500:
        frame = rescale_by_height(frame, 500)
        AutoResize(frame)
    
    if width > 700:
        frame = rescale_by_width(frame, 700)
        AutoResize(frame)
    
    return frame

objectDetector = MinimalObjectDetector()
objectDetector.Initialize()

cap = cv2.VideoCapture('Footage.mp4')
_, frame = cap.read()

# cv2.imshow('output', frame)
# cv2.waitKey()

fps = cap.get(cv2.CAP_PROP_FPS)
TotalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

print(fps)
print(TotalFrames)

OutputDirectory = "Frames"
CurrentDirectory = os.path.curdir
OutputDirectoryPath = os.path.join(CurrentDirectory, OutputDirectory)

if os.path.exists(OutputDirectoryPath):
    shutil.rmtree(OutputDirectoryPath)
    time.sleep(0.5)
os.mkdir(OutputDirectoryPath)

CurrentFrame = 1
fpsCounter = 0
FrameWrittenCount = 1
while CurrentFrame < TotalFrames:
    _, frame = cap.read()
    if (frame == None):
        continue
    

    if fpsCounter > fps:
        fpsCounter = 0

        frame_rgb = np.copy(frame)
        result = objectDetector.Process(frame_rgb)

        IsPersonExists = False
        for i in range(len(result)):
            if result[i]['label'] == 'person':
                IsPersonExists = True

        if IsPersonExists == True:
            frame = AutoResize(frame)

            filename = "frame_" + str(FrameWrittenCount) + ".jpg"
            cv2.imwrite(os.path.join(OutputDirectoryPath, filename), frame)

            FrameWrittenCount += 1
    
    fpsCounter += 1
    CurrentFrame += 1

print('[INFO] Frames extracted')

