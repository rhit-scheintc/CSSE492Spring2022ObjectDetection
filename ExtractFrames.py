import cv2

video = cv2.VideoCapture('HighDensity2.mp4')
success, img = video.read()

frameCount = 0
imgCount = 0
while success:
    if frameCount % 25 == 0:
        cv2.imwrite("HighDensity2/Image%d.jpg" % imgCount, img)
        imgCount += 1

    success, img = video.read()

    frameCount += 1

print("Finished")
print("Over " + str(frameCount) + " frames the program saved " + str(imgCount + 1) + " images")