import cv2
import datetime
import numpy as np

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X','V','I','D')

out = cv2.VideoWriter("Video Captured on Motion Detection.mp4", fourcc, 20.0, (1280,720))    # Out put file name, and frame size.

ret, frame1 = cap.read()
ret, frame2 = cap.read()
print(frame1.shape)


while cap.isOpened():

    diff = cv2.absdiff(frame1, frame2)                  # performing Image Processing to draw contours
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=4)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont1 = []

    for contour in contours:                            # Sorting contours based on area.
        if cv2.contourArea(contour) < 250:
            continue
        else:
            cont1.append(contour)

    print(len(cont1))

    if len(cont1) > 1:
        image = cv2.resize(frame1, (1280, 720))         # To adjest for output Frame Size
        date_time = str(datetime.datetime.now())        # To obtain current Date & time

        font = cv2.FONT_HERSHEY_TRIPLEX                 # To specify Font Style
        image = cv2.putText(image, "Date: " + date_time.split()[0], (10, 50), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
        image = cv2.putText(image, "Time: " + date_time.split()[1], (10, 100), font, 1, (0, 0, 0), 2, cv2.LINE_AA)

        out.write(image)                                # To Save the Video on Motion detection


    image = cv2.resize(frame1, (1280, 720))             # To show Original Video
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(1) == 27:                # Press "Escape Key" to Stop Capturing the video.
        break


cv2.destroyAllWindows()
cap.release()
out.release()