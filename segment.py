import cv2
import numpy as np
import imutils

# globals
bg = None


def run_avg(image, aWeight):
    """
    Calculate running average between the background model and current frame.
    :param image:
    :param aWeight:
    :return:
    """

    global bg
    # if there is no bg (because it's currently the first frame)
    # then initialize it with the current frame
    if bg is None:
        bg = image.copy().astype("float")
        return

    cv2.accumulateWeighted(image, bg, aWeight)


def segment(image, threshold=25):
    global bg
    diff = cv2.absdiff(bg.astype('uint8'), image)

    # threshold the diff'd image and get the contours of only hands
    thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]
    cnts, _ = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # if no contours detected return
    if len(cnts) == 0:
        return
    else:
        segmented = max(cnts, key=cv2.contourArea)
        return thresholded, segmented


if __name__ == '__main__':
    # running average weight
    aWeight = 0.5
    # webcam reference
    camera = cv2.VideoCapture(0)
    # region of interest
    top, right, bottom, left = 10, 350, 225, 590
    # initialize num_frames
    num_frames = 0

    while True:
        # get current frame and resize and flip so it's not mirrored
        (grabbed, frame) = camera.read()
        frame = imutils.resize(frame, width=700)
        frame = cv2.flip(frame, 1)

        clone = frame.copy()
        (height, width) = frame.shape[:2]
        roi = frame[top:bottom, right:left]

        # convert region of interest to grayscale and blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        # calibrate background until threshold reached
        if num_frames < 30:
            run_avg(gray, aWeight)
        else:
            # segment the hand region and then check
            hand = segment(gray)
            if hand is not None:
                # if it exists, unpack the thresholded image, draw, and display
                (thresholded, segmented) = hand
                cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                cv2.imshow('Thresholded', thresholded)

        # draw segmented hand, increment frames and display
        cv2.rectangle(clone, (left, top), (right, bottom), (0, 255, 0), 2)
        num_frames += 1
        cv2.imshow("Video Feed", clone)

        # wait for keypress to signal interrupt; "q" to stop loop
        keypress = cv2.waitKey(1) & 0xFF
        if keypress == ord('q'):
            break

    # free memory
    camera.release()
    cv2.destroyAllWindows()
