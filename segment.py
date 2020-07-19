import cv2
import numpy as np
import imutils

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