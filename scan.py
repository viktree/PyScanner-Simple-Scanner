#!/usr/local/bin/python3

# -*- coding: utf-8 -*-
# @Author: tintinux
# @Date:   2017-11-08 22:40:21
# @Last Modified time: 2017-11-09 00:28:43

import imutils
import argparse
from transform import four_point_transform
import cv2


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the image to be scanned")
args = vars(ap.parse_args())


def filename_no_ext(file_name):
    return file_name.split(':')[0]


def load_resize_image(img_name):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    orig = cv2.imread(img_name)
    ratio = orig.shape[0] / 500.0
    resized = imutils.resize(orig, height=500)
    return orig, resized, ratio


def detect_edges(image):
    # convert the image to grayscale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 100, 200)

    return edged


def find_contours(image, edged):
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    _, cnts, _ = cv2.findContours(
        edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    # print(cnts)
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    # loop over the contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # if our approximated contour has four points, then we
        # can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break

    return screenCnt


def apply_perspective_transform(orig, screenCnt, ratio):
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # convert the warped image to grayscale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 11)


def main(show_steps=False):
    file_name = filename_no_ext(args["image"])
    original, resized, ratio = load_resize_image(args["image"])
    edged = detect_edges(resized)

    if show_steps:
        # show the original image and the edge detected image
        print("STEP 1: Edge Detection")
        cv2.imshow("Image", original)
        cv2.imshow("Edged", edged)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    screenContours = find_contours(resized, edged)

    if show_steps:
        # show the contour (outline) of the piece of paper
        print("STEP 2: Find contours of paper")
        cv2.drawContours(resized, [screenContours], -1, (0, 255, 0), 2)
        cv2.imshow("Outline", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    final_image = apply_perspective_transform(original,
                                              screenContours, ratio)

    if show_steps:
        # show the original and scanned images
        print("STEP 3: Apply perspective transform")
        cv2.imshow("Original", imutils.resize(original, height=650))
        cv2.imshow("Scanned", imutils.resize(final_image, height=650))
        cv2.waitKey(0)

    cv2.imwrite(file_name + "_transformed.jpg", final_image)


if __name__ == '__main__':
    main(show_steps=True)
