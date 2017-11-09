#!/usr/local/bin/python3

# -*- coding: utf-8 -*-
# @Author: tintinux
# @Date:   2017-11-08 22:41:22
# @Last Modified time: 2017-11-08 23:44:49

import numpy as np
import cv2


def euclidean_dist(a, b):
    return np.sqrt(((a[0] - b[0]) ** 2) + ((a[1] - b[1]) ** 2))


def order_points(coords):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = coords.sum(axis=1)  # s = x + y
    rect[0] = coords[np.argmin(s)]
    rect[2] = coords[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    d = np.diff(coords, axis=1)  # d = |x - y|
    rect[1] = coords[np.argmin(d)]
    rect[3] = coords[np.argmax(d)]

    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    old_corners = order_points(pts)
    (tl, tr, br, bl) = old_corners

    # The larger width of the two sides is the new width
    widthA = euclidean_dist(br, bl)
    widthB = euclidean_dist(tr, tl)
    maxWidth = max(int(widthA), int(widthB))

    # The larger height of the two sides is the new height
    heightA = euclidean_dist(tr, br)
    heightB = euclidean_dist(tl, bl)
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    new_corners = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(old_corners, new_corners)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))
