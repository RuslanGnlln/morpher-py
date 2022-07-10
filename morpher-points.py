#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gimpfu import *
import numpy as np
import cv2

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask



# GIMP
def channelData(layer):  # convert gimp image to numpy
    region = layer.get_pixel_rgn(0, 0, layer.width, layer.height)
    pixChars = region[:, :]  # Take whole layer
    bpp = region.bpp
    #return np.frombuffer(pixChars,dtype=np.uint8).reshape(len(pixChars)/bpp,bpp)
    return np.frombuffer(pixChars, dtype=np.uint8).reshape(layer.height, layer.width, bpp)


# GIMP
def createResultLayer(image, name, result):
    rlBytes = np.uint8(result).tobytes();
    rl = gimp.Layer(image, name, image.width, image.height, 0, 100, NORMAL_MODE)
    region = rl.get_pixel_rgn(0, 0, rl.width, rl.height, True)
    region[:, :] = rlBytes
    image.add_layer(rl, 0)
    gimp.displays_flush()

points1 = []
points2 = []

def CallBackFunc1(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points1.append((int(x), int(y)))

def CallBackFunc2(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points2.append((int(x), int(y)))

def scanPoints(img1, img2):
    img1copy = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2copy = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    windowName1 = 'MouseCallback1'
    windowName2 = 'MouseCallback2'
    cv2.namedWindow(windowName1)
    cv2.namedWindow(windowName2)
    cv2.setMouseCallback(windowName1, CallBackFunc1)
    cv2.setMouseCallback(windowName2, CallBackFunc2)
    while (True):
        cv2.imshow('MouseCallback1', img1copy)
        for p in points1:
            cv2.circle(img1copy, p, 4, (0,0,255), -1)
        for p in points2:
            cv2.circle(img2copy, p, 4, (0,0,255), -1)
        cv2.imshow('MouseCallback2', img2copy)
        if cv2.waitKey(20) == 13:
            cv2.destroyAllWindows()
            break


def morpherScript(image, image1, image2, alpha):

    # GIMP
    img1 = channelData(image1)
    img2 = channelData(image2)

    height1, width1, channels1 = img1.shape
    height2, width2, channels2 = img2.shape

    points1.append((0, 0))
    points1.append((0, int(height1/2)))
    points1.append((0, int(height1-1)))
    points1.append((int(width1/2), int(height1-1)))
    points1.append((int(width1-1), int(height1-1)))
    points1.append((int(width1-1), int(height1/2)))
    points1.append((int(width1-1), 0))
    points1.append((int(width1/2), 0))

    points2.append((0, 0))
    points2.append((0, int(height2 / 2)))
    points2.append((0, int(height2 - 1)))
    points2.append((int(width2 / 2), int(height2 - 1)))
    points2.append((int(width2 - 1), int(height2 - 1)))
    points2.append((int(width2 - 1), int(height2 / 2)))
    points2.append((int(width2 - 1), 0))
    points2.append((int(width2 / 2), 0))

    # # Read array of corresponding points
    scanPoints(img1, img2)

    if (len(points1) != len(points2)):
        gimp.message("unequal number of dots\n please try again")
        return

    # # Convert Mat to float data type
    img1 = np.float32(img1)
    img2 = np.float32(img2)

    points = []

    # Compute weighted average point coordinates
    for i in range(0, len(points1)):
          x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
          y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
          points.append((int(x), int(y)))

    # Allocate space for final output
    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

    # Rectangle to be used with Subdiv2D
    size = img1.shape
    rect = (0, 0, size[1], size[0])

    # Create an instance of Subdiv2D
    subdiv1 = cv2.Subdiv2D(rect)

    # Insert points into subdiv
    for p in points1:
        subdiv1.insert(p)

    triangleList1 = subdiv1.getTriangleList()
    trianglePoints = []

    for p in triangleList1:
        trianglePoints.append((int(p[0]), int(p[1])))
        trianglePoints.append((int(p[2]), int(p[3])))
        trianglePoints.append((int(p[4]), int(p[5])))


    while trianglePoints:
        x = points1.index(trianglePoints[0])
        trianglePoints.pop(0)
        y = points1.index(trianglePoints[0])
        trianglePoints.pop(0)
        z = points1.index(trianglePoints[0])
        trianglePoints.pop(0)

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    # GIMP
    createResultLayer(image, 'new_output', imgMorph)


# GIMP
register(
        "python-fu-morpher",
        "Morphing",
        "Morphing using affine transform",
        "RuslanGnlln",
        "ruslan040506@gmail.com",
        "06-07-2022",
        "morphing",
        "*",
        [
            (PF_IMAGE, "image", "image", None),
            (PF_DRAWABLE, "image1", "image1", None),
            (PF_DRAWABLE, "image2", "image2", None),
            (PF_SLIDER, "alpha", "alpha", 0.5, (0,1,0.01))
        ],
        [],
        morpherScript, menu="<Image>/Morph/")

main()

