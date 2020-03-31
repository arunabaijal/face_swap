# import the necessary packages
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import math
from numpy.linalg import norm
import sys
import copy
import os

def start(swap, image, video):
    # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-p", "--shape-predictor", required=True,
    # 	help="path to facial landmark predictor")
    # ap.add_argument("-i", "--image", required=True,
    # 	help="path to input image")
    # args = vars(ap.parse_args())
    
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    cam = cv2.VideoCapture(video)
    width = int(cam.get(3))  # float
    height = int(cam.get(4))
    print(width, height)
    vidWriter = cv2.VideoWriter("./video_output_data1.mp4",cv2.VideoWriter_fourcc(*'mp4v'), int(cam.get(5)), (width, height))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    # swap = True
    face_swap_file = image
    # load the input image, resize it, and convert it to grayscale
    # images = ['Scarlett.jpg', 'Rambo.jpg']
    # imageA = cv2.imread('Scarlett.jpg')
    
            # cv2.circle(imageB, (x, y), 1, (0, 0, 255), -1)
    currentframe = 0
    while (True):
        # reading from frame
        ret, frame = cam.read()
        if ret:
            imageA = frame
            grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
            rectsA = detector(grayA, 1)
            if (swap and len(rectsA) < 2) or (not swap and len(rectsA) == 0):
                vidWriter.write(frame)
                print("Can't find enough faces in frame")
                continue
            PA = []
            xPointsA = []
            yPointsA = []
            shapeA = None
            # for (i, rect) in enumerate(rectsA):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
            if swap:
                shapeA = predictor(grayA, rectsA[0])
            else:
                shapeA = predictor(grayA, rectsA[0])
            shapeA = face_utils.shape_to_np(shapeA)
            print(len(shapeA))
            for (x, y) in shapeA:
                PA.append([x, y, 1])
                xPointsA.append(x)
                yPointsA.append(y)
                # convert dlib's rectangle to a OpenCV-style bounding box
                # [i.e., (x, y, w, h)], then draw the face bounding box
                # (x, y, w, h) = face_utils.rect_to_bb(rect)
                # cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # # show the face number
                # cv2.putText(imageA, "Face #{}".format(i + 1), (x - 10, y - 10),
                # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw them on the image
                # for (x, y) in shapeA:
                # cv2.circle(imageA, (x, y), 1, (0, 0, 255), -1)
            if swap:
                imageB = frame
            else:
                imageB = cv2.imread(face_swap_file)

            # detect faces in the grayscale image

            grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
            # detect faces in the grayscale image
            # rectsB = detector(grayB, 1)

            PB = []
            shapeB = None
            xPointsB = []
            yPointsB = []

            # loop over the face detections
            # for (i, rect) in enumerate(rectsB):
                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
            if swap:
                shapeB = predictor(grayB, rectsA[1])
            else:
                rectsB = detector(grayB, 1)
                shapeB = predictor(grayB, rectsB[0])
            shapeB = face_utils.shape_to_np(shapeB)
            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)], then draw the face bounding box
            # (x, y, w, h) = face_utils.rect_to_bb(rect)
            # cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # # show the face number
            # cv2.putText(imageB, "Face #{}".format(i + 1), (x - 10, y - 10),
            # 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shapeB:
                PB.append([x, y, 1])
                xPointsB.append(x)
                yPointsB.append(y)
            weights_x, weights_y = find_weights(PA, shapeA, xPointsB, yPointsB)
            # print(weights)
            mask_warped_img, warped_img = warp_face(imageA, imageB, shapeA, shapeB, weights_x, weights_y)
            r = cv2.boundingRect(mask_warped_img)
            center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
            output = cv2.seamlessClone(warped_img.copy(), imageA, mask_warped_img, center, cv2.NORMAL_CLONE)
            if swap:
                weights_x, weights_y = find_weights(PB, shapeB, xPointsA, yPointsA)
                # print(weights)
                mask_warped_img, warped_img = warp_face(imageB, imageA, shapeB, shapeA, weights_x, weights_y)
                r = cv2.boundingRect(mask_warped_img)
                center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))
                output = cv2.seamlessClone(warped_img.copy(), output, mask_warped_img, center, cv2.NORMAL_CLONE)
                # imageA[rects[0].top() - 40:rects[0].bottom() + 40, rects[0].left() - 40:rects[0].right() + 40, :] = output
                # cv2.imshow("warped_img", warped_img)
                # cv2.imshow("mask_warped", mask_warped_img)
                # cv2.imshow("mask", mask)
            vidWriter.write(output)
            cv2.imwrite('output_data1_' + str(currentframe) + '.jpg', output)
            # cv2.waitKey(0)

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break
    vidWriter.release()
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    # loop over the face detections
    
    
    # a1 = weights[-1]
    # ay = weights[-2]
    # ax = weights[-3]
    # weights = weights[:len(weights)-3]
    # warpedA = []
    # for i in range(grayB.shape[0]):
    # 	for j in range(grayB.shape[1]):
    # 		sum = 0
    # 		for ind, weight in enumerate(weights):
    # 			# print(shapeB[ind])
    # 			r = norm(shapeB[ind] - [i, j], 1) + np.exp(10 ** -7)
    # 			u = (r ** 2) * (math.log(r ** 2))
    # 			sum = sum + weight*u
    # 		warpedA.append(a1 + ax*i + ay*j + sum)
    #
    # # print(warpedA)
    #
    # new = np.zeros(grayB.shape)
    #
    # warpedA = np.reshape(warpedA, (333,500,2))
    # print(warpedA.shape)
    # print(new.shape)
    #
    # for i in range(new.shape[0]):
    # 	for j in range(new.shape[1]):
    # 		if 0 <= int(warpedA[i][j][1]) < grayA.shape[1] and 0 <= int(warpedA[i][j][0]) < grayA.shape[0]:
    # 			print(warpedA[i][j])
    # 			new[i][j] = grayA[int(warpedA[i][j][0])][int(warpedA[i][j][1])]
    
    # show the output image with the face detections + facial landmarks
    # cv2.imshow("OutputA", imageA)


def warp_face(imageA, imageB, shapeA, shapeB, weights_x, weights_y):
    w, h = imageB.shape[:2]
    mask = mask_from_points((w, h), shapeB)
    # def warp_tps(img_source,img_target,points1,points2,weights_x,weights_y,mask):
    xy1_min = np.float32([min(shapeA[:, 0]), min(shapeA[:, 1])])
    xy1_max = np.float32([max(shapeA[:, 0]), max(shapeA[:, 1])])
    xy2_min = np.float32([min(shapeB[:, 0]), min(shapeB[:, 1])])
    xy2_max = np.float32([max(shapeB[:, 0]), max(shapeB[:, 1])])
    x = np.arange(xy1_min[0], xy1_max[0]).astype(int)
    y = np.arange(xy1_min[1], xy1_max[1]).astype(int)
    X, Y = np.mgrid[x[0]:x[-1] + 1, y[0]:y[-1] + 1]
    # X,Y = np.mgrid[0:src_shape[2],0:src_shape[3]]
    pts_src = np.vstack((X.ravel(), Y.ravel()))
    xy = pts_src.T
    u = np.zeros_like(xy[:, 0])
    v = np.zeros_like(xy[:, 0])
    # print(u.shape)
    # print(v.shape)
    for i in range(xy.shape[0]):
        u[i] = fxy(xy[i, :], shapeA, weights_x)
    u[u < xy2_min[0]] = xy2_min[0]
    u[u > xy2_max[0]] = xy2_max[0]
    for j in range(xy.shape[0]):
        v[j] = fxy(xy[j, :], shapeA, weights_y)
    v[v < xy2_min[1]] = xy2_min[1]
    v[v > xy2_max[1]] = xy2_max[1]
    #     print(u.shape)
    #     print(img_source.shape)
    warped_img = imageA.copy()
    mask_warped_img = np.zeros_like(warped_img[:, :, 0])
    for a in range(u.shape[0]):
        #     for b in range(v.shape[0]):
        #     warped_img[xy[a,1],xy[a,0],:] = warped_src_face[v[a],u[a],:]
        if mask[v[a], u[a]] > 0:
            warped_img[xy[a, 1], xy[a, 0], :] = imageB[v[a], u[a], :]
            mask_warped_img[xy[a, 1], xy[a, 0]] = 255
        # plt.imshow(warped_img)
        # plt.show()
        # return warped_img, mask_warped_img
    return mask_warped_img, warped_img


def find_weights(PA, shapeA, xPointsB, yPointsB):
    K = np.zeros([len(shapeA), len(shapeA)])
    for i in range(len(shapeA)):
        for j in range(len(shapeA)):
            r = norm(shapeA[i] - shapeA[j], 2) + np.exp(10 ** -7)
            K[i][j] = (r ** 2) * (math.log(r ** 2))
    mat = np.vstack([np.hstack((K, PA)), np.hstack([np.transpose(PA), np.zeros([3, 3])])])
    lam = np.exp(10 ** -7)
    V = []
    for v in xPointsB:
        V.append(v)
    V.append(0)
    V.append(0)
    V.append(0)
    # print('K', K)
    weights_x = np.matmul(np.linalg.inv(mat + lam * np.identity(len(shapeA) + 3)), V)
    V = []
    for v in yPointsB:
        V.append(v)
    V.append(0)
    V.append(0)
    V.append(0)
    weights_y = np.matmul(np.linalg.inv(mat + lam * np.identity(len(shapeA) + 3)), V)
    return weights_x, weights_y


def U(r):
    return (r**2)*(math.log(r**2))

def fxy(pt1,pts2,weights):
    K = np.zeros([pts2.shape[0],1])
    for i in range(pts2.shape[0]):
        K[i] = U(np.linalg.norm((pts2[i]-pt1),2) + np.exp(10**-7))
    f = weights[-1] + weights[-3]*pt1[0] +weights[-2]*pt1[1]+np.matmul(K.T,weights[0:-3])
    return f

def mask_from_points(size, points,erode_flag=1):
    radius = 10  # kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    if erode_flag:
        mask = cv2.erode(mask, kernel,iterations=1)

    return mask

if __name__ == '__main__':
    start()