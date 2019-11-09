from os import listdir
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import os
import numpy as np
import skimage as ski
from skimage import data, color, filters, io, feature, measure, draw
import cv2
import colorsys
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import skimage.morphology as mp

dir_path = ''


# path to images
def list_image(dir_path):
    #return [os.path.join(dir_path, file) for file in ['1.jpg']]
    return [os.path.join(dir_path, file) for file in ['1.jpg','2.jpg','3.jpg','4.jpg','5.jpg','6.jpg','7.jpg','8.jpg',]]



# read images
def input_data(imagePath):
    # return data.imread(imagePath,as_gray=True)
    return cv2.imread(imagePath, cv2.IMREAD_COLOR)


def reduction_of_color(image):
    image_data = image / 255.0

    image_data = image_data.reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.25)
    flags = cv2.KMEANS_RANDOM_CENTERS
    image_data = image_data.astype(np.float32)
    compactness, labels, centers = cv2.kmeans(
        image_data, 3, None, criteria, 20, flags)
    new_colors = centers[labels].reshape((-1, 3))
    image_recolored = new_colors.reshape(image.shape)

    return image_recolored


def findCircles(image):
    # workFlow = reduction_of_color(image)
    # workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # workFlow = cv2.GaussianBlur(workFlow, (9, 9), 0)
    # workFlow = workFlow*255
    # workFlow = workFlow.astype(np.uint8)
    # ret, workFlow = cv2.threshold(workFlow, 150, 255, cv2.THRESH_BINARY)

    workFlow = reduction_of_color(image)
    workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, workFlow = cv2.threshold(
        workFlow, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cv2.morphologyEx(workFlow, cv2.MORPH_CLOSE, kernel)
    # printWorkflow(workFlow)
    # im2,
   # for i in range(0,5):
    #  workFlow = mp.erosion(workFlow)
    contours, hierarchy = cv2.findContours(
        workFlow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    circles = []
    crosses = []
    f = 0
    for i, k in enumerate(contours):
       # if circleDetectionByCompactness(k) and cv2.contourArea(k) > 100:
        if cv2.contourArea(k) > 50:
            circles.append(k)
        cv2.drawContours(image, [k], 0, (0, 255, 0), 1)
        # f = 1 - f

    circles, crosses = removeProteus(circles)

   # printWorkflow(image)
    return image, circles, crosses


def circleDetectionByCompactness(contour):
    perimeter = cv2.arcLength(contour, True)
    area = cv2.contourArea(contour)

    if area < 20:
        return False
    compactness = (perimeter ** 2) / (4*np.pi*area)

    if abs(compactness - 1) < 0.25:
        return True
    else:
        return False


def circleDetection(x, y, contour):
    distances = []
    for i in contour:
        pointX = i[0][0]
        pointY = i[0][1]

        distances = np.append(
            distances, dist.euclidean((x, y), (pointX, pointY)))

    median = np.median(distances)
    eps = epsilon(distances)

    for i in distances:
        if abs(i - median) > median / 3:
            return False
    return True


def epsilon(distances):
    maxDist = max(distances)
    return maxDist/10


def removeProteus(conturs):
    areas = []
    for c in conturs:
        areas = np.append(areas, cv2.contourArea(c))

    median = np.median(areas)

    circles = []
    crosses = []

    for c in conturs:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        print(solidity)
        if solidity > 0.8 and solidity < 1:
           # if abs(area - median) < 0.5 * median:
            circles.append(c)
        elif solidity > 0.16 and solidity <= 0.35:
            if area < 2*sum(areas) / areas.size:
                crosses.append(c)

    return circles, crosses


def findGroups(circles, image, cntColor):
    ones = []
    centroids = []
    for i, circles in enumerate(circles):
        moments = cv2.moments(circles)
        centerX = int(moments['m10'] / moments['m00'])
        centerY = int(moments['m01'] / moments['m00'])

        element = {}
        element['id'] = i
        element['centroid'] = [centerX, centerY]
        element['area'] = cv2.contourArea(circles)
        element['radius'] = getRadius(circles, [centerX, centerY])
        element['contour'] = circles
        centroids.append(element)

    # print(centroids)

    for c in centroids:
        ones.append(c)
        cv2.drawContours(image, [c['contour']], 0, cntColor, 2)

        # # finding the closest neighbour:
        # else:
        #     min = getCentroidsDistance(c['centroid'], neigbours[0]['centroid'])
        #     for n in neigbours:
        #         if min > getCentroidsDistance(c['centroid'], n['centroid']):
        #             min = getCentroidsDistance(c['centroid'], n['centroid'])

        # elif len(neigbours) == 1:
        #     twos.append(c)
        #     cv2.drawContours(image, [c['contour']], 0, (1, 0, 0), 2)
        # elif len(neigbours) == 7:
        #     twos.append(c)
        #     cv2.drawContours(image, [c['contour']], 0, (0, 0, 1), 2)

    return image


def findNeighbours(circles, circlesList):
    neighbours = []
    for c in circlesList:
        distance = getCentroidsDistance(c['centroid'], circles['centroid'])
        # print(distance)
        if 5*circles['radius'] > distance > 1:
            # print(distance)
            neighbours.append(c)

    return neighbours


def getCentroidsDistance(a, b):
    return dist.euclidean((a[0], a[1]), (b[0], b[1]))


def getRadius(circles, cetroid):
    distances = []
    for i in circles:
        pointX = i[0][0]
        pointY = i[0][1]

        distances = np.append(distances, dist.euclidean(
            (cetroid[0], cetroid[1]), (pointX, pointY)))

    return np.mean(distances)


def dbscan(circles, image):
    centroids = []
    distances = []
    # print(len(circles))
    for i, circles in enumerate(circles):
        moments = cv2.moments(circles)
        if moments['m00']:
            moments = cv2.moments(circles)
            centerX = int(moments['m10'] / moments['m00'])
            centerY = int(moments['m01'] / moments['m00'])
            distances.append(getRadius(circles, [centerX, centerY]))
            centroids.append([centerX, centerY])
            cv2.drawContours(image, [circles], 0, (125, 25, 25), 1)

    meanRadius = np.mean(distances)
    # print(len(centroids))
    # getRadius()
    db = DBSCAN(eps=8*meanRadius, min_samples=1).fit(centroids)
    labels = db.labels_
    cubes = len(set(labels)) - (1 if -1 in labels else 0)

    merged = list(zip(centroids, labels))

    groups = []
    print(cubes)

    for i in range(cubes):
        group = []
        for m in merged:
            if m[1] == i:
                group.append(m)
        groups.append(group)

    font = cv2.FONT_HERSHEY_SIMPLEX
    for g in groups:
        count = len(g)
        x = []
        y = []
        for el in g:
            x.append(el[0][0])
            y.append(el[0][1])
        cv2.rectangle(image, (int(min(x) - meanRadius), int(max(y) + meanRadius)),
                      (int(max(x) + meanRadius), int(min(y) - meanRadius)), (255, 0, 0), 2)
        cv2.putText(image, str(count), (int(min(x) - meanRadius),
                                        int(max(y) + meanRadius)), font, 1, (0, 255,), 2, cv2.LINE_AA)
        # cv2.putText(image, str(m[1]), (m[0][0], m[0][1]), font, 1, (0, 255,), 2, cv2.LINE_AA)

    # for m in merged:

        # cv2.putText(image, str(m[1]), (m[0][0], m[0][1]), font, 1, (0, 255, ), 2, cv2.LINE_AA)

    return image


def printWorkflow(workFlow):
    plt.figure()
    io.imshow(workFlow)
    plt.show()


if __name__ == "__main__":
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=0, bottom=0, right=0.985, hspace=0, wspace=0)
    for i, imagePath in enumerate(list_image(dir_path)):
        image = input_data(imagePath)

        imgTmp = image.copy()
        image = cv2.resize(
         image, (int(image.shape[1]/4), int(image.shape[0]/4)))
        image, circles, crosses = findCircles(image)

        # print(len(circles))

        image = findGroups(circles, image, (0, 0, 128))
        image = findGroups(crosses, image, (256, 0, 0))
        # image = dbscan(circles, image)
        plt.imshow(image, cmap="Greys_r")
        plt.axis("off")
    # plt.savefig("kosci.pdf", bbox_inches="tight")
        plt.show()


# kod w ktorym wykorzystuje momenty

    # for i, k in enumerate(circles):
    #     moments = cv2.moments(k)
    #
    #     if not moments['m00']:
    #         continue
    #
    #     centerX = int(moments['m10']/moments['m00'])
    #     centerY = int(moments['m01']/moments['m00'])
    #
    #     # if circleDetectionByCompactness(k) and cv2.contourArea(k) > 50:
    #     cv2.drawContours(image, [k], 0, np.asarray(colorsys.hsv_to_rgb(i / len(contours), 1, 1)), 1)

    #
    # if circleDetection(centerX, centerY, k) and cv2.contourArea(k) > 50:
    #     print((cv2.contourArea(k)))
    #     cv2.drawContours(image, [k], 0, np.asarray(colorsys.hsv_to_rgb(i / len(contours), 1, 1)), 1)

    # if moments['m00']:
    #     cv2.circle(image, (int(moments['m10']/moments['m00']), int(moments['m01']/moments['m00'])), 2, (255, 255, 255), -10)


# def findCirclesHough(image_recolored, mainimg):
#     # img = cv2.medianBlur(image_recolored, 5)
#     # img = ski.filters.gaussian(img, 2)
#     # tmp = img * 255
#     # img = tmp.astype(np.uint8)
#     # kernel = np.ones((3, 3), np.uint8)
#     # # _, img = cv2.threshold(img, 60, 60, cv2.THRESH_TRUNC)
#     #
#     # # return img
#     #
#     # # _, img = cv2.threshold(img, 20, 20, cv2.THRESH_TOZERO)
#     #
#     # img = cv2.erode(img, kernel, iterations=1)
#     # img = cv2.dilate(img, kernel, iterations=2)
#
#     # img = cv2.subtract(255, img)
#
#     circles = cv2.HoughCircles(image_recolored, cv2.HOUGH_GRADIENT, 20, 15)
#     print(circles)
#     circles = np.uint16(np.around(circles))
#
#     for i in circles[0, :]:
#         # draw the outer circle
#         cv2.circle(mainimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#
#     return mainimg
