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

dir_path = 'photo/'

#Do wyrzucenia: 31, 40, 41, 51, 58, 60, 61
#TRUDNE PRZYDPAKI: 11, 21, 23, 24

# path to images
def list_image(dir_path):
    photoList = []
    for i in range(1, 62):
        photoList.append(str(i) + '.jpg')
    #print(photoList)
    #return [os.path.join(dir_path, file) for file in ['1.jpg']]
    return [os.path.join(dir_path, file) for file in photoList]


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
def gamma_correction(img, correction):
    img = img/255.0
    img = cv2.pow(img, correction)
    return np.uint8(img*255)

def findShapes(image):
    # workFlow = reduction_of_color(image)
    # workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # workFlow = cv2.GaussianBlur(workFlow, (9, 9), 0)
    # workFlow = workFlow*255
    # workFlow = workFlow.astype(np.uint8)
    # ret, workFlow = cv2.threshold(workFlow, 150, 255, cv2.THRESH_BINARY)

    workFlow = reduction_of_color(image)
    #printWorkflow(workFlow)
    workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #printWorkflow(workFlow)


    ret, workFlow = cv2.threshold(
        workFlow, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #printWorkflow(workFlow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cv2.morphologyEx(workFlow, cv2.MORPH_CLOSE, kernel)
    #printWorkflow(workFlow)
    # #printWorkflow(workFlow)
    # im2,
   # for i in range(0,5):
    #  workFlow = mp.erosion(workFlow)
    contours, hierarchy = cv2.findContours(
        workFlow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    circles = []
    crosses = []
    fields = []
    f = 0
    for i, k in enumerate(contours):
       # if circleDetectionByCompactness(k) and cv2.contourArea(k) > 100:
        if cv2.contourArea(k) > 50:
            circles.append(k)
            cv2.drawContours(image, [k], 0, (0, 255, 0), 1)
        # f = 1 - f

    circles, crosses, fields = removeProteus(circles)

   # #printWorkflow(image)
    return image, circles, crosses, fields


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
    fields = []
    for c in conturs:
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        #
        #print(solidity)
        if solidity > 0.8 and solidity < 1:
            if abs(area - median) < 0.5 * median:
                circles.append(c)
        elif solidity > 0.25:  # and solidity <= 0.5:
            # and  area > 2*sum(areas) / areas.size:
            if area < 1.2*sum(areas) / areas.size:
                crosses.append(c)
            else:
                fields.append(c)

    return circles, crosses, fields


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
    #print(cubes)

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

#MJ
def MJfindGroups(image):
    workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    workFlow = workFlow*255
    workFlow = workFlow.astype(np.uint8)
    ret, workFlow = cv2.threshold(workFlow, 160, 255, cv2.THRESH_BINARY)
    #printWorkflow(workFlow)
    # Taking a matrix of size 5 as the kernel
    kX = 6
    kY = 6
    if len(image[0]) < len(image):
        kX = 9
    else:
        kY = 9
    kern = np.ones((kX, kY), np.uint8)

    workFlow = cv2.dilate(workFlow, kern, iterations=9)
    #printWorkflow(workFlow)
    contours, hierarchy = cv2.findContours(workFlow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #for i, k in enumerate(contours):
    #    cv2.drawContours(image, k, -1, (0, 150, 0), 2)

    coordinate = boardArea(contours, (len(image[0]), len(image)))

    partList = imagePart(image, coordinate) #zabieramy tylko większe obszary

    return partList
#MJ
def imagePart(image, coordinates):
    partList = []
    for coord in coordinates:
        if abs(coord[3]-coord[2]) > 40 and abs(coord[1] - coord[0]) > 40:
            partList.append(image[coord[2]: coord[3],coord[0]: coord[1]] )

    plt.imshow(image, cmap="Greys_r")
    plt.savefig("pho/test" + str(i+1) + ".jpg", bbox_inches="tight")
    j =0
    for item in partList:
        #printWorkflow(item)
        plt.imshow(item, cmap="Greys_r")
        plt.savefig("pho/test" +str(i+1)  + "sub" + str(j) + ".jpg", bbox_inches="tight")
        j = j + 1
    return partList

#MJ
#[[],[],[]...], [x,y]
def boardArea(contours, size):
    contoursNode = []

    for con in contours:
        lewo = size[0]
        prawo = 0
        gora = 0
        dol = size[1]
        for pixel in con:
            p = pixel[0]
            if(p[0] > prawo):
                prawo = p[0]
            if(p[0] < lewo):
                lewo = p[0]
            if(p[1] > gora ):
                gora = p[1]
            if(p[1] < dol):
                dol = p[1]
        contoursNode.append((lewo, prawo, dol, gora))
    return contoursNode


if __name__ == "__main__":
    plt.figure(figsize=(20, 10))
    plt.subplots_adjust(left=0, bottom=0, right=0.985, hspace=0, wspace=0)
    for i, imagePath in enumerate(list_image(dir_path)):
        image = input_data(imagePath)

        imgTmp = image.copy()
        image = cv2.resize(
            image, (int(image.shape[1]/4), int(image.shape[0]/4)))

        MJfindGroups(image) #Najpierw ta funkcja!

        #image, circles, crosses, fields = findShapes(image)


        # print(len(circles))
        #for j in MJfindGroups(image):
            #i, circles, crosses, fields = findShapes(j)
            #j = findGroups(circles, j, (0, 0, 128))
            #j = findGroups(crosses, j, (256, 0, 0))
            #j = findGroups(fields, j, (256, 256, 0))
         #   plt.imshow(j, cmap="Greys_r")
            #plt.show()

        # image = dbscan(circles, image)
        #plt.imshow(image, cmap="Greys_r")
        #plt.imshow(image, cmap="Greys_r")
        #plt.axis("off")
        #plt.savefig("tests/test"+str(i)+".jpg", bbox_inches="tight")
        #plt.show()
