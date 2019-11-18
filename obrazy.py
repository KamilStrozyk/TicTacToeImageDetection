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

# Do wyrzucenia: 31, 40, 41, 51, 58, 60, 61
# TRUDNE PRZYDPAKI: 11, 21, 23, 24

# path to images


def list_image(dir_path):
    photoList = []
    for i in range(1, 62):
        # for i in range(1, 62):
        photoList.append(str(i) + '.jpg')
    # print(photoList)
    # return [os.path.join(dir_path, file) for file in ['1.jpg']]
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
    workFlow = reduction_of_color(image)
    # printWorkflow(workFlow)
    workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # printWorkflow(workFlow)
    ret, workFlow = cv2.threshold(
        workFlow, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # printWorkflow(workFlow)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cv2.morphologyEx(workFlow, cv2.MORPH_CLOSE, kernel)
    # printWorkflow(workFlow)
    contours, hierarchy = cv2.findContours(
        workFlow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    shapes = []

    for i, k in enumerate(contours):
        if cv2.contourArea(k) > 100:
            shapes.append(k)
            cv2.drawContours(image, [k], 0, (0, 255, 0), 1)

    circles, crosses, fields = detectShapes(shapes)
   # printWorkflow(workFlow)
    return circles, crosses, fields


def prepareSamples():
    # path = ['circle.jpg', 'cross1.jpg',
    #         'cross2.jpg', 'field1.jpg', 'field2.jpg']
    # sampleNames = ['circle', 'cross', 'cross', 'field', 'field']

    path = ['circle.jpg', 'circle1.jpg',
            'cross3.jpg', 'cross4.jpg', 'field2.jpg', 'field3.jpg', 'field4.jpg', ]
    sampleNames = ['circle', 'circle',  'cross',
                   'cross',  'field',  'field',  'field']
    sampleContours = []

    for i in range(0, len(path)):
        img = cv2.imread('samples/'+path[i], 0)
        ret, thresh = cv2.threshold(img, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, 2, 1)
        sampleContours.append(contours[0])

    return sampleContours, sampleNames


def detectShapes(conturs):
    areas = []
    for c in conturs:
        areas = np.append(areas, cv2.contourArea(c))

    median = np.median(areas)

    circles = []
    crosses = []

    sampleContours, sampleNames = prepareSamples()
    x = 0
    for c in conturs:
        # area = cv2.contourArea(c)
        # hull = cv2.convexHull(c)
        # hull_area = cv2.contourArea(hull)
        # solidity = float(area)/hull_area
        # # print(solidity)
        # if solidity > 0.8 and solidity < 0.96:
        #    # if abs(area - median) < 0.5 * median:
        #     circles.append(c)
        # elif solidity <1:#elif solidity > 0.25:  # and solidity <= 0.5:
        #     # and  area > 2*sum(areas) / areas.size:
        #     if area < sum(areas) / areas.size:
        #         crosses.append(c)
        #     else:
        #         fields.append(c)
        bestRet = 45
        bestShape = "XD"
        for i in range(0, len(sampleContours)):
            ret = cv2.matchShapes(c, sampleContours[i], 1, 0.0)
            if ret < bestRet:
                bestRet = ret
                bestShape = sampleNames[i]

        print(bestRet)
        print(bestShape)

        if bestShape == 'circle':
            circles.append(c)
        elif bestShape == 'cross':
            crosses.append(c)
        elif bestShape == 'field':
            fields.append(c)

    return circles, crosses, fields


def drawContoursOnImage(contours, image, cntColor):
    for i, k in enumerate(contours):
        cv2.drawContours(image, [k], 0, cntColor, 1)
    return image


def getRadius(circles, cetroid):
    distances = []
    for i in circles:
        pointX = i[0][0]
        pointY = i[0][1]

        distances = np.append(distances, dist.euclidean(
            (cetroid[0], cetroid[1]), (pointX, pointY)))

    return np.mean(distances)


def printWorkflow(workFlow):
    plt.figure()
    io.imshow(workFlow)
    plt.show()


# MJ


def MJfindGroups(image):
    workFlow = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    workFlow = workFlow*255
    workFlow = workFlow.astype(np.uint8)
    ret, workFlow = cv2.threshold(workFlow, 160, 255, cv2.THRESH_BINARY)
    # printWorkflow(workFlow)
    # Taking a matrix of size 5 as the kernel
    kX = 6
    kY = 6
    if len(image[0]) < len(image):
        kX = 9
    else:
        kY = 9
    kern = np.ones((kX, kY), np.uint8)

    workFlow = cv2.dilate(workFlow, kern, iterations=9)
    # printWorkflow(workFlow)
    contours, hierarchy = cv2.findContours(
        workFlow, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # for i, k in enumerate(contours):
    #    cv2.drawContours(image, k, -1, (0, 150, 0), 2)

    coordinate = boardArea(contours, (len(image[0]), len(image)))

    partList = imagePart(image, coordinate)  # zabieramy tylko większe obszary

    return partList
# MJ


def imagePart(image, coordinates):
    partList = []
    for coord in coordinates:
        if abs(coord[3]-coord[2]) > 40 and abs(coord[1] - coord[0]) > 40:
            partList.append(image[coord[2]: coord[3], coord[0]: coord[1]])

    plt.imshow(image, cmap="Greys_r")
    #plt.savefig("pho/test" + str(i+1) + ".jpg", bbox_inches="tight")
    j = 0
    for item in partList:
        # printWorkflow(item)
        plt.imshow(item, cmap="Greys_r")
        # plt.savefig("pho/test" + str(i+1) + "sub" +
        # str(j) + ".jpg", bbox_inches="tight")
        j = j + 1
    return partList

# MJ
# [[],[],[]...], [x,y]


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
            if(p[1] > gora):
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

        # MJfindGroups(image)  # Najpierw ta funkcja!

        #image, circles, crosses, fields = findShapes(image)
        circles = []
        crosses = []
        fields = []
        # print(len(circles))
        for j in MJfindGroups(image):
            circles, crosses, fields = findShapes(j)

            j = drawContoursOnImage(circles, j, (0, 0, 128))
            j = drawContoursOnImage(crosses, j, (256, 0, 0))
            j = drawContoursOnImage(fields, j, (256, 256, 0))
            plt.imshow(j, cmap="Greys_r")
            # plt.show()

        #image = dbscan(circles, image)
        plt.imshow(image, cmap="Greys_r")
        plt.imshow(image, cmap="Greys_r")
        plt.axis("off")
        #plt.savefig("tests/test"+str(i)+".jpg", bbox_inches="tight")
        plt.show()
