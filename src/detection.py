import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['figure.figsize'] = 8, 6


def adaptive_threshold(image):
    image = cv2.GaussianBlur(image, (9, 9), 0)
    image = cv2.bilateralFilter(image, 21, 75, 75)

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_bin = cv2.adaptiveThreshold(image_gray, 155, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
    # plt.imshow(image_bin, 'gray')
    # plt.show()

    return image_bin


def construct_contours(image_bin, original_image):
    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_fruits = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        if 200 < width < 1000 and 200 < height < 1000:
            contours_fruits.append(contour)

    image_copy = original_image.copy()
    cv2.drawContours(image_copy, contours_fruits, -1, 255, 3)

    # plt.imshow(image_copy)
    # plt.show()

    return contours_fruits


def crop_image(contour, image):
    x, y, w, h = cv2.boundingRect(contour)
    cropped = image[y:y + h, x:x + w]

    return cropped


def color_contour(image):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    k = 2
    _, labels, _ = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8([0, 1])

    labels = labels.flatten()

    image_bin = centers[labels.flatten()]
    image_bin = image_bin.reshape((image.shape[0], image.shape[1]))

    return image_bin


def IoU():
    print("** EVALUACIJA **")

    # ucitavam sliku

    img = cv2.imread("D:/Semestar7/Soft kompjuting/Projekat/fruits-360/IoU/kivi.jpg")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # pravim prvi bounding box
    # prvi bouding box ce da bude rezultat nase segmentacije

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    image_bin = cv2.adaptiveThreshold(img_gray, 155, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)

    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_barcode = []  # ovde ce biti samo konture koje pripadaju bar-kodu

    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        if 200 < width < 1000 and 200 < height < 1000:  # uslov da kontura pripada bar-kodu
            contours_barcode.append(contour)  # ova kontura pripada bar-kodu

    iou_scores = []

    for con in contours_barcode:

        # izdvajamo koordinate kvadrata konture
        rect = cv2.minAreaRect(con)
        box = cv2.boxPoints(rect)  # koordinate kvadrata

        box_lista = []

        for p in box:
            print(p)
            box_lista.append([int(float(p[0])), int(float(p[1]))])

        first_bb_points = box_lista
        stencil = np.zeros(img.shape).astype(img.dtype)
        contours = [np.array(first_bb_points)]
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result1 = cv2.bitwise_and(img, stencil)
        result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
        plt.imshow(result1)
        plt.show()

        # pravimo drugi bounding box
        # drugi bounding box ce da bude rucno unesen

        second_bb_points = [[491, 1013], [884, 771], [1043, 1034], [652, 1273]]
        stencil = np.zeros(img.shape).astype(img.dtype)
        contours = [np.array(second_bb_points)]
        color = [255, 255, 255]
        cv2.fillPoly(stencil, contours, color)
        result2 = cv2.bitwise_and(img, stencil)
        result2 = cv2.cvtColor(result2, cv2.COLOR_BGR2RGB)
        plt.imshow(result2)
        plt.show()

        # racunanje greske
        intersection = np.logical_and(result1, result2)
        union = np.logical_or(result1, result2)
        iou_score = np.sum(intersection) / np.sum(union)
        iou_scores.append(iou_score)


    # ovo radimo zato sto ponekad imamo
    # konturu unutar konture
    iou_score = [iou_scores[0]]
    for ious in iou_scores:
        if ious > iou_score:
            iou_score = ious

    print('IoU je % s' % iou_score)
