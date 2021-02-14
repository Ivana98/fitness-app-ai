import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.cluster.KMeans
from os import listdir, path
import keras
from categories import CATEGORIES

matplotlib.rcParams['figure.figsize'] = 8, 6

# BASE_PATH = "D:/Fakultet/7sms-Soft_computing/fruits-360/test-multiple_fruits"
# BASE_PATH = "D:/Semestar7/Soft kompjuting/Projekat/fruits-360/tmf2"
BASE_PATH = "D:/soft/fruits-360/test-multiple"
loaded_model = keras.models.load_model('../saved_models/model1.h5')

def trashold_segmantation():

    for file_name in listdir(BASE_PATH):
        file_path = path.join(BASE_PATH, file_name)

        img = cv2.imread(file_path)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.GaussianBlur(img, (9, 9), 0)
        img = cv2.bilateralFilter(img, 21, 75, 75)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image_bin = cv2.adaptiveThreshold(img_gray, 155, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
        # _, image_bin = cv2.threshold(img_gray, 225, 255, cv2.THRESH_BINARY_INV)
        #### ret, image_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        plt.imshow(image_bin, 'gray')
        plt.show()

        napravi_konture(image_bin, img2)


def napravi_konture(image_bin, img):
    """
    Pravimo i izdvajamo konture.

    :param image_bin:
    :param img: Originalna slika ili slika za koju zelimo da ide u mrezu
    """

    contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours_barcode = []  # ovde ce biti samo konture koje pripadaju bar-kodu
    for contour in contours:  # za svaku konturu
        center, size, angle = cv2.minAreaRect(
            contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
        width, height = size
        print("size: ")
        print(size)
        print("angle: ")
        print(angle)
        if 200 < width < 1000 and 200 < height < 1000:  # uslov da kontura pripada bar-kodu
            izdvoj_sliku(contour, img)
            contours_barcode.append(contour)  # ova kontura pripada bar-kodu

    print("Broj kontura koje imamo: " + str(len(contours_barcode)))

    img3 = img.copy()
    cv2.drawContours(img3, contours_barcode, -1, 255, 3)  # (255, 0, 0) je bilo umesto 255

    ### prikaz slike sa iscrtanim konturama
    # plt.imshow(img3)
    # plt.show()



def izdvoj_sliku(contour, img):
    """
    Izdvajamo konturu sa originalne slike i printujemo je.

    :param contour: izdvojena kontura
    :param img: originalna slika
    :return: None
    """

    x, y, w, h = cv2.boundingRect(contour)
    cropped = img[y:y + h, x:x + w]  # ovo treba da ide u mrezu

    get_image_class(cropped)

    plt.imshow(cropped)
    plt.show()


def color_contour():
    for file_name in listdir(BASE_PATH):
        file_path = path.join(BASE_PATH, file_name)

        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pixel_values = image.reshape((-1, 3))
        pixel_values = np.float32(pixel_values)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        k = 2  # broj klastera
        _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8([0, 1])

        labels = labels.flatten()

        segmented_image = centers[labels.flatten()]
        segmented_image = segmented_image.reshape((image.shape[0], image.shape[1]))

        plt.imshow(segmented_image, 'gray')
        plt.show()

        napravi_konture(segmented_image, image)


def get_image_class(image):
    image = cv2.resize(image, (25, 25))
    image = np.expand_dims(image, axis=0)

    # image = image / 255

    predictions = loaded_model.predict(image)
    print(predictions)
    class_name = CATEGORIES[np.argmax(predictions)]

    print(predictions)
    print("predicted number: " + str(np.argmax(predictions)) + " class_name: " + str(class_name))
    return class_name


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

if __name__ == '__main__':
    trashold_segmantation()
    # IoU()
    # color_contour()
