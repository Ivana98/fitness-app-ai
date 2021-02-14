import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.cluster.KMeans
from os import listdir, path
import keras
from src.categories import CATEGORIES

matplotlib.rcParams['figure.figsize'] = 8, 6

# BASE_PATH = "D:/Fakultet/7sms-Soft_computing/fruits-360/test-multiple_fruits"
BASE_PATH = "D:/Semestar7/Soft kompjuting/Projekat/fruits-360/tmf2"
# BASE_PATH = "D:/soft/fruits-360/test-multiple_fruits"
loaded_model = keras.models.load_model('../saved_models/saved_modelsmodel1.h5')

def trashold_segmantation():

    for file_name in listdir(BASE_PATH):
        file_path = path.join(BASE_PATH, file_name)

        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image_bin = cv2.adaptiveThreshold(img_gray, 155, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 5)
        # _, image_bin = cv2.threshold(img_gray, 225, 255, cv2.THRESH_BINARY_INV)
        #### ret, image_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        plt.imshow(image_bin, 'gray')
        plt.show()

        # dodavanje erozije
        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  # MORPH_ELIPSE, MORPH_RECT...
        # plt.imshow(cv2.erode(image_bin, kernel, iterations=1), 'gray')
        # cv2.erode(image_bin, kernel, iterations=1)
        # plt.show()

        napravi_konture(image_bin, img)


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
        if 200 < width < 1000 and 200 < height < 1000:  # uslov da kontura pripada bar-kodu
            izdvoj_sliku(contour, img)
            contours_barcode.append(contour)  # ova kontura pripada bar-kodu

    print("Broj kontura koje imamo: " + str(len(contours_barcode)))

    img3 = img.copy()
    # plt.imshow(img3)
    # plt.show()
    cv2.drawContours(img3, contours_barcode, -1, 255, 3)  # (255, 0, 0) je bilo umesto 255



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
    image = cv2.resize(image, (100, 100))
    image = np.expand_dims(image, axis=0)

    predictions = loaded_model.predict(image)
    class_name = CATEGORIES[np.argmax(predictions)]

    print(predictions)
    print("predicted number: " + str(np.argmax(predictions)) + " class_name: " + str(class_name))
    return class_name


if __name__ == '__main__':
    trashold_segmantation()
    # color_contour()
