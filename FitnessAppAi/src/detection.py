import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# import sklearn.cluster.KMeans
from os import listdir, path

matplotlib.rcParams['figure.figsize'] = 8, 6

# BASE_PATH = "D:/Semestar7/Soft kompjuting/Projekat/fruits-360/test-multiple_fruits"
BASE_PATH = "D:/Semestar7/Soft kompjuting/Projekat/fruits-360/tmf2"


def main():

    for file_name in listdir(BASE_PATH):
        file_path = path.join(BASE_PATH, file_name)

        img = cv2.imread(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 45, 5)
        ##### ret, image_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        plt.imshow(image_bin, 'gray')
        plt.show()

        # pokusacemo da dodamo malo erozije za nas primer
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  # MORPH_ELIPSE, MORPH_RECT...
        # plt.imshow(cv2.erode(image_bin, kernel, iterations=1), 'gray')
        # cv2.erode(image_bin, kernel, iterations=1)
        # plt.show()

        contours, hierarchy = cv2.findContours(image_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # img2 = img.copy()
        # cv2.drawContours(img2, contours, -1, (255, 0, 0), 5)
        # plt.imshow(img2)
        # plt.show()

        contours_barcode = []  # ovde ce biti samo konture koje pripadaju bar-kodu
        for contour in contours:  # za svaku konturu
            center, size, angle = cv2.minAreaRect(
                contour)  # pronadji pravougaonik minimalne povrsine koji ce obuhvatiti celu konturu
            width, height = size
            if 200 < width < 700 and 200 < height < 700:  # uslov da kontura pripada bar-kodu
                izdvoj_sliku(contour, img)
                contours_barcode.append(contour)  # ova kontura pripada bar-kodu

        print("Broj kontura koje imamo: " + str(len(contours_barcode)))

        img3 = img.copy()
        cv2.drawContours(img3, contours_barcode, -1, 255, 5)  # (255, 0, 0) je bilo umesto 255
        plt.imshow(img3)
        plt.show()


def izdvoj_sliku(contour, img):
    """
    Izdvajamo konturu sa originalne slike i printujemo je.

    :param contour: izdvojena kontura
    :param img: originalna slika
    :return: None
    """

    x, y, w, h = cv2.boundingRect(contour)
    cropped = img[y:y + h, x:x + w]
    plt.imshow(cropped)
    plt.show()


if __name__ == '__main__':
    main()
