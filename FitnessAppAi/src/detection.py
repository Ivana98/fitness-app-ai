import cv2
import matplotlib
import matplotlib.pyplot as plt
# import sklearn.cluster.KMeans
from os import listdir, path

matplotlib.rcParams['figure.figsize'] = 16, 12

BASE_PATH = "D:/soft/fruits-360/test-multiple_fruits"


def main():

    for file_name in listdir(BASE_PATH):
        file_path = path.join(BASE_PATH, file_name)

        img = cv2.imread(file_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(img_gray, 'gray')
        # plt.show()

        # image_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 5)
        ret, image_bin = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
        plt.imshow(image_bin, 'gray')
        plt.show()

        # image = image.reshape((image.shape[0] * image.shape[1], 3))

        # clt = KMeans(n_clusters=args["clusters"])
        # clt.fit(image)

main()