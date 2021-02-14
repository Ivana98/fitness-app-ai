import cv2
from os import listdir, path
from src.imports import TEST_FOLDER
from src.detection import adaptive_threshold, crop_image, construct_contours
import numpy as np
import keras
from src.categories import CATEGORIES_SMALLER

loaded_model = keras.models.load_model('../saved_models/model.h5')


def main():
    for file_name in listdir(TEST_FOLDER):
        file_path = path.join(TEST_FOLDER, file_name)

        image = cv2.imread(file_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_bin = adaptive_threshold(image)

        contours = construct_contours(image_bin, original_image)

        for contour in contours:
            cropped_image = crop_image(contour, original_image)

            cropped_image = cv2.resize(cropped_image, (25, 25))
            cropped_image = np.expand_dims(cropped_image, axis=0)

            predictions = loaded_model.predict(cropped_image)
            class_name = CATEGORIES_SMALLER[np.argmax(predictions)]
            print(class_name)



if __name__ == '__main__':
    main()
