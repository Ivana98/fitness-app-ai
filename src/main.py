import cv2
from os import listdir, path
from src.imports import TEST_FOLDER
from src.detection import adaptive_threshold, crop_image, construct_contours
import numpy as np
import json
import keras
from src.nutritions_read import read_fruit_nutrition, get_fruit_nutrition, read_actual_fruit_nutritions, \
    get_actual_nutrition, get_nutrition_accuracy, add_to_current_nutrition
from src.categories import CATEGORIES_SMALLER

loaded_model = keras.models.load_model('../saved_models/model.h5')


def main():
    fruits_nutrition = read_fruit_nutrition()
    actual_nutritions = read_actual_fruit_nutritions()

    sum_accuracy = 0
    accuracy_count = 0
    current_fruits_nutrition_str = '{"energetska_vrednost": 0, "belancevine": 0, "ugljeni_hidrati": 0, "masti": 0}'

    for file_name in listdir(TEST_FOLDER):
        actual_nutrition = get_actual_nutrition(actual_nutritions, file_name)

        current_fruits_nutrition = json.loads(current_fruits_nutrition_str)

        file_path = path.join(TEST_FOLDER, file_name)

        image = cv2.imread(file_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_bin = adaptive_threshold(image)

        contours = construct_contours(image_bin, original_image)

        for contour in contours:
            cropped_image = crop_image(contour, original_image)
            class_name = get_class_name(cropped_image)

            nutrition = get_fruit_nutrition(fruits_nutrition, class_name)
            add_to_current_nutrition(current_fruits_nutrition, nutrition)

        accuracy = get_nutrition_accuracy(current_fruits_nutrition, actual_nutrition)
        sum_accuracy += accuracy
        accuracy_count += 1
        print('Accuracy for ' + file_name + ': ' + str(accuracy))

    print('\n\nGeneral accuracy: ' + str(sum_accuracy / accuracy_count))


def get_class_name(image):
    image = cv2.resize(image, (25, 25))
    image = np.expand_dims(image, axis=0)

    predictions = loaded_model.predict(image)
    return CATEGORIES_SMALLER[np.argmax(predictions)]


if __name__ == '__main__':
    main()
