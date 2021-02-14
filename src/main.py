import cv2
from os import listdir, path
from src.imports import TEST_FOLDER
from src.detection import adaptive_threshold, crop_image, get_image_class, construct_contours


def main():
    for file_name in listdir(TEST_FOLDER):
        file_path = path.join(TEST_FOLDER, file_name)

        image = cv2.imread(file_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_bin = adaptive_threshold(image)

        contours = construct_contours(image_bin, original_image)

        for contour in contours:
            cropped_image = crop_image(contour, original_image)
            class_name = get_image_class(cropped_image)
            print(class_name)



if __name__ == '__main__':
    main()
