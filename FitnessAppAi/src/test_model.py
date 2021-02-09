import warnings
from keras.preprocessing.image import ImageDataGenerator
import keras
import json

warnings.filterwarnings("ignore")

IMG_W, IMG_H = 100, 100
NUM_OF_CLASSES = 131
BATCH_SIZE = 32

BASE_PATH = "D:/Semestar7/Soft kompjuting/Projekat"
TEST_DIR = BASE_PATH + "/fruits-360/Test"

def main():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_W, IMG_H),
        batch_size=BATCH_SIZE,
        class_mode='categorical')

    # Recreate the exact same model, including its weights and the optimizer
    loaded_model = keras.models.load_model('../saved_models/saved_modelsmodel1.h5')

    # Show the model architecture
    loaded_model.summary()

    # Show results
    test_score = loaded_model.evaluate_generator(test_generator, verbose=0)

    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])

def parse_json():
    print("*** PARSIRANJE JSON FILE-A ***")

    with open('../fajlovi/nutritivne_vrednosti.json', 'r') as f:
        array = json.load(f)

    print(array)


if __name__ == '__main__':
    # main()
    parse_json()
