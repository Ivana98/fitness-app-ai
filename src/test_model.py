from keras.preprocessing.image import ImageDataGenerator
import keras

from src.imports import TEST_SMALLER_FOLDER

img_width, img_height = 25, 25
no_classes = 10
batch_size = 25


def main():
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_directory(
        TEST_SMALLER_FOLDER,
        target_size=(img_width, img_width),
        batch_size=batch_size,
        class_mode='sparse')

    # Recreate the exact same model, including its weights and the optimizer
    loaded_model = keras.models.load_model('../saved_models/model.h5')

    # Show the model architecture
    loaded_model.summary()

    # Show results
    test_score = loaded_model.evaluate_generator(test_generator, verbose=0)

    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])


if __name__ == '__main__':
    main()
