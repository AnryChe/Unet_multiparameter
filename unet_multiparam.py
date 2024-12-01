
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

# Пути к данным
IMAGE_PATH = "./camvid/images/"  # Путь к изображениям
MASK_PATH = "./camvid/masks/"    # Путь к маскам

# Параметры
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 32  # Количество классов в CamVid

# Гиперпараметры
BASE_FILTERS = 64  # Базовое количество фильтров
KERNEL_SIZE = (3, 3)  # Размер ядра свертки
NUM_LAYERS = 4  # Количество уровней в энкодере/декодере


def load_data(image_path, mask_path):
    images_list = []
    masks_list = []
    
    for img_name in os.listdir(image_path):
        # Загружаем изображения
        img = load_img(os.path.join(image_path, img_name), target_size=(IMG_HEIGHT, IMG_WIDTH))
        img = img_to_array(img) / 255.0  # Нормализация
        images_list.append(img)
        
        # Загружаем маски
        mask_name = img_name.replace(".png", "_L.png")  # Предполагается определенный формат
        mask = load_img(os.path.join(mask_path, mask_name), target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode="grayscale")
        mask = img_to_array(mask).astype(np.int32)
        masks_list.append(mask)
    
    return np.array(images_list), np.array(masks_list)


# Загрузка данных
images, masks = load_data(IMAGE_PATH, MASK_PATH)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(images, masks, test_size=0.2, random_state=42)

# One-hot кодирование масок
y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)


# Функция для создания U-Net с изменяемыми гиперпараметрами
def unet_model(input_size=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES, base_filters=BASE_FILTERS, kernel_size=KERNEL_SIZE, num_layers=NUM_LAYERS):
    inputs = Input(input_size)
    skips = []
    x = inputs

    # Энкодер
    for i in range(num_layers):
        x = Conv2D(base_filters * (2 ** i), kernel_size, activation="relu", padding="same")(x)
        x = Conv2D(base_filters * (2 ** i), kernel_size, activation="relu", padding="same")(x)
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    # Боттлнек
    x = Conv2D(base_filters * (2 ** num_layers), kernel_size, activation="relu", padding="same")(x)
    x = Conv2D(base_filters * (2 ** num_layers), kernel_size, activation="relu", padding="same")(x)
    
    # Декодер
    for i in reversed(range(num_layers)):
        x = UpSampling2D(size=(2, 2))(x)
        x = Concatenate()([x, skips[i]])
        x = Conv2D(base_filters * (2 ** i), kernel_size, activation="relu", padding="same")(x)
        x = Conv2D(base_filters * (2 ** i), kernel_size, activation="relu", padding="same")(x)
    
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(x)
    model = Model(inputs, outputs)
    return model


# Создание модели
model = unet_model()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Data Augmentation
data_gen_args = dict(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Генераторы данных
train_image_generator = image_datagen.flow(X_train, batch_size=16, seed=42)
train_mask_generator = mask_datagen.flow(y_train, batch_size=16, seed=42)

train_generator = zip(train_image_generator, train_mask_generator)

# Обучение модели
EPOCHS = 20
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // 16,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

# Сохранение модели
model.save("unet_camvid_augmented.h5")


# Функция для визуализации
def visualize_prediction(image, mask, prediction):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.title("Image")
    plt.imshow(image)
    
    plt.subplot(1, 3, 2)
    plt.title("True Mask")
    plt.imshow(np.argmax(mask, axis=-1))
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask")
    plt.imshow(np.argmax(prediction, axis=-1))
    plt.show()


# Пример предсказания
sample_image = X_test[0]
sample_mask = y_test[0]
predicted_mask = model.predict(np.expand_dims(sample_image, axis=0))

visualize_prediction(sample_image, sample_mask, predicted_mask[0])
