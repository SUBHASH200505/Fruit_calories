import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No data augmentation for the test set, just rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the training dataset
train_generator = train_datagen.flow_from_directory(
   r"C:\Users\Subhash\Downloads\archive (5)\asl_alphabet_train\asl_alphabet_train",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Load the test dataset
test_generator = test_datagen.flow_from_directory(
    r"C:\Users\Subhash\Downloads\archive (5)\asl_alphabet_test",
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)
