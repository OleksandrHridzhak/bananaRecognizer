#imports
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image  import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras._tf_keras.keras.optimizers import Adam

#pahts
train_dir = 'MY_data/train'
validation_dir = 'MY_data/test'

#variables
img_size = (150, 150)
batch_size = 32
epochs = 10


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


test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    train_dir, 
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'  
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,  
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary' 
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  
])


model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)


#Congrats and save the model
model.save('model_weights.h5')
print("We did it!")