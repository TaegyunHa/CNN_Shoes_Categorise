# Convolutional Neural Network

# Part 1 - Building the CNN
from keras.models import Sequential    # Secquence of layers
from keras.layers import Conv2D        # Convolution of layers
from keras.layers import MaxPooling2D  # Pooling step
from keras.layers import Flatten       # Flattening (input)
from keras.layers import Dense         # Fully connected Network

# Initialising the CNN
classifier = Sequential() # CNN initialised

# Step 1 - Convolution
classifier.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation = 'relu')) # Tensorflow
# Convolution2D(64,3,3) 64 filters consist of 3x3 feature detector
# input_shape(3,256,256), (number of channel, 256x256 pxels) = Theano (256,256,3) for Tensorflow

# Step2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
# poolsize (2,2) 2x2 -> not losing feature

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3,3), activation = 'relu')) # we already have pooled feature (no input shape)
classifier.add(MaxPooling2D(pool_size=(2,2)))

#classifier.add(Conv2D(64, (3,3), activation = 'relu')) # we already have pooled feature (no input shape)
#classifier.add(MaxPooling2D(pool_size=(2,2)))


# Step3 - Flattening
classifier.add(Flatten())

# Step4 - Full Connection - Adding a full conection layer
classifier.add(Dense(output_dim = 128, activation = 'relu'))
# output_dim = the number of node in hidden layer (power of two)
classifier.add(Dense(output_dim = 4, activation = 'sigmoid'))
# output_dim = the number of outcome layer

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# optimizer = randomly deactivate node - remove relationship between independent nodes
# loss = categorical_crossentropy(more than two)

# NN structure save
model_json = classifier.to_json()
with open("shoes_nn2.json", "w") as json_file:
    json_file.write(model_json)


# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,  # pixel value between 0 and 1
        shear_range=0.2,
        zoom_range=0.2,  # Random zoom
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory( 'Shoes_data/training_set',
                                                  target_size=(64, 64), # (128,128) for better accuracy. size of images that is expected in CNN model (64x64)
                                                  batch_size=32,
                                                  class_mode='categorical')  # class of dependent variable binary?
       
test_set = test_datagen.flow_from_directory('Shoes_data/test_set',
                                            target_size=(64, 64), # (128,128) for better accuracy. 
                                            batch_size=32,
                                            class_mode='categorical')

# Call back
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
chkpt = ModelCheckpoint('shoe_weight2.hdf5', monitor='val_loss', verbose=1, save_best_only=True)
es = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=1, verbose=1, mode='auto')

classifier.fit_generator(training_set,
                          steps_per_epoch=8000, # number of image for train
                          epochs=50,
                          validation_data=test_set,
                          validation_steps=2000,
                          callbacks = [chkpt,es])# number of image for test

 
# Load structure and weight of NN
# load json and create NN structure model
# =============================================================================
# from keras.models import model_from_json
# json_file = open('shoes_nn2.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# classifier = model_from_json(loaded_model_json)
# =============================================================================


# load the weights of NN into new model
classifier.load_weights("shoe_weight2.hdf5")
print("Loaded model from disk")
 


# Part 3 - Making new prediction

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('Shoes_data/single_prediction/ab.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0) # input must be in the batch
test_image = test_image/255

result = classifier.predict(test_image)
#training_set.class_indices
if result[0][0] == 0:
  prediction = 'Boots'
elif result[0][0] == 1:
  prediction = 'sandals'
elif result[0][0] == 2:
  prediction = 'Shoes'
else:
  prediction = 'Slippers'