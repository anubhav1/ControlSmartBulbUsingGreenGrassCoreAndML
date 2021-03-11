from pyimagesearch.az_dataset import load_az_dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras import Model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import build_montages
import matplotlib.pyplot as plt
import cv2
import numpy as np

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS =50
INIT_LR = 1e-1
BS = 32

# load the myhandwriting datasets
print("[INFO] loading datasets...")
(data, labels) = load_az_dataset("myhandwriting.csv")

# add a channel dimension to every image in the dataset and scale the
# pixel intensities of the images from [0, 255] down to [0, 1]
data = np.expand_dims(data, axis=-1)
data /= 255.0

# convert the labels from integers to vectors(one-hot encoding)
le = LabelBinarizer()
labels = le.fit_transform(labels)
counts = labels.sum(axis=0)

# account for skew in the labeled data
classTotals = labels.sum(axis=0)
classWeight = {}

# loop over all classes and calculate the class weight
for i in range(0, len(classTotals)):
	classWeight[i] = classTotals.max() / classTotals[i]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.20, stratify=labels, random_state=42)

#looking at images to see whether everyting is fine
# for i in range(49):  
# 	plt.subplot(7,7, 1 + i)
# 	plt.imshow(data[i], cmap=plt.get_cmap('gray'))
# plt.show()

# construct the image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=10,
	zoom_range=0.05,
	width_shift_range=0.1,
	height_shift_range=0.1,
	shear_range=0.15,
	horizontal_flip=False,
	fill_mode="nearest")

#Starting Transfer learning using fine tuning
pretrained_model = load_model("handwriting.h5")
pretrained_model.summary()
# removing last dense node with 36 outputs because we have less outputs
pretrained_model._layers.pop()

#Freezing the layers
for layer in pretrained_model.layers:
	layer.trainable = False

dense_layer_1 = Dense(512, activation='relu', name='dense_layer_1')
dense_layer_2 = Dense(256, activation='relu', name='dense_layer_2')
dropout_layer_1 = Dropout(0.3)
dropout_layer_2 = Dropout(0.5)
dense_layer_4 = Dense(counts.size, activation='softmax', name='dense_layer_4')

#Adding new layers in pretrained_model
x = pretrained_model.layers[-2].output
x = dense_layer_1(x)
x = dropout_layer_1(x)
x = dense_layer_2(x)
x = dropout_layer_2(x)
preds = dense_layer_4(x)
model = Model(pretrained_model.input, preds)
model.summary()

# initialize and compile our deep neural network
print("[INFO] compiling model...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	class_weight=classWeight,
	verbose=1)

# define the list of label names
labelNames = "ABCDEFGILMNOPRSTUV"
labelNames = [l for l in labelNames]

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=labelNames))

# save the model to disk
print("serializing network...")
model.save("myhandwriting.h5", save_format="h5")

# construct a plot that plots and saves the training history
print("Plotting evaluations...")
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot1.png")

#Testing the model
images = []
#randomly select a few testing characters for testing
for i in np.random.choice(np.arange(0, len(testY)), size=(49,)):
	# classify the character
	probs = model.predict(testX[np.newaxis, i])
	prediction = probs.argmax(axis=1)
	label = labelNames[prediction[0]]

	# extract the image from the test data and initialize the text
	# label color as green (correct)
	image = (testX[i] * 255).astype("uint8")
	cv2.imwrite("image.jpg", image)
	color = (0, 255, 0)

	# otherwise, the class label prediction is incorrect
	if prediction[0] != np.argmax(testY[i]):
		color = (0, 0, 255)

	# merge the channels into one image, resize the image from 32x32
	# to 96x96 so we can better see it and then draw the predicted
	# label on the image
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
		color, 2)

	# add the image to our list of output images
	images.append(image)

# construct the montage for the images
montage = build_montages(images, (96, 96), (7, 7))[0]

# show the output montage
cv2.imshow("OCR Results", montage)
cv2.waitKey(0)