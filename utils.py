# import the necessary packages
from tensorflow.keras.datasets import mnist
from imutils import build_montages
import numpy as np
import cv2
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt



def make_pairs(images, labels):
	"""
	return tuble nd-array contain per-images and there labels
	==> pairImages: list of two images
	==> pairLabels: list if images are similar(1) or not(0)

	return pairImages double number of given images, by build for each image similar pairs and negative pairs 'randomly'.
	---------------

		params:
			images: array-like contain images data
			labels: array-like contain images label
	"""

	pairImages = [] #Semilar Images in one list like [image1,image2] ==> same class
	pairLabels = [] #Label for lists if it positive or negative pairs

	numClasses = len(np.unique(labels)) #Number of classes
	idx = [np.where(labels == i)[0] for i in range(0, numClasses)] # list contain indexs for images for each class

  # loop over all images
	for idxA in range(len(images)):
  #  idxA is for image index

		currentImage = images[idxA] # Select image using index
		label = labels[idxA] # Select the label


		idxB = np.random.choice(idx[label])# randomly pick an index that belongs to the *same* class
		posImage = images[idxB] # Select image using index



		pairImages.append([currentImage, posImage])# prepare a positive pair
		pairLabels.append([1]) # that image is semilar to each other


		negIdx = np.where(labels != label)[0]# Select list of indexs from differant class
		negImage = images[np.random.choice(negIdx)] #Select random image

		# prepare a negative pair of images and update our lists
		pairImages.append([currentImage, negImage])
		pairLabels.append([0])

	# return a 2-tuple of our image pairs and labels
	return (np.array(pairImages), np.array(pairLabels))

def main():

  # load MNIST dataset and scale the pixel values to the range of [0, 1]
  print("[INFO] loading MNIST dataset...")
  (trainX, trainY), (testX, testY) = mnist.load_data()
  # build the positive and negative image pairs
  print("[INFO] preparing positive and negative pairs...")
  (pairTrain, labelTrain) = make_pairs(trainX, trainY)
  (pairTest, labelTest) = make_pairs(testX, testY)
  # initialize the list of images that will be used when building our
  # montage
  images = []
  # loop over a sample of our training pairs
  for i in np.random.choice(np.arange(0, len(pairTrain)), size=(49,)):
    # grab the current image pair and label
    imageA = pairTrain[i][0]
    imageB = pairTrain[i][1]
    label = labelTrain[i]
    # to make it easier to visualize the pairs and their positive or
    # negative annotations, we're going to "pad" the pair with four
    # pixels along the top, bottom, and right borders, respectively
    output = np.zeros((36, 60), dtype="uint8")
    pair = np.hstack([imageA, imageB])
    output[4:32, 0:56] = pair
    # set the text label for the pair along with what color we are
    # going to draw the pair in (green for a "positive" pair and
    # red for a "negative" pair)
    text = "neg" if label[0] == 0 else "pos"
    color = (0, 0, 255) if label[0] == 0 else (0, 255, 0)
    # create a 3-channel RGB image from the grayscale pair, resize
    # it from 60x36 to 96x51 (so we can better see it), and then
    # draw what type of pair it is on the image
    vis = cv2.merge([output] * 3)
    vis = cv2.resize(vis, (96, 51), interpolation=cv2.INTER_LINEAR)
    cv2.putText(vis, text, (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
      color, 2)
    # add the pair visualization to our list of output images
    images.append(vis)
  # construct the montage for the images
  montage = build_montages(images, (96, 51), (7, 7))[0]
  # show the output montage
  plt.imshow(montage)
  plt.show()

# if __name__ == "__main__":
#   main()



def euclidean_distance(vectors):
	# unpack the vectors into separate lists
	(featsA, featsB) = vectors
	# compute the sum of squared distances between the vectors
	sumSquared = K.sum(K.square(featsA - featsB), axis=1,
		keepdims=True)
	# return the euclidean distance between the vectors
	return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def plot_training(H, plotPath):
	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.plot(H.history["accuracy"], label="train_acc")
	plt.plot(H.history["val_accuracy"], label="val_acc")
	plt.title("Training Loss and Accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss/Accuracy")
	plt.legend(loc="lower left")
	plt.savefig(plotPath)





