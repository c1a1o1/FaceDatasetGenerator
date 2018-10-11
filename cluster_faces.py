from sklearn.cluster import DBSCAN
from imutils import build_montages
import numpy as np
import os
import pickle
import cv2
import shutil
import time

def rescale_by_height(image, target_height, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_height` (preserving aspect ratio)."""
    w = int(round(target_height * image.shape[1] / image.shape[0]))
    return cv2.resize(image, (w, target_height), interpolation=method)

def rescale_by_width(image, target_width, method=cv2.INTER_LANCZOS4):
    """Rescale `image` to `target_width` (preserving aspect ratio)."""
    h = int(round(target_width * image.shape[0] / image.shape[1]))
    return cv2.resize(image, (target_width, h), interpolation=method)

encodings_filename = "encodings.pickle"
output_directory = os.getcwd()

print(str(encodings_filename) + ' ' + str(output_directory))

OutputFolder = os.path.join(output_directory, "ClusteredFaces")
if not os.path.exists(OutputFolder):
    os.makedirs(OutputFolder)
else:
	shutil.rmtree(OutputFolder)
	time.sleep(0.5)
	os.makedirs(OutputFolder)

InputEncodingFile = encodings_filename
if not (os.path.isfile(InputEncodingFile) and os.access(InputEncodingFile, os.R_OK)):
	print('The input encoding file, ' + str(InputEncodingFile) + ' does not exists or unreadable')
	exit()

NumberOfParallelJobs = -1

# load the serialized face encodings + bounding box locations from
# disk, then extract the set of encodings to so we can cluster on
# them
print("[INFO] Loading encodings")
data = pickle.loads(open(InputEncodingFile, "rb").read())
data = np.array(data)

encodings = [d["encoding"] for d in data]

# cluster the embeddings
print("[INFO] Clustering")
clt = DBSCAN(eps=0.5, metric="euclidean", n_jobs=NumberOfParallelJobs)
clt.fit(encodings)

# determine the total number of unique faces found in the dataset
labelIDs = np.unique(clt.labels_)
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print("[INFO] # unique faces: {}".format(numUniqueFaces))


MontageFolderPath = os.path.join(OutputFolder, "Montage")
os.makedirs(MontageFolderPath)

# loop over the unique face integers
for labelID in labelIDs:
	# find all indexes into the `data` array that belong to the
	# current label ID, then randomly sample a maximum of 25 indexes
	# from the set
	print("[INFO] faces for face ID: {}".format(labelID))

	FaceFolder = os.path.join(OutputFolder, "Face_" + str(labelID))
	os.makedirs(FaceFolder)

	idxs = np.where(clt.labels_ == labelID)[0]
	idxs = np.random.choice(idxs, size=min(25, len(idxs)),
		replace=False)

	# initialize the list of faces to include in the montage
	# faces = []
	portraits = []

	# loop over the sampled indexes
	counter = 1
	for i in idxs:
		# load the input image and extract the face ROI
		image = cv2.imread(data[i]["imagePath"])
		(o_top, o_right, o_bottom, o_left) = data[i]["loc"]

		height, width, channel = image.shape

		widthMargin = 100
		heightMargin = 150

		top = o_top - heightMargin
		if top < 0:
			top = 0
		
		bottom = o_bottom + heightMargin
		if bottom > height:
			bottom = height
		
		left = o_left - widthMargin
		if left < 0:
			left = 0
		
		right = o_right + widthMargin
		if right > width:
			right = width

		portrait = image[top:bottom, left:right]

		if len(portraits) < 25:
			portraits.append(portrait)

		# face = image[top:bottom, left:right]

		# force resize the face ROI to 96x96 and then add it to the
		# faces montage list
		# face = cv2.resize(face, (96, 96))
		# faces.append(face)

		portrait = rescale_by_width(portrait, 400)

		FaceFilename = "face_" + str(counter) + ".jpg"

		FaceImagePath = os.path.join(FaceFolder, FaceFilename)
		cv2.imwrite(FaceImagePath, portrait)





		widthMargin = 20
		heightMargin = 20

		top = o_top - heightMargin
		if top < 0:
			top = 0
		
		bottom = o_bottom + heightMargin
		if bottom > height:
			bottom = height
		
		left = o_left - widthMargin
		if left < 0:
			left = 0
		
		right = o_right + widthMargin
		if right > width:
			right = width

		AnnotationFilename = "face_" + str(counter) + ".txt"
		AnnotationFilePath = os.path.join(FaceFolder, AnnotationFilename)
		
		f = open(AnnotationFilePath, 'w')
		f.write(str(labelID) + ' ' + str(left) + ' ' + str(top) + ' ' + str(right) + ' ' + str(bottom) + "\n")
		f.close()


		counter += 1

	# create a montage using 96x96 "tiles" with 5 rows and 5 columns
	# montage = build_montages(faces, (96, 96), (5, 5))[0]
	montage = build_montages(portraits, (96, 120), (5, 5))[0]
	
	MontageFilenamePath = os.path.join(MontageFolderPath, "Face_" + str(labelID) + ".jpg")
	cv2.imwrite(MontageFilenamePath, montage)

	# show the output montage
	# title = "Face ID #{}".format(labelID)
	# title = "Unknown Faces" if labelID == -1 else title
	# cv2.imshow(title, montage)
	# cv2.waitKey(0)