import os
import dlib
import cv2
from pyPiper import Node, Pipeline
from tqdm import tqdm
from imutils import paths
import face_recognition
import pickle
import shutil
import time
from merge_encodings import PicklesListCollator
import warnings

class FramesProvider(Node):
    def setup(self, sourcePath):
        self.sourcePath = sourcePath
        self.filesList = []
        for item in os.listdir(self.sourcePath):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.jpg':
                self.filesList.append(os.path.join(item))
        self.TotalFilesCount = self.size = len(self.filesList)
        self.ProcessedFilesCount = self.pos = 0

    def run(self, data):
        if self.ProcessedFilesCount < self.TotalFilesCount:
            self.emit({'id': self.ProcessedFilesCount, 'imagePath': os.path.join(self.sourcePath, self.filesList[self.ProcessedFilesCount])})
            self.ProcessedFilesCount += 1
            
            self.pos = self.ProcessedFilesCount
        else:
            self.close()

class FaceEncoder(Node):
    def setup(self, detection_method = 'cnn'):
        self.detection_method = detection_method
        # detection_method can be cnn or hog
    def run(self, data):
        id = data['id']
        imagePath = data['imagePath']
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb, model=self.detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]

        self.emit({'id': id, 'encodings': d})

class DatastoreManager(Node):
    def setup(self, encodingsOutputPath):
        self.encodingsOutputPath = encodingsOutputPath
    def run(self, data):
        encodings = data['encodings']
        id = data['id']
        with open(os.path.join(self.encodingsOutputPath, 'encodings_' + str(id) + '.pickle'), 'wb') as f:
            f.write(pickle.dumps(encodings))

class TqdmUpdate(tqdm):
    def update(self, done, total_size=None):
        if total_size is not None:
            self.total = total_size
        self.n = done
        super().refresh()


if __name__ == "__main__":
    CurrentPath = os.getcwd()
    FramesDirectory = "Frames"
    FramesDirectoryPath = os.path.join(CurrentPath, FramesDirectory)
    EncodingsFolder = "Encodings"
    EncodingsFolderPath = os.path.join(CurrentPath, EncodingsFolder)

    if os.path.exists(EncodingsFolderPath):
        shutil.rmtree(EncodingsFolderPath, ignore_errors=True)
        time.sleep(0.5)
    os.makedirs(EncodingsFolderPath)

    pipeline = Pipeline(FramesProvider("Files source", sourcePath=FramesDirectoryPath) | 
                        FaceEncoder("Encode faces") | 
                        DatastoreManager("Store encoding", encodingsOutputPath=EncodingsFolderPath), n_threads = 4, quiet = True)
    pbar = TqdmUpdate()
    pipeline.run(update_callback=pbar.update)

    print()
    print('[INFO] Encodings extracted')

    picklesListCollator = PicklesListCollator(picklesInputDirectory = EncodingsFolderPath)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        picklesListCollator.GeneratePickle("encodings.pickle")

    print('[INFO] Pickles merged')
