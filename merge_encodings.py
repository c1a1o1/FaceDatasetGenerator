import pickle
import os

class PicklesListCollator:
    def __init__(self, picklesInputDirectory):
        self.picklesInputDirectory = picklesInputDirectory
    
    def GeneratePickle(self, outputFilepath):
        datastore = []

        ListOfPickleFiles = []
        for item in os.listdir(self.picklesInputDirectory):
            _, fileExt = os.path.splitext(item)
            if fileExt == '.pickle':
                ListOfPickleFiles.append(os.path.join(self.picklesInputDirectory, item))

        for picklePath in ListOfPickleFiles:
            with open(picklePath, "rb") as f:
                data = pickle.loads(f.read())
                datastore.extend(data)

        with open(outputFilepath, 'wb') as f:
            f.write(pickle.dumps(datastore))

if __name__ == "__main__":
    CurrentPath = os.getcwd()

    EncodingsInputDirectory = "Encodings"
    EncodingsInputDirectoryPath = os.path.join(CurrentPath, EncodingsInputDirectory)

    OutputEncodingPickleFilename = "encoding.pickle"

    os.remove(OutputEncodingPickleFilename)
    picklesListCollator = PicklesListCollator(picklesInputDirectory = EncodingsInputDirectoryPath)
    picklesListCollator.GeneratePickle(OutputEncodingPickleFilename)