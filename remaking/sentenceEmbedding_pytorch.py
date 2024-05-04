from sentence_transformers import SentenceTransformer

class SentenceEmbedding(object):
    """docstring for SentenceEmbedding"""

    def __init__(self):
        self.model = None
        self.loadModel()

    def loadModel(self):
        if self.model is None:
            print('Loading Pytorch model....')
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print('Model Loaded.')

    def embed(self, input_):
        ### input_ should be list of string
        embedding = self.model.encode(input_)
        return embedding    # return a numpy array

    def getEmbeddings(self, data):
        vector = []
        start = 0
        step = 10000

        for i in range(int(len(data) / step)):
            samples = data[start:start + step]
            start += step
            features = self.embed(samples).tolist()
            vector += features

        if len(data) % step != 0:
            samples = data[start:]
            features = self.embed(samples).tolist()
            vector += features
        return vector
    
    # use moel API, then convert to numpy
    # def getEmbeddings(self, data):
    #     return self.model(data).numpy()
