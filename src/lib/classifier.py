from sklearn.svm import SVC

class FaceClassifier():
    def __init__(self):
        self._clf = SVC(C=1, kernel="linear", probability=True)
    
    def fit(self, embeddings, labels):
        self._clf.fit(embeddings, labels)

    def predict(self, vec):
        return self._clf.predict_proba(vec)
