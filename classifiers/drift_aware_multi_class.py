from river.base.classifier import Classifier
from river.multiclass import OneVsRestClassifier
from drift_detectors.multi_class_detector import InformedDrift
import numpy as np
from collections import deque

class OneVsRestDriftAwareClassifier(OneVsRestClassifier):
    def __init__(self, classifier: Classifier, driftDetector:InformedDrift):
        super().__init__(classifier)
        self.windows = {}
        self.driftDetector = driftDetector
        self.idx = 0
        self.max_size = 1000
    
    def learn_one(self, x, y, **kwargs):
        x_feat = x.copy()
        if y not in self.windows.keys():
            self.windows[y] = deque(maxlen=self.max_size)
        self.windows[y].append((x,y))
        self.driftDetector.update(x_feat,y)
        if (any(self.driftDetector.drift)):
            affected_classes = np.where(self.driftDetector.drift)[0]
            #print ("Drift detected in class {} in {}".format(affected_classes, self.idx))
            for drifted_class in affected_classes:
                del self.classifiers[drifted_class]
                while len(self.windows[drifted_class]) > 0:
                    x_w, y_w = self.windows[drifted_class].pop()
                    #print (y_w)
                    super().learn_one(x_w, y_w, **kwargs)
                

            #print (self.classifiers)
        self.idx += 1
        super().learn_one(x, y, **kwargs)