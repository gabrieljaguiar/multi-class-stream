from river.base.classifier import Classifier
from river.multiclass import OneVsRestClassifier
from ..drift_detectors.multi_class_detector import InformedDrift
import numpy as np

class OneVsRestDriftAwareClassifier(OneVsRestClassifier):
    def __init__(self, classifier: Classifier, driftDetector:InformedDrift):
        super().__init__(classifier)
        self.driftDetector = driftDetector
    
    def learn_one(self, x, y, **kwargs):
        self.driftDetector.update(x,y)
        if (any(self.driftDetector.drift)):
            affected_classes = 2**np.where(self.driftDetector.drift)[0]
            for drifted_classes in affected_classes:
                self.classifiers[drifted_classes] = self.classifier.clone()
        return super().learn_one(x, y, **kwargs)