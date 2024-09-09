from river.base import DriftDetector, DriftAndWarningDetector
import numpy as np
import collections
from pymfe.mfe import MFE
import pandas as pd


class MultiDetector(DriftAndWarningDetector):
    def __init__(self):
        super().__init__()
        self.class_affected = []

    def getClassesAffected(self):
        return self.classes_affected


class InformedDrift(MultiDetector):
    def __init__(
        self, n_classes: int, alpha:float = 0.2,window_size: int = 100, grace_period: int = 5, whole_drift: bool = False
    ):
        super().__init__()
        self.classifiers = {}
        self.windows = {
            key: collections.deque(maxlen=window_size) for key in range(0, n_classes)
        }
        self.current_concept = {key: [] for key in range(0, n_classes)}
        self.total_concept = {key: [] for key in range(0, n_classes)}
        self.window_size = window_size
        self.grace_period = grace_period
        self.predictions ={key: [] for key in range(0, n_classes)}
        self.distance_to_centroid_hist = {key: [] for key in range(0, n_classes)}
        self._features = [ "iq_range.mean", "kurtosis.mean", "mean.mean",
                             "median.mean", "sd.mean", "t_mean.mean", "eigenvalues.mean"]
        
        self.att_max = {key: [] for key in range(0, n_classes)}
        self.att_min = {key: [] for key in range(0, n_classes)}
        self.centroid = {key: [] for key in range(0, n_classes)}
        self.warning = np.zeros(n_classes)
        self.drift = np.zeros(n_classes)
        self.whole_drift = whole_drift
        self.alpha = alpha

    def update(self, x: np.array, y: int):
        x["class"] = y
        self.windows[y].append(x)
        if (self.drift.any()):
            self.drift = np.zeros(len(self.drift))
        
        if len(self.windows[y]) == self.windows[y].maxlen:
            #print (len(self.windows[y]))
            df = pd.DataFrame(self.windows[y])
            mfe = MFE(groups=["general", "statistical"], summary=["mean"])
            mfe.fit(df.iloc[:, :-1].to_numpy(), df.iloc[:, -1].to_numpy())
            ft = mfe.extract()
            ft = dict(zip(ft[0], ft[1]))
            current_window = {key: ft[key] for key in self._features}  
            #self.current_concept[y].append(current_window)
            self.total_concept[y].append(current_window)
            
            if len(self.centroid[y]) == 0:
                self.centroid[y] = pd.Series(current_window)
            

            if len(self.current_concept[y]) > self.grace_period:
                distance_to_centroid = abs(self.centroid[y] - pd.Series(current_window)) 
                
                relative_dif = distance_to_centroid/abs(self.att_max[y][-1] - self.att_min[y][-1])
                
                drifted_att = sum(relative_dif > 1 + self.alpha)
                if (drifted_att > 2):
                    if self.whole_drift:
                        for c in range (len(self.current_concept.keys())):
                            self.current_concept[c].clear()
                    self.current_concept[y].clear()
                    self.warning[y] = 0
                    self.drift[y] = 1
                elif (drifted_att > 1):
                    if not self.warning[y]:
                        self.warning[y] = 1
                    else:
                        #for c in range(len(self.current_concept)):
                        self.current_concept[y].clear()
                        self.warning[y] = 0
                        self.drift[y] = 1
                else:
                    self.current_concept[y].append(current_window)
                    self.centroid[y] = pd.DataFrame(self.current_concept[y]).mean()
                    self.warning[y] = 0
                    self.drift[y] = 0
                self.att_max[y].append(self.att_max[y][-1])
                self.att_min[y].append(self.att_min[y][-1])
            else:
                self.current_concept[y].append(current_window)
                self.centroid[y] = pd.DataFrame(self.current_concept[y]).mean()
                self.att_max[y].append(pd.DataFrame(self.current_concept[y]).max())
                self.att_min[y].append(pd.DataFrame(self.current_concept[y]).min())
                self._drift_detected = False
                self._warning_detected = False
                
            

            self.windows[y].clear()
            
        self._drift_detected = False

class DummyDetector(MultiDetector):
    def __init__(self, n_classes:int, driftPositions: dict):
        """
            Drift positions should be in the format:
            {
                index: [classes drifted in that index]
            }
        """
        self.n_classes = n_classes
        self.driftPositions = driftPositions
        self.drift = np.zeros(self.n_classes)
        self.instanceCounter = 0
        super().__init__()
    
    def update(self, *args, **kwargs):
        self.instanceCounter += 1
        drifted = self.driftPositions.get(self.instanceCounter)
        if drifted:
            for c in drifted:
                self.drift[c] = 1
        else:
            self.drift = np.zeros(self.n_classes)
