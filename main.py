from river import tree, drift, naive_bayes, ensemble, forest, multiclass
from experiment import Experiment
from joblib import Parallel, delayed
import itertools
from glob import glob
import os
from utils.csv import CSVStream

from copy import deepcopy
import warnings
from classifiers.drift_aware_multi_class import OneVsRestDriftAwareClassifier
from drift_detectors.multi_class_detector import DummyDetector, InformedDrift

models = [
 
    ("HT", tree.HoeffdingTreeClassifier()),
    (
        "DDM-HT",
        drift.DriftRetrainingClassifier(
            tree.HoeffdingAdaptiveTreeClassifier(), drift_detector=drift.binary.DDM()
        ),
    ),
    (
        "GT-HT",
        drift.DriftRetrainingClassifier(
            tree.HoeffdingAdaptiveTreeClassifier(), drift_detector=drift.DummyDriftDetector(t_0=100000)
        ),
    ),
    ("EFHT", tree.ExtremelyFastDecisionTreeClassifier()),
    ("SRP", ensemble.SRPClassifier()),
    ("ARF", forest.ARFClassifier()),
    #("LB", ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier())),
    ("ADWINBagging", ensemble.ADWINBaggingClassifier(model=tree.HoeffdingTreeClassifier())),
    ("AdaBoost", ensemble.AdaBoostClassifier(model=tree.HoeffdingTreeClassifier())),
    ("OneVsAll-NC", multiclass.OneVsRestClassifier(tree.HoeffdingTreeClassifier())),
    (
        "OneVsAll-DDM",
        drift.DriftRetrainingClassifier(
            multiclass.OneVsRestClassifier(tree.HoeffdingTreeClassifier()),
            drift_detector=drift.binary.DDM(),
        ),
    ),
    (
        "OneVsAll-GT",
        OneVsRestDriftAwareClassifier(tree.HoeffdingTreeClassifier(), None),
    ),
    (
        "OneVsAll-CIDDM",
        OneVsRestDriftAwareClassifier(tree.HoeffdingTreeClassifier(), None),
    ),
    (
        "Bagging-CIDDM",
        ensemble.BaggingClassifier(
            OneVsRestDriftAwareClassifier(tree.HoeffdingTreeClassifier(), drift.DummyDriftDetector())
        ),
    ),
    (
        "Bagging-GT",
        ensemble.BaggingClassifier(
            OneVsRestDriftAwareClassifier(tree.HoeffdingTreeClassifier(), drift.DummyDriftDetector())
        ),
    ),

]

models = [("ADWINBagging", ensemble.ADWINBaggingClassifier(model=tree.HoeffdingTreeClassifier()))]

def task(stream_path, model, dd=None):
    warnings.filterwarnings("ignore")
    stream = CSVStream("{}".format(stream_path))
    stream_name = os.path.splitext(os.path.basename(stream_path))[0]
    stream_output = os.path.dirname(stream_path).replace("datasets", "output")
    n_class = int(stream_name.split("_")[-1])

    model_name, model = model
    model_local = model.clone()

    if isinstance(model_local, OneVsRestDriftAwareClassifier) or isinstance(
        model_local, ensemble.BaggingClassifier
    ):
        if model_name == "OneVsAll-GT" or model_name == "Bagging-GT":
            if n_class > 5:
                drift_points = {
                    100000: [n_class - 1, n_class - 2],
                    200000: [n_class - 1, n_class - 2],
                    300000: [n_class - 1, n_class - 2],
                }
            else:
                drift_points = {
                    100000: [n_class - 1],
                    200000: [n_class - 1],
                    300000: [n_class - 1],
                }
            if isinstance(model_local, ensemble.BaggingClassifier):
                model_local = ensemble.BaggingClassifier(
                    OneVsRestDriftAwareClassifier(
                        tree.HoeffdingTreeClassifier(),
                        DummyDetector(n_class, drift_points),
                    )
                )
            else:
                model_local.driftDetector = DummyDetector(n_class, drift_points)
        else:
            if isinstance(model_local, ensemble.BaggingClassifier):
                model_local = ensemble.BaggingClassifier(
                    OneVsRestDriftAwareClassifier(
                        tree.HoeffdingTreeClassifier(), InformedDrift(n_classes=n_class)
                    )
                )
            else:
                model_local.driftDetector = InformedDrift(n_classes=n_class)

    exp_name = "{}_{}".format(model_name, stream_name)
    print("Running {}...".format(exp_name))
    if not (os.path.exists("{}/{}.csv".format(stream_output, exp_name))):  # or True:
        exp = Experiment(
            exp_name, stream_output, model_local, dd, stream, stream_size=400000
        )

        exp.run()

        exp.save()


PATH = "./datasets/"
EXT = "*.csv"
streams = [file for file in glob(os.path.join(PATH, EXT))]

out = Parallel(n_jobs=8)(
    delayed(task)(stream, model) for stream, model in itertools.product(streams, models)
)
