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
        "ADWIN-HT",
        drift.DriftRetrainingClassifier(
            tree.HoeffdingAdaptiveTreeClassifier(), drift_detector=drift.ADWIN()
        ),
    )("EFHT", tree.ExtremelyFastDecisionTreeClassifier()),
    ("SRP", ensemble.SRPClassifier()),
    ("ARF", forest.ARFClassifier()),
    ("LB", ensemble.LeveragingBaggingClassifier()),
    ("ADWINBagging", ensemble.ADWINBaggingClassifier()),
    ("AdaBoost", ensemble.AdaBoostClassifier()),
    ("OneVsAll-NC", multiclass.OneVsRestClassifier(tree.HoeffdingTreeClassifier())),
    (
        "OneVsAll-ADWIN",
        drift.DriftRetrainingClassifier(
            multiclass.OneVsRestClassifier(tree.HoeffdingTreeClassifier()),
            drift_detector=drift.ADWIN(),
        ),
    )(
        "OneVsAll-GT",
        OneVsRestDriftAwareClassifier(tree.HoeffdingTreeClassifier(), None),
    )(
        "OneVsAll-CIDDM",
        OneVsRestDriftAwareClassifier(tree.HoeffdingTreeClassifier(), None),
    ),
]


def task(stream_path, model, dd=None):
    warnings.filterwarnings("ignore")
    stream = CSVStream("{}".format(stream_path))
    stream_name = os.path.splitext(os.path.basename(stream_path))[0]
    stream_output = os.path.dirname(stream_path).replace("datasets", "output")
    n_class = int(stream_name.split("_")[-1])

    model_name, model = model
    model_local = model.clone()

    if isinstance(model_local, OneVsRestDriftAwareClassifier):
        if model_name == "OneVsAll-GT":
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

            model_local.driftDetector = DummyDetector(n_class, drift_points)
        else:
            model_local.driftDetector = InformedDrift(n_class)

    exp_name = "{}_{}".format(model_name, stream_name)
    print("Running {}...".format(exp_name))
    if (os.path.exists("{}/{}.csv".format(stream_output, exp_name))) or True:
        exp = Experiment(
            exp_name, stream_output, model_local, dd, stream, stream_size=20000
        )

        exp.run()

        exp.save()


for model in models:
    PATH = "./datasets/"
    EXT = "*.csv"
    streams = [
        file
        for path, subdir, files in os.walk(PATH)
        for file in glob(os.path.join(path, EXT))
    ]

    out = Parallel(n_jobs=1)(
        delayed(task)(stream, model)
        for stream, model in itertools.product(streams, models)
    )
