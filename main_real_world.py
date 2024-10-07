from generators.concept_drift import RealWorldConceptDriftStream


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
from tqdm import tqdm
from functools import lru_cache
import copy 

models = [
    ("HT", tree.HoeffdingTreeClassifier()),
    (
        "DDM-HT",
        drift.DriftRetrainingClassifier(
            tree.HoeffdingAdaptiveTreeClassifier(), drift_detector=drift.binary.DDM()
        ),
    ),
    ("EFHT", tree.ExtremelyFastDecisionTreeClassifier()),
    ("SRP", ensemble.SRPClassifier()),
    ("ARF", forest.ARFClassifier()),
    # ("LB", ensemble.LeveragingBaggingClassifier(model=tree.HoeffdingTreeClassifier())),
    (
        "ADWINBagging",
        ensemble.ADWINBaggingClassifier(model=tree.HoeffdingTreeClassifier()),
    ),
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
    )
]

streams = [
    (
        "semi_synth_1_to_3_sudden",
        1,
        [
            "./datasets/semi_synthetic/semi_synth_concept_1.csv",
            "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        ],
    ),
    (
        "semi_synth_1_to_3_gradual",
        15000,
        [
            "./datasets/semi_synthetic/semi_synth_concept_1.csv",
            "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        ],
    ),
    (
        "semi_synth_1_to_6_sudden",
        1,
        [
            "./datasets/semi_synthetic/semi_synth_concept_1.csv",
            "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        ],
    ),
    (
        "semi_synth_1_to_6_gradual",
        15000,
        [
            "./datasets/semi_synthetic/semi_synth_concept_1.csv",
            "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        ],
    ),
    (
        "semi_synth_6_to_3_sudden",
        1,
        [
            "./datasets/semi_synthetic/semi_synth_concept_6.csv",
            "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        ],
    ),
    (
        "semi_synth_6_to_3_gradual",
        15000,
        [
            "./datasets/semi_synthetic/semi_synth_concept_6.csv",
            "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        ],
    ),
]


def task(stream_path, model, dd=None):
    warnings.filterwarnings("ignore")
    stream_name, stream_width, stream_paths = stream_path
    stream_output = "./output/semi-synth/"
    n_class = 6

    stream = RealWorldConceptDriftStream(
        stream_paths[0],
        stream_paths[1],
        classes_affected=[5], width=stream_width, position=750000, size=125000, n_classes=6
    )
    #print (model)
    model_name, model = model
    model_local = copy.deepcopy(model)

    if isinstance(model_local, OneVsRestDriftAwareClassifier):
        if model_name == "OneVsAll-GT":
            drift_points = {
                75000: [n_class - 1],
            }

            model_local.driftDetector = DummyDetector(n_class, drift_points)
        else:
            model_local.driftDetector = InformedDrift(n_class)

    exp_name = "{}_{}".format(model_name, stream_name)
    print("Running {}...".format(exp_name))
    if not (os.path.exists("{}/{}.csv".format(stream_output, exp_name))):  # or True:
        exp = Experiment(
            exp_name, stream_output, model_local, dd, stream, stream_size=125000
        )

        exp.run()

        exp.save()




out = Parallel(n_jobs=1)(
        delayed(task)(stream, model)
        for stream, model in itertools.product(streams, models)
    )
