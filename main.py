from river import tree, drift, naive_bayes
from experiment import Experiment
from joblib import Parallel, delayed
import itertools
from classifiers import AdaptedDriftRetrainingClassifier

from glob import glob
import os
from utils.csv import CSVStream

from drift_detectors import (
    RDDM_M,
    EDDM_M,
    STEPD_M,
    ECDDWT_M,
    ADWINDW,
    KSWINDW,
    PHDW,
    FHDDMDW,
    FHDDMSDW,
)
from drift_detectors import ECDDWTConfig, EDDMConfig, RDDMConfig, STEPDConfig, InformedDrift
from copy import deepcopy
import warnings



models = [
    ("HT", tree.HoeffdingTreeClassifier()),
    ("NB", naive_bayes.GaussianNB()),
    (
        "NB_RT",
        AdaptedDriftRetrainingClassifier(
           model=naive_bayes.GaussianNB(), drift_detector=ADWINDW(), train_in_background=False,
        ),
    ),
    (
        "HT_RT",
        AdaptedDriftRetrainingClassifier(
            model=tree.HoeffdingTreeClassifier(), drift_detector=ADWINDW(), train_in_background=False,
        ),
    ),
]

dds = [
    ("ADWIN", ADWINDW()),
    ("CIDDM", InformedDrift(n_classes=2)),
    ("NO_DRIFT", drift.NoDrift()),
    ("PageHinkley", PHDW()),
    ("HDDM", drift.binary.HDDM_W()),
    ("KSWIN", KSWINDW()),
    ("DDM", drift.binary.DDM()),
    ("RDDM", RDDM_M(RDDMConfig())),
    ("STEPD", STEPD_M(STEPDConfig())),
    ("ECDD", ECDDWT_M(ECDDWTConfig())),
    ("EDDM", EDDM_M(EDDMConfig())),
    ("FHDDM", FHDDMDW()),
    ("FHDDMS", FHDDMSDW()),
    ("GT", drift.DummyDriftDetector(t_0=100000))
]
def task(stream_path, model, dd):
    warnings.filterwarnings("ignore")
    stream = CSVStream("{}".format(stream_path))
    stream_name = os.path.splitext(os.path.basename(stream_path))[0]
    stream_output = os.path.dirname(stream_path).replace("datasets", "output")
    n_class = int(stream_name.split("_")[-1])
    
    model_name, model = model
    model_local = model.clone()
    dd_name, dd_method = dd

    if (type(dd_method) == InformedDrift):
        dd_local = InformedDrift(n_classes=n_class, window_size=500, grace_period=10)
    elif (type(dd_method) == drift.DummyDriftDetector):
        dd_local = drift.DummyDriftDetector(t_0=100000)
    else:
        dd_local = dd_method.clone()
    if type(model_local) == AdaptedDriftRetrainingClassifier:
        if (type(dd_method) == InformedDrift):
            model_local.drift_detector = InformedDrift(n_classes=n_class, window_size=500, grace_period=10)
        elif (type(dd_method) == drift.DummyDriftDetector):
            model_local.drift_detector = drift.DummyDriftDetector(t_0=100000)
        else:
            model_local.drift_detector = dd_local.clone()
    exp_name = "{}_{}_{}".format(model_name, dd_name, stream_name)
    print("Running {}...".format(exp_name))
    exp = Experiment(exp_name, stream_output, model_local, dd_local, stream, stream_size=400000)

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

    out = Parallel(n_jobs=16)(
        delayed(task)(stream, model, dd)
        for stream, dd in itertools.product(streams, dds)
    )
