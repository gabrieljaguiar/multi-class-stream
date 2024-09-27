from imblearn.over_sampling import SMOTE
import pandas as pd

concepts_breaks = [14.352, 19.500, 33.240, 38.682, 39.510]

df = pd.read_csv("./INSECTS abrupt_balanced.csv")