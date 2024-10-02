from generators.concept_drift import RealWorldConceptDriftStream

def main():
    st = RealWorldConceptDriftStream(
        "./datasets/semi_synthetic/semi_synth_concept_1.csv",
        "./datasets/semi_synthetic/semi_synth_concept_3.csv",
        classes_affected=[2], width=1, position=10000, size=20000
    )
    
    for idx, (X,y) in enumerate(st):
        if idx+1 % 1000 == 0:
            print (idx)
            
main()