import pandas as pd

# load raw data
df = pd.read_csv("1000_train.csv.gz")

# save sample to processed folder
df.head(1000).to_csv(r"C:\Users\DELL\Desktop\GSOC Projects\hep-event-classifier\data\processed\sample.csv", index=False)
