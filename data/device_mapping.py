import pandas as pd

df = pd.read_csv("master_msrp.csv")

airpods_models = (
    df.loc[
        df["Device"].astype(str).str.strip().eq("AirPods"),
        "Standardized Model"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .sort_values()
)

print(f"Unique AirPods models: {len(airpods_models)}")
print(airpods_models.to_string(index=False))