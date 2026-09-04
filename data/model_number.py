import pandas as pd

df = pd.read_csv("master_msrp.csv")

models = (
    df["Standardized Model"]
    .dropna()
    .astype(str)
    .str.strip()
    .drop_duplicates()
    .sort_values()
)

with open("model_list.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(models))

print(f"Exported {len(models)} models to model_list.txt")