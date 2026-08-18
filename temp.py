import joblib
import os

input_file = "model.pkl"
output_file = "model_compressed.pkl"

model = joblib.load(input_file)

joblib.dump(
    model,
    output_file,
    compress=9
)

print(
    f"Original:   {os.path.getsize(input_file) / (1024 * 1024):.2f} MB"
)

print(
    f"Compressed: {os.path.getsize(output_file) / (1024 * 1024):.2f} MB"
)