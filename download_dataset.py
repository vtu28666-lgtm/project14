import kagglehub
import os
import traceback

try:
    print("Starting dataset download...")

    path = kagglehub.dataset_download(
        "vinayak123tyagi/bearing-dataset"
    )

    print("\nDataset downloaded successfully!")
    print("Dataset location:", path)

    print("\nFiles available in dataset:")

    for root, dirs, files in os.walk(path):
        for file in files:
            print(os.path.join(root, file))

except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()

input("\nPress Enter to exit...")