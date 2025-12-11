# import os
# import csv
# import pandas as pd
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
#
# # ============================================================
# #                    CONFIG (EDIT THESE ONLY)
# # ============================================================
#
# # Your REAL dataset path
# DATASET_BASE = r"C:\Users\Abu Hurrairah\Downloads\files\datasets\License-Plate-dataset"
#
# # Your noisy folders path (already generated)
# NOISY_BASE = r"K:\pythonProject\OfficeWork\Mam Sadaf\pythonProject1\Zeeshan\yolo_noisy_outputs"
#
# # Where YOLO training results should be saved
# OUTPUT_RUNS = r"C:\Users\Abu Hurrairah\Downloads\files\yolo_training_results"
#
# # Your clean data.yaml
# DATA_YAML = fr"{DATASET_BASE}\data.yaml"
#
# # YOLO weights
# MODEL_WEIGHTS = "yolov8n.pt"
#
# # CSV results file
# RESULTS_CSV = fr"{OUTPUT_RUNS}\yolo_noise_results.csv"
#
#
# # ============================================================
# #                 PREPARE TRAIN FOLDER PATHS
# # ============================================================
#
# def get_train_dir(noise_type, noise_level):
#     """
#     Matches EXACTLY your folder names:
#         random10, random20, ..., asymmetric10, ..., adversarial40
#     """
#     if noise_type == "clean":
#         return fr"{DATASET_BASE}\train"
#
#     return fr"{NOISY_BASE}\{noise_type}{noise_level}"
#
#
# # ============================================================
# #    Update YAML with custom train folder (val/test unchanged)
# # ============================================================
#
# def make_temp_yaml(original_yaml, train_dir, out_yaml):
#     with open(original_yaml, "r") as f:
#         content = f.readlines()
#
#     new_lines = []
#     for line in content:
#         if line.strip().startswith("train:"):
#             new_lines.append(f"train: {train_dir}\n")
#         else:
#             new_lines.append(line)
#
#     with open(out_yaml, "w") as f:
#         f.writelines(new_lines)
#
#
# # ============================================================
# #                     Write CSV Results
# # ============================================================
#
# def write_results(row):
#     os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
#     file_exists = os.path.exists(RESULTS_CSV)
#
#     with open(RESULTS_CSV, "a", newline="") as f:
#         writer = csv.writer(f)
#         if not file_exists:
#             writer.writerow(["noise_type", "noise_level", "mAP50", "mAP50_95", "precision", "recall"])
#         writer.writerow(row)
#
#
# # ============================================================
# #                MAIN YOLO TRAINING LOOP
# # ============================================================
#
# def train_all():
#     print("🚀 Starting YOLO training (using existing noisy folders)...\n")
#     os.makedirs(OUTPUT_RUNS, exist_ok=True)
#
#     noise_types = ["clean", "random", "asymmetric", "adversarial"]
#     noise_levels = [0, 10, 20, 40]
#
#     for ntype in noise_types:
#         for nlevel in noise_levels:
#
#             if ntype == "clean" and nlevel != 0:
#                 continue
#             if ntype != "clean" and nlevel == 0:
#                 continue
#
#             train_folder = get_train_dir(ntype, nlevel)
#
#             print("\n====================================================")
#             print(f" Training → {ntype} {nlevel}%")
#             print(f" Train folder → {train_folder}")
#             print("====================================================")
#
#             if not os.path.exists(train_folder):
#                 print(f"❌ SKIPPED — Folder not found: {train_folder}")
#                 continue
#
#             temp_yaml = fr"{OUTPUT_RUNS}\data_temp_{ntype}_{nlevel}.yaml"
#             make_temp_yaml(DATA_YAML, train_folder, temp_yaml)
#
#             model = YOLO(MODEL_WEIGHTS)
#
#             results = model.train(
#                 data=temp_yaml,
#                 epochs=20,
#                 imgsz=640,
#                 batch=16,
#                 project=OUTPUT_RUNS,
#                 name=f"exp_{ntype}_{nlevel}",
#                 verbose=True
#             )
#
#             metrics = results.results_dict
#
#             write_results([
#                 ntype, nlevel,
#                 metrics.get("metrics/mAP50(B)", 0),
#                 metrics.get("metrics/mAP50-95(B)", 0),
#                 metrics.get("metrics/precision(B)", 0),
#                 metrics.get("metrics/recall(B)", 0)
#             ])
#
#             print(f"✔ Finished: {ntype} {nlevel}%")
#
#     print("\n🎉 ALL TRAINING COMPLETE")
#     print(f"📄 Results saved → {RESULTS_CSV}")
#
#
# # ============================================================
# #                          MAIN
# # ============================================================
#
# if __name__ == "__main__":
#     train_all()















import os
import csv
from ultralytics import YOLO

# ============================================================
#                    CONFIG (EDIT THESE ONLY)
# ============================================================

# Your REAL dataset path
DATASET_BASE = r"C:\Users\Abu Hurrairah\Downloads\files\datasets\License-Plate-dataset"

# Your noisy folders path (already generated)
NOISY_BASE = r"K:\pythonProject\OfficeWork\Mam Sadaf\pythonProject1\Zeeshan\yolo_noisy_outputs"

# Where YOLO training results should be saved
OUTPUT_RUNS = r"C:\Users\Abu Hurrairah\Downloads\files\yolo_training_results"

# Your clean data.yaml
DATA_YAML = fr"{DATASET_BASE}\data.yaml"

# YOLO weights
MODEL_WEIGHTS = "yolov8n.pt"

# CSV results file
RESULTS_CSV = fr"{OUTPUT_RUNS}\yolo_noise_results.csv"


# ============================================================
#                 HELPER: PATH NORMALISATION
# ============================================================

def to_posix(path: str) -> str:
    """Return absolute path with forward slashes (YOLO-friendly)."""
    return os.path.abspath(path).replace("\\", "/")


# ============================================================
#                 PREPARE TRAIN FOLDER PATHS
# ============================================================

def get_train_dir(noise_type, noise_level):
    """
    Matches EXACTLY your folder names:
        random10, random20, ..., asymmetric10, ..., adversarial40
    """
    if noise_type == "clean":
        return fr"{DATASET_BASE}\train"

    return fr"{NOISY_BASE}\{noise_type}{noise_level}"


# ============================================================
#    Update YAML with ABSOLUTE train/val/test paths
# ============================================================

def make_temp_yaml(original_yaml, train_dir, out_yaml):
    """
    We keep class names etc from original yaml, but we overwrite:
        train:
        val:
        test:
    with ABSOLUTE paths so YOLO never gets confused.
    """
    train_abs = to_posix(train_dir)
    val_abs = to_posix(os.path.join(DATASET_BASE, "valid"))
    test_abs = to_posix(os.path.join(DATASET_BASE, "test"))

    with open(original_yaml, "r") as f:
        content = f.readlines()

    new_lines = []
    for line in content:
        stripped = line.strip()
        if stripped.startswith("train:"):
            new_lines.append(f"train: {train_abs}\n")
        elif stripped.startswith("val:"):
            new_lines.append(f"val: {val_abs}\n")
        elif stripped.startswith("test:"):
            new_lines.append(f"test: {test_abs}\n")
        else:
            new_lines.append(line)

    with open(out_yaml, "w") as f:
        f.writelines(new_lines)


# ============================================================
#                     Write CSV Results
# ============================================================

def write_results(row):
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    file_exists = os.path.exists(RESULTS_CSV)

    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["noise_type", "noise_level", "mAP50", "mAP50_95", "precision", "recall"])
        writer.writerow(row)


# ============================================================
#                MAIN YOLO TRAINING LOOP
# ============================================================

def train_all():
    print("🚀 Starting YOLO training (using existing noisy folders)...\n")
    os.makedirs(OUTPUT_RUNS, exist_ok=True)

    noise_types = ["clean", "random", "asymmetric", "adversarial"]
    noise_levels = [0, 10, 20, 40]

    for ntype in noise_types:
        for nlevel in noise_levels:

            if ntype == "clean" and nlevel != 0:
                continue
            if ntype != "clean" and nlevel == 0:
                continue

            train_folder = get_train_dir(ntype, nlevel)

            print("\n====================================================")
            print(f" Training → {ntype} {nlevel}%")
            print(f" Train folder → {train_folder}")
            print("====================================================")

            if not os.path.exists(train_folder):
                print(f"❌ SKIPPED — Folder not found: {train_folder}")
                continue

            temp_yaml = fr"{OUTPUT_RUNS}\data_temp_{ntype}_{nlevel}.yaml"
            make_temp_yaml(DATA_YAML, train_folder, temp_yaml)

            print(f"Using YAML: {temp_yaml}")

            model = YOLO(MODEL_WEIGHTS)

            results = model.train(
                data=temp_yaml,
                epochs=2,
                imgsz=640,
                batch=16,
                project=OUTPUT_RUNS,
                name=f"exp_{ntype}_{nlevel}",
                verbose=True
            )

            metrics = results.results_dict

            write_results([
                ntype, nlevel,
                float(metrics.get("metrics/mAP50(B)", 0)),
                float(metrics.get("metrics/mAP50-95(B)", 0)),
                float(metrics.get("metrics/precision(B)", 0)),
                float(metrics.get("metrics/recall(B)", 0))
            ])

            print(f"✔ Finished: {ntype} {nlevel}%")

    print("\n🎉 ALL TRAINING COMPLETE")
    print(f"📄 Results saved → {RESULTS_CSV}")


# ============================================================
#                          MAIN
# ============================================================

if __name__ == "__main__":
    train_all()





