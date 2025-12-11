# import os
# import random
# import argparse
# from glob import glob
#
#
# # ===================== Utility functions =====================
#
# def read_yolo_label(path: str):
#     """Read all YOLO label lines from a file."""
#     if not os.path.exists(path):
#         return []
#
#     with open(path, "r") as f:
#         lines = [l.strip() for l in f.readlines() if l.strip()]
#     return lines
#
#
# def write_yolo_label(path: str, lines):
#     """Write YOLO label lines to a file."""
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w") as f:
#         if lines:
#             f.write("\n".join(lines))
#
#
# def random_class_flip(class_id: int, max_classes: int = 3) -> int:
#     """
#     Randomly flip class_id to another id in [0, max_classes-1].
#     If you only have 1 class, set max_classes=2 to create a fake wrong class.
#     """
#     if max_classes <= 1:
#         return class_id
#
#     new_id = class_id
#     while new_id == class_id:
#         new_id = random.randint(0, max_classes - 1)
#     return new_id
#
#
# def random_bbox_noise(x, y, w, h, noise_level=0.1):
#     """
#     Add small random noise (±noise_level) to bbox parameters.
#     All values stay in [0, 1].
#     """
#     nx = x + random.uniform(-noise_level, noise_level)
#     ny = y + random.uniform(-noise_level, noise_level)
#     nw = w + random.uniform(-noise_level, noise_level)
#     nh = h + random.uniform(-noise_level, noise_level)
#
#     nx = max(0.0, min(1.0, nx))
#     ny = max(0.0, min(1.0, ny))
#     nw = max(0.01, min(1.0, nw))
#     nh = max(0.01, min(1.0, nh))
#
#     return nx, ny, nw, nh
#
#
# def asymmetric_bbox_noise(x, y, w, h, noise_level=0.2):
#     """
#     Asymmetric noise: push bbox more in one direction
#     (e.g., to the right / bottom).
#     """
#     nx = x + random.uniform(0.0, noise_level)
#     ny = y + random.uniform(0.0, noise_level)
#     nw = w + random.uniform(-noise_level, noise_level)
#     nh = h + random.uniform(-noise_level, noise_level)
#
#     nx = max(0.0, min(1.0, nx))
#     ny = max(0.0, min(1.0, ny))
#     nw = max(0.01, min(1.0, nw))
#     nh = max(0.01, min(1.0, nh))
#
#     return nx, ny, nw, nh
#
#
# def adversarial_bbox_noise():
#     """
#     Adversarial: a very bad bbox, far from correct region.
#     """
#     nx = random.uniform(0.0, 1.0)
#     ny = random.uniform(0.0, 1.0)
#     nw = random.uniform(0.4, 1.0)
#     nh = random.uniform(0.4, 1.0)
#     return nx, ny, nw, nh
#
#
# # ===================== Main injection logic =====================
#
# def inject_noise_to_labels(
#     label_folder: str,
#     output_folder: str,
#     noise_rate: float,
#     noise_type: str,
#     max_classes: int = 2,
#     seed: int = 42,
# ):
#     """
#     label_folder: original labels (train/labels)
#     output_folder: where to save noisy labels
#     noise_rate: e.g. 0.1 = 10% of files will be corrupted
#     noise_type: 'random', 'asymmetric', 'adversarial'
#     max_classes: total number of classes (for random flipping)
#     """
#     random.seed(seed)
#
#     os.makedirs(output_folder, exist_ok=True)
#
#     label_paths = sorted(glob(os.path.join(label_folder, "*.txt")))
#     total_files = len(label_paths)
#     noisy_count = int(round(total_files * noise_rate))
#
#     noisy_files = set(random.sample(label_paths, noisy_count))
#     print(f"Found {total_files} label files")
#     print(f"Noise type: {noise_type}, rate: {noise_rate*100:.1f}% "
#           f"→ {noisy_count} files will be corrupted")
#
#     for path in label_paths:
#         lines = read_yolo_label(path)
#         new_lines = []
#
#         apply_noise = path in noisy_files
#
#         for line in lines:
#             parts = line.split()
#             if len(parts) != 5:
#                 # skip malformed line
#                 new_lines.append(line)
#                 continue
#
#             class_id = int(parts[0])
#             x, y, w, h = map(float, parts[1:])
#
#             if apply_noise:
#                 if noise_type == "random":
#                     # flip class + small bbox noise
#                     class_id = random_class_flip(class_id, max_classes=max_classes)
#                     x, y, w, h = random_bbox_noise(x, y, w, h, noise_level=0.10)
#
#                 elif noise_type == "asymmetric":
#                     # keep class, only distort bbox asymmetrically
#                     x, y, w, h = asymmetric_bbox_noise(x, y, w, h, noise_level=0.20)
#
#                 elif noise_type == "adversarial":
#                     # big bbox corruption + sometimes class flip
#                     if random.random() < 0.5:
#                         class_id = random_class_flip(class_id, max_classes=max_classes)
#                     x, y, w, h = adversarial_bbox_noise()
#
#                 else:
#                     raise ValueError(f"Unknown noise_type: {noise_type}")
#
#             new_line = f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
#             new_lines.append(new_line)
#
#         out_path = os.path.join(output_folder, os.path.basename(path))
#         write_yolo_label(out_path, new_lines)
#
#     print(f"Done. Noisy labels saved in: {output_folder}")
#
#
# # ===================== CLI =====================
#
# def main():
#     parser = argparse.ArgumentParser(description="Inject label noise into YOLO detection labels")
#     parser.add_argument("--label_dir", type=str, required=True,
#                         help="Path to clean YOLO labels folder (e.g., .../train/labels)")
#     parser.add_argument("--output_dir", type=str, required=True,
#                         help="Output folder for noisy labels")
#     parser.add_argument("--noise_rate", type=float, default=0.1,
#                         help="Fraction of label files to corrupt (0.1 = 10%)")
#     parser.add_argument("--noise_type", type=str,
#                         choices=["random", "asymmetric", "adversarial"],
#                         default="random")
#     parser.add_argument("--max_classes", type=int, default=2,
#                         help="Total number of classes in YOLO model")
#     parser.add_argument("--seed", type=int, default=42,
#                         help="Random seed")
#
#     args = parser.parse_args()
#
#     inject_noise_to_labels(
#         label_folder=args.label_dir,
#         output_folder=args.output_dir,
#         noise_rate=args.noise_rate,
#         noise_type=args.noise_type,
#         max_classes=args.max_classes,
#         seed=args.seed,
#     )
#
#
# if __name__ == "__main__":
#     main()

















# import os
# import random
# from glob import glob
#
#
# # =====================================================
# #              Utility Functions
# # =====================================================
#
# def read_yolo_label(path):
#     """Reads YOLO label file (.txt)."""
#     if not os.path.exists(path):
#         return []
#     with open(path, "r") as f:
#         return [l.strip() for l in f.readlines() if l.strip()]
#
#
# def write_yolo_label(path, lines):
#     """Writes YOLO label file."""
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     with open(path, "w") as f:
#         f.write("\n".join(lines))
#
#
# def random_class_flip(class_id: int, max_classes: int = 2) -> int:
#     """Randomly flips a class ID."""
#     new_id = class_id
#     while new_id == class_id:
#         new_id = random.randint(0, max_classes - 1)
#     return new_id
#
#
# def random_bbox_noise(x, y, w, h, noise=0.1):
#     """Applies soft noise to bounding box."""
#     nx = x + random.uniform(-noise, noise)
#     ny = y + random.uniform(-noise, noise)
#     nw = w + random.uniform(-noise, noise)
#     nh = h + random.uniform(-noise, noise)
#
#     nx = max(0.0, min(1.0, nx))
#     ny = max(0.0, min(1.0, ny))
#     nw = max(0.01, min(1.0, nw))
#     nh = max(0.01, min(1.0, nh))
#
#     return nx, ny, nw, nh
#
#
# def asymmetric_bbox_noise(x, y, w, h, noise=0.2):
#     """Applies stronger asymmetric noise (biased)."""
#     nx = x + random.uniform(0, noise)
#     ny = y + random.uniform(0, noise)
#     nw = w + random.uniform(-noise, noise)
#     nh = h + random.uniform(-noise, noise)
#
#     nx = max(0.0, min(1.0, nx))
#     ny = max(0.0, min(1.0, ny))
#     nw = max(0.01, min(1.0, nw))
#     nh = max(0.01, min(1.0, nh))
#
#     return nx, ny, nw, nh
#
#
# def adversarial_bbox_noise():
#     """Very bad bbox (extreme corruption)."""
#     nx = random.uniform(0.0, 1.0)
#     ny = random.uniform(0.0, 1.0)
#     nw = random.uniform(0.4, 1.0)
#     nh = random.uniform(0.4, 1.0)
#     return nx, ny, nw, nh
#
#
# # =====================================================
# #              Core Noise Injection
# # =====================================================
#
# def inject_noise(label_folder, output_folder, noise_rate, noise_type, max_classes=2, seed=42):
#     random.seed(seed)
#
#     label_files = glob(os.path.join(label_folder, "*.txt"))
#     total = len(label_files)
#     noisy_count = int(total * noise_rate)
#
#     noisy_selected = set(random.sample(label_files, noisy_count))
#
#     print(f"\n[{noise_type.upper()} — {noise_rate*100:.0f}%]  Corrupting {noisy_count}/{total} files")
#
#     for file in label_files:
#         lines = read_yolo_label(file)
#         new_lines = []
#         apply_noise = file in noisy_selected
#
#         for line in lines:
#             parts = line.split()
#             if len(parts) != 5:
#                 new_lines.append(line)
#                 continue
#
#             cls = int(parts[0])
#             x, y, w, h = map(float, parts[1:])
#
#             if apply_noise:
#                 if noise_type == "random":
#                     cls = random_class_flip(cls, max_classes)
#                     x, y, w, h = random_bbox_noise(x, y, w, h, noise=0.10)
#
#                 elif noise_type == "asymmetric":
#                     x, y, w, h = asymmetric_bbox_noise(x, y, w, h, noise=0.20)
#
#                 elif noise_type == "adversarial":
#                     cls = random_class_flip(cls, max_classes)
#                     x, y, w, h = adversarial_bbox_noise()
#
#             new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
#
#         out = os.path.join(output_folder, os.path.basename(file))
#         write_yolo_label(out, new_lines)
#
#     print(f"Saved noisy labels to: {output_folder}")
#
#
# # =====================================================
# #              AUTO-GENERATE ALL NOISE
# # =====================================================
#
# def generate_all_noise():
#     base = "datasets/License-Plate-dataset/train/labels"
#
#     # noise types + levels
#     noise_types = ["random", "asymmetric", "adversarial"]
#     noise_levels = [0.10, 0.20, 0.40]  # 10%, 20%, 40%
#
#     for ntype in noise_types:
#         for nlevel in noise_levels:
#             folder_name = f"datasets/License-Plate-dataset/train_noisy_{ntype}{int(nlevel*100)}/labels"
#             inject_noise(
#                 label_folder=base,
#                 output_folder=folder_name,
#                 noise_rate=nlevel,
#                 noise_type=ntype,
#                 max_classes=2
#             )
#
#
# # =====================================================
# #                       MAIN
# # =====================================================
#
# if __name__ == "__main__":
#     print("⚙️  Generating ALL YOLO noise versions...")
#     generate_all_noise()
#     print("\n✅ DONE! All noisy datasets generated successfully.")
#



















import os
import random
from glob import glob


# =====================================================
#              CONFIG — EDIT THESE ONLY
# =====================================================

# Your REAL dataset labels path
BASE_LABEL_FOLDER = r"C:\Users\Abu Hurrairah\Downloads\files\datasets\License-Plate-dataset\train\labels"

# Where you want noisy datasets to be created
OUTPUT_BASE_FOLDER = r"K:\pythonProject\OfficeWork\Mam Sadaf\pythonProject1\Zeeshan\yolo_noisy_outputs"


# =====================================================
#              Utility Functions
# =====================================================

def read_yolo_label(path):
    """Reads YOLO label file (.txt)."""
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def write_yolo_label(path, lines):
    """Writes YOLO label file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def random_class_flip(class_id: int, max_classes: int = 2) -> int:
    """Randomly flips a class ID."""
    new_id = class_id
    while new_id == class_id:
        new_id = random.randint(0, max_classes - 1)
    return new_id


def random_bbox_noise(x, y, w, h, noise=0.1):
    """Applies soft noise to bounding box."""
    nx = x + random.uniform(-noise, noise)
    ny = y + random.uniform(-noise, noise)
    nw = w + random.uniform(-noise, noise)
    nh = h + random.uniform(-noise, noise)

    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    nw = max(0.01, min(1.0, nw))
    nh = max(0.01, min(1.0, nh))

    return nx, ny, nw, nh


def asymmetric_bbox_noise(x, y, w, h, noise=0.2):
    """Applies stronger asymmetric noise (biased)."""
    nx = x + random.uniform(0, noise)
    ny = y + random.uniform(0, noise)
    nw = w + random.uniform(-noise, noise)
    nh = h + random.uniform(-noise, noise)

    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    nw = max(0.01, min(1.0, nw))
    nh = max(0.01, min(1.0, nh))

    return nx, ny, nw, nh


def adversarial_bbox_noise():
    """Very bad bbox (extreme corruption)."""
    nx = random.uniform(0.0, 1.0)
    ny = random.uniform(0.0, 1.0)
    nw = random.uniform(0.4, 1.0)
    nh = random.uniform(0.4, 1.0)
    return nx, ny, nw, nh


# =====================================================
#              Core Noise Injection
# =====================================================

def inject_noise(label_folder, output_folder, noise_rate, noise_type, max_classes=2, seed=42):
    random.seed(seed)

    label_files = glob(os.path.join(label_folder, "*.txt"))
    total = len(label_files)
    noisy_count = int(total * noise_rate)

    noisy_selected = set(random.sample(label_files, noisy_count))

    print(f"\n[{noise_type.upper()} — {noise_rate*100:.0f}%] Corrupting {noisy_count}/{total} files")

    for file in label_files:
        lines = read_yolo_label(file)
        new_lines = []
        apply_noise = file in noisy_selected

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                new_lines.append(line)
                continue

            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if apply_noise:
                if noise_type == "random":
                    cls = random_class_flip(cls, max_classes)
                    x, y, w, h = random_bbox_noise(x, y, w, h, noise=0.10)

                elif noise_type == "asymmetric":
                    x, y, w, h = asymmetric_bbox_noise(x, y, w, h, noise=0.20)

                elif noise_type == "adversarial":
                    cls = random_class_flip(cls, max_classes)
                    x, y, w, h = adversarial_bbox_noise()

            new_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        out = os.path.join(output_folder, os.path.basename(file))
        write_yolo_label(out, new_lines)

    print(f"Saved noisy labels to: {output_folder}")


# =====================================================
#              AUTO-GENERATE ALL NOISE
# =====================================================

def generate_all_noise():
    noise_types = ["random", "asymmetric", "adversarial"]
    noise_levels = [0.10, 0.20, 0.40]  # 10%, 20%, 40%

    for ntype in noise_types:
        for nlevel in noise_levels:
            out_folder = os.path.join(
                OUTPUT_BASE_FOLDER,
                f"{ntype}{int(nlevel*100)}",
                "labels"
            )

            inject_noise(
                label_folder=BASE_LABEL_FOLDER,
                output_folder=out_folder,
                noise_rate=nlevel,
                noise_type=ntype,
                max_classes=2
            )


# =====================================================
#               MAIN
# =====================================================

if __name__ == "__main__":
    print("⚙️ Generating ALL YOLO noise versions…")
    generate_all_noise()
    print("\n✅ DONE! All noisy datasets created successfully.")


