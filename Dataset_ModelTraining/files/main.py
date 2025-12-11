from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv
import csv
import numpy as np
from scipy.interpolate import interp1d
import ast
import pandas as pd
from tqdm import tqdm


# --------- FIXED PATHS ---------
VIDEO_PATH = r"C:\Users\Haris\Desktop\YALPR\Dataset_ModelTraining\files\videos\input.mp4"
OUTPUT_DIR = r"C:\Users\Haris\Desktop\YALPR\Dataset_ModelTraining\files\outputs"
TEST_CSV = OUTPUT_DIR + r"\test.csv"
TEST_INTERPOLATED_CSV = OUTPUT_DIR + r"\test_interpolated.csv"
OUTPUT_VIDEO = OUTPUT_DIR + r"\output.mp4"
LICENSE_MODEL_PATH = r"C:\Users\Haris\Desktop\YALPR\runs\detect\train5\weights\best.pt"
# --------------------------------


results = {}
mot_tracker = Sort()

# Load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO(LICENSE_MODEL_PATH)

# Load input video
cap = cv2.VideoCapture(VIDEO_PATH)
vehicles = [2, 3, 5, 7]

frame_nmr = -1
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Processing video", unit="frame") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        pbar.update(1)
        results[frame_nmr] = {}

        # Vehicle detection
        detections = coco_model(frame, verbose=False)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # Tracking
        track_ids = mot_tracker.update(np.asarray(detections_))

        # License plate detection
        license_plates = license_plate_detector(frame, verbose=False)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                lp_gray = cv2.cvtColor(lp_crop, cv2.COLOR_BGR2GRAY)
                _, lp_thresh = cv2.threshold(lp_gray, 64, 255, cv2.THRESH_BINARY_INV)

                lp_text, lp_score = read_license_plate(lp_thresh)

                if lp_text is not None:
                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': lp_text,
                            'bbox_score': score,
                            'text_score': lp_score
                        }
                    }


# Save initial CSV
write_csv(results, TEST_CSV)


# --------- INTERPOLATION ---------
def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    for car_id in tqdm(unique_car_ids):
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(car_id)]
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]

        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame = car_frame_numbers[0]

        for i in range(len(car_frame_numbers)):
            fn = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            lp_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_fn = car_frame_numbers[i - 1]
                prev_car = car_bboxes_interpolated[-1]
                prev_lp = license_plate_bboxes_interpolated[-1]

                if fn - prev_fn > 1:
                    gap = fn - prev_fn
                    x = np.array([prev_fn, fn])
                    x_new = np.linspace(prev_fn, fn, num=gap, endpoint=False)

                    interp_c = interp1d(x, np.vstack((prev_car, car_bbox)), axis=0)
                    interp_lp = interp1d(x, np.vstack((prev_lp, lp_bbox)), axis=0)

                    car_bboxes_interpolated.extend(interp_c(x_new)[1:])
                    license_plate_bboxes_interpolated.extend(interp_lp(x_new)[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(lp_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_nr = first_frame + i
            row = {}
            row['frame_nmr'] = str(frame_nr)
            row['car_id'] = str(car_id)
            row['car_bbox'] = ' '.join(map(str, car_bboxes_interpolated[i]))
            row['license_plate_bbox'] = ' '.join(map(str, license_plate_bboxes_interpolated[i]))

            if str(frame_nr) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                orig = [p for p in data if 
                        int(p['frame_nmr']) == frame_nr and 
                        int(float(p['car_id'])) == int(car_id)][0]
                row['license_plate_bbox_score'] = orig.get('license_plate_bbox_score', '0')
                row['license_number'] = orig.get('license_number', '0')
                row['license_number_score'] = orig.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data


# Load CSV
with open(TEST_CSV, 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Interpolate
interpolated_data = interpolate_bounding_boxes(data)

# Save new CSV
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox',
          'license_plate_bbox_score', 'license_number', 'license_number_score']

with open(TEST_INTERPOLATED_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)


# --------- VISUALIZATION / OUTPUT VIDEO ---------
results_df = pd.read_csv(TEST_INTERPOLATED_CSV)

cap = cv2.VideoCapture(VIDEO_PATH)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

license_plate = {}
unique_car_ids = np.unique(results_df['car_id'])

for car_id in tqdm(unique_car_ids, desc="Extracting best license plates"):
    max_score = np.amax(results_df[results_df['car_id'] == car_id]['license_number_score'])
    best_row = results_df[(results_df['car_id'] == car_id) &
                          (results_df['license_number_score'] == max_score)].iloc[0]

    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': best_row['license_number']
    }

    frame_nr = best_row['frame_nmr']
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nr)
    ret, frame = cap.read()

    x1, y1, x2, y2 = ast.literal_eval(
        best_row['license_plate_bbox'].replace(' ', ',')
    )

    lp_crop = frame[int(y1):int(y2), int(x1):int(x2)]
    lp_crop = cv2.resize(lp_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
    license_plate[car_id]['license_crop'] = lp_crop

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

frame_nmr = -1
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

with tqdm(total=total_frames, desc="Rendering video") as pbar:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_nmr += 1
        df_ = results_df[results_df['frame_nmr'] == frame_nmr]

        for idx in range(len(df_)):
            # CAR bounding box
            car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(
                df_.iloc[idx]['car_bbox'].replace(' ', ',')
            )

            cv2.rectangle(frame, (int(car_x1), int(car_y1)),
                          (int(car_x2), int(car_y2)), (0, 255, 0), 10)

            # LP bounding box
            x1, y1, x2, y2 = ast.literal_eval(
                df_.iloc[idx]['license_plate_bbox'].replace(' ', ',')
            )
            cv2.rectangle(frame, (int(x1), int(y1)),
                          (int(x2), int(y2)), (0, 0, 255), 10)

            # LP crop + text
            car_id = df_.iloc[idx]['car_id']
            lp_crop = license_plate[car_id]['license_crop']
            H, W, _ = lp_crop.shape

            try:
                frame[int(car_y1) - H - 100:int(car_y1) - 100,
                      int((car_x1 + car_x2 - W) / 2):int((car_x1 + car_x2 + W) / 2)] = lp_crop

                lp_text = license_plate[car_id]['license_plate_number']

                cv2.putText(frame, lp_text,
                            (int(car_x1), int(car_y1) - H - 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (255, 255, 255), 4)
            except:
                pass

        out.write(frame)
        pbar.update(1)

cap.release()
out.release()
