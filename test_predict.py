import cv2
import torch
import re
import pandas as pd
from paddleocr import PaddleOCR
from pathlib import Path

# Setup device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load models
def load_model(model_name, weights_path, device, conf=0.75, iou=0.1, max_det=1):
    model = torch.hub.load('ultralytics/yolov5', model_name, path=weights_path, device=device)
    model.conf = conf  # NMS confidence threshold
    model.iou = iou  # NMS IoU threshold
    model.max_det = max_det  # maximum number of detections per image
    return model

# Adjust paths according to your setup
first_model_path = 'models/yolov5_lcd.pt'
second_model_path = 'models/yolov5_odometer.pt'

first_model = load_model('custom', first_model_path, device)
second_model = load_model('custom', second_model_path, device)

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=torch.cuda.is_available())

def perform_ocr(image):
    # Ensure image is in RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ocr_results = ocr_model.ocr(rgb_image, det=False, cls=False)
    text, confidence = ocr_results[0][0]
    return text, confidence

def process_and_visualize(image_path):
    # Read the image
    image = cv2.imread(str(image_path))
    
    # First model inference: LCD Detection
    results = first_model(image)
    detections = results.xyxy[0].cpu().numpy()

    data_for_csv = []

    for det in detections:
        x1, y1, x2, y2, _, conf = map(int, det)
        cropped_image = image[y1:y2, x1:x2]

        # Draw a blue rectangle around the detected LCD
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color

        # Second model inference on the LCD Crop: Reading Detection
        second_results = second_model(cropped_image)
        second_detections = second_results.xyxy[0].cpu().numpy()

        for sec_det in second_detections:
            x1_s, y1_s, x2_s, y2_s, _, conf_s = map(int, sec_det)
            second_crop = cropped_image[y1_s:y2_s, x1_s:x2_s]

            # Draw a green rectangle around the detected reading
            cv2.rectangle(image, (x1 + x1_s, y1 + y1_s), (x1 + x2_s, y1 + y2_s), (0, 255, 0), 2)  # Green color

            # Perform OCR on the second crop
            text, confidence = perform_ocr(second_crop)
            numbers_text = ''.join(re.findall(r'\d+', text))
            print(f'{image_path.name}, \t Odometer reading: {numbers_text} kms, \t Confidence: {confidence}')
            data_for_csv.append((image_path.name, numbers_text))

            # Put the detected odometer reading as text on the image
            cv2.putText(image, f'{numbers_text} kms', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the predicted image
    predicted_image_path = str(image_path.parent / f"predicted_{image_path.name}")
    cv2.imwrite(predicted_image_path, image)

    return data_for_csv

def main(image_dir_path, csv_file_path):
    # Directories
    image_dir = Path(image_dir_path)
    all_data = []

    for image_path in image_dir.glob('*.*'):
        if image_path.is_file() and image_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            data_for_csv = process_and_visualize(image_path)
            all_data.extend(data_for_csv)
    
    # Create DataFrame and save to CSV using pandas
    df = pd.DataFrame(all_data, columns=['filename', 'odometer_reading_prediction'])
    df.to_csv(csv_file_path, index=False)

    print("Inference completed, CSV file saved, and predicted images generated.")

if __name__ == "__main__":
    image_dir_path = "data/sample_test_images"
    csv_file_path = 'sample_test_images_results.csv'
    main(image_dir_path, csv_file_path)
