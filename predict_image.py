import os
import cv2
import pytesseract
import psycopg2
from datetime import datetime
from ultralytics import YOLO
from tkinter import Tk, filedialog

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Establish a connection to your PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="test",
    user="postgres",
    password="2004"
)


def browse_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


def process_image(image_path):
    global license_plate_number, score
    threshold = 0.5
    class_name_dict = {
        0: 'num_plate'
    }

    image = cv2.imread(image_path)
    model = YOLO("anpr_model.pt")
    results = model(image)[0]

    license_plate = None

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            class_name = class_name_dict.get(int(class_id))
            if class_name == 'num_plate':
                license_plate = image[int(y1):int(y2), int(x1):int(x2)]
                license_plate_number = extract_license_plate_number(license_plate)
            else:
                vehicle_type = class_name

            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(image, class_name.upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Image', image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

    # Get current date and time
    current_datetime = datetime.now()
    current_time = current_datetime.strftime("%I:%M %p")

    # Insert the detection details into the database
    cursor = conn.cursor()
    insert_query = """
    INSERT INTO detections (detection_date, detection_time, file_name, license_number, confidence_level)
    VALUES (%s, %s, %s, %s, %s);
    """
    cursor.execute(insert_query, (
    current_datetime.date(), current_datetime.time(), os.path.basename(image_path), license_plate_number, score))
    conn.commit()
    cursor.close()

    return license_plate_number


def extract_license_plate_number(license_plate):
    gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
    # Apply additional preprocessing techniques to enhance the license plate image
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    ret, threshold = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Configure Tesseract parameters for license plate OCR
    custom_config = r'--oem 3 --psm 7 -l eng'
    license_plate_number = pytesseract.image_to_string(threshold, config=custom_config)
    license_plate_number = ''.join(e for e in license_plate_number if e.isalnum())

    return license_plate_number


# Prompt the user to browse for the input file
image_path = browse_file()

# Process the selected image
if image_path:
    license_plate_number = process_image(image_path)

    print("\nLicense Plate Number:", license_plate_number)
    print("\nDetected data is successfully uploaded to the Database")
