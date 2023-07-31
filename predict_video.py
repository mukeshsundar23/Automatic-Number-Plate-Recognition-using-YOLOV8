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


def process_video(video_path, video_id):
    global license_plate_number, score
    model = YOLO("anpr_model.pt")
    threshold = 0.5
    class_name_dict = {
        0: 'num_plate'
    }

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    video_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    video_frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to save the annotated video
    video_output_path = os.path.splitext(video_path)[0] + "_annotated.mp4"
    video_writer = cv2.VideoWriter(video_output_path,
                                   cv2.VideoWriter_fourcc(*"mp4v"),
                                   video_fps, (video_width, video_height))

    frame_count = 0

    while True:
        # Read the next frame from the video
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_count += 1
        print(f"Processing frame {frame_count}/{video_frame_count}")

        results = model(frame)[0]

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                class_name = class_name_dict.get(int(class_id))
                if class_name == 'num_plate':
                    license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    license_plate_number = extract_license_plate_number(license_plate)

                    # Insert the detection details into the database
                    cursor = conn.cursor()
                    insert_query = """
                    INSERT INTO video_detections (video_id, detection_date, detection_time, frame_number, file_name, license_number, confidence_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s);
                    """
                    cursor.execute(insert_query, (
                        video_id, datetime.now().date(), datetime.now().time(),
                        frame_count, os.path.basename(video_path), license_plate_number, score))
                    conn.commit()
                    cursor.close()

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame, class_name.upper(), (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        # Write the annotated frame to the video writer
        video_writer.write(frame)

        # Display the annotated frame
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and writer objects
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()


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


# Prompt the user to browse for the input video file
video_path = browse_file()

# Prompt the user to enter the video ID
video_id = int(input("Enter the video ID: "))

# Process the selected video
if video_path:
    process_video(video_path, video_id)

    print("\nDetected data is successfully uploaded to the Database")
