import os
import cv2
import pytesseract
import psycopg2
from datetime import datetime
from ultralytics import YOLO
from tkinter import Tk, Label, Button, simpledialog, Entry

# Specify the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Establish a connection to your PostgreSQL database
conn = psycopg2.connect(
    host="localhost",
    database="test",
    user="postgres",
    password="2004"
)

# Keep track of previous detections
previous_detections = set()

def process_cctv(cctv_id, video_source):
    global license_plate_number, score
    model = YOLO("anpr_model.pt")
    threshold = 0.5
    class_name_dict = {
        0: 'num_plate'
    }

    # Create a VideoCapture object based on the selected video source
    video_capture = cv2.VideoCapture(video_source)

    while True:
        # Read a frame from the video feed
        ret, frame = video_capture.read()

        # Check if the frame is successfully read
        if not ret:
            print("Failed to read frame from video source.")
            break

        # Perform object detection
        results = model(frame)

        # Process the detections
        for result in results[0].boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result

            if score > threshold:
                class_name = class_name_dict.get(int(class_id))
                if class_name == 'num_plate':
                    license_plate = frame[int(y1):int(y2), int(x1):int(x2)]
                    license_plate_number = extract_license_plate_number(license_plate)

                    # Check if license plate meets the confidence level threshold and is not a duplicate
                    if score > 0.85 and license_plate_number not in previous_detections:
                        # Draw the bounding box and label on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, class_name.upper(), (int(x1), int(y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        # Get current date and time
                        current_datetime = datetime.now()

                        # Insert the detection details into the database
                        cursor = conn.cursor()
                        insert_query = """
                        INSERT INTO cctv_detections (detection_date, detection_time, cctv_id, license_number, confidence_level)
                        VALUES (%s, %s, %s, %s, %s);
                        """
                        cursor.execute(insert_query, (
                            current_datetime.date(), current_datetime.time(), cctv_id, license_plate_number, score))
                        conn.commit()
                        cursor.close()

                        # Add the license plate to the previous detections set
                        previous_detections.add(license_plate_number)

        # Display the frame with the detected license plate
        cv2.imshow('CCTV Feed', frame)

        # Check if the 'q' key is pressed to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close the display window
    video_capture.release()
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


def select_video_source():
    def select_webcam():
        process_cctv(cctv_id_entry.get(), 0)
        window.destroy()

    def select_cctv():
        cctv_ip = simpledialog.askstring("Enter CCTV IP Address", "Please enter the CCTV IP address:")
        process_cctv(cctv_id_entry.get(), f"rtsp://{cctv_ip}/live")
        window.destroy()

    # Create the Tkinter window
    window = Tk()
    window.title("Select Video Source")
    window.geometry("400x200")

    # Add label and button for webcam selection
    webcam_label = Label(window, text="Webcam")
    webcam_label.pack()
    webcam_button = Button(window, text="Select", command=select_webcam)
    webcam_button.pack()

    # Add label and button for CCTV selection
    cctv_label = Label(window, text="Custom CCTV IP")
    cctv_label.pack()
    cctv_button = Button(window, text="Select", command=select_cctv)
    cctv_button.pack()

    # Add entry for CCTV ID
    cctv_id_label = Label(window, text="Enter CCTV ID:")
    cctv_id_label.pack()
    cctv_id_entry = Entry(window)
    cctv_id_entry.pack()

    # Run the Tkinter event loop
    window.mainloop()


# Run the function to select the video source and start processing the CCTV feed
select_video_source()

# Close the database connection
conn.close()
