# Automatic Number Plate Recognition using YOLOv8

## Description

The Automatic Number Plate Recognition (ANPR) system implemented in this project uses the state-of-the-art YOLOv8 (You Only Look Once version 8) deep learning model for real-time and accurate detection and recognition of vehicle number plates in images and video streams. The YOLOv8 model, built on the YOLO (You Only Look Once) architecture, is known for its speed and precision, making it an ideal choice for ANPR applications.

The ANPR system processes images or video frames, identifies and localizes license plates, and then extracts the alphanumeric characters from the plates. It's designed to handle different license plate formats, including various fonts, colors, and sizes. The project provides an easy-to-use command-line interface and can be easily integrated into existing applications or used as a standalone tool.

## Main Features

- **Real-Time ANPR:** Fast and efficient detection and recognition of number plates in real-time video streams.
- **Accurate Localization:** Precisely locates the position of number plates within images or video frames.
- **Alphanumeric Extraction:** Extracts the alphanumeric characters from the license plates for further processing.
- **Customization:** Users can fine-tune the ANPR system using their datasets to adapt it to specific regions or license plate formats.
- **Minimal Dependencies:** The project uses lightweight libraries to ensure easy setup and deployment.
- **CLI Interface:** Provides a user-friendly command-line interface for convenient usage.

Whether you're building a traffic monitoring system, a parking management application, or conducting research in computer vision, this Automatic Number Plate Recognition using YOLOv8 offers a powerful and flexible solution.

## Installation

Instructions on how to install and set up the ANPR system can be found in the [Installation Guide](link/to/installation/guide).Installing YOLOv8 and setting up Tesseract for text extraction can involve multiple steps. Below is a step-by-step installation guide for YOLOv8 using a custom dataset and integrating Tesseract for text extraction.

Step 1: Clone the YOLOv8 Repository

1. Open a terminal or command prompt.
2. Navigate to the directory where you want to install YOLOv8.
3. Clone the YOLOv8 repository from GitHub: git clone https://github.com/AlexeyAB/darknet.git

Step 2: Configure YOLOv8 for Custom Dataset

1. Go into the darknet directory: cd darknet
2. Make sure you have CUDA and cuDNN installed if you want to use GPU acceleration for training and inference. If you don't have a GPU, you can still use YOLOv8, but it will be slower.
3. Modify the Makefile to enable GPU and other configurations. Open the Makefile using a text editor.
4. Change the GPU, CUDNN, and OPENCV flags to 1 to enable GPU, cuDNN, and OpenCV support, respectively.
5. Adjust other settings in the Makefile according to your system (e.g., set ARCH based on your GPU architecture).
6. Compile YOLOv8: make

Step 3: Prepare Your Custom Dataset

1. Organize your custom dataset in the following format:
/path/to/dataset/
├── images/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── labels/
    ├── image1.txt
    ├── image2.txt
    └── ...
Each image in the images folder should have a corresponding .txt file in the labels folder. The .txt file should contain one line per object detected in the image, with each line in the format: class x_center y_center width height.
2. Create a .names file that contains the names of the classes in your custom dataset, one per line.
3. Create a .data file that specifies the configuration for your custom dataset:
classes = {number of classes}
train = /path/to/dataset/train.txt
valid = /path/to/dataset/valid.txt
names = /path/to/your.names
backup = /path/to/save/weights
Replace {number of classes} with the actual number of classes in your dataset.
4. Prepare train.txt and valid.txt, which contain the paths to the training and validation images, respectively. Each line in these files should be the path to an image, e.g., '/path/to/dataset/images/image1.jpg'.

Step 4: Train YOLOv8 on Your Custom Dataset

1. Download pre-trained weights (e.g., yolov4.conv.137) from the YOLO website or use the darknet53.conv.74 provided with the YOLOv8 repository.
2. Start training: ./darknet detector train /path/to/your.data /path/to/yolov4.cfg /path/to/pre-trained-weights -map

Step 5: Integrating Tesseract for Text Extraction

Install Tesseract OCR:

For Ubuntu: sudo apt-get install tesseract-ocr
For macOS: brew install tesseract
Install the Tesseract Python wrapper (tesserocr): pip install tesserocr

Step 6: Using YOLOv8 with Tesseract for Text Extraction

1. After YOLOv8 detects license plates in an image, crop the license plate region.
2. Save the cropped license plate region as an image file.
3. Use Tesseract to perform text extraction on the saved image file:
import tesserocr
from PIL import Image

# Load the cropped license plate region
plate_image = Image.open("path/to/cropped_plate_image.jpg")

# Perform text extraction using Tesseract
extracted_text = tesserocr.image_to_text(plate_image)

print(extracted_text)

Remember that this is just a general guide, and depending on your specific use case and environment, some details might differ. Always refer to the official documentation of YOLOv8 and Tesseract for more information and troubleshooting.

## Usage

To get started with the ANPR system, please refer to the [Usage Guide](link/to/usage/guide).

## Screenshots

![ANPR in Action](images)

## Contact

If you have any questions or suggestions, feel free to reach out to me at [mukeshsundar2362004@gmail.com](mailto:mukeshsundar2362004@gmail.com).
