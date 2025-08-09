A mini computer vision project that detects and extracts objects from images using classical image processing techniques — thresholding, HSV color masking, and watershed segmentation.
This tool works entirely on OpenCV + NumPy (no deep learning), making it fast and ideal for embedded systems like Raspberry Pi or Jetson Nano.

🛠 Features
Two Processing Modes:

1.Threshold → Binary thresholding on grayscale images

2.Color → HSV color masking + watershed segmentation

Target Colors: Red, Green, Blue (HSV range based)

Watershed Segmentation: Separates touching/overlapping objects

ROI Extraction: Saves each detected object as an individual image

Area Filtering: Ignores noise using MIN_AREA

Bounding Boxes & Labels: Object ID + area in pixels



How to run:

# Clone repository
git clone https://github.com/Phoenixx-cv1/ClassicSegmenter.git
cd ClassicSegmenter

# Run script
python OpenSegClassic.py

Edit the script to change MODE → "threshold" or "color"
TARGET_COLOR → "red", "green", "blue"
IMG_PATH → Path to your image
