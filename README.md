# VideoSnapStitcher

VideoSnapStitcher is a GUI tool designed to analyze timestamps within video frames using OCR (Optical Character Recognition) and extract/stitch specific scenes from multiple videos based on designated times. It is particularly useful for efficiently editing footage where **dates and times are burned into the video**, such as surveillance camera recordings or dashcam footage.

## 🚀 Features
- **OCR-based Automatic Analysis**: Automatically identifies the exact time range of a video by extracting timestamps from the first and last frames using OCR.
- **Dynamic ROI (Region of Interest) Selection**: Intuitively specify the area for OCR scanning by dragging a bounding box on the preview screen with your mouse.
- **Advanced Image Preprocessing**: Includes grayscale conversion, binarization, contrast adjustment, and zoom functions to improve OCR accuracy.
- **Flexible Path Resolution**: Automatically searches for video files in hierarchical structures (e.g., `YYYY/MM/DD/HH/mm.mp4`) that match the specified timestamps.
- **Built-in Test Video Maker**: Includes a helper tool to generate "dummy videos" with burned-in date/time text for testing purposes.

---

## 🛠 Installation

### System Requirements
1.  **Python 3.x** or higher
2.  **Tesseract OCR Engine**: Please install it from the [official website](https://github.com/UB-Mannheim/tesseract/wiki).
    *   On Windows, it is typically installed in `C:\Program Files\Tesseract-OCR\tesseract.exe`.

### Library Installation
Run the following command to install the required dependencies:

```bash
pip install PySide6 opencv-python numpy pytesseract
```

---

## 📂 File Structure
- `VideoSnapStitcher.py`: The main tool for video analysis and stitching.
- `TestVideoMaker.py`: A helper tool to create test videos with date/time text.

---

## 📖 Usage

### 1. Create Test Data (Optional)
First, run `TestVideoMaker.py` to generate videos for testing.
- Set the start/end dates, FPS, resolution, etc., and click "Start Process" to export a dummy video with date text into your specified directory.

### 2. Using the Main Tool (VideoSnapStitcher)
Run `VideoSnapStitcher.py`.

#### Step A: Initial Setup
1.  **Parent Path**: Select the root folder where your video files are stored.
2.  **Output Path**: Select the location and filename for the finished video.
3.  **Path Format**: Enter the input file path format (e.g., `YYYY/MM/DD/HH/mm.mp4`).

#### Step B: OCR Optimization (Settings Tab)
1.  Load a source video into the preview window.
2.  Drag and resize the **ROI (Green Box)** to enclose the area where the timestamp is displayed.
3.  Go to the "OCR & Path Settings" tab and adjust contrast, zoom, etc., until the OCR correctly recognizes the text in the preview.

#### Step C: Execution of Stitching Process
1.  **Timestamp List**: Enter the timestamps you wish to extract, one per line (e.g., `2026/09/28 14:30:05`).
2.  Click "Start Process." The tool will automatically perform the following:
    - Search for video files corresponding to each timestamp.
    - Extract start and end times via OCR from those videos to calculate total duration.
    - Identify the exact frames matching your requested timestamps.
    - Stitch all scenes together into a single MP4 file.

---

## ⚙️ Configuration Details
| Item | Description |
| :--- | :--- |
| **ROI X, Y, W, H** | Coordinates and size of the area for OCR execution |
| **Contrast / Zoom** | Settings to enhance images for better OCR accuracy |
| **Binarize** | Converts image to black and white (for OCR) |
| **Path Format** | Template used to identify files within a folder structure |

---

## ⚠️ Notes
- If the Tesseract path is not correctly configured, the tool will return an error. Please verify `tesseract_path` in the Settings tab.
- When videos have significantly different resolutions or FPS, there may be visual inconsistencies in the final merged video (the output is generally resized to match the source).
