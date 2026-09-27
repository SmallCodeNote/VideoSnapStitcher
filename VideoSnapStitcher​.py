import sys
import os
import json
import cv2
import numpy as np
import pytesseract
import traceback
from datetime import datetime, timedelta
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QLabel, QPushButton, QLineEdit, QFileDialog, QTabWidget, 
                                QTextEdit, QSpinBox, QFormLayout, QGroupBox)
from PySide6.QtCore import Qt, QThread, Signal, QSize, QPoint, QRect
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen
from time import monotonic

# --- Settings Management Class ---
class SettingsManager:
    def __init__(self):
        self.default_settings = {
            "path_format": "YYYY/MM/DD/HH/mm.mp4",
            "tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            "roi_x": 100, "roi_y": 100, "roi_w": 200, "roi_h": 50,
            "padding_top": 10, "padding_bottom": 10, "padding_left": 10, "padding_right": 10,
            "gray_scale": True, "binarize": True, "contrast": 1.5, "zoom": 2.0,
            "fps": 30,
            "parent_path": "",
            "output_path": "",
            "timestamp_list": "",
            "ocr_preview_interval_ms": 500,
        }
        self.settings = self.default_settings.copy()

    def load(self):
        file_path = f"{os.path.splitext(sys.argv[0])[0]}.json"
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.settings.update(loaded)

    def save(self, file_path=None):
        if not file_path:
            file_path = f"{os.path.splitext(sys.argv[0])[0]}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, indent=4)

# --- Video Processing Worker Class ---
class VideoWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    preview_update = Signal(np.ndarray, str)
    ocr_preview_update = Signal(np.ndarray, str)
    finished = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._is_running = True
        if "tesseract_path" in self.config:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]

        # OCR preview update control
        self.ocr_preview_interval_ms = int(self.config.get("ocr_preview_interval_ms", 500))
        self._last_ocr_emit_time = 0.0
        self._last_ocr_roi = None
        self._last_ocr_text = ""

    def emit_latest_ocr(self, force=False):
        if self._last_ocr_roi is None:
            return
        now = monotonic()
        interval_sec = self.ocr_preview_interval_ms / 1000.0
        if force or (now - self._last_ocr_emit_time >= interval_sec):
            self.ocr_preview_update.emit(self._last_ocr_roi.copy(), self._last_ocr_text)
            self._last_ocr_emit_time = now

    def resolve_video_path(self, parent_path, ts_str):
        dt = None
        formats = ["%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]

        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                break
            except ValueError:
                continue

        if dt is None:
            return None, None

        path_fmt = self.config.get("path_format", "YYYY/MM/DD/HH/mm.mp4")

        target_rel_path = (
            path_fmt
            .replace("YYYYMMDD", dt.strftime("%Y%m%d"))
            .replace("YYYYMM", dt.strftime("%Y%m"))
            .replace("YYYY", dt.strftime("%Y"))
            .replace("HH", dt.strftime("%H"))
            .replace("mm", dt.strftime("%M"))
        )

        full_path = os.path.join(parent_path, target_rel_path).replace("\\", "/")
        search_dt = dt

        for _ in range(30):
            if os.path.exists(full_path):
                return full_path, full_path

            search_dt -= timedelta(minutes=1)
            target_rel_path = (
                path_fmt
                .replace("YYYYMMDD", search_dt.strftime("%Y%m%d"))
                .replace("YYYYMM", search_dt.strftime("%Y%m"))
                .replace("YYYY", search_dt.strftime("%Y"))
                .replace("HH", search_dt.strftime("%H"))
                .replace("mm", search_dt.strftime("%M"))
            )
            full_path = os.path.join(parent_path, target_rel_path).replace("\\", "/")

        return None, full_path

    def get_video_metadata(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            return None

        ret1, frame1 = cap.read()
        if not ret1:
            cap.release()
            return None

        start_dt = self.extract_datetime_from_ocr(frame1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret2, frame2 = cap.read()

        if ret2:
            end_dt = self.extract_datetime_from_ocr(frame2)
        else:
            end_dt = start_dt

        cap.release()
        return {
            "start_dt": start_dt,
            "end_dt": end_dt,
            "total_frames": total_frames
        }

    def preprocess_image(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim < 2:
            return np.zeros((100, 100), dtype=np.uint8)

        img_h, img_w = frame.shape[:2]
        x, y, w, h = self.config['roi_x'], self.config['roi_y'], self.config['roi_w'], self.config['roi_h']
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            roi = frame[max(0, y1):min(img_h, y1+50), max(0, x1):min(img_w, x1+50)]

        if self.config['gray_scale'] and len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        alpha = self.config['contrast']
        beta = 128
        roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)

        if self.config['binarize'] and roi.ndim == 2:
            _, roi = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        pt = [self.config['padding_top'], self.config['padding_bottom'], 
               self.config['padding_left'], self.config['padding_right']]
        
        if roi.ndim == 2:
            roi = np.pad(roi, ((pt[0], pt[1]), (pt[2], pt[3])), mode='constant', constant_values=0)
        else:
            roi = np.pad(roi, (pt[0], pt[1]), axis=0, mode='constant')
            roi = np.pad(roi, (pt[2], pt[3]), axis=1, mode='constant')

        zoom = int(self.config['zoom'])
        if zoom > 1:
            h_new, w_new = roi.shape[0] * zoom, roi.shape[1] * zoom
            roi = cv2.resize(roi, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
        return roi

    def extract_datetime_from_ocr(self, frame):
        roi = self.preprocess_image(frame)
        text = pytesseract.image_to_string(roi).strip()
        
        # Store latest OCR results & limit GUI updates
        self._last_ocr_roi = roi.copy()
        self._last_ocr_text = text
        self.emit_latest_ocr()

        if not text:
            return datetime.now()

        formats = ["%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]
        for fmt in formats:
            try:
                return datetime.strptime(text[:20], fmt)
            except (ValueError, IndexError):
                pass

        print(f"Warning: OCR text '{text}' could not be parsed.")
        return datetime.now()

    def run(self):
        try:
            parent_path = self.config['parent_path']
            output_path = self.config['output_path']
            fps = float(self.config['fps'])
            ts_list = [t.strip() for t in self.config['timestamp_list'].split('\n') if t.strip()]

            if not ts_list:
                self.status.emit("Error: Timestamp list is empty.")
                return

            video_metadata, missing_paths = [], []
            self.status.emit("Scanning videos...")

            for ts_str in ts_list:
                if not self._is_running: break
                path, last_tried = self.resolve_video_path(parent_path, ts_str)
                if path and os.path.exists(path):
                    meta = self.get_video_metadata(path)
                    if meta:
                        target_dt = None
                        for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M"]:
                            try:
                                target_dt = datetime.strptime(ts_str, fmt)
                                break
                            except ValueError: pass
                        if target_dt:
                            video_metadata.append({"path": path, "target_dt": target_dt, "meta": meta})
                else:
                    missing_paths.append(last_tried)
                    self.status.emit(f"Skipped (Not found): {ts_str}")

            if not self._is_running or not video_metadata:
                if not video_metadata and self._is_running:
                    self.status.emit("Error: No target videos were found.")
                return

            # Preparation for writing
            cap_init = cv2.VideoCapture(video_metadata[0]["path"])
            w, h = int(cap_init.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap_init.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_init.release()

            writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            # Main processing loop
            for i, data in enumerate(video_metadata):
                if not self._is_running: break
                self.status.emit(f"Processing ({i+1}/{len(video_metadata)})")
                meta = data["meta"]
                total_sec = (meta["end_dt"] - meta["start_dt"]).total_seconds()
                elapsed_sec = max(0, min(total_sec, (data["target_dt"] - meta["start_dt"]).total_seconds()))

                frame_idx = int((elapsed_sec / total_sec) * (meta["total_frames"] - 1)) if total_sec > 0 else 0
                cap = cv2.VideoCapture(data["path"])
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

                ret, frame = cap.read()
                if ret:
                    self.preview_update.emit(frame, data["target_dt"].strftime("%Y/%m/%d %H:%M:%S"))
                    writer.write(cv2.resize(frame, (w, h)))
                cap.release()

                self.progress.emit(int((i + 1) / len(video_metadata) * 100))

            writer.release()
            self.status.emit("Completed." if self._is_running else "Process stopped.")

        except Exception as e:
            traceback.print_exc()
            self.status.emit(f"Error occurred: {e}")
        finally:
            self.emit_latest_ocr(force=True)
            self.finished.emit()

    def stop(self):
        self._is_running = False
        self.emit_latest_ocr(force=True)


# --- PreviewWidget & MainWindow ---
class PreviewWidget(QLabel):
    video_loaded_signal = Signal()
    roi_changed = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(800, 600)
        self.setStyleSheet("background-color: black; border: 2px solid gray;")

        self.current_frame = None
        self.roi_rect = QRect(100, 100, 200, 50)
        self.is_dragging = False
        self.drag_mode = None
        self.last_mouse_pos = QPoint()
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.display_width = 0
        self.display_height = 0

    def set_frame(self, frame):
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
            return

        self.current_frame = frame
        img_h, img_w = frame.shape[:2]

        scale = min(self.width() / img_w, self.height() / img_h)
        self.scale_factor = scale

        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        self.display_width = new_w
        self.display_height = new_h

        self.offset_x = (self.width() - new_w) // 2
        self.offset_y = (self.height() - new_h) // 2

        scaled_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        rgb_image = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
        rgb_image = np.ascontiguousarray(rgb_image)

        qimg = QImage(rgb_image.data, new_w, new_h, QImage.Format_RGB888).copy()

        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.black)

        painter = QPainter(pixmap)
        painter.drawPixmap(self.offset_x, self.offset_y, QPixmap.fromImage(qimg))
        painter.end()

        self.setPixmap(pixmap)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.current_frame is None:
            return

        painter = QPainter(self)
        r = self.roi_rect

        px = int(r.x() * self.scale_factor) + self.offset_x
        py = int(r.y() * self.scale_factor) + self.offset_y
        pw = int(r.width() * self.scale_factor)
        ph = int(r.height() * self.scale_factor)

        pen = QPen(QColor(0, 255, 0))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(px, py, pw, ph)

        painter.setBrush(QColor(0, 255, 0))
        painter.drawEllipse(px + pw - 5, py + ph - 5, 10, 10)

    def mousePressEvent(self, event):
        if self.current_frame is None:
            if event.button() == Qt.LeftButton:
                self.video_loaded_signal.emit()
            return

        if event.button() == Qt.RightButton:
            h, w = self.current_frame.shape[:2]
            scale = min(self.width() / w, self.height() / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            scaled_frame = cv2.resize(self.current_frame, (new_w, new_h))
            rgb_image = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2RGB)
            rgb_image = np.ascontiguousarray(rgb_image)

            qimg = QImage(rgb_image.data, new_w, new_h, QImage.Format_RGB888).copy()
            QApplication.clipboard().setImage(qimg)
            return

        if event.button() != Qt.LeftButton:
            return

        mx = event.position().x()
        my = event.position().y()

        # Invalidate black bar area
        if (mx < self.offset_x or mx > self.offset_x + self.display_width or
            my < self.offset_y or my > self.offset_y + self.display_height):
            return

        vx = (mx - self.offset_x) / self.scale_factor
        vy = (my - self.offset_y) / self.scale_factor

        self.is_dragging = True
        self.last_mouse_pos = event.position().toPoint()
        r = self.roi_rect

        if vx > r.x() + r.width() - 30 and vy > r.y() + r.height() - 30:
            self.drag_mode = "resize"
        else:
            self.drag_mode = "move"

    def mouseMoveEvent(self, event):
        if not self.is_dragging or self.current_frame is None:
            return

        delta = event.position().toPoint() - self.last_mouse_pos
        dx = int(delta.x() / self.scale_factor)
        dy = int(delta.y() / self.scale_factor)

        img_h, img_w = self.current_frame.shape[:2]
        r = self.roi_rect

        if self.drag_mode == "move":
            new_x = r.x() + dx
            new_y = r.y() + dy
            max_x = img_w - r.width()
            max_y = img_h - r.height()
            self.roi_rect.moveTo(max(0, min(new_x, max_x)), max(0, min(new_y, max_y)))

        elif self.drag_mode == "resize":
            new_w = max(10, r.width() + dx)
            new_h = max(10, r.height() + dy)
            self.roi_rect.setRect(r.x(), r.y(), min(new_w, img_w - r.x()), min(new_h, img_h - r.y()))

        self.last_mouse_pos = event.position().toPoint()
        self.roi_changed.emit(self.roi_rect.x(), self.roi_rect.y(), 
                                 self.roi_rect.width(), self.roi_rect.height())
        self.update()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False
        self.drag_mode = None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VideoSnapStitcher")
        self.settings_mgr = SettingsManager()
        self.settings_mgr.load()
        self.init_ui()
        self.apply_settings()
    
    def sync_preview_from_roi(self):
        self.preview_widget.roi_rect.setRect(
            self.roi_x_spin.value(), self.roi_y_spin.value(),
            self.roi_w_spin.value(), self.roi_h_spin.value()
        )
        self.preview_widget.update()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Top Layout (Preview & Tabs) ---
        top_layout = QHBoxLayout()
        self.preview_widget = PreviewWidget()
        self.preview_widget.video_loaded_signal.connect(self.open_source_video)
        self.preview_widget.roi_changed.connect(self.sync_roi_from_preview)
        top_layout.addWidget(self.preview_widget, stretch=2)

        # --- Tabs Setup ---
        self.tabs = QTabWidget()
        tab1 = QWidget()
        t1_layout = QVBoxLayout(tab1)
        
        self.ts_list_edit = QTextEdit()
        self.ts_list_edit.setPlaceholderText("YYYY/MM/DD HH:mm\n...")
        self.fps_input = QLineEdit("30")
        self.status_label = QLabel("Status: Ready")
        
        self.ocr_preview = QLabel() 
        self.ocr_preview.setFixedSize(320, 120)
        self.ocr_preview.setAlignment(Qt.AlignCenter)
        self.ocr_preview.setStyleSheet("""border: 1px solid gray; background:black""")

        self.ocr_result = QLabel("Result: -")
        t1_layout.addWidget(QLabel("Timestamp List:"))
        t1_layout.addWidget(self.ts_list_edit)
        t1_layout.addWidget(QLabel("Output MP4 Frame Rate (FPS):"))
        t1_layout.addWidget(self.fps_input)
        t1_layout.addWidget(self.status_label)
        t1_layout.addWidget(QLabel("OCR Image Preview:"))
        t1_layout.addWidget(self.ocr_preview)
        t1_layout.addWidget(self.ocr_result)
        
        # --- Settings Tab ---
        tab2 = QWidget()
        t2_layout = QVBoxLayout(tab2)
        settings_group = QGroupBox("OCR & Path Settings")
        form_layout = QFormLayout()

        path_format_container = QVBoxLayout()
        path_format_label = QLabel("Input File Path Format:")
        self.path_format_input = QLineEdit()
        path_format_container.addWidget(path_format_label)
        path_format_container.addWidget(self.path_format_input)

        tesseract_path_label = QLabel("Tesseract Executable Path:")
        self.tesseract_path_input = QLineEdit()
        path_format_container.addWidget(tesseract_path_label)
        path_format_container.addWidget(self.tesseract_path_input)
        
        t2_layout.addLayout(path_format_container)

        # ROI settings
        self.roi_x_spin = QSpinBox(); self.roi_x_spin.setRange(0, 10000)
        self.roi_y_spin = QSpinBox(); self.roi_y_spin.setRange(0, 10000)
        self.roi_w_spin = QSpinBox(); self.roi_w_spin.setRange(0, 10000)
        self.roi_h_spin = QSpinBox(); self.roi_h_spin.setRange(0, 10000)

        for s in [self.roi_x_spin, self.roi_y_spin, self.roi_w_spin, self.roi_h_spin]:
            s.valueChanged.connect(self.sync_preview_from_roi)

        form_layout.addRow("ROI X:", self.roi_x_spin)
        form_layout.addRow("ROI Y:", self.roi_y_spin)
        form_layout.addRow("ROI W:", self.roi_w_spin)
        form_layout.addRow("ROI H:", self.roi_h_spin)

        self.pad_top_spin = QSpinBox(); self.pad_top_spin.setRange(0, 100)
        form_layout.addRow("OCR Padding (Top):", self.pad_top_spin)

        self.contrast_spin = QSpinBox(); self.contrast_spin.setRange(10, 300)
        form_layout.addRow("Contrast:", self.contrast_spin)

        self.zoom_spin = QSpinBox(); self.zoom_spin.setRange(1, 10)
        form_layout.addRow("OCR Zoom Factor:", self.zoom_spin)

        settings_group.setLayout(form_layout)
        t2_layout.addWidget(settings_group)

        self.tabs.addTab(tab1, "Processing Status & Timestamps")
        self.tabs.addTab(tab2, "Settings")
        top_layout.addWidget(self.tabs, stretch=1)
        main_layout.addLayout(top_layout)

        # --- Bottom Layout (Paths & Buttons) ---
        bottom_layout = QVBoxLayout()
        path_row = QHBoxLayout()
        self.parent_path_input = QLineEdit()
        btn_get_parent = QPushButton("Get Parent Folder")
        path_row.addWidget(QLabel("Parent Path:"))
        path_row.addWidget(self.parent_path_input)
        path_row.addWidget(btn_get_parent)

        self.output_path_input = QLineEdit()
        btn_get_output = QPushButton("Select Output")
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output Path:"))
        out_row.addWidget(self.output_path_input)
        out_row.addWidget(btn_get_output)

        bottom_layout.addLayout(path_row)
        bottom_layout.addLayout(out_row)

        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("Start Process")
        self.btn_stop = QPushButton("Stop Process")
        self.btn_save = QPushButton("SaveSetting")
        self.btn_load = QPushButton("LoadSetting")
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_load)
        bottom_layout.addLayout(btn_layout)

        main_layout.addLayout(bottom_layout)

        # Connections
        self.btn_start.clicked.connect(self.start_process)
        self.btn_stop.clicked.connect(lambda: self.worker.if_exists().stop())
        self.btn_save.clicked.connect(self.save_settings_dialog)
        self.btn_load.clicked.connect(self.load_settings_dialog)
        btn_get_parent.clicked.connect(self.get_parent_folder)
        btn_get_output.clicked.connect(self.set_output_path)

    def open_source_video(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Source Video", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_path:
            cap = cv2.VideoCapture(file_path)
            ret, frame = cap.read()
            if ret:
                self.preview_widget.set_frame(frame)
            cap.release()

    def sync_roi_from_preview(self, x, y, w, h):
        # Block signals temporarily to update values without triggering infinite loops
        self.roi_x_spin.blockSignals(True)
        self.roi_y_spin.blockSignals(True)
        self.roi_w_spin.blockSignals(True)
        self.roi_h_spin.blockSignals(True)
        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)
        self.roi_x_spin.blockSignals(False)
        self.roi_y_spin.blockSignals(False)
        self.roi_w_spin.blockSignals(False)
        self.roi_h_spin.blockSignals(False)

    def apply_settings(self):
        s = self.settings_mgr.settings
        self.path_format_input.setText(s["path_format"])
        self.tesseract_path_input.setText(s.get("tesseract_path", ""))
        self.fps_input.setText(str(s["fps"]))
        self.parent_path_input.setText(s.get("parent_path", ""))
        self.output_path_input.setText(s.get("output_path", ""))
        if "timestamp_list" in s:
            self.ts_list_edit.setPlainText(s["timestamp_list"])
        
        self.roi_x_spin.setValue(s["roi_x"])
        self.roi_y_spin.setValue(s["roi_y"])
        self.roi_w_spin.setValue(s["roi_w"])
        self.roi_h_spin.setValue(s["roi_h"])
        
        self.pad_top_spin.blockSignals(True)
        self.pad_top_spin.setValue(s["padding_top"])
        self.pad_top_spin.blockSignals(False)

        contrast_val = int(s["contrast"] * 100)
        self.contrast_spin.blockSignals(True)
        self.contrast_spin.setValue(max(10, min(300, contrast_val)))
        self.contrast_spin.blockSignals(False)

        zoom_val = int(s["zoom"])
        self.zoom_spin.blockSignals(True)
        self.zoom_spin.setValue(max(1, min(10, zoom_val)))
        self.zoom_spin.blockSignals(False)

        self.preview_widget.roi_rect.setRect(s["roi_x"], s["roi_y"], s["roi_w"], s["roi_h"])
        self.preview_widget.update()        

    def get_parent_folder(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Parent Folder")
        if dir_path: self.parent_path_input.setText(dir_path)

    def set_output_path(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Specify Save Location", "", "Video Files (*.mp4)")
        if file_path: self.output_path_input.setText(file_path)

    def save_settings_dialog(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Settings", f"{os.path.splitext(sys.argv[0])[0]}.json", "JSON Files (*.json)")
        if file_path:
            current_settings = {
                "path_format": self.path_format_input.text(),
                "tesseract_path": self.tesseract_path_input.text(),
                "roi_x": self.roi_x_spin.value(),
                "roi_y": self.roi_y_spin.value(),
                "roi_w": self.roi_w_spin.value(),
                "roi_h": self.roi_h_spin.value(),
                "padding_top": int(self.pad_top_spin.value()),
                "contrast": self.contrast_spin.value() / 100.0,
                "zoom": float(self.zoom_spin.value()),
                "fps": self.fps_input.text(),
                "parent_path": self.parent_path_input.text(),
                "output_path": self.output_path_input.text(),
                "timestamp_list": self.ts_list_edit.toPlainText()
            }
            self.settings_mgr.settings.update(current_settings)
            self.settings_mgr.save(file_path)

    def load_settings_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Settings", f"{os.path.splitext(sys.argv[0])[0]}.json", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                self.settings_mgr.settings.update(loaded)
            self.apply_settings()

    def start_process(self):
        config = {
            "parent_path": self.parent_path_input.text(),
            "output_path": self.output_path_input.text(),
            "timestamp_list": self.ts_list_edit.toPlainText(),
            "fps": self.fps_input.text(),
            "tesseract_path": self.tesseract_path_input.text(),
            **self.settings_mgr.settings
        }
        self.worker = VideoWorker(config)
        self.worker.status.connect(lambda s: self.status_label.setText(f"Status: {s}"))
        self.worker.progress.connect(lambda p: print(f"Progress: {p}%"))
        self.worker.preview_update.connect(self.update_preview)
        self.worker.ocr_preview_update.connect(self.update_ocr_preview)
        self.worker.start()

    def update_preview(self, frame, timestamp):
        self.preview_widget.set_frame(frame)

    def update_ocr_preview(self, roi, text):
        if roi is None or not isinstance(roi, np.ndarray):
            return

        if roi.ndim == 2:
            h, w = roi.shape
            qimg = QImage(roi.data, w, h, roi.strides[0], QImage.Format_Grayscale8).copy()
        else:
            rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888).copy()

        pixmap = QPixmap.fromImage(qimg).scaled(300, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.ocr_preview.setPixmap(pixmap)
        self.ocr_result.setText(f"Result: {text if text else '(Recognition Failed)'}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
