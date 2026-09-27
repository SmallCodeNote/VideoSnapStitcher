import sys
import os
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Organize PySide6 imports
from PySide6 import QtCore
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                                QDateTimeEdit, QComboBox, QSpinBox, QProgressBar, 
                                QFileDialog, QMessageBox, QFormLayout)
from PySide6.QtCore import Qt, QThread, Signal, Slot

# --- Logic Class (Video Generation Engine) ---

class VideoGenerationWorker(QThread):
    """Thread class for processing video generation in the background."""
    progress_update = Signal(dict)  # {"file": str, "percent": float}
    finished = Signal()
    error = Signal(str)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._is_running = True

    def stop(self):
        """Flag to safely interrupt the process."""
        self._is_running = False

    def _resolve_placeholders(self, template: str, dt: datetime) -> str:
        """Replace placeholders like YYYY, MM, DD, HH, mm with actual values."""
        replacements = {
            "YYYYMMDD": dt.strftime("%Y%m%d"),
            "YYYYMM": dt.strftime("%Y%m"),
            "YYYY": dt.strftime("%Y"),
            "HH": dt.strftime("%H"),
            "mm": dt.strftime("%M"),
            "ss": dt.strftime("%S"),
            "MM": dt.strftime("%m"), 
            "DD": dt.strftime("%d")   
        }
        path = template
        for key in sorted(replacements, key=len, reverse=True):
            path = path.replace(key, replacements[key])
        return path

    def get_text_position(self, frame_shape: tuple, pos_type: str, draw_text: str, 
                           fontFace: int, fontScale: float, thickness: int, margin: int) -> tuple:
        """
        Calculate the size of the string to be drawn and return coordinates that fit within the specified margin range.
        """
        h, w = frame_shape[:2]
        # Get text size (width, height), baseline
        (text_width, text_height), _ = cv2.getTextSize(draw_text, fontFace, fontScale, thickness)

        # Define allowed coordinate range (considering margins)
        x_min, x_max = margin, w - margin
        y_min, y_max = margin, h - margin

        # Prepare to limit by maximum drawable size in case of extremely large strings
        # (If the text itself is larger than the margin, push it back to the margin position)

        if pos_type == "top_left":
            return (x_min, y_min)
        elif pos_type == "top_right":
            # Calculate from right edge and clamp so as not to exceed x_max
            x = max(x_min, min(w - text_width - margin, x_max))
            y = y_min
            return (x, y)
        elif pos_type == "bottom_left":
            # Calculate from bottom edge and clamp so as not to exceed y_max
            x = x_min
            y = max(y_min, min(h - text_height - margin, y_max))
            return (x, y)
        elif pos_type == "bottom_right":
            # Calculate from both right and bottom edges, clamping each
            x = max(x_min, min(w - text_width - margin, x_max))
            y = max(y_min, min(h - text_height - margin, y_max))
            return (x, y)
        elif pos_type == "center":
            # Calculate center and clamp to fit within the range
            x = max(x_min, min((w - text_width) // 2, x_max))
            y = max(y_min, min((h - text_height) // 2, y_max))
            return (x, y)
        else:
            return (w // 2, h // 2)

    def format_path(self, dt: datetime) -> str:
        """Generate a save path according to the specified format."""
        fmt = self.config['file_path_format']
        return self._resolve_placeholders(fmt, dt)

    def run(self):
        try:
            q_start = self.config['start_datetime']
            start_dt = datetime(q_start.date().year(), q_start.date().month(), q_start.date().day(), 
                                 q_start.time().hour(), q_start.time().minute(), 0)
            
            q_end = self.config['end_datetime']
            end_dt = datetime(q_end.date().year(), q_end.date().month(), q_end.date().day(), 
                               q_end.time().hour(), q_end.time().minute(), 0)
            
            if start_dt >= end_dt:
                self.error.emit("Process interrupted because the start date is after the end date.")
                return

            fps = float(self.config['fps'])
            segment_min = int(self.config['segment_duration_minutes'])
            draw_fmt = self.config['date_format_draw']
            text_size = float(self.config['font_size'])
            pos_type = self.config['text_position']
            width = int(self.config['video_width'])
            height = int(self.config['video_height'])
            margin = int(self.config['margin'])

            current_dt = start_dt
            total_duration = (end_dt - start_dt).total_seconds()
            total_frames = int(total_duration * fps)
            processed_frames = 0

            while current_dt < end_dt and self._is_running:
                segment_end_dt = min(current_dt + timedelta(minutes=segment_min), end_dt)
                # Keep the start time of the segment (for calculation)
                segment_start_dt = current_dt 
                segment_duration_sec = (segment_end_dt - segment_start_dt).total_seconds()
                segment_frames = int(segment_duration_sec * fps)

                file_path = self.format_path(current_dt)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(file_path, fourcc, fps, (width, height))

                if not out.isOpened():
                    self.error.emit(f"Could not open file: {file_path}")
                    return

                for i in range(segment_frames):
                    if not self._is_running:
                        out.release()
                        return

                    # --- Fix Point ---
                    # Calculate the time of the current frame (exactly segment_start_dt when i=0)
                    # Adopt a method of back-calculating from index to prevent cumulative errors in timedelta
                    frame_time = segment_start_dt + timedelta(seconds=i / fps)
                    if frame_time > segment_end_dt:
                        frame_time = segment_end_dt

                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    draw_text = self._resolve_placeholders(draw_fmt, frame_time)
                    
                    pos = self.get_text_position((height, width), pos_type, 
                                                 draw_text, cv2.FONT_HERSHEY_SIMPLEX, 
                                                 text_size/100, 3, margin)

                    cv2.putText(frame, draw_text, pos, 
                                 fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                 fontScale=text_size/100, 
                                 color=(255, 255, 255), 
                                 thickness=3)

                    out.write(frame)
                    # ------------------

                    processed_frames += 1
                    
                    if processed_frames % 10 == 0:
                        percent = (processed_frames / total_frames) * 100
                        self.progress_update.emit({
                            "file": os.path.basename(file_path),
                            "percent": percent
                        })

                # Update time for the next segment
                current_dt = segment_end_dt
                out.release()
            
            if self._is_running:
                self.finished.emit()

        except Exception as e:
            print(f"\n[ERROR] VideoGenerationWorker Error:\n{e}", file=sys.stderr)
            self.error.emit(str(e))


# --- GUI Class ---

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TestVideoMaker - Video Generation Tool")
        self.setMinimumWidth(700)
        self.worker = None
        self._is_running_flag = True 
        self.init_ui()
        self.load_settings_auto()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        form_layout = QFormLayout()

        now = QtCore.QDateTime.currentDateTime()
        start_qdt = now.addDays(-1) 
        end_qdt = now.addDays(1)    

        start_dt_obj = QtCore.QDateTime(start_qdt.date().year(), start_qdt.date().month(), 
                                          start_qdt.date().day(), start_qdt.time().hour(), 
                                          start_qdt.time().minute(), 0)
        end_dt_obj = QtCore.QDateTime(end_qdt.date().year(), end_qdt.date().month(), 
                                        end_qdt.date().day(), end_qdt.time().hour(), 
                                        end_qdt.time().minute(), 0)

        self.start_dt_input = QDateTimeEdit(start_dt_obj)
        self.end_dt_input = QDateTimeEdit(end_dt_obj)
        
        self.fps_input = QLineEdit("30")
        self.segment_combo = QComboBox()
        self.segment_combo.addItems(["1min", "5min", "10min", "20min", "30min"])

        self.path_format_input = QLineEdit("C:/Videos/YYYY/YYYYMM/YYYYMMDD/HH/mm.mp4")
        self.font_size_input = QSpinBox()
        self.font_size_input.setRange(1, 500)
        self.font_size_input.setValue(50)

        self.pos_combo = QComboBox()
        self.pos_combo.addItems(["center", "top_left", "top_right", "bottom_left", "bottom_right"])

        self.draw_format_input = QLineEdit("YYYY/MM/DD HH:mm:ss")

        # Resolution settings
        self.width_input = QSpinBox()
        self.width_input.setRange(1, 8000)
        self.width_input.setValue(1920)
        self.height_input = QSpinBox()
        self.height_input.setRange(1, 8000)
        self.height_input.setValue(1080)

        # Margin settings
        self.margin_input = QSpinBox()
        self.margin_input.setRange(0, 500)
        self.margin_input.setValue(30)

        form_layout.addRow("Start Date/Time:", self.start_dt_input)
        form_layout.addRow("End Date/Time:", self.end_dt_input)
        form_layout.addRow("FPS:", self.fps_input)
        form_layout.addRow("Segment Duration:", self.segment_combo)
        form_layout.addRow("Save Path Format:", self.path_format_input)
        form_layout.addRow("Font Size:", self.font_size_input)
        form_layout.addRow("Text Position:", self.pos_combo)
        form_layout.addRow("Date Display Format:", self.draw_format_input)
        form_layout.addRow("Margin Width:", self.margin_input)
        
        res_layout = QHBoxLayout()
        res_layout.addWidget(QLabel("Width:"))
        res_layout.addWidget(self.width_input)
        res_layout.addWidget(QLabel("Height:"))
        res_layout.addWidget(self.height_input)
        form_layout.addRow("Resolution:", res_layout)

        layout.addLayout(form_layout)

        self.status_label = QLabel("Waiting...")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Process")
        self.stop_btn = QPushButton("Stop Process")
        self.save_btn = QPushButton("Save Setting")
        self.load_btn = QPushButton("Load Setting")

        self.start_btn.clicked.connect(self.start_process)
        self.stop_btn.clicked.connect(self.stop_process)
        self.save_btn.clicked.connect(self.save_settings)
        self.load_btn.clicked.connect(self.load_settings)

        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addWidget(self.load_btn)
        layout.addLayout(btn_layout)

    def get_config(self) -> Dict[str, Any]:
        """Get settings from the current GUI."""
        return {
            "start_datetime": self.start_dt_input.dateTime(),
            "end_datetime": self.end_dt_input.dateTime(),
            "fps": float(self.fps_input.text() or 30),
            "segment_duration_minutes": int(self.segment_combo.currentText().replace("min", "")),
            "file_path_format": self.path_format_input.text(),
            "font_size": float(self.font_size_input.value()),
            "text_position": self.pos_combo.currentText(),
            "date_format_draw": self.draw_format_input.text(),
            "video_width": int(self.width_input.value()),
            "video_height": int(self.height_input.value()),
            "margin": int(self.margin_input.value())
        }

    def start_process(self):
        config = self.get_config()
        self.start_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self._is_running_flag = True
        
        self.worker = VideoGenerationWorker(config)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(lambda: self.on_finished("Completed."))
        self.worker.error.connect(lambda e: self.on_finished(f"Error: {e}", is_error=True))
        self.worker.start()

    def stop_process(self):
        if self.worker and self._is_running_flag:
            self.worker.stop()
            self._is_running_flag = False
            self.status_label.setText("Stopping...")

    @Slot(dict)
    def update_progress(self, data):
        self.status_label.setText(f"Generating: {data['file']}")
        self.progress_bar.setValue(int(data['percent']))

    def on_finished(self, message, is_error=False):
        self.start_btn.setEnabled(True)
        if is_error:
            print(f"\n[INFO] Process finished with error.")
            QMessageBox.critical(self, "Error", message)
        else:
            QMessageBox.information(self, "Done", message)
        self.status_label.setText("Waiting...")
        self.progress_bar.setValue(0)

    def save_settings(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings", f"TestVideoMaker.json")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                config = self.get_config()
                save_data = config.copy()
                dt1 = config["start_datetime"]
                dt2 = config["end_datetime"]
                save_data["start_datetime"] = dt1.toString(QtCore.Qt.ISODate)
                save_data["end_datetime"] = dt2.toString(QtCore.Qt.ISODate)
                json.dump(save_data, f, ensure_ascii=False, indent=4)

    def load_settings(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", f"TestVideoMaker.json")
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self._apply_json_config(config)

    def load_settings_auto(self):
        filename = "TestVideoMaker.json"
        if os.path.exists(filename):
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    self._apply_json_config(config)
            except Exception as e:
                print(f"Auto load failed: {e}", file=sys.stderr)

    def _apply_json_config(self, config: Dict[str, Any]):
        """Apply settings from JSON individually. Continue even if an error occurs."""
        if 'fps' in config:
            try: self.fps_input.setText(str(config['fps']))
            except Exception as e: print(f"Error loading fps: {e}", file=sys.stderr)

        if 'segment_duration_minutes' in config:
            try:
                val = int(float(config['segment_duration_minutes']))
                for i in range(self.segment_combo.count()):
                    item_text = self.segment_combo.itemText(i)
                    if item_text.replace("min", "").strip() == str(val):
                        self.segment_combo.setCurrentIndex(i)
                        break
            except Exception as e: print(f"Error loading segment duration: {e}", file=sys.stderr)

        if 'file_path_format' in config:
            try: self.path_format_input.setText(config['file_path_format'])
            except Exception as e: print(f"Error loading path format: {e}", file=sys.stderr)

        if 'font_size' in config:
            try: self.font_size_input.setValue(int(config['font_size']))
            except Exception as e: print(f"Error loading font size: {e}", file=sys.stderr)

        if 'text_position' in config:
            try:
                pos_str = str(config['text_position']).strip()
                for i in range(self.pos_combo.count()):
                    item_text = self.pos_combo.itemText(i).strip()
                    if item_text == pos_str:
                        self.pos_combo.setCurrentIndex(i)
                        break
            except Exception as e: print(f"Error loading text position: {e}", file=sys.stderr)

        if 'date_format_draw' in config:
            try: self.draw_format_input.setText(config['date_format_draw'])
            except Exception as e: print(f"Error loading date format draw: {e}", file=sys.stderr)

        if 'video_width' in config:
            try: self.width_input.setValue(int(config['video_width']))
            except Exception as e: print(f"Error loading video width: {e}", file=sys.stderr)

        if 'video_height' in config:
            try: self.height_input.setValue(int(config['video_height']))
            except Exception as e: print(f"Error loading video height: {e}", file=sys.stderr)

        if 'margin' in config:
            try: self.margin_input.setValue(int(config['margin']))
            except Exception as e: print(f"Error loading margin: {e}", file=sys.stderr)

        if 'start_datetime' in config:
            try:
                dt = QtCore.QDateTime.fromString(config["start_datetime"], QtCore.Qt.ISODate)
                self.start_dt_input.setDateTime(dt)
                self.start_dt_input.setTime(QtCore.QTime(dt.time().hour(), dt.time().minute(), 0))
            except Exception as e: print(f"Error loading start datetime: {e}", file=sys.stderr)

        if 'end_datetime' in config:
            try:
                dt = QtCore.QDateTime.fromString(config["end_datetime"], QtCore.Qt.ISODate)
                self.end_dt_input.setDateTime(dt)
                self.end_dt_input.setTime(QtCore.QTime(dt.time().hour(), dt.time().minute(), 0))
            except Exception as e: print(f"Error loading end datetime: {e}", file=sys.stderr)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
