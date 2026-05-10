import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QPushButton, QWidget, QHBoxLayout, QFileDialog)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

from video_processor import VideoProcessor

# --- МИНИМАЛИСТИЧНЫЕ СТИЛИ ---
BUTTON_STYLE = """
    QPushButton {
        background-color: #F3F4F6;
        color: #1F2937;
        border: 1px solid #D1D5DB;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 14px;
        font-weight: 500;
    }
    QPushButton:hover {
        background-color: #E5E7EB;
    }
    QPushButton:pressed {
        background-color: #D1D5DB;
    }
    QPushButton:disabled {
        background-color: #F9FAFB;
        color: #9CA3AF;
        border: 1px solid #E5E7EB;
    }
"""

LABEL_STYLE_NORMAL = "font-size: 15px; color: #4B5563;"
LABEL_STYLE_BOLD = "font-size: 15px; font-weight: 600; color: #111827;"


class ClickableLabel(QLabel):
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        self.clicked.emit(event.pos().x(), event.pos().y())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ видео")
        self.setStyleSheet("background-color: #FFFFFF;")  # Белый фон окна

        # Устанавливаем базовый чистый шрифт для всего приложения
        font = QFont("Segoe UI", 10)
        font.setStyleHint(QFont.SansSerif)
        QApplication.setFont(font)

        self.points = []
        self.processor_thread = None
        self.video_path = None
        self.current_count = 0

        # Начальная картинка (строгая темная заглушка)
        self.first_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.first_frame[:] = (245, 245, 245)  # Светло-серый фон вместо черного
        cv2.putText(self.first_frame, "", (500, 360),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (150, 150, 150), 1, cv2.LINE_AA)

        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        vbox = QVBoxLayout(central_widget)
        vbox.setContentsMargins(20, 20, 20, 20)
        vbox.setSpacing(15)

        # --- ВЕРХНЯЯ ПАНЕЛЬ ---
        top_hbox = QHBoxLayout()

        self.btn_select_file = QPushButton("Выбрать файл")
        self.btn_select_file.setStyleSheet(BUTTON_STYLE)
        self.btn_select_file.setCursor(Qt.PointingHandCursor)
        self.btn_select_file.clicked.connect(self.select_file)
        top_hbox.addWidget(self.btn_select_file)

        self.file_label = QLabel("Файл не выбран")
        self.file_label.setStyleSheet("color: #9CA3AF; font-size: 13px;")
        top_hbox.addWidget(self.file_label)

        top_hbox.addStretch(1)
        vbox.addLayout(top_hbox)

        # --- ИНФОРМАЦИОННАЯ ПАНЕЛЬ ---
        self.info_label = QLabel("Выберите видеофайл для начала работы")
        self.info_label.setStyleSheet(LABEL_STYLE_NORMAL)
        self.info_label.setAlignment(Qt.AlignCenter)
        vbox.addWidget(self.info_label)

        # --- ХОЛСТ ДЛЯ ВИДЕО ---
        self.image_label = ClickableLabel(self)
        self.image_label.clicked.connect(self.add_point)
        self.image_label.setFixedSize(1280, 720)
        self.image_label.setStyleSheet("border: 1px solid #E5E7EB; border-radius: 4px;")
        self.update_image_display(self.first_frame.copy())
        vbox.addWidget(self.image_label)

        # --- ПАНЕЛЬ УПРАВЛЕНИЯ ---
        bottom_hbox = QHBoxLayout()

        self.btn_clear = QPushButton("Очистить зону")
        self.btn_clear.setStyleSheet(BUTTON_STYLE)
        self.btn_clear.setCursor(Qt.PointingHandCursor)
        self.btn_clear.clicked.connect(self.clear_points)
        self.btn_clear.setEnabled(False)
        bottom_hbox.addWidget(self.btn_clear)

        self.btn_start = QPushButton("Начать анализ")
        self.btn_start.setStyleSheet(BUTTON_STYLE)
        self.btn_start.setCursor(Qt.PointingHandCursor)
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_start.setEnabled(False)
        bottom_hbox.addWidget(self.btn_start)

        # ДОБАВЛЕНА КНОПКА ОСТАНОВКИ
        self.btn_stop = QPushButton("Остановить")
        self.btn_stop.setStyleSheet(BUTTON_STYLE)
        self.btn_stop.setCursor(Qt.PointingHandCursor)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_stop.setEnabled(False)
        bottom_hbox.addWidget(self.btn_stop)

        # Центрируем нижние кнопки
        bottom_wrapper = QHBoxLayout()
        bottom_wrapper.addStretch(1)
        bottom_wrapper.addLayout(bottom_hbox)
        bottom_wrapper.addStretch(1)

        vbox.addLayout(bottom_wrapper)

    def select_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Выберите видео", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)", options=options)

        if file_name:
            self.video_path = file_name
            short_name = self.video_path.split('/')[-1].split('\\')[-1]
            self.file_label.setText(short_name)
            self.file_label.setStyleSheet("color: #4B5563; font-size: 13px;")

            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()

            if ret:
                self.first_frame = cv2.resize(frame, (1280, 720))
                self.clear_points()

                self.btn_start.setEnabled(True)
                self.btn_clear.setEnabled(True)
                self.info_label.setText("Нарисуйте зону подсчета (минимум 3 точки) и нажмите «Начать анализ»")
                self.info_label.setStyleSheet(LABEL_STYLE_NORMAL)

    def add_point(self, x, y):
        if not self.video_path or (self.processor_thread and self.processor_thread.isRunning()):
            return
        self.points.append((x, y))
        self.draw_points()

    def clear_points(self):
        if self.processor_thread and self.processor_thread.isRunning():
            return
        self.points = []
        self.update_image_display(self.first_frame.copy())

    def draw_points(self):
        frame_copy = self.first_frame.copy()
        for i, point in enumerate(self.points):
            cv2.circle(frame_copy, point, 5, (255, 255, 255), -1)
            cv2.circle(frame_copy, point, 6, (50, 50, 50), 1)
            if i > 0:
                cv2.line(frame_copy, self.points[i - 1], point, (255, 255, 255), 2)

        if len(self.points) > 2:
            cv2.line(frame_copy, self.points[-1], self.points[0], (255, 255, 255), 2)

        self.update_image_display(frame_copy)

    def update_image_display(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def start_processing(self):
        if len(self.points) < 3:
            self.info_label.setText("Требуется замкнутая область (минимум 3 точки)")
            self.info_label.setStyleSheet("font-size: 15px; color: #DC2626;")
            return

        self.current_count = 0
        self.info_label.setText("Инициализация нейросети...")
        self.info_label.setStyleSheet(LABEL_STYLE_NORMAL)

        # Переключаем состояние кнопок
        self.btn_start.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_select_file.setEnabled(False)
        self.btn_stop.setEnabled(True)  # Активируем кнопку остановки

        self.processor_thread = VideoProcessor(self.video_path, self.points)
        self.processor_thread.change_pixmap_signal.connect(self.update_image_display)
        self.processor_thread.update_count_signal.connect(self.update_count_text)
        self.processor_thread.finished_signal.connect(self.processing_finished)
        self.processor_thread.start()

    # НОВАЯ ФУНКЦИЯ ДЛЯ ОСТАНОВКИ ПРОЦЕССА
    def stop_processing(self):
        if self.processor_thread and self.processor_thread.isRunning():
            self.info_label.setText("Остановка процесса, подождите...")
            self.btn_stop.setEnabled(False)
            self.processor_thread.stop() # Даем команду потоку завершить цикл

    def update_count_text(self, count):
        self.current_count = count
        self.info_label.setText(
            f"Обработка видео... Уникальных объектов в зоне: {count}"
        )
        self.info_label.setStyleSheet(LABEL_STYLE_BOLD)

    def processing_finished(self):
        # Возвращаем кнопки в исходное состояние
        self.btn_start.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_select_file.setEnabled(True)
        self.btn_stop.setEnabled(False)

        self.info_label.setText(f"Анализ завершен. Итоговое количество объектов в зоне: {self.current_count}")
        self.info_label.setStyleSheet(LABEL_STYLE_BOLD)

    def closeEvent(self, event):
        if self.processor_thread and self.processor_thread.isRunning():
            self.processor_thread.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())