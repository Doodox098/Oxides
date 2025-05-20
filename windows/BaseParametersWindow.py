import json
import os
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, QSize, QDir
from PyQt6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QPushButton, QMenu, QMessageBox, QFileDialog, QLineEdit, \
    QComboBox


class BaseParametersWindow(QMainWindow):
    params_changed = pyqtSignal(dict)

    def __init__(self, window_title, default_config_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle(window_title)
        self.default_config_path = default_config_path
        self.params = {}
        self.default_params = self.load_default_params()
        self.init_ui()

    def load_default_params(self):
        """Загружает параметры по умолчанию из файла конфигурации"""
        try:
            with open(self.default_config_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading default params from {self.default_config_path}: {e}")
            return {}

    def init_ui(self):
        self.setMinimumSize(QSize(400, 300))
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

    def create_menu_button(self, layout, button_text):
        button_box = QHBoxLayout()

        self.load_button = QPushButton(button_text)
        self.load_menu = QMenu()
        self.load_button.setMenu(self.load_menu)
        self.load_menu.aboutToShow.connect(self.update_load_menu)

        self.load_dialog_button = QPushButton("Load from file...")
        self.load_dialog_button.clicked.connect(self.load_config_dialog)

        button_box.addWidget(self.load_button)
        button_box.addWidget(self.load_dialog_button)
        layout.addLayout(button_box)

    def update_load_menu(self):
        self.load_menu.clear()
        directory = os.path.dirname(self.default_config_path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        file_list = QDir(directory).entryList(["*.json"], QDir.Filter.Files)

        for filename in file_list:
            action = self.load_menu.addAction(filename)
            action.triggered.connect(lambda checked, f=filename: self.load_file(f))

        self.load_menu.addSeparator()
        refresh_action = self.load_menu.addAction("Update list")
        refresh_action.triggered.connect(self.update_load_menu)

    def load_file(self, filename):
        file_path = Path(os.path.dirname(self.default_config_path)) / filename
        try:
            with open(file_path, 'r') as file:
                content = json.load(file)
            self.update_params(content)
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Unable to load file: {e}",
                QMessageBox.StandardButton.Ok
            )

    def load_config_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load parameters",
            os.path.dirname(self.default_config_path),
            "JSON Files (*.json);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if file_path:
            self.load_file(os.path.basename(file_path))

    def save_config(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameters",
            os.path.dirname(self.default_config_path),
            "JSON Files (*.json);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if not file_path:
            return

        params = self.collect_parameters()
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(params, file, ensure_ascii=False, indent=4)
            QMessageBox.information(
                self,
                "Parameters saved",
                f"Parameters successfully saved to:\n{file_path}",
                QMessageBox.StandardButton.Ok
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Unable to save file:\n{str(e)}",
                QMessageBox.StandardButton.Ok
            )

    def update_params(self, params):
        for key, value in params.items():
            if key in self.params:
                widget = self.params[key]
                if isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QComboBox):
                    widget.setCurrentIndex(value)

    def collect_parameters(self):
        params = {}
        integer_params = getattr(self, 'integer_params', [])

        for name, widget in self.params.items():
            if isinstance(widget, QLineEdit):
                try:
                    if name in integer_params:
                        params[name] = int(widget.text()) if widget.text() else 0
                    else:
                        params[name] = float(widget.text()) if widget.text() else 0.0
                except ValueError:
                    params[name] = 0 if name in integer_params else 0.0
            elif isinstance(widget, QComboBox):
                params[name] = widget.currentIndex()
        return params
