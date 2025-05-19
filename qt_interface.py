import json
from pathlib import Path
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtCore import QSize, Qt, QDir, pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton,
                             QLineEdit, QLabel, QToolBar, QMenu, QFileDialog,
                             QMessageBox, QComboBox, QHBoxLayout, QVBoxLayout,
                             QGroupBox, QFormLayout, QScrollArea, QSplitter)
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QTabWidget
from PyQt6.QtGui import QPixmap
import sys
import os

from oxide_server import main_process


class AnalysisThread(QThread):
    result_ready = pyqtSignal(object, object, name="result_ready")

    def __init__(self, file_path, oxides_params, params, chemistry):
        super().__init__()
        self.file_path = file_path
        self.oxides_params = oxides_params
        self.params = params
        self.chemistry = chemistry

    def run(self):
        self.params.setdefault("model", "first")
        self.params.setdefault("show_every", 0)
        self.params.setdefault("optim", "RMSprop")
        self.params.setdefault("optim_params", {
                                    "lr": 0.001,
                                    "momentum": 0.7
                                })
        try:
            oxides_result, image = main_process(
                self.file_path,
                self.oxides_params,
                self.params,
                self.chemistry
            )
            self.result_ready.emit(oxides_result, image)
        except Exception as e:
            self.result_ready.emit(None, str(e))


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


class AlgoWindow(BaseParametersWindow):
    def __init__(self, *args, **kwargs):
        super().__init__("Algo parameters", "configs/algo_params/config.json", *args, **kwargs)
        self.integer_params = [
            'num_epoch', 'stop_after', 'draw_every', 'delete_after',
            'unstable_after', 'instability_duration', 'stable_after',
            'window_size', 'temp_step'
        ]
        self.init_ui()

    def init_ui(self):
        super().init_ui()

        input_structure = {
            'Penalties:': [
                ('Tb Violation', 'edit', self.default_params.get('t_begin_weight', 0), 't_begin_weight'),
                ('Tb Change', 'edit', self.default_params.get('t_begin_change_weight', 0), 't_begin_change_weight'),
                ('Oxide Displacement', 'edit', self.default_params.get('oxide_usage_weight', 0), 'oxide_usage_weight'),
                ('Basic Oxide Displacement', 'edit', self.default_params.get('guaranteed_usage_weight', 0),
                 'guaranteed_usage_weight'),
                ('Approximation Precision', 'edit', self.default_params.get('residual_weight', 0), 'residual_weight'),
            ],
            'Initial Parameters:': [
                ('Initial E', 'edit', self.default_params.get('e_var', 0), 'e_var'),
                ('E Scale', 'edit', self.default_params.get('e_scale', 0), 'e_scale'),
                ('V Scale', 'edit', self.default_params.get('v_scale', 0), 'v_scale'),
                ('Tm Range', 'edit', self.default_params.get('t_max_delta', 0), 't_max_delta'),
                ('Tm Scale', 'edit', self.default_params.get('t_max_delta_scale', 0), 't_max_delta_scale'),
                ('Tb Range', 'edit', self.default_params.get('t_beg_delta', 0), 't_beg_delta'),
                ('Tb Scale', 'edit', self.default_params.get('t_beg_delta_scale', 0), 't_beg_delta_scale'),
            ],
            'Visualization and Control:': [
                ('Iteration Count', 'edit', self.default_params.get('num_epoch', 0), 'num_epoch'),
                ('Delete After', 'edit', self.default_params.get('delete_after', 0), 'delete_after'),
                ('Stop After', 'edit', self.default_params.get('stop_after', 0), 'stop_after'),
                ('Render Period', 'edit', self.default_params.get('draw_every', 0), 'draw_every'),
            ],
            'Oscillation Tracking:': [
                ('Sensitivity', 'edit', self.default_params.get('unstable_after', 0), 'unstable_after'),
                ('Tolerance', 'edit', self.default_params.get('instability_duration', 0), 'instability_duration'),
                ('Recovery Time', 'edit', self.default_params.get('stable_after', 0), 'stable_after'),
                ('Window Size', 'edit', self.default_params.get('window_size', 0), 'window_size'),
            ],
            'File Reading Parameters:': [
                ('Grid Step Multiplier', 'edit', self.default_params.get('temp_step', 0), 'temp_step'),
                ('Smoothing', 'edit', self.default_params.get('reference_sigma', 0), 'reference_sigma'),
            ],
        }

        main_layout = QVBoxLayout()
        inputs_layout = QHBoxLayout()
        buttons_layout = QHBoxLayout()
        main_layout.addLayout(inputs_layout)
        main_layout.addLayout(buttons_layout)

        # Calculate optimal layout
        group_sizes = [len(group) for group in input_structure.values()]
        n_columns = min(3, len(input_structure))
        columns = [QVBoxLayout() for _ in range(n_columns)]
        for layout in columns:
            inputs_layout.addLayout(layout)

        for i, (group_name, params) in enumerate(input_structure.items()):
            group_box = QGroupBox(group_name)
            form_layout = QFormLayout()
            columns[i % n_columns].addWidget(group_box)

            for label, widget_type, default_value, param_name in params:
                if widget_type == 'edit':
                    self.params[param_name] = QLineEdit(str(default_value))
                    self.params[param_name].setMinimumHeight(25)
                form_layout.addRow(QLabel(label), self.params[param_name])

            group_box.setLayout(form_layout)

        self.create_menu_button(buttons_layout, "Load configuration")

        self.save_button = QPushButton("Save configuration")
        buttons_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_config)

        self.apply_button = QPushButton("Apply")
        buttons_layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_params)

        self.central_widget.setLayout(main_layout)

    def apply_params(self):
        params = self.collect_parameters()
        self.params_changed.emit(params)
        self.close()


class OxidesWindow(BaseParametersWindow):
    def __init__(self, *args, **kwargs):
        super().__init__("Oxides parameters", "configs/oxides_params/oxides.json", *args, **kwargs)

        self.init_ui()

    def init_ui(self):
        super().init_ui()

        main_layout = QVBoxLayout()
        form_layout = QFormLayout()
        main_layout.addLayout(form_layout)

        colors = [
            QColor(0, 0, 255),  # Blue (Present)
            QColor(255, 0, 0),  # Red (Missing)
            QColor(128, 128, 128),  # Grey (Unknown)
        ]

        for oxide_name, default_value in self.default_params.items():
            self.params[oxide_name] = QComboBox(self)
            self.params[oxide_name].addItems(['Present', 'Missing', 'Unknown'])
            self.params[oxide_name].setStyleSheet(f"""
                QComboBox {{
                    color: {colors[default_value].name()};
                }}
            """)
            self.params[oxide_name].currentIndexChanged.connect(
                lambda idx, oxide=oxide_name: self.update_oxide_color(oxide, idx)
            )
            self.params[oxide_name].setCurrentIndex(default_value)
            form_layout.addRow(QLabel(oxide_name), self.params[oxide_name])

        self.create_menu_button(main_layout, "Load oxides")

        self.save_button = QPushButton("Save")
        main_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_config)

        self.apply_button = QPushButton("Apply")
        main_layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_params)

        self.central_widget.setLayout(main_layout)

    def update_oxide_color(self, oxide_name, index):
        colors = [
            QColor(0, 0, 255),  # Blue
            QColor(255, 0, 0),  # Red
            QColor(128, 128, 128),  # Grey
        ]
        self.params[oxide_name].setStyleSheet(f"QComboBox {{ color: {colors[index].name()}; }}")

    def apply_params(self):
        params = self.collect_parameters()
        oxides = {
            'guaranteed_oxides': [name for name, index in params.items() if index == 0],
            'other_oxides': [name for name, index in params.items() if index == 2]
        }
        self.params_changed.emit(oxides)
        self.close()


class ChemWindow(BaseParametersWindow):
    def __init__(self, *args, **kwargs):
        super().__init__("Chemical parameters", "configs/chemistry_params/esh-15.json", *args, **kwargs)
        self.new_element_name = None
        self.new_element_value = None

    def init_ui(self):
        super().init_ui()

        main_layout = QVBoxLayout()
        self.form_layout = QFormLayout()
        main_layout.addLayout(self.form_layout)

        # Add existing elements
        self.update_elements_form()

        # Add controls for new element
        add_group = QGroupBox("Add New Element")
        add_layout = QHBoxLayout()

        self.new_element_name_edit = QLineEdit()
        self.new_element_name_edit.setPlaceholderText("Element name")
        self.new_element_value_edit = QLineEdit()
        self.new_element_value_edit.setPlaceholderText("Value")

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_element)

        add_layout.addWidget(QLabel("Name:"))
        add_layout.addWidget(self.new_element_name_edit)
        add_layout.addWidget(QLabel("Value:"))
        add_layout.addWidget(self.new_element_value_edit)
        add_layout.addWidget(add_button)
        add_group.setLayout(add_layout)
        main_layout.addWidget(add_group)

        self.create_menu_button(main_layout, "Load composition")

        # Button layout
        button_layout = QHBoxLayout()

        self.save_button = QPushButton("Save")
        button_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_config)

        self.apply_button = QPushButton("Apply")
        button_layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_params)

        main_layout.addLayout(button_layout)

        self.central_widget.setLayout(main_layout)

    def update_elements_form(self):
        # Update changes of parameters
        for element_name in list(self.params.keys()):
            self.default_params[element_name] = self.params[element_name].text().strip()

        # Clear existing elements
        for i in reversed(range(self.form_layout.rowCount())):
            self.form_layout.removeRow(i)

        # Add current elements with delete buttons
        for element_name in list(self.default_params.keys()):
            value_edit = QLineEdit(str(self.default_params[element_name]))
            self.params[element_name] = value_edit

            delete_button = QPushButton("×")
            delete_button.setFixedWidth(30)
            delete_button.clicked.connect(lambda _, name=element_name: self.remove_element(name))

            hbox = QHBoxLayout()
            hbox.addWidget(value_edit)
            hbox.addWidget(delete_button)

            self.form_layout.addRow(QLabel(element_name), hbox)

    def add_element(self):
        element_name = self.new_element_name_edit.text().strip()
        element_value = self.new_element_value_edit.text().strip()

        if not element_name:
            QMessageBox.warning(self, "Warning", "Please enter element name")
            return

        try:
            value = float(element_value)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number")
            return

        if element_name in self.default_params:
            QMessageBox.warning(self, "Warning", f"Element '{element_name}' already exists")
            return

        self.default_params[element_name] = value
        self.new_element_name_edit.clear()
        self.new_element_value_edit.clear()
        self.update_elements_form()

    def remove_element(self, element_name):
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to remove {element_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.default_params[element_name]
            if element_name in self.params:
                del self.params[element_name]
            self.update_elements_form()

    def load_file(self, filename):
        file_path = Path(os.path.dirname(self.default_config_path)) / filename
        try:
            with open(file_path, 'r') as file:
                content = json.load(file)

            # Clear current elements
            self.default_params.clear()
            self.params.clear()

            # Load new elements
            self.default_params.update(content)

            # Update form
            self.update_elements_form()

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Unable to load file: {e}",
                QMessageBox.StandardButton.Ok
            )

    def collect_parameters(self):
        params = {}
        for name, widget in self.params.items():
            if isinstance(widget, QLineEdit):
                try:
                    params[name] = float(widget.text()) if widget.text() else 0.0
                except ValueError:
                    params[name] = 0.0
        return params

    def apply_params(self):
        chemistry = self.collect_parameters()
        self.params_changed.emit(chemistry)
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oxides Separation")
        self.setMinimumSize(QSize(1100, 600))
        self.statusBar().showMessage("No file selected")

        # Инициализация окон параметров для получения значений по умолчанию
        self.algo_window = AlgoWindow(self)
        self.oxides_window = OxidesWindow(self)
        self.chem_window = ChemWindow(self)

        # Инициализация параметров из окон по умолчанию
        self.params = self.algo_window.default_params
        self.oxides_params = {
            'guaranteed_oxides': [name for name, index in self.oxides_window.default_params.items() if index == 0],
            'other_oxides': [name for name, index in self.oxides_window.default_params.items() if index == 2]
        }
        self.chemistry = self.chem_window.default_params

        self.init_ui()
        self.file_path = None

    def init_ui(self):
        # Create toolbar
        toolbar = QToolBar("")
        self.addToolBar(toolbar)

        # File action
        file_action = QAction("File", self)
        file_action.setToolTip("Choose file to analyze")
        file_action.triggered.connect(self.choose_file)
        toolbar.addAction(file_action)

        # Algorithm parameters
        algo_action = QAction("Algo parameters", self)
        algo_action.setToolTip("Parameters of algorithm")
        algo_action.triggered.connect(self.open_algo_window)
        toolbar.addAction(algo_action)

        # Oxides parameters
        oxides_action = QAction("Oxides to search", self)
        oxides_action.setToolTip("Choose oxides to search")
        oxides_action.triggered.connect(self.change_oxides)
        toolbar.addAction(oxides_action)

        # Chemistry parameters
        chem_action = QAction("Composition", self)
        chem_action.setToolTip("Chemical composition")
        chem_action.triggered.connect(self.change_chemistry)
        toolbar.addAction(chem_action)

        # Run action
        self.run_action = QAction("Run", self)
        self.run_action.setToolTip("Run analysis")
        self.run_action.triggered.connect(self.run)
        toolbar.addAction(self.run_action)

        # Connect signals
        self.algo_window.params_changed.connect(self.update_from_algo_window)
        self.oxides_window.params_changed.connect(self.update_from_oxides_window)
        self.chem_window.params_changed.connect(self.update_from_chem_window)

        # Set central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

    def open_algo_window(self):
        self.algo_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.algo_window.show()

    def change_oxides(self):
        self.oxides_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.oxides_window.show()

    def change_chemistry(self):
        self.chem_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.chem_window.show()

    def update_from_algo_window(self, params):
        self.params.update(params)

    def update_from_oxides_window(self, params):
        self.oxides_params.update(params)

    def update_from_chem_window(self, params):
        self.chemistry.update(params)

    def choose_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose FGA file",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_path:
            self.file_path = file_path
            file_name = os.path.basename(file_path)
            self.statusBar().showMessage(f"Selected file: {file_name}")

    def run(self):
        self.run_action.setDisabled(True)

        if not self.file_path:
            self.choose_file()
            if not self.file_path:
                QMessageBox.critical(
                    self,
                    "Error",
                    "Choose file to analyze",
                    QMessageBox.StandardButton.Ok
                )
                self.run_action.setEnabled(True)
                return

        print("Algorithm parameters:")
        for key, value in self.params.items():
            print(f"{key}: {value}")

        print("\nOxides parameters:")
        for key, value in self.oxides_params.items():
            print(f"{key}: {value}")

        print("\nChemistry:")
        for key, value in self.chemistry.items():
            print(f"{key}: {value}")

        print(f"\nFile path: {self.file_path}")

        self.analysis_thread = AnalysisThread(
            self.file_path,
            self.oxides_params,
            self.params,
            self.chemistry
        )
        self.analysis_thread.result_ready.connect(self.display_results)
        self.analysis_thread.start()

    def display_results(self, oxides_results, image):
        self.run_action.setEnabled(True)
        if oxides_results is None:
            QMessageBox.critical(
                self,
                "Error",
                f"Error in algorithm: {image}",
                QMessageBox.StandardButton.Ok
            )
            return

        # Clear previous results if any
        if hasattr(self, 'results_widget'):
            self.central_widget.layout().removeWidget(self.results_widget)
            self.results_widget.deleteLater()

        # Create a new widget to hold all results
        self.results_widget = QWidget()
        main_layout = QHBoxLayout()  # Changed to horizontal layout

        # Create splitter for resizable panes
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Image with zoom capabilities
        if image:
            image_container = QWidget()
            image_layout = QVBoxLayout()

            # Convert PIL Image to QPixmap
            from PIL.ImageQt import ImageQt
            self.original_pixmap = QPixmap.fromImage(ImageQt(image))

            # Create scroll area for zoomable image
            self.image_scroll = QScrollArea()
            self.image_scroll.setWidgetResizable(True)

            self.image_label = QLabel()
            self.image_label.setPixmap(self.original_pixmap.scaled(
                self.image_scroll.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            self.image_scroll.setWidget(self.image_label)

            # Add zoom controls
            zoom_controls = QHBoxLayout()
            zoom_in_btn = QPushButton("Zoom In (+)")
            zoom_out_btn = QPushButton("Zoom Out (-)")
            reset_zoom_btn = QPushButton("Reset Zoom")

            zoom_in_btn.clicked.connect(self.zoom_in_image)
            zoom_out_btn.clicked.connect(self.zoom_out_image)
            reset_zoom_btn.clicked.connect(self.reset_image_zoom)

            zoom_controls.addWidget(zoom_in_btn)
            zoom_controls.addWidget(zoom_out_btn)
            zoom_controls.addWidget(reset_zoom_btn)

            image_layout.addWidget(self.image_scroll)
            image_layout.addLayout(zoom_controls)
            image_container.setLayout(image_layout)
            splitter.addWidget(image_container)

        # Right side: Oxygen content table
        table_container = QWidget()
        table_layout = QVBoxLayout()

        oxygen_table = QTableWidget()
        # Определяем какой алгоритм вызывался
        count_columns = len(list(oxides_results.values())[0])
        if count_columns == 6: # Выводим результат разделения
            oxygen_table.setColumnCount(5)
            oxygen_table.setHorizontalHeaderLabels(["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"])
            oxygen_table.setRowCount(len(oxides_results))
            # Сортируем результаты по Tb
            oxides_results = {key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])}

            for row, (oxide, value) in enumerate(oxides_results.items()):
                oxygen_table.setItem(row, 0, QTableWidgetItem(oxide))
                oxygen_table.setItem(row, 1, QTableWidgetItem(f"{value['ppm']:.5f}"))
                oxygen_table.setItem(row, 2, QTableWidgetItem(f"{value['vf']:.4f}"))
                oxygen_table.setItem(row, 3, QTableWidgetItem(f"{value['Tb']:.1f}"))
                oxygen_table.setItem(row, 4, QTableWidgetItem(f"{value['Tm']:.1f}"))

            for col in range(5):
                oxygen_table.setColumnWidth(col, 20)
        elif count_columns == 2: # Выводим теоретические Tb
            oxygen_table.setColumnCount(3)
            oxygen_table.setHorizontalHeaderLabels(["Oxide", "Tb (K)", "Tm (K)"])
            oxygen_table.setRowCount(len(oxides_results))
            # Сортируем результаты по Tb
            oxides_results = {key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])}

            for row, (oxide, value) in enumerate(oxides_results.items()):
                oxygen_table.setItem(row, 0, QTableWidgetItem(oxide))
                oxygen_table.setItem(row, 1, QTableWidgetItem(f"{value['Tb']:.4f}"))
                oxygen_table.setItem(row, 2, QTableWidgetItem(f"{value['Tm']:.4f}"))

            for col in range(3):
                oxygen_table.setColumnWidth(col, 20)

        oxygen_table.resizeColumnsToContents()
        oxygen_table.horizontalHeader().setStretchLastSection(True)

        # Add title above the table
        table_title = QLabel("Oxygen Content Analysis")
        table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        table_title.setStyleSheet("font-weight: bold; font-size: 14px;")

        table_layout.addWidget(table_title)
        table_layout.addWidget(oxygen_table)
        table_container.setLayout(table_layout)
        splitter.addWidget(table_container)

        # Set initial sizes (2:1 ratio)
        splitter.setSizes([self.width()*2//3, self.width()//3])

        main_layout.addWidget(splitter)
        self.results_widget.setLayout(main_layout)

        # Add the results widget to the main window
        if not hasattr(self, 'central_layout'):
            self.central_layout = QVBoxLayout()
            self.central_widget.setLayout(self.central_layout)

        self.central_layout.addWidget(self.results_widget)
        self.reset_image_zoom()

    # Keep the same zoom methods as before
    def zoom_in_image(self):
        if hasattr(self, 'image_label') and hasattr(self, 'original_pixmap'):
            current_size = self.image_label.pixmap().size()
            new_size = current_size * 1.2
            self.image_label.setPixmap(self.original_pixmap.scaled(
                new_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def zoom_out_image(self):
        if hasattr(self, 'image_label') and hasattr(self, 'original_pixmap'):
            current_size = self.image_label.pixmap().size()
            new_size = current_size * 0.8
            if new_size.width() > 50 and new_size.height() > 50:  # Minimum size
                self.image_label.setPixmap(self.original_pixmap.scaled(
                    new_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                ))

    def reset_image_zoom(self):
        if hasattr(self, 'image_label') and hasattr(self, 'original_pixmap'):
            self.image_label.setPixmap(self.original_pixmap.scaled(
                self.image_scroll.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()