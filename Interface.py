import pandas as pd
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton,
                             QLabel, QToolBar, QFileDialog,
                             QMessageBox, QHBoxLayout, QVBoxLayout,
                             QScrollArea, QSplitter)
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QPixmap
import sys
import os

from server.oxide_server import main_process, oxid_process
from windows.AlgoWindow import AlgoWindow
from windows.ChemWindow import ChemWindow
from windows.OxidesWindow import OxidesWindow

class XlsxSaveThread(QThread):
    def __init__(self, object, name):
        super(XlsxSaveThread, self).__init__()
        self.object = object
        self.name = name

    def run(self):
        self.object.transpose().to_excel(self.name)

class AnalysisThread(QThread):
    result_ready = pyqtSignal(object, object, name="result_ready")

    def __init__(self, file_path, oxides_params, params, chemistry, mode='oxsep'):  # mode = 'oxid' / 'oxsep'
        super().__init__()
        self.file_path = file_path
        self.oxides_params = oxides_params
        self.params = params
        self.chemistry = chemistry
        self.mode = mode

    def run(self): # Oxid - only params evaluation; Oxsep - full algorithm
        if self.mode == 'oxsep':
            self.run_oxsep()
        elif self.mode == 'oxid':
            self.run_oxid()
        else:
            self.result_ready.emit(None, 'Wrong mode')

    def run_oxid(self):
        try:
            oxides_result, data = oxid_process(
                self.chemistry
            )
            self.result_ready.emit(oxides_result, data)
        except Exception as e:
            self.result_ready.emit(None, str(e))

    def run_oxsep(self):
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
            'guaranteed_oxides': [name for name in self.oxides_window.default_params.keys()
                                  if self.oxides_window.default_params[name]['type'] == 0],
            'other_oxides': [name for name in self.oxides_window.default_params.keys()
                             if self.oxides_window.default_params[name]['type'] == 2],
            'density': {name: self.oxides_window.default_params[name]['density']
                        for name in self.oxides_window.default_params.keys()}
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
        self.run_action.setToolTip("Run full analysis")
        self.run_action.triggered.connect(self.run)
        toolbar.addAction(self.run_action)

        # Run oxid action
        self.run_oxid_action = QAction("Run OxID", self)
        self.run_oxid_action.setToolTip("Run only temperatures calculation")
        self.run_oxid_action.triggered.connect(self.run_oxid)
        toolbar.addAction(self.run_oxid_action)

        # Connect signals
        self.algo_window.params_changed.connect(self.update_from_algo_window)
        self.oxides_window.params_changed.connect(self.update_from_oxides_window)
        self.chem_window.params_changed.connect(self.update_from_chem_window)

        # Set central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

    def open_algo_window(self):
        self.algo_window.init_ui()
        self.algo_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.algo_window.show()

    def change_oxides(self):
        self.oxides_window.init_ui()
        self.oxides_window.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.oxides_window.show()

    def change_chemistry(self):
        self.chem_window.init_ui()
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

    def run_oxid(self):
        self.run_oxid_action.setDisabled(True)
        self.run_action.setDisabled(True)
        self.analysis_thread = AnalysisThread(
            None,
            None,
            None,
            self.chemistry,
            mode='oxid',
        )
        self.analysis_thread.result_ready.connect(self.display_results)
        self.analysis_thread.start()

    def run(self):
        self.run_oxid_action.setDisabled(True)
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
                self.run_oxid_action.setEnabled(True)
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
            self.chemistry,
            mode='oxsep',
        )
        self.analysis_thread.result_ready.connect(self.display_results)
        self.analysis_thread.start()

    def display_results(self, oxides_results, data):
        self.run_action.setEnabled(True)
        self.run_oxid_action.setEnabled(True)
        if oxides_results is None:
            QMessageBox.critical(
                self,
                "Error",
                f"Error in algorithm: {data}",
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
        if data:
            image_container = QWidget()
            image_layout = QVBoxLayout()

            # Convert PIL Image to QPixmap
            from PIL.ImageQt import ImageQt
            self.original_pixmap = QPixmap.fromImage(ImageQt(data))

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

        self.data_to_export = pd.DataFrame({key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])})
        # Определяем какой алгоритм вызывался
        count_columns = len(list(oxides_results.values())[0])
        if count_columns == 6:  # Выводим результат разделения
            oxygen_table.setColumnCount(5)
            oxygen_table.setHorizontalHeaderLabels(["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"])
            oxygen_table.setRowCount(len(oxides_results))
            oxides_results = {key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])}

            for row, (oxide, value) in enumerate(oxides_results.items()):
                for col, (col_key, col_value) in enumerate([
                    ("oxide", oxide),
                    ("ppm", f"{value['ppm'] * 10000:.5f}"),
                    ("vf", f"{value['vf']:.5f}"),
                    ("Tb", f"{value['Tb']:.1f}"),
                    ("Tm", f"{value['Tm']:.1f}")
                ]):
                    item = QTableWidgetItem(col_value)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)  # Read-only
                    oxygen_table.setItem(row, col, item)

            for col in range(5):
                oxygen_table.setColumnWidth(col, 20)
            # Export to default file
            self.save_thread = XlsxSaveThread(self.data_to_export, 'analysis_results.xlsx').start()
        elif count_columns == 2:  # Выводим теоретические Tb
            oxygen_table.setColumnCount(3)
            oxygen_table.setHorizontalHeaderLabels(["Oxide", "Tb (K)", "Tm (K)"])
            oxygen_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
            oxygen_table.setRowCount(len(oxides_results))
            # Сортируем результаты по Tb
            oxides_results = {key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])}

            for row, (oxide, value) in enumerate(oxides_results.items()):
                for col, (col_key, col_value) in enumerate([
                    ("oxide", oxide),
                    ("Tb", f"{value['Tb']:.2f}"),
                    ("Tm", f"{value['Tm']:.2f}")
                ]):
                    item = QTableWidgetItem(col_value)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)  # Read-only
                    oxygen_table.setItem(row, col, item)

            for col in range(3):
                oxygen_table.setColumnWidth(col, 20)
            # Export to default file
            self.save_thread = XlsxSaveThread(self.data_to_export, 'oxid_results.xlsx')
        oxygen_table.resizeColumnsToContents()

        # Add title above the table
        table_title = QLabel("Oxygen Content Analysis")
        table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        table_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        # Add export button
        export_button = QPushButton("Export data...")
        export_button.clicked.connect(self.export_data)

        table_layout.addWidget(table_title)
        table_layout.addWidget(oxygen_table)
        table_layout.addWidget(export_button)
        table_container.setLayout(table_layout)
        splitter.addWidget(table_container)

        # Set initial sizes (2:1 ratio)
        splitter.setSizes([self.width() * 2 // 3, self.width() // 3])

        main_layout.addWidget(splitter)
        self.results_widget.setLayout(main_layout)

        # Add the results widget to the main window
        if not hasattr(self, 'central_layout'):
            self.central_layout = QVBoxLayout()
            self.central_widget.setLayout(self.central_layout)

        self.central_layout.addWidget(self.results_widget)
        self.reset_image_zoom()
        self.save_thread.start()

    def export_data(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save parameters",
            "",
            "Excel Files (*.xlsx);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )
        if not file_path:
            return
        try:
            self.data_to_export.transpose().to_excel(file_path)
            QMessageBox.information(
                self,
                "Data saved",
                f"Data successfully saved to:\n{file_path}",
                QMessageBox.StandardButton.Ok
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Unable to save file:\n{str(e)}",
                QMessageBox.StandardButton.Ok
            )

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
