from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtGui import QAction
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton,
                             QLabel, QToolBar, QFileDialog,
                             QMessageBox, QHBoxLayout, QVBoxLayout,
                             QScrollArea, QSplitter, QTabWidget)
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QPixmap
import sys
import os

from server.oxide_server import main_process, oxid_process, process_multiple_files
from windows.AlgoWindow import AlgoWindow
from windows.ChemWindow import ChemWindow
from windows.OxidesWindow import OxidesWindow

class XlsxSaveThread(QThread):
    def __init__(self, data):
        super(XlsxSaveThread, self).__init__()
        self.data = data

    def run(self):
        for obj, name in self.data:
            if isinstance(obj, dict):
                obj = pd.DataFrame(obj)
            obj.to_excel(name, index=False)


class AnalysisThread(QThread):
    result_ready = pyqtSignal(object, object, object, name="result_ready")

    def __init__(self, file_paths, oxides_params, params, chemistry, mode='oxsep'):
        super().__init__()
        self.file_paths = file_paths
        self.oxides_params = oxides_params
        self.params = params
        self.chemistry = chemistry
        self.mode = mode
        self._is_running = True  # Flag to control thread execution

    def stop(self):
        """Stop the thread gracefully"""
        self._is_running = False
        self.terminate()  # Force stop if needed

    def run(self):
        if not self._is_running:
            return

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
            self.result_ready.emit(oxides_result, data, 'oxid')
        except Exception as e:
            self.result_ready.emit(None, str(e), 'oxid')

    def run_oxsep(self):
        self.params.setdefault("model", "first")
        self.params.setdefault("show_every", 0)
        self.params.setdefault("optim", "RMSprop")
        self.params.setdefault("optim_params", {
            "lr": 0.001,
            "momentum": 0.7
        })
        type = 'one_file_oxsep' if len(self.file_paths) == 1 else 'multiple_files_oxsep'
        try:
            if type == 'one_file_oxsep':
                oxides_result, image = main_process(
                    self.file_paths[0],
                    self.oxides_params,
                    self.params,
                    self.chemistry
                )
            else:
                oxides_result, image = process_multiple_files(
                    self.file_paths,
                    self.oxides_params,
                    self.params,
                    self.chemistry
                )
            self.result_ready.emit(oxides_result, image, type)
        except Exception as e:
            self.result_ready.emit(None, str(e), type)


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

        # Add Stop action
        self.stop_action = QAction("Stop Analysis", self)
        self.stop_action.setToolTip("Stop current analysis")
        self.stop_action.triggered.connect(self.stop_analysis)
        self.stop_action.setEnabled(False)  # Disabled by default
        toolbar.addAction(self.stop_action)

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
        """Handle selection of one or multiple files"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose FGA file(s)",
            "",
            "CSV Files (*.csv);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_paths:
            self.file_paths = file_paths  # Store list of paths
            file_names = [os.path.basename(path) for path in file_paths]

            # Update status bar message
            if len(file_names) == 1:
                self.statusBar().showMessage(f"Selected file: {os.path.basename(file_paths[0])}")
            else:
                self.statusBar().showMessage(f"Selected {len(file_names)} files")

            self.file_path = file_paths[0]

    def run_oxid(self):
        self.run_oxid_action.setDisabled(True)
        self.run_action.setDisabled(True)
        self.stop_action.setEnabled(True)  # Enable stop button
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
        self.stop_action.setEnabled(True)  # Enable stop button

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
            self.file_paths,
            self.oxides_params,
            self.params,
            self.chemistry,
            mode='oxsep',
        )
        self.analysis_thread.result_ready.connect(self.display_results)
        self.analysis_thread.start()

    def stop_analysis(self):
        """Stop the currently running analysis"""
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            self.run_action.setEnabled(True)
            self.run_oxid_action.setEnabled(True)
            self.stop_action.setEnabled(False)

    def display_results(self, oxides_results, data, type: str):
        self.run_action.setEnabled(True)
        self.run_oxid_action.setEnabled(True)
        self.stop_action.setEnabled(False)  # Disable stop button
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
        main_layout = QHBoxLayout()
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Image with zoom capabilities
        if data:
            image_container = self.make_image(data)
            splitter.addWidget(image_container)

        # Right side: Oxygen content table(s)
        table_container = QWidget()
        table_layout = QVBoxLayout()

        # Store export data with context
        self.export_data_context = {
            'type': type,
            'all_data': {},
            'aggregated_data': None
        }

        if type == 'multiple_files_oxsep':
            # Create a tab widget for multiple files
            tab_widget = QTabWidget()

            # Process individual files and create tabs for them
            all_oxide_data = {oxide: {'ppm': [], 'vf': [], 'Tb': [], 'Tm': []} for oxide in chain.from_iterable(oxides_results.values())}
            for file_name, results in oxides_results.items():
                # Create a table for this file's results
                columns = ["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"]
                file_table = QTableWidget()
                file_table.setColumnCount(5)
                file_table.setHorizontalHeaderLabels(columns)

                # Sort results by Tb
                sorted_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1]['Tb'])}
                file_table.setRowCount(len(sorted_results))

                # Populate the table
                export_results = []
                for row, (oxide, value) in enumerate(sorted_results.items()):
                    export_results.append({})
                    for col, (col_key, col_value) in enumerate([
                        ("Oxide", oxide),
                        ("Oxygen (ppm)", f"{value['ppm']:.5f}"),
                        ("Vol. fraction", f"{value['vf']:.5f}"),
                        ("Tb (K)", f"{value['Tb']:.1f}"),
                        ("Tm (K)", f"{value['Tm']:.1f}")
                    ]):
                        item = QTableWidgetItem(col_value)
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        file_table.setItem(row, col, item)
                        export_results[-1][col_key] = col_value
                file_table.resizeColumnsToContents()

                # Create a tab for this file
                tab = QWidget()
                tab_layout = QVBoxLayout()
                tab_layout.addWidget(file_table)

                # Add export button for this specific file
                file_export_button = QPushButton(f"Export {Path(file_name).stem} results...")
                file_export_button.clicked.connect(lambda _, fn=file_name, res=sorted_results:
                                                   self.export_single_file(fn, res))
                tab_layout.addWidget(file_export_button)

                tab.setLayout(tab_layout)

                # Add tab with shortened filename
                tab_name = Path(file_name).stem
                if len(tab_name) > 20:
                    tab_name = "..." + tab_name[-17:]
                tab_widget.addTab(tab, tab_name)

                # Store data for this file
                self.export_data_context['all_data'][file_name] = export_results

                # Aggregate data for summary tab
                for oxide_name in all_oxide_data.keys():
                    # If there is no oxide in this file, then ppm and volume fraction can be considered 0
                    all_oxide_data[oxide_name]['ppm'].append(sorted_results.get(oxide_name, {}).get('ppm', 0))
                    all_oxide_data[oxide_name]['vf'].append(sorted_results.get(oxide_name, {}).get('vf', 0))
                    # ...but not Tb and Tm
                    if oxide_name in sorted_results:
                        all_oxide_data[oxide_name]['Tb'].append(sorted_results[oxide_name]['Tb'])
                        all_oxide_data[oxide_name]['Tm'].append(sorted_results[oxide_name]['Tm'])

            # Create aggregated results tab if we have data
            if all_oxide_data:
                # Prepare aggregated data
                columns = [
                    "Oxide",
                    "Oxygen (ppm)", "Oxygen std (ppm)",
                    "Vol. fraction", "Vol. fraction std",
                    "Tb (K)", "Tb std (K)", "Tm (K)", "Tm std (K)"
                ]
                aggregated_results = []
                for oxide_name, values in all_oxide_data.items():
                    aggregated_results.append({
                        'Oxide': oxide_name,
                        'Oxygen (ppm)': f"{float(np.nanmean(values['ppm'])):.5f}",
                        'Oxygen std (ppm)': 0.0 if len(values['ppm']) == 1 else f"{float(np.nanstd(values['ppm'], ddof=1)):.5f}",
                        'Vol. fraction': f"{float(np.nanmean(values['vf'])):.5f}",
                        'Vol. fraction std': 0.0 if len(values['vf']) == 1 else f"{float(np.nanstd(values['vf'], ddof=1)):.5f}",
                        'Tb (K)': f"{float(np.nanmean(values['Tb'])):.1f}",
                        'Tb std (K)': 0.0 if len(values['Tb']) == 1 else f"{float(np.nanstd(values['Tb'], ddof=1)):.1f}",
                        'Tm (K)': f"{float(np.nanmean(values['Tm'])):.1f}",
                        'Tm std (K)': 0.0 if len(values['Tm']) == 1 else f"{float(np.nanstd(values['Tm'], ddof=1)):.1f}"
                    })

                # Sort by Tb
                aggregated_results = sorted(aggregated_results, key=lambda x: x['Tb (K)'])

                # Create aggregated table
                agg_table = QTableWidget()
                agg_table.setColumnCount(9)
                agg_table.setHorizontalHeaderLabels(columns)
                agg_table.setRowCount(len(aggregated_results))

                for row, oxide_dict in enumerate(aggregated_results):
                    for col, (col_key, col_value) in enumerate(oxide_dict.items()):
                        item = QTableWidgetItem(str(col_value))
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        agg_table.setItem(row, col, item)

                agg_table.resizeColumnsToContents()

                # Create aggregated tab
                agg_tab = QWidget()
                agg_layout = QVBoxLayout()
                agg_layout.addWidget(QLabel("Aggregated Results (All Files)"))
                agg_layout.addWidget(agg_table)

                # Add export button for aggregated results
                agg_export_button = QPushButton("Export aggregated results...")
                agg_export_button.clicked.connect(lambda _, res=aggregated_results:
                                                  self.export_aggregated_results(res))
                agg_layout.addWidget(agg_export_button)

                agg_tab.setLayout(agg_layout)
                tab_widget.addTab(agg_tab, "Aggregated")

                # Store aggregated data
                self.export_data_context['aggregated_data'] = aggregated_results

            table_layout.addWidget(tab_widget)

        else:  # Single file or oxid mode
            oxygen_table = QTableWidget()

            if type == 'one_file_oxsep':
                columns = ["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"]
                oxygen_table.setColumnCount(5)
                oxygen_table.setHorizontalHeaderLabels(["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"])
                oxygen_table.setRowCount(len(oxides_results))
                oxides_results = {key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])}
                export_results = []
                for row, (oxide, value) in enumerate(oxides_results.items()):
                    export_results.append({})
                    for col, (col_key, col_value) in enumerate([
                        ("Oxide", oxide),
                        ("Oxygen (ppm)", f"{value['ppm']:.5f}"),
                        ("Vol. fraction", f"{value['vf']:.5f}"),
                        ("Tb (K)", f"{value['Tb']:.1f}"),
                        ("Tm (K)", f"{value['Tm']:.1f}")
                    ]):
                        item = QTableWidgetItem(col_value)
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        oxygen_table.setItem(row, col, item)
                        export_results[-1][col_key] = col_value

                # Store single file data
                self.export_data_context['all_data']['single_file'] = export_results

            elif type == 'oxid':
                columns = ["Oxide", "Tb (K)", "Tm (K)"]
                oxygen_table.setColumnCount(3)
                oxygen_table.setHorizontalHeaderLabels(["Oxide", "Tb (K)", "Tm (K)"])
                oxygen_table.horizontalHeader().setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
                oxygen_table.setRowCount(len(oxides_results))
                oxides_results = {key: value for key, value in sorted(oxides_results.items(), key=lambda x: x[1]['Tb'])}
                export_results = []
                for row, (oxide, value) in enumerate(oxides_results.items()):
                    export_results.append({})
                    for col, (col_key, col_value) in enumerate([
                        ("Oxide", oxide),
                        ("Tb (K)", f"{value['Tb']:.2f}"),
                        ("Tm (K)", f"{value['Tm']:.2f}")
                    ]):
                        item = QTableWidgetItem(col_value)
                        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                        oxygen_table.setItem(row, col, item)
                        export_results[-1][col_key] = col_value
                # Store oxid data
                self.export_data_context['all_data']['oxid'] = export_results

            oxygen_table.resizeColumnsToContents()
            table_title = QLabel("Oxygen Content Analysis")
            table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            table_title.setStyleSheet("font-weight: bold; font-size: 14px;")
            table_layout.addWidget(table_title)
            table_layout.addWidget(oxygen_table)

        # Add main export button that handles all exports
        export_button = QPushButton("Export all data...")
        export_button.clicked.connect(self.export_all_data)
        table_layout.addWidget(export_button)

        table_container.setLayout(table_layout)
        splitter.addWidget(table_container)

        # Set initial sizes
        splitter.setSizes([self.width() * 2 // 3, self.width() // 3])
        main_layout.addWidget(splitter)
        self.results_widget.setLayout(main_layout)

        # Add the results widget to the main window
        if not hasattr(self, 'central_layout'):
            self.central_layout = QVBoxLayout()
            self.central_widget.setLayout(self.central_layout)

        self.central_layout.addWidget(self.results_widget)

        # Start background saving of all data
        self.save_thread = XlsxSaveThread(self.prepare_export_data())
        self.save_thread.start()

    def prepare_export_data(self):
        """Prepare data for automatic background export"""
        data_to_export = []
        context = self.export_data_context

        if context['type'] == 'multiple_files_oxsep':
            for file_name, results in context['all_data'].items():
                base_name = os.path.basename(file_name)
                data_to_export.append((pd.DataFrame(results), f'{base_name}_results.xlsx'))

            if context['aggregated_data']:
                data_to_export.append((pd.DataFrame(context['aggregated_data']), 'aggregated_results.xlsx'))

        elif context['type'] == 'one_file_oxsep':
            data_to_export.append((pd.DataFrame(context['all_data']['single_file']), 'analysis_results.xlsx'))

        elif context['type'] == 'oxid':
            data_to_export.append((pd.DataFrame(context['all_data']['oxid']), 'oxid_results.xlsx'))

        return data_to_export

    def export_single_file(self, file_name, results):
        """Export results for a single file"""
        base_name = Path(file_name).stem
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {base_name} Results",
            f"{base_name}_results.xlsx",
            "Excel Files (*.xlsx)"
        )

        if file_path:
            df = pd.DataFrame(results)
            df.to_excel(file_path, index=False)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Results exported successfully to:\n{file_path}",
                QMessageBox.StandardButton.Ok
            )

    def export_aggregated_results(self, results):
        """Export aggregated results"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Aggregated Results",
            "aggregated_results.xlsx",
            "Excel Files (*.xlsx)"
        )

        if file_path:
            df = pd.DataFrame(results)
            df.to_excel(file_path, index=False)
            QMessageBox.information(
                self,
                "Export Complete",
                f"Aggregated results exported successfully to:\n{file_path}",
                QMessageBox.StandardButton.Ok
            )

    def export_all_data(self):
        """Export all available data (individual files + aggregated)"""
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Select Directory to Export All Data",
            "",
        )

        if dir_path:
            try:
                # Export individual files
                for file_name, results in self.export_data_context['all_data'].items():
                    if self.export_data_context['type'] == 'multiple_files_oxsep':
                        base_name = Path(file_name).stem
                        export_path = os.path.join(dir_path, f"{base_name}_results.xlsx")
                    else:
                        if self.export_data_context['type'] == 'one_file_oxsep':
                            export_path = os.path.join(dir_path, "analysis_results.xlsx")
                        else:
                            export_path = os.path.join(dir_path, "oxid_results.xlsx")

                    pd.DataFrame(results).to_excel(export_path, index=False)

                # Export aggregated data if exists
                if self.export_data_context.get('aggregated_data'):
                    export_path = os.path.join(dir_path, "aggregated_results.xlsx")
                    pd.DataFrame(self.export_data_context['aggregated_data']).to_excel(export_path, index=False)

                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"All data exported successfully to:\n{dir_path}",
                    QMessageBox.StandardButton.Ok
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error during export: {str(e)}",
                    QMessageBox.StandardButton.Ok
                )

    def make_image(self, data):
        """Create a tabbed container for multiple images or a single image widget"""
        if isinstance(data, dict):  # Multiple images
            tab_widget = QTabWidget()

            for file_name, image_data in data.items():
                # Create individual image container for each image
                image_container = self._create_single_image_container(image_data)

                # Generate tab name (you can customize this)
                tab_name = os.path.basename(file_name)

                tab_widget.addTab(image_container, tab_name)

            return tab_widget
        else:  # Single image (backward compatibility)
            return self._create_single_image_container(data)

    def _create_single_image_container(self, image_data):
        """Helper function to create container for a single image"""
        image_container = QWidget()
        image_layout = QVBoxLayout()

        # Convert PIL Image to QPixmap
        from PIL.ImageQt import ImageQt
        original_pixmap = QPixmap.fromImage(ImageQt(image_data))

        # Create scroll area for zoomable image
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)

        image_label = QLabel()
        image_label.setPixmap(original_pixmap.scaled(
            image_scroll.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_scroll.setWidget(image_label)

        # Store references for zoom functionality
        image_label.original_pixmap = original_pixmap  # Attach to label
        image_label.image_scroll = image_scroll  # Attach to label

        # Add zoom controls
        zoom_controls = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_out_btn = QPushButton("Zoom Out (-)")
        reset_zoom_btn = QPushButton("Reset Zoom")

        # Connect zoom functions with current image's components
        zoom_in_btn.clicked.connect(lambda: self.zoom_in_image(image_label))
        zoom_out_btn.clicked.connect(lambda: self.zoom_out_image(image_label))
        reset_zoom_btn.clicked.connect(lambda: self.reset_image_zoom(image_label))

        zoom_controls.addWidget(zoom_in_btn)
        zoom_controls.addWidget(zoom_out_btn)
        zoom_controls.addWidget(reset_zoom_btn)

        image_layout.addWidget(image_scroll)
        image_layout.addLayout(zoom_controls)
        image_container.setLayout(image_layout)

        return image_container

    def zoom_in_image(self, image_label):
        """Zoom in for specific image"""
        current_size = image_label.pixmap().size()
        new_size = current_size * 1.2
        image_label.setPixmap(image_label.original_pixmap.scaled(
            new_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))

    def zoom_out_image(self, image_label):
        """Zoom out for specific image"""
        current_size = image_label.pixmap().size()
        new_size = current_size * 0.8
        if new_size.width() > 50 and new_size.height() > 50:
            image_label.setPixmap(image_label.original_pixmap.scaled(
                new_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def reset_image_zoom(self, image_label):
        """Reset zoom for specific image"""
        image_label.setPixmap(image_label.original_pixmap.scaled(
            image_label.image_scroll.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        ))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()
