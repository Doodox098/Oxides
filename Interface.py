import ctypes
from itertools import chain
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QThread
from PyQt6.QtWidgets import (QApplication, QWidget, QMainWindow, QPushButton,
                             QLabel, QToolBar, QFileDialog,
                             QMessageBox, QHBoxLayout, QVBoxLayout,
                             QScrollArea, QSplitter, QTabWidget, QFormLayout, QLineEdit, QFrame, QGroupBox)
from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem
from PyQt6.QtGui import QPixmap
import sys
import os
from server.oxide_server import main_process, oxid_process, process_multiple_files, get_references_plot
from windows.AlgoWindow import AlgoWindow
from windows.ChemWindow import ChemWindow
from windows.OxidesWindow import OxidesWindow

class ServerRequestType(Enum):
    OXIDES_PARAMS_CALCULATION = 1
    SEPARATE_OXIDES = 2
    EXTRACT_TOTAL_OXYGEN = 3
    PLOT_REFERENCES = 4

class ServerResponseType(Enum):
    ONE_FILE_SEPARATE_OXIDES = 1
    MULTIPLE_FILES_SEPARATE_OXIDES = 2
    OXIDES_PARAMS_CALCULATION = 3
    EXTRACT_TOTAL_OXYGEN_COMPLETE = 4
    PLOT_REFERENCES_COMPLETE = 5

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

    def __init__(self, file_paths, oxides_params, params, chemistry, total_oxygen, mode=ServerRequestType.OXIDES_PARAMS_CALCULATION):
        super().__init__()
        self.file_paths = file_paths
        self.oxides_params = oxides_params
        self.params = params # This now includes 'limits'
        self.total_oxygen = total_oxygen
        self.chemistry = chemistry
        self.mode = mode
        self._is_running = True

    def stop(self):
        """Stop the thread gracefully"""
        self._is_running = False
        self.wait() # Prefer wait() over terminate() for graceful shutdown

    def run(self):
        if not self._is_running:
            return
        if self.mode == ServerRequestType.OXIDES_PARAMS_CALCULATION:
            self.run_oxsep()
        elif self.mode == ServerRequestType.SEPARATE_OXIDES:
            self.run_oxid()
        elif self.mode == ServerRequestType.EXTRACT_TOTAL_OXYGEN:
            self.run_file_preprocess()
        elif self.mode == ServerRequestType.PLOT_REFERENCES:
            self.run_file_preprocess()
        else:
            self.result_ready.emit(None, 'Wrong mode', None)

    def run_oxid(self):
        try:
            # Pass limits to oxid_process if needed, or handle within the function
            # For now, assuming oxid_process doesn't need limits directly in this call signature
            oxides_result, data = oxid_process(
                self.chemistry
            )
            self.result_ready.emit(oxides_result, data, ServerResponseType.OXIDES_PARAMS_CALCULATION)
        except Exception as e:
            self.result_ready.emit(None, str(e), ServerResponseType.OXIDES_PARAMS_CALCULATION)

    def run_file_preprocess(self):
        """Extract total oxygen from filenames"""
        total_oxygen_list = None
        if self.mode == ServerRequestType.EXTRACT_TOTAL_OXYGEN:
            total_oxygen_list = []
            failed_files = []
            for path in self.file_paths:
                try:
                    # Attempt to extract from filename (e.g., "Sample 5.2 data.csv")
                    filename = os.path.basename(path)
                    parts = filename.split()
                    if len(parts) >= 3:
                        oxygen_str = parts[2].replace(',', '.') # Handle comma decimal separator
                        oxygen_value = float(oxygen_str)
                        total_oxygen_list.append(oxygen_value)
                    else:
                        raise ValueError("Filename format incorrect")
                except (ValueError, IndexError):
                    # If extraction fails, default to 0 or NaN, or mark as failed
                    total_oxygen_list.append(0.0) # Or np.nan if preferred
                    failed_files.append(path)

            if failed_files:
                print(f"Warning: Could not extract total oxygen for files: {failed_files}. Defaulting to 0.0.")

        reference_plots = None
        try:
            plot_config = self.params.copy()
            reference_plots = get_references_plot(self.file_paths, plot_config)
            print("Reference plots generated successfully.")
        except Exception as e:
            print(f"Warning: Could not generate reference plots: {e}")
            reference_plots = None

        self.result_ready.emit(total_oxygen_list, reference_plots, ServerResponseType.EXTRACT_TOTAL_OXYGEN_COMPLETE)

    def run_oxsep(self):
        # Ensure default params are set, including 'limits'
        self.params.setdefault("model", "first")
        self.params.setdefault("show_every", 0)
        self.params.setdefault("optim", "RMSprop")
        self.params.setdefault("limits", [0, 2500]) # Ensure 'limits' is in params
        self.params.setdefault("optim_params", {
            "lr": self.params.get("warmup_lr", 0.001), # Default if not set
            "momentum": self.params.get("momentum", 0.7) # Default if not set
        })

        # Determine response type based on number of files
        type_response = ServerResponseType.ONE_FILE_SEPARATE_OXIDES if len(self.file_paths) == 1 else ServerResponseType.MULTIPLE_FILES_SEPARATE_OXIDES

        try:
            # Extract temperature limits from params
            min_temp, max_temp = self.params.get('limits', [0, 2500])
            # Pass limits as part of the config (params) dictionary
            analysis_params = self.params.copy() # Pass the full params including limits

            if type_response == ServerResponseType.ONE_FILE_SEPARATE_OXIDES:
                oxides_result, image = main_process(
                    self.file_paths[0], # Single path string
                    self.oxides_params,
                    analysis_params, # Pass params including limits
                    self.chemistry,
                    self.total_oxygen
                )
            else: # MULTIPLE FILES
                oxides_result, image = process_multiple_files(
                    self.file_paths, # List of paths
                    self.oxides_params,
                    analysis_params, # Pass params including limits
                    self.chemistry,
                    self.total_oxygen
                )
            self.result_ready.emit(oxides_result, image, type_response)
        except Exception as e:
            self.result_ready.emit(None, str(e), type_response)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Oxides Separation")
        self.setMinimumSize(QSize(1300, 700)) # Increased minimum width for sidebar
        self.statusBar().showMessage("No file selected")

        # Store file paths and total oxygen data
        self.file_paths = []
        self.total_oxygen_data = [] # List to store oxygen values corresponding to file_paths

        # Initialize parameter windows
        self.algo_window = AlgoWindow(self)
        self.oxides_window = OxidesWindow(self)
        self.chem_window = ChemWindow(self)

        # Initialize parameters from windows
        self.params = self.algo_window.default_params.copy()
        # Add default 'limits' to params/config
        self.params.setdefault('limits', [0, 2500])

        self.oxides_params = {
            'guaranteed_oxides': [name for name in self.oxides_window.default_params.keys()
                                  if self.oxides_window.default_params[name]['type'] == 0],
            'other_oxides': [name for name in self.oxides_window.default_params.keys()
                             if self.oxides_window.default_params[name]['type'] == 2],
            'density': {name: self.oxides_window.default_params[name]['density']
                        for name in self.oxides_window.default_params.keys()}
        }
        self.chemistry = self.chem_window.default_params.copy()

        self.init_ui()

    def init_ui(self):
        # Create toolbar
        toolbar = QToolBar("")
        self.addToolBar(toolbar)

        # File action
        file_action = QAction("File", self)
        file_action.setToolTip("Choose file(s) to analyze")
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
        self.stop_action.setEnabled(False)
        toolbar.addAction(self.stop_action)

        # Connect signals from parameter windows
        self.algo_window.params_changed.connect(self.update_from_algo_window)
        self.oxides_window.params_changed.connect(self.update_from_oxides_window)
        self.chem_window.params_changed.connect(self.update_from_chem_window)

        # --- Central Widget Layout ---
        # Main central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget) # Use QHBoxLayout for sidebar

        # Create the permanent sidebar widget
        self.sidebar_widget = self.create_sidebar()
        self.main_layout.addWidget(self.sidebar_widget)

        # Content area (will hold results or placeholder)
        self.content_area = QWidget()
        self.content_layout = QVBoxLayout(self.content_area)
        self.main_layout.addWidget(self.content_area)

        # Initially show a placeholder or message in content area
        self.show_initial_content()

    def create_sidebar(self):
        """Creates the permanent sidebar widget."""
        sidebar = QFrame()
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar.setFixedWidth(300)
        layout = QVBoxLayout(sidebar)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel("Configuration")
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title)

        self.sidebar_scroll = QScrollArea()
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_inner_widget = QWidget()
        self.sidebar_form_layout = QFormLayout(self.sidebar_inner_widget)
        self.sidebar_scroll.setWidget(self.sidebar_inner_widget)
        layout.addWidget(self.sidebar_scroll)

        # === Temperature Limits Group ===
        self.limits_group = QGroupBox("Temperature Limits (K)")
        self.limits_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        limits_layout = QFormLayout(self.limits_group)

        self.min_temp_input = QLineEdit()
        self.min_temp_input.setText(str(self.params['limits'][0]))
        self.min_temp_input.editingFinished.connect(self.on_limits_edited)
        limits_layout.addRow("Min Temp:", self.min_temp_input)

        self.max_temp_input = QLineEdit()
        self.max_temp_input.setText(str(self.params['limits'][1]))
        self.max_temp_input.editingFinished.connect(self.on_limits_edited)
        limits_layout.addRow("Max Temp:", self.max_temp_input)

        self.sidebar_form_layout.addRow(self.limits_group)

        # === File Oxygen Content Group ===
        self.files_group = QGroupBox("File Oxygen Content")
        self.files_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.files_layout = QFormLayout(self.files_group)

        self.oxygen_data_placeholder = QLabel("No files loaded.")
        self.oxygen_data_placeholder.setWordWrap(True)
        self.files_layout.addRow(self.oxygen_data_placeholder)

        self.sidebar_form_layout.addRow(self.files_group)

        # Keep track of dynamically added oxygen input widgets
        self.oxygen_inputs = []
        self.filename_labels = []

        return sidebar

    def update_sidebar_file_data(self):
        """Updates the file-specific part of the sidebar."""
        # Clear existing oxygen input widgets
        for widget in self.oxygen_inputs + self.filename_labels:
            widget.deleteLater()
        self.oxygen_inputs.clear()
        self.filename_labels.clear()

        # Remove placeholder if it exists
        if self.oxygen_data_placeholder.parent() is not None:
            self.oxygen_data_placeholder.setParent(None)

        if self.file_paths and len(self.file_paths) == len(self.total_oxygen_data):
            for i, (path, oxygen) in enumerate(zip(self.file_paths, self.total_oxygen_data)):
                filename_label = QLabel(os.path.basename(path))
                oxygen_input = QLineEdit()
                oxygen_input.setText(str(oxygen))
                oxygen_input.setProperty('file_index', i)
                oxygen_input.editingFinished.connect(self.on_oxygen_edited)

                self.files_layout.addRow(filename_label, oxygen_input)
                self.oxygen_inputs.append(oxygen_input)
                self.filename_labels.append(filename_label)
        else:
            # Re-add placeholder if no files or mismatch
            if self.oxygen_data_placeholder.parent() is None:
                self.files_layout.addRow(self.oxygen_data_placeholder)


    def on_oxygen_edited(self):
        """Handles the event when an oxygen value is edited in the sidebar."""
        sender = self.sender()
        if isinstance(sender, QLineEdit):
            try:
                new_value = float(sender.text())
                file_index = sender.property('file_index')
                if 0 <= file_index < len(self.total_oxygen_data):
                    self.total_oxygen_data[file_index] = new_value
                    print(f"Updated total_oxygen for file {file_index} to {new_value}")
                else:
                    print(f"Error: Invalid file index {file_index}")
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for total oxygen.")
                # Revert to the previous value in the input field
                file_index = sender.property('file_index')
                if 0 <= file_index < len(self.total_oxygen_data):
                     sender.setText(str(self.total_oxygen_data[file_index]))

    def on_limits_edited(self):
        """Handles the event when temperature limits are edited."""
        try:
            new_min = float(self.min_temp_input.text())
            new_max = float(self.max_temp_input.text())
            if new_min >= new_max:
                raise ValueError("Min temperature must be less than Max temperature.")
            self.params['limits'] = [new_min, new_max]
            print(f"Updated temperature limits to [{new_min}, {new_max}]")

            self.plotting_thread = AnalysisThread(
                self.file_paths,
                None,  # oxides_params not needed
                self.params,  # params not needed
                None,  # chemistry not needed
                None,
                mode=ServerRequestType.PLOT_REFERENCES,
            )
            self.plotting_thread.result_ready.connect(self.handle_preprocessing_results)
            self.plotting_thread.start()

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", f"Please enter valid numbers for temperature limits. {str(e)}")
            # Revert to the previous values in the input fields
            old_min, old_max = self.params['limits']
            self.min_temp_input.setText(str(old_min))
            self.max_temp_input.setText(str(old_max))


    def show_initial_content(self):
        """Shows initial content or message when no results are displayed."""
        # Clear previous content
        self.clear_content_area()

        # Add initial message or widget
        initial_label = QLabel("Please load files and run analysis.")
        initial_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.content_layout.addWidget(initial_label)

    def clear_content_area(self):
        """Clears all widgets from the main content area."""
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

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
            self.file_paths = file_paths
            file_names = [os.path.basename(path) for path in file_paths]

            # Update status bar message
            if len(file_names) == 1:
                self.statusBar().showMessage(f"Selected file: {file_names[0]}")
            else:
                self.statusBar().showMessage(f"Selected {len(file_names)} files")

            # Initialize total_oxygen_data list with 0.0 defaults
            self.total_oxygen_data = [0.0] * len(self.file_paths)
            # Trigger preprocessing to attempt to extract oxygen values
            self.preprocess_files()

    def preprocess_files(self):
        """Starts the preprocessing thread to extract total oxygen."""
        if not self.file_paths:
            return

        self.run_oxid_action.setDisabled(True)
        self.run_action.setDisabled(True)
        self.stop_action.setEnabled(True)

        # Create and start preprocessing thread
        self.preprocessing_thread = AnalysisThread(
            self.file_paths,
            None, # oxides_params not needed
            self.params, # params not needed
            None, # chemistry not needed
            None,
            mode=ServerRequestType.EXTRACT_TOTAL_OXYGEN,
        )
        self.preprocessing_thread.result_ready.connect(self.handle_preprocessing_results)
        self.preprocessing_thread.start()

    def handle_preprocessing_results(self, total_oxygen_list, reference_plots, response_type):
        """Handles the results from the preprocessing thread."""
        self.run_action.setEnabled(True)
        self.run_oxid_action.setEnabled(True)
        self.stop_action.setEnabled(False)

        if total_oxygen_list is not None:
            if len(total_oxygen_list) == len(self.file_paths):
                 self.total_oxygen_data = total_oxygen_list
            else:
                 print(f"Warning: Preprocessing returned {len(total_oxygen_list)} oxygen values for {len(self.file_paths)} files.")
                 pass

            self.update_sidebar_file_data()
        else:
             # Handle potential errors in preprocessing if needed
             print("Preprocessing did not complete successfully or returned unexpected data.")
             self.update_sidebar_file_data() # Still update sidebar, likely with defaults

        if reference_plots is not None:
            print("Displaying reference plots...")
            self.clear_content_area()
            results_widget = QWidget()
            main_layout = QHBoxLayout(results_widget)
            splitter = QSplitter(Qt.Orientation.Horizontal)

            image_container = self.make_image(reference_plots)
            splitter.addWidget(image_container)

            info_container = QWidget()
            info_layout = QVBoxLayout()
            info_label = QLabel("Reference plots for selected files.\nRun analysis to see results.")
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            info_label.setWordWrap(True)
            info_layout.addWidget(info_label)
            info_container.setLayout(info_layout)
            splitter.addWidget(info_container)

            splitter.setSizes([self.width() * 2 // 3, self.width() // 3])
            main_layout.addWidget(splitter)
            self.content_layout.addWidget(results_widget)


    def run_oxid(self):
        self.run_oxid_action.setDisabled(True)
        self.run_action.setDisabled(True)
        self.stop_action.setEnabled(True)
        self.analysis_thread = AnalysisThread(
            None,
            None,
            self.params, # Pass params including limits
            self.chemistry,
            self.total_oxygen_data,
            mode=ServerRequestType.SEPARATE_OXIDES,
        )
        self.analysis_thread.result_ready.connect(self.display_results)
        self.analysis_thread.start()

    def run(self):
        if not self.file_paths:
            QMessageBox.critical(
                self,
                "Error",
                "No files selected. Please choose file(s) to analyze.",
                QMessageBox.StandardButton.Ok
            )
            return

        self.run_oxid_action.setDisabled(True)
        self.run_action.setDisabled(True)
        self.stop_action.setEnabled(True)

        print("Algorithm parameters:")
        for key, value in self.params.items():
            print(f"{key}: {value}")
        print("\nOxides parameters:")
        for key, value in self.oxides_params.items():
            print(f"{key}: {value}")
        print("\nChemistry:")
        for key, value in self.chemistry.items():
            print(f"{key}: {value}")
        print(f"\nFile paths: {self.file_paths}")
        print(f"\nTotal Oxygen Data: {self.total_oxygen_data}")

        # Start the main analysis thread, passing params (which includes 'limits')
        self.analysis_thread = AnalysisThread(
            self.file_paths,
            self.oxides_params,
            self.params, # Pass the full params dict including 'limits'
            self.chemistry,
            self.total_oxygen_data,
            mode=ServerRequestType.OXIDES_PARAMS_CALCULATION,
        )
        self.analysis_thread.result_ready.connect(self.display_results)
        self.analysis_thread.start()

    def stop_analysis(self):
        """Stop the currently running analysis or preprocessing"""
        stopped_any = False
        # Stop main analysis thread if running
        if hasattr(self, 'analysis_thread') and self.analysis_thread.isRunning():
            self.analysis_thread.stop()
            stopped_any = True
        # Stop preprocessing thread if running
        if hasattr(self, 'preprocessing_thread') and self.preprocessing_thread.isRunning():
            self.preprocessing_thread.stop()
            stopped_any = True

        if stopped_any:
            self.run_action.setEnabled(True)
            self.run_oxid_action.setEnabled(True)
            self.stop_action.setEnabled(False)

    def display_results(self, oxides_results, data, type_response: ServerResponseType):
        self.run_action.setEnabled(True)
        self.run_oxid_action.setEnabled(True)
        self.stop_action.setEnabled(False)

        if oxides_results is None:
            QMessageBox.critical(
                self,
                "Error",
                f"Error in algorithm: {data}",
                QMessageBox.StandardButton.Ok
            )
            return

        # Clear previous results in the content area
        self.clear_content_area()

        # Create a new widget to hold all results
        results_widget = QWidget()
        main_layout = QHBoxLayout(results_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left side: Image with zoom capabilities
        if data:
            image_container = self.make_image(data)
            splitter.addWidget(image_container)

        # Right side: Oxygen content table(s)
        table_container = QWidget()
        table_layout = QVBoxLayout()

        # Store export data with context (as before, logic unchanged)
        self.export_data_context = {
            'type': type_response,
            'all_data': {},
            'aggregated_data': None
        }

        # --- (Rest of display_results logic remains largely the same) ---

        if type_response == ServerResponseType.MULTIPLE_FILES_SEPARATE_OXIDES:
            # --- Multiple Files Table Logic (Unchanged) ---
            tab_widget = QTabWidget()
            all_oxide_data = {oxide: {'ppm': [], 'vf': [], 'Tb': [], 'Tm': []} for oxide in chain.from_iterable(oxides_results.values())}
            for file_name, results in oxides_results.items():
                columns = ["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"]
                file_table = QTableWidget()
                file_table.setColumnCount(5)
                file_table.setHorizontalHeaderLabels(columns)
                sorted_results = {k: v for k, v in sorted(results.items(), key=lambda x: x[1]['Tb'])}
                file_table.setRowCount(len(sorted_results))
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
                tab = QWidget()
                tab_layout = QVBoxLayout()
                tab_layout.addWidget(file_table)
                file_export_button = QPushButton(f"Export {Path(file_name).stem} results...")
                file_export_button.clicked.connect(lambda _, fn=file_name, res=sorted_results:
                                                   self.export_single_file(fn, res))
                tab_layout.addWidget(file_export_button)
                tab.setLayout(tab_layout)
                tab_name = Path(file_name).stem
                if len(tab_name) > 20:
                    tab_name = "..." + tab_name[-17:]
                tab_widget.addTab(tab, tab_name)
                self.export_data_context['all_data'][file_name] = export_results
                for oxide_name in all_oxide_data.keys():
                    all_oxide_data[oxide_name]['ppm'].append(sorted_results.get(oxide_name, {}).get('ppm', 0))
                    all_oxide_data[oxide_name]['vf'].append(sorted_results.get(oxide_name, {}).get('vf', 0))
                    if oxide_name in sorted_results:
                        all_oxide_data[oxide_name]['Tb'].append(sorted_results[oxide_name]['Tb'])
                        all_oxide_data[oxide_name]['Tm'].append(sorted_results[oxide_name]['Tm'])
            if all_oxide_data:
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
                        'Oxygen std (ppm)': 0.0 if len(values['ppm']) <= 1 else f"{float(np.nanstd(values['ppm'], ddof=1)):.5f}",
                        'Vol. fraction': f"{float(np.nanmean(values['vf'])):.5f}",
                        'Vol. fraction std': 0.0 if len(values['vf']) <= 1 else f"{float(np.nanstd(values['vf'], ddof=1)):.5f}",
                        'Tb (K)': f"{float(np.nanmean(values['Tb'])):.1f}",
                        'Tb std (K)': 0.0 if len(values['Tb']) <= 1 else f"{float(np.nanstd(values['Tb'], ddof=1)):.1f}",
                        'Tm (K)': f"{float(np.nanmean(values['Tm'])):.1f}",
                        'Tm std (K)': 0.0 if len(values['Tm']) <= 1 else f"{float(np.nanstd(values['Tm'], ddof=1)):.1f}"
                    })
                aggregated_results = sorted(aggregated_results, key=lambda x: x['Tb (K)'])
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
                agg_tab = QWidget()
                agg_layout = QVBoxLayout()
                agg_layout.addWidget(QLabel("Aggregated Results (All Files)"))
                agg_layout.addWidget(agg_table)
                agg_export_button = QPushButton("Export aggregated results...")
                agg_export_button.clicked.connect(lambda _, res=aggregated_results:
                                                  self.export_aggregated_results(res))
                agg_layout.addWidget(agg_export_button)
                agg_tab.setLayout(agg_layout)
                tab_widget.addTab(agg_tab, "Aggregated")
                self.export_data_context['aggregated_data'] = aggregated_results
            table_layout.addWidget(tab_widget)
        else: # Single file or oxid mode
            oxygen_table = QTableWidget()
            if type_response == ServerResponseType.ONE_FILE_SEPARATE_OXIDES:
                # --- Single File Table Logic (Unchanged) ---
                columns = ["Oxide", "Oxygen (ppm)", "Vol. fraction", "Tb (K)", "Tm (K)"]
                oxygen_table.setColumnCount(5)
                oxygen_table.setHorizontalHeaderLabels(columns)
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
                self.export_data_context['all_data']['single_file'] = export_results
            elif type_response == ServerResponseType.OXIDES_PARAMS_CALCULATION:
                # --- OxID Results Table Logic (Unchanged) ---
                columns = ["Oxide", "Tb (K)", "Tm (K)"]
                oxygen_table.setColumnCount(3)
                oxygen_table.setHorizontalHeaderLabels(columns)
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
                self.export_data_context['all_data']['oxid'] = export_results
            oxygen_table.resizeColumnsToContents()
            table_title = QLabel("Oxygen Content Analysis")
            table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            table_title.setStyleSheet("font-weight: bold; font-size: 14px;")
            table_layout.addWidget(table_title)
            table_layout.addWidget(oxygen_table)

        # Add main export button (Unchanged)
        export_button = QPushButton("Export all data...")
        export_button.clicked.connect(self.export_all_data)
        table_layout.addWidget(export_button)
        table_container.setLayout(table_layout)
        splitter.addWidget(table_container)
        splitter.setSizes([self.width() * 2 // 3, self.width() // 3])
        main_layout.addWidget(splitter)
        # Add the results widget to the main content area
        self.content_layout.addWidget(results_widget)
        # Start background saving (Unchanged)
        self.save_thread = XlsxSaveThread(self.prepare_export_data())
        self.save_thread.start()

    # --- Remaining methods (export, image handling) remain largely unchanged ---
    # (Keeping them for completeness, but they are not the focus of this modification)
    def prepare_export_data(self):
        """Prepare data for automatic background export"""
        data_to_export = []
        context = self.export_data_context
        # Determine context type string for export logic
        context_type_str = None
        if context['type'] == ServerResponseType.MULTIPLE_FILES_SEPARATE_OXIDES:
            context_type_str = 'multiple_files_oxsep'
        elif context['type'] == ServerResponseType.ONE_FILE_SEPARATE_OXIDES:
            context_type_str = 'one_file_oxsep'
        elif context['type'] == ServerResponseType.OXIDES_PARAMS_CALCULATION:
            context_type_str = 'oxid'

        if context_type_str == 'multiple_files_oxsep':
            for file_name, results in context['all_data'].items():
                base_name = os.path.basename(file_name)
                data_to_export.append((pd.DataFrame(results), f'{base_name}_results.xlsx'))
            if context['aggregated_data']:
                data_to_export.append((pd.DataFrame(context['aggregated_data']), 'aggregated_results.xlsx'))
        elif context_type_str == 'one_file_oxsep':
            data_to_export.append((pd.DataFrame(context['all_data']['single_file']), 'analysis_results.xlsx'))
        elif context_type_str == 'oxid':
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
            # Convert dict results to list of dicts if necessary for DataFrame
            if isinstance(results, dict):
                df_data = []
                for oxide, value in results.items():
                    df_data.append({
                        "Oxide": oxide,
                        "Oxygen (ppm)": f"{value['ppm']:.5f}" if 'ppm' in value else "",
                        "Vol. fraction": f"{value['vf']:.5f}" if 'vf' in value else "",
                        "Tb (K)": f"{value['Tb']:.1f}" if 'Tb' in value else "",
                        "Tm (K)": f"{value['Tm']:.1f}" if 'Tm' in value else ""
                    })
                df = pd.DataFrame(df_data)
            else:
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
                context_type_str = None
                if self.export_data_context['type'] == ServerResponseType.MULTIPLE_FILES_SEPARATE_OXIDES:
                    context_type_str = 'multiple_files_oxsep'
                elif self.export_data_context['type'] == ServerResponseType.ONE_FILE_SEPARATE_OXIDES:
                    context_type_str = 'one_file_oxsep'
                elif self.export_data_context['type'] == ServerResponseType.OXIDES_PARAMS_CALCULATION:
                    context_type_str = 'oxid'

                for file_name, results in self.export_data_context['all_data'].items():
                    if context_type_str == 'multiple_files_oxsep':
                        base_name = Path(file_name).stem
                        export_path = os.path.join(dir_path, f"{base_name}_results.xlsx")
                    else:
                        if context_type_str == 'one_file_oxsep':
                            export_path = os.path.join(dir_path, "analysis_results.xlsx")
                        else:
                            export_path = os.path.join(dir_path, "oxid_results.xlsx")
                     # Convert dict results to list of dicts if necessary for DataFrame
                    if isinstance(results, dict):
                        df_data = []
                        for oxide, value in results.items():
                            df_data.append({
                                "Oxide": oxide,
                                "Oxygen (ppm)": f"{value['ppm']:.5f}" if 'ppm' in value else "",
                                "Vol. fraction": f"{value['vf']:.5f}" if 'vf' in value else "",
                                "Tb (K)": f"{value['Tb']:.1f}" if 'Tb' in value else "",
                                "Tm (K)": f"{value['Tm']:.1f}" if 'Tm' in value else ""
                            })
                        df = pd.DataFrame(df_data)
                    else:
                        df = pd.DataFrame(results)
                    df.to_excel(export_path, index=False)

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
        if isinstance(data, dict):
            tab_widget = QTabWidget()
            for file_name, image_data in data.items():
                image_container = self._create_single_image_container(image_data)
                tab_name = os.path.basename(file_name)
                tab_widget.addTab(image_container, tab_name)
            return tab_widget
        else:
            return self._create_single_image_container(data)

    def _create_single_image_container(self, image_data):
        """Helper function to create container for a single image"""
        image_container = QWidget()
        image_layout = QVBoxLayout()
        from PIL.ImageQt import ImageQt
        original_pixmap = QPixmap.fromImage(ImageQt(image_data))
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
        image_label.original_pixmap = original_pixmap
        image_label.image_scroll = image_scroll
        zoom_controls = QHBoxLayout()
        zoom_in_btn = QPushButton("Zoom In (+)")
        zoom_out_btn = QPushButton("Zoom Out (-)")
        reset_zoom_btn = QPushButton("Reset Zoom")
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

def set_taskbar_icon():
    if sys.platform == 'win32':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('myapp.1.0')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_taskbar_icon()
    window = MainWindow()
    window.setWindowIcon(QIcon('icon-round.ico'))
    window.show()
    app.exec()
