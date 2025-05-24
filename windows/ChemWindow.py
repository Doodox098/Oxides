import json
import os
from copy import deepcopy
from pathlib import Path

from PyQt6.QtWidgets import QVBoxLayout, QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QPushButton, QLabel, \
    QMessageBox

from .BaseParametersWindow import BaseParametersWindow


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

        self.temp_params = deepcopy(self.default_params)

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


        density_layout = QFormLayout()
        density_layout.addWidget(QLabel("Density:"))
        self.density = QLineEdit()
        density_layout.addWidget(self.density)
        main_layout.addLayout(density_layout)
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

        # Add existing elements
        self.update_elements_form()

    def update_elements_form(self):
        # Update changes of parameters
        for element_name in list(self.params.keys()):
            self.temp_params[element_name] = self.params[element_name].text().strip()

        # Clear existing elements
        for i in reversed(range(self.form_layout.rowCount())):
            self.form_layout.removeRow(i)

        # Add current elements with delete buttons
        for element_name in filter(lambda x: x != 'density', list(self.temp_params.keys())):
            value_edit = QLineEdit(str(self.temp_params[element_name]))
            self.params[element_name] = value_edit

            delete_button = QPushButton("×")
            delete_button.setFixedWidth(30)
            delete_button.clicked.connect(lambda _, name=element_name: self.remove_element(name))

            hbox = QHBoxLayout()
            hbox.addWidget(value_edit)
            hbox.addWidget(delete_button)

            self.form_layout.addRow(QLabel(element_name), hbox)
        self.density.setText(str(self.temp_params['density']))

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

        if element_name in self.temp_params:
            QMessageBox.warning(self, "Warning", f"Element '{element_name}' already exists")
            return

        self.temp_params[element_name] = value
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
            del self.temp_params[element_name]
            if element_name in self.params:
                del self.params[element_name]
            self.update_elements_form()

    def load_file(self, filename):
        file_path = Path(os.path.dirname(self.default_config_path)) / filename
        try:
            with open(file_path, 'r') as file:
                content = json.load(file)

            # Clear current elements
            self.temp_params.clear()
            self.params.clear()

            # Load new elements
            self.temp_params.update(content)

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
        params['density'] = float(self.density.text())
        return params

    def apply_params(self):
        chemistry = self.collect_parameters()
        self.default_params = self.temp_params
        self.params_changed.emit(chemistry)
        self.close()

    def update_params(self, params):
        self.density.setText(str(params['density']))
        params.pop('density')
        super().update_params(params)