from copy import deepcopy

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout, QFormLayout, QComboBox, QLabel, QPushButton, QLineEdit, QHBoxLayout, QGroupBox, \
    QMessageBox

from .BaseParametersWindow import BaseParametersWindow

class OxidesWindow(BaseParametersWindow):
    def __init__(self, *args, **kwargs):
        super().__init__("Oxides parameters", "configs/oxides_params/oxides.json", *args, **kwargs)

        self.init_ui()

    def init_ui(self):
        super().init_ui()

        main_layout = QVBoxLayout()
        self.form_layout = QFormLayout()
        main_layout.addLayout(self.form_layout)
        self.temp_params = deepcopy(self.default_params)

        self.colors = [
            QColor(0, 0, 255),  # Blue (Present)
            QColor(255, 0, 0),  # Red (Missing)
            QColor(128, 128, 128),  # Grey (Unknown)
        ]

        # Add controls for new element
        add_group = QGroupBox("Add New Oxide")
        add_layout = QHBoxLayout()

        self.new_oxide_name_edit = QLineEdit()
        self.new_oxide_name_edit.setPlaceholderText("Oxide name")
        self.new_oxide_type_edit = QComboBox(self)
        self.new_oxide_type_edit.addItems(['Present', 'Missing', 'Unknown'])
        self.new_oxide_type_edit.setStyleSheet(f"""
                QComboBox {{
                    color: {self.colors[2].name()};
                }}
            """)
        self.new_oxide_type_edit.setCurrentIndex(2)
        self.new_oxide_type_edit.currentIndexChanged.connect(
                lambda idx: self.new_oxide_type_edit.setStyleSheet(f"QComboBox {{ color: {self.colors[idx].name()}; }}")
            )
        self.new_oxide_density_edit = QLineEdit()
        self.new_oxide_density_edit.setPlaceholderText("Density")

        add_button = QPushButton("Add")
        add_button.clicked.connect(self.add_element)
        # Adding new oxides is not implemented because of need to update database
        add_button.setDisabled(True)

        add_layout.addWidget(QLabel("Name:"))
        add_layout.addWidget(self.new_oxide_name_edit)
        add_layout.addWidget(QLabel("Type:"))
        add_layout.addWidget(self.new_oxide_type_edit)
        add_layout.addWidget(QLabel("Density:"))
        add_layout.addWidget(self.new_oxide_density_edit)
        add_layout.addWidget(add_button)
        add_group.setLayout(add_layout)
        main_layout.addWidget(add_group)

        self.create_menu_button(main_layout, "Load oxides")

        self.save_button = QPushButton("Save")
        main_layout.addWidget(self.save_button)
        self.save_button.clicked.connect(self.save_config)

        self.apply_button = QPushButton("Apply")
        main_layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_params)

        self.central_widget.setLayout(main_layout)
        self.update_elements_form()

    def update_oxide_color(self, oxide_name, index):
        self.params[oxide_name]['type'].setStyleSheet(f"QComboBox {{ color: {self.colors[index].name()}; }}")

    def collect_parameters(self):
        params = {}

        for name, widgets in self.params.items():
            type_widget, density_widget = widgets['type'], widgets['density']
            params[name] = {}
            params[name]['type'] = type_widget.currentIndex()
            params[name]['density'] = float(density_widget.text())
        return params

    def apply_params(self):
        params = self.collect_parameters()
        oxides = {
            # 'guaranteed_oxides': [name for name, index in params.items() if index == 0],
            # 'other_oxides': [name for name, index in params.items() if index == 2],
            'guaranteed_oxides': [name for name in params.keys() if params[name]['type'] == 0],
            'other_oxides': [name for name in params.keys() if params[name]['type'] == 2],
            'density': {name: params[name]['density'] for name in params.keys()}
        }
        self.default_params = self.temp_params
        self.params_changed.emit(oxides)
        self.close()

    def update_params(self, params):
        for key, values in params.items():
            type = values['type']
            density = values['density']
            if key in self.params:
                widgets = self.params[key]
                widgets['type'].setCurrentIndex(type)
                widgets['density'].setText(str(density))
    # Add/remove oxides module
    def update_elements_form(self):
        # Update changes of parameters
        for oxide_name in list(self.params.keys()):
            self.temp_params[oxide_name] = {}
            self.temp_params[oxide_name]['type'] = self.params[oxide_name]['type'].currentIndex()
            self.temp_params[oxide_name]['density'] = float(self.params[oxide_name]['density'].text())

        # Clear existing elements
        for i in reversed(range(self.form_layout.rowCount())):
            self.form_layout.removeRow(i)

        # Add current elements with delete buttons
        for oxide_name, default_values in self.temp_params.items():
            density_widget = QLineEdit(self)
            density_widget.setText(str(default_values['density']))
            type_widget = QComboBox(self)
            self.params[oxide_name] = {'type': type_widget, 'density': density_widget}
            type_widget.addItems(['Present', 'Missing', 'Unknown'])
            type_widget.setStyleSheet(f"""
                QComboBox {{
                    color: {self.colors[default_values['type']].name()};
                }}
            """)
            type_widget.currentIndexChanged.connect(
                lambda idx, oxide=oxide_name: self.update_oxide_color(oxide, idx)
            )
            type_widget.setCurrentIndex(default_values['type'])

            delete_button = QPushButton("×")
            delete_button.setFixedWidth(30)
            delete_button.clicked.connect(lambda _, name=oxide_name: self.remove_element(name))
            # Adding new oxides is not able now, so is deleting
            delete_button.setDisabled(True)

            fields = QHBoxLayout()
            for widget in [type_widget, density_widget, delete_button]:
                widget.setMinimumHeight(20)
                fields.addWidget(widget, 1)
            label = QLabel(oxide_name)
            label.setMinimumHeight(20)
            self.form_layout.addRow(label, fields)

    def add_element(self):
        oxide_name = self.new_oxide_name_edit.text().strip()
        oxide_type = self.new_oxide_type_edit.currentIndex()
        oxide_density = self.new_oxide_density_edit.text().strip()

        if not oxide_name:
            QMessageBox.warning(self, "Warning", "Please enter oxide name")
            return

        try:
            density = float(oxide_density)
        except ValueError:
            QMessageBox.warning(self, "Warning", "Please enter a valid number")
            return

        if oxide_name in self.temp_params:
            QMessageBox.warning(self, "Warning", f"Oxide '{oxide_name}' already exists")
            return

        self.temp_params[oxide_name] = {"type": oxide_type, "density": density}
        self.new_oxide_name_edit.clear()
        self.new_oxide_type_edit.setCurrentIndex(2)
        self.new_oxide_type_edit.setStyleSheet(f"""
                QComboBox {{
                    color: {self.colors[2].name()};
                }}
            """)
        self.new_oxide_density_edit.clear()
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
