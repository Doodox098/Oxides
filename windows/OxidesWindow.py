from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QVBoxLayout, QFormLayout, QComboBox, QLabel, QPushButton, QLineEdit, QHBoxLayout

from .BaseParametersWindow import BaseParametersWindow

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

        for oxide_name, default_values in self.default_params.items():
            density_widget = QLineEdit(self)
            density_widget.setText(str(default_values['density']))
            type_widget = QComboBox(self)
            self.params[oxide_name] = {'type': type_widget, 'density': density_widget}
            type_widget.addItems(['Present', 'Missing', 'Unknown'])
            type_widget.setStyleSheet(f"""
                QComboBox {{
                    color: {colors[default_values['type']].name()};
                }}
            """)
            type_widget.currentIndexChanged.connect(
                lambda idx, oxide=oxide_name: self.update_oxide_color(oxide, idx)
            )
            type_widget.setCurrentIndex(default_values['type'])
            fields = QHBoxLayout()
            fields.addWidget(type_widget, 1)
            fields.addWidget(density_widget, 1)
            form_layout.addRow(QLabel(oxide_name), fields)


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
        self.params[oxide_name]['type'].setStyleSheet(f"QComboBox {{ color: {colors[index].name()}; }}")

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