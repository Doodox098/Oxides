from PyQt6.QtWidgets import QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QLineEdit, QLabel, QPushButton

from .BaseParametersWindow import BaseParametersWindow

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