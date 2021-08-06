class Configure:
    def __init__(self, configure=None):
        self._configure = self.generate_default_configure()

        if configure is not None:
            self.update(configure)

    def generate_default_configure(self) -> dict:
        default_configure = {
            "distillation_config": {
                "temperature": 4,
                "temperature_scheduler": "none",
                "kd_loss_type": "ce",
                "kd_loss_weight": 1,
                "kd_loss_weight_scheduler": "none",
                "hard_label_weight": 1,
                "hard_label_weight_scheduler": "none",
                "probability_shift": True,
            },

            "training_config": {
                "gradient_accumulation_steps": 1,
                "ckpt_frequency": 1,
                "ckpt_epoch_frequency": 1,
                "log_dir": "./saved_log/",
                "output_dir": "./saved_models/",
                "fp16": False,
                "fp16_opt_level": "O1",
            },

            "distiller_train_config": {
                "num_epochs": 50,
                "max_grad_norm": -1.0,
            },

            "matches": {
                "L6_hidden_smmd": [
                    {"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [2, 2], "layer_S": [1, 1], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [4, 4], "layer_S": [2, 2], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [6, 6], "layer_S": [3, 3], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [8, 8], "layer_S": [4, 4], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [10, 10], "layer_S": [5, 5], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [12, 12], "layer_S": [6, 6], "feature": "hidden", "loss": "mmd", "weight": 1}
                ],
                "L6_hidden_mse": [
                    {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 2, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 4, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 6, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 8, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 10, "layer_S": 5, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 12, "layer_S": 6, "feature": "hidden", "loss": "hidden_mse", "weight": 1}
                ],
                "L3_hidden_smmd": [
                    {"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [4, 4], "layer_S": [1, 1], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [8, 8], "layer_S": [2, 2], "feature": "hidden", "loss": "mmd", "weight": 1},
                    {"layer_T": [12, 12], "layer_S": [3, 3], "feature": "hidden", "loss": "mmd", "weight": 1}
                ],
                "L3_hidden_mse": [
                    {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 4, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 8, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                    {"layer_T": 12, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1}
                ]
            }
        }

        return default_configure

    def update(self, configure: dict) -> None:
        updated_key = []

        for key, value in configure.items():
            if key in self._configure:

                # 默认配置字典和用户配置字典中，某个配置的值都是字典时，进行更新
                if type(self._configure[key]) == dict and type(value) == dict:
                    self._configure[key].update(value)
                    updated_key.append(key)

                # 默认配置字典和用户配置字典中，某个配置的值都是字符串时，进行更新
                elif type(self._configure[key]) == type(value):
                    self._configure[key] = value
                    updated_key.append(key)

                # 默认配置字典和用户配置字典中，某个配置的值类型不同时，不进行更新
                else:
                    updated_key.append(key)

        for key in updated_key:
            del configure[key]

        self._configure.update(configure)

    def set(self, configure):
        self._configure = configure

    def get(self):
        return self._configure

    def __setitem__(self, key, value):
        self._configure[key] = value

    def __getitem__(self, key):
        return self._configure[key]
