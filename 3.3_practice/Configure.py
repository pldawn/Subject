class Configure(BaseConfigure):
    def __init__(self, configure=None):
        self._configure = self.generate_default_configure()

        if configure is not None:
            self.update(configure)

    def generate_default_configure(self) -> dict:
        default_configure = {
            "device": "cuda:0",
            "top_k": 10,
            "top_p": 0.90,
            "repetition_penalty": 1.2,
            "repetition_window": 4,
            "max_length": 512,
            "max_summary_length": 30,
            "summary_nums": 10
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
