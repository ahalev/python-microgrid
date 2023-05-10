from collections import UserDict

import numpy as np
import pandas as pd


class ModularLogger(UserDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_length = max(len(v) for _, v in self.items()) if len(self.data) else 0

    def flush(self):
        d = self.data.copy()
        self.clear()
        self._log_length = 0
        return d

    def log(self, log_dict=None, **log_items):
        if log_items:
            if log_dict:
                raise TypeError('Cannot pass both positional and keyword arguments.')

            log_dict = log_items

        for key, value in log_dict.items():
            if key not in self:
                self[key] = []

            try:
                self[key].append(value.item())
            except AttributeError:
                self[key].append(value)
            except ValueError:
                raise ValueError('Only scalar values can be logged.')

        self._log_length += 1

    def to_dict(self):
        return self.data.copy()

    def raw(self):
        return {k: list(map(float, v)) for k, v in self.data.items()}

    def to_frame(self):
        return pd.DataFrame(self.data)

    def serialize(self, key):
        return {key: self.to_frame()} if len(self) > 0 else {}

    def __len__(self):
        return self._log_length

    @classmethod
    def from_raw(cls, raw):
        if raw is None:
            return cls()
        elif isinstance(raw, str):
            raw = pd.read_csv(raw).to_dict()
        return cls(raw)
