from typing import Any


class BaseTransform:

    def fit(self, *args, **kwargs):
        ...

    def transform(self, *args, **kwargs):
        ...

    def fit_transform(self, *args, **kwargs) -> Any:
        self.fit(*args, **kwargs)
        return self.transform(*args, **kwargs)
    