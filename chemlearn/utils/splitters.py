import numpy as np


class Splitter:
    def __init__(self, test_size: float = 0.2, random_state: int = None, **kwargs):
        self.test_size = test_size
        self.train_size = 1 - test_size
        self.random_state = random_state
        for key, value in kwargs.items():
            setattr(self, key, value)

    def split(self, data, **kwargs):
        pass


class TrainTestSplitter(Splitter):
    """Split `items` into random train and test subsets using sklearn train_test_split utility."""

    def __init__(self,
                 test_size: float = 0.2,
                 random_state: int = None,
                 stratify: np.ndarray = None,
                 shuffle: bool = True):
        super().__init__(test_size=test_size, random_state=random_state, stratify=stratify, shuffle=shuffle)

    def split(self, data, **kwargs):
        from sklearn.model_selection import train_test_split
        train, valid = train_test_split(list(range(len(data))), **self.__dict__)
        return {'train_idx': train, 'valid_idx': valid}
