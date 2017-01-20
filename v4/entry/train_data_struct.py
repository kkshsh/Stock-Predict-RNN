import numpy as np


class DD:
    def __init__(self, date_range, is_split_data_to_train_test, code="", train_data=None, softmax=None, target=None):
        # if softmax is not None:
        #     assert train_data.shape[0] == softmax.shape[0]
        assert len(date_range) == train_data.shape[0]
        self.date_range = date_range
        self.train_data = train_data
        self.softmax = softmax
        self.target = target
        self.code = code
        self.days = len(date_range)
        train_days = self.days - 200
        index = list(range(self.days))
        np.random.shuffle(index)
        """在线更新时，不分配测试数据"""
        if is_split_data_to_train_test:
            self.train_index = index[:train_days]
            self.test_index = index[train_days:]
        else:
            self.train_index = index
            self.test_index = []

        # mask = np.ones(train_data.shape[0], dtype=bool)
        # index = np.random.randint(low=0, high=self.days, size=100)
        # mask[index] = False


    # TODO:convert date to index and index to date
