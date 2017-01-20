from v4.util.data_preprocess import DataPreprocess
from v4.db.db_service.read_mongodb import ReadDB


class GenTrainData:
    def __init__(self, all_code, time_step, operate_type, end_date=None, limit=None):
        """
        operate_type:
        case 1: 线下训练
        case 2: 在线训练
        case 3: 预测
        """
        read_db = ReadDB()
        preprocess = DataPreprocess(time_step)
        """list of DD object"""
        self.dd_list = []
        for code in all_code:
            db_data, date_range = read_db.read_one_stock_data(code, end_date=end_date, limit=limit)
            dd = preprocess.process(db_data, date_range, operate_type=operate_type)
            dd.code = code
            self.dd_list.append(dd)
        read_db.destory()
