import tensorflow as tf
from v4.config import config
from v4.model.model_12_15 import LstmModel
import numpy as np
import pandas as pd


with tf.Graph().as_default(), tf.Session() as session:
    print("-------------predict---------------")
    print("start to predict")
    config.config_print()
    lstmModel = LstmModel(session)
    lstmModel.build_graph()
    lstmModel.load_data(operate_type=config.PREDICT, end_date=config.op_last_transaction_date, limit=config.time_step + 4)
    lstmModel.saver.restore(lstmModel.session, config.op_ckpt_file_path)
    print("load model file over, file name == %s" % config.op_ckpt_file_path)

    last_transaction_date = config.op_last_transaction_date
    code_list = []
    probability = []
    up_down = []
    for dd in lstmModel.dd_list:
        if last_transaction_date != dd.date_range[-1]:
            print("stock code : %d, 预测数据有误, 数据最新日期== %s，但last_transaction_date== %s"
                  % (dd.code, dd.date_range[-1], last_transaction_date))
            continue
        predict = lstmModel.session.run(lstmModel.predict_target,
                                        feed_dict={lstmModel.one_train_data: [dd.train_data[-1]],
                                                   lstmModel.rnn_keep_prop: 1.0,
                                                   lstmModel.hidden_layer_keep_prop: 1.0})

        predict = np.exp(predict)
        prob = predict / np.sum(predict, axis=1)[:, np.newaxis]
        code_list.append(dd.code)
        probability.append(prob.max())
        # 1: up, 0:down
        up_down.append(1 - prob.argmax())
    data_frame = pd.DataFrame({"code": code_list, "prob": probability, "up_down": up_down})
    data_frame.to_csv(config.op_export_excel_file_path)
    print("predict over, export csv file path == %s" % config.op_export_excel_file_path)

    session.close()
