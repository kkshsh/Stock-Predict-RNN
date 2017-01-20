from v4.model.model_12_15 import *
import sys

with tf.Graph().as_default(), tf.Session() as session:
    config.config_print()
    print("-------------online train---------------")
    print("please confirm !!!!!!!!!!!!!")
    print("1. retore ckpt file is %s" % config.ot_ckpt_file_path)
    print("2. train epochs == %d" % config.ot_online_train_epoche)
    print("3. 最后交易日期：%s " % config.ot_last_transaction_date)
    print("4. 训练天数：%d" % (config.ot_limit - config.time_step - 1))
    print("please type 'yes' to continue, then type Ctl+D to finish:\n")
    message = sys.stdin.readlines()[0]
    if message.strip() != "yes":
        print("exit success")
        sys.exit()

    print("start to online training........")
    lstmModel = LstmModel(session)
    lstmModel.build_graph()
    lstmModel.load_data(config.ONLINE_TRAIN, end_date=config.ot_last_transaction_date, limit=config.ot_limit)
    lstmModel.saver.restore(lstmModel.session, config.ot_ckpt_file_path)
    print("restore ckpt file over........")
    lstmModel.train_model(operate_type=config.ONLINE_TRAIN, epochs=config.ot_online_train_epoche)
    print("online training over.......")
    session.close()
