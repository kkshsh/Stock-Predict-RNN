import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)

class Option:
    def __init__(self):
        self.timeStep = 100
        self.hiddenCellNum = 200
        self.epochs = 200
        self.batchSize = 50
        self.hiddenLayerNum = 1
        self.keepProp = 1
        self.learningRate = 0.001
        self.outputCellNum = 512
        self.predict_type = "close"  # could be one of ["open", "close", "high", "low"]
        self.forget_bias = 0.8
        self.loop_time = 50
        self.train_data_type = "rate"  # could be ["zscore", "rate"]
        self.is_save_file = False
        logger.info("options:::::\n%s", self)

    def __str__(self):
        return "timeStep: " + str(self.timeStep) + "\n" \
                "hiddenCellNum: " + str(self.hiddenCellNum) + "\n" \
                "epochs: " + str(self.epochs) + "\n" \
                "batchSize: " + str(self.batchSize) + "\n" \
                "hiddenLayerNum: " + str(self.hiddenLayerNum) + "\n" \
                "keepProp: " + str(self.keepProp) + "\n" \
                "learningRate: " + str(self.learningRate) + "\n" \
                "outputCellNum: " + str(self.outputCellNum) + "\n" \
                "predict_type: " + self.predict_type + "\n" \
                "forget_bias: " + str(self.forget_bias) + "\n" \
                "loop_time: " + str(self.loop_time) + "\n" \
                "train_data_type: " + self.train_data_type + "\n" \
                "is_save_file: " + str(self.is_save_file)
