from rqalpha.__main__ import *
import matplotlib.pyplot as plt
# rqalpha run -f rqalpha-test.py -s 2016-01-01 -e 2016-01-05 -o result.pkl --no-plot

def init(context):
    context.s1 = '000001.XSHE'
    context.s2 = '000024.XSHE'
    context.stocks = [context.s1, context.s2]


def handle_bar(context, bar_dict):
    api.update_universe(context.stocks)
    # history = api.history(50, '1d', 'close')
    # plt.plot(history.values)
    # plt.show()
    # print(history.index.values)
    # print(history.columns.values)
    # print(history)
    inst = api.instruments(context.stocks)
    for i in inst:
        print(i)
    print(inst)

