import pandas as pd
"""过滤掉数据量没有满足要求的code， 没有对3打头的code过滤"""
# 2500阀值过滤后大约还剩1318个code
THRESHOLD = 2500

def load_all_code():
    filePath = "/home/daiab/code/ml/something-interest/v3/csv_data/all_code.csv"
    csv = pd.read_csv(filepath_or_buffer=filePath, index_col=0, dtype=str)
    # return fiterCode(csv['code'].values[:3])
    return fiter_code(csv)

def fiter_code(csv):
    filter_result = []
    for index in range(csv.shape[0]):
        code = int(csv.iloc[index, 0])
        days = int(csv.iloc[index, 1])
        if code < 100000 and days > THRESHOLD:
            # if code.startswith("3"):continue
            filter_result.append(code)
    print("all code number == %d" % len(filter_result))
    return filter_result


if __name__ == '__main__':
    print(len(load_all_code()))