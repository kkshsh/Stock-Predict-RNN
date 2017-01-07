import pandas as pd
"""过滤掉数据量没有满足要求的code， 没有对3打头的code过滤"""
# 2500阀值过滤后大约还剩1318个code
THRESHOLD = 2500

def readallcode():
    filePath = "/home/daiab/code/ml/something-interest/db/meta_csv/all_code.csv"
    csv = pd.read_csv(filepath_or_buffer=filePath, index_col=0, dtype=str)
    # return fiterCode(csv['code'].values[:3])
    return fiterCode(csv)

def fiterCode(csv):
    filter_result = []
    for index in range(csv.shape[0]):
        if int(csv.iloc()[index][1]) > THRESHOLD:
            # if code.startswith("3"):continue
            filter_result.append(int(csv.iloc()[index][0]))
    return filter_result


if __name__ == '__main__':
    print(len(readallcode()))