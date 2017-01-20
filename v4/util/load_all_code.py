import pandas as pd
from v4.config import config
"""过滤掉数据量没有满足要求的code， 没有对3打头的code过滤"""
# 2500阀值过滤后大约还剩1318个code
THRESHOLD = 2500


def load_all_code():
    csv = pd.read_csv(filepath_or_buffer=config.code_csv_file_path, index_col=0, dtype=str)
    return filter_code(csv)


def filter_code(csv):
    filter_result = []
    for index in range(csv.shape[0]):
        code = int(csv.iloc[index, 0])
        days = int(csv.iloc[index, 1])
        if days > THRESHOLD:
            # if code.startswith("3"):continue
            filter_result.append(code)
    print("all code number == %d" % len(filter_result))
    return filter_result


if __name__ == '__main__':
    print(len(load_all_code()))
