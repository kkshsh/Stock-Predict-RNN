from zipfile import *
import os
import zipfile
def add_dir_zip(dir):
    zip_f = zipfile.ZipFile('zip.zip', 'w', zipfile.ZIP_DEFLATED)
    for dirpath, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            zip_f.write(os.path.join(dirpath, filename))
    zip_f.close

if __name__ == "__main__":
    add_dir_zip('/home/daiab/code/ml/something-interest/test')
