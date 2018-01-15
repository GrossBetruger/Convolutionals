import shutil
import os

def duplicate_file_list(flist, num_of_duplicates):
    for i in range(num_of_duplicates):
        for fname in flist:
            shutil.copy(fname, fname + "_" + str(i))

if __name__ == "__main__":
    os.chdir(os.path.join("train", "cat"))
    files = os.listdir(".")
    duplicate_file_list(files, 16)
