import os

from os import listdir
from os.path import join, isfile


# read file, and combine string and label
def combine_txt_and_label(file_path, text_label):
    if os.path.exists(file_path):
        (base_path, file_name) = os.path.split(file_path)
        with open(file_path, mode='r', encoding='utf-8') as f:
            tmp = f.read()
            # print(tmp)
            return "{} b\'{}\'".format(text_label, tmp)
    return ""


# combine strings from different files
def combine_txt_files(path, label):
    str_rlt = ""
    if os.path.exists(path):
        onlyfiles = [join(path, f) for f in listdir(path) if (isfile(join(path, f)) and f.endswith(".txt"))]
        if onlyfiles.__len__() > 0:
            str_list = []
            for file_name in onlyfiles:
                full_path = "{}/{}".format(path, file_name)
                tmp_str = combine_txt_and_label(file_name, label)
                print(tmp_str)
                if tmp_str != "":
                    str_list.append(tmp_str)
            return "\n".join(str_list)
    return str_rlt


if __name__ == '__main__':
    # rlt = combine_txt_files("../test_data/jose-phishing", "__label__0")
    # combine_txt_path = "{}/combine_txt_jose_phishing.txt".format(".")
    # rlt = combine_txt_files("../test_data/from_weicheng/normal_txt", "__label__1")
    # combine_txt_path = "{}/combine_txt_wc_normal.txt".format(".")
    rlt = combine_txt_files("../test_data/from_weicheng/phishing_txt", "__label__0")
    combine_txt_path = "{}/combine_txt_wc_phishing.txt".format(".")
    with open(combine_txt_path, 'w') as eml_file:
        m_len = eml_file.write(rlt)
        print("save eml to {}, len {}".format(combine_txt_path, m_len))
