#!/usr/bin/env python3
import quopri
import os
import re


def t1():
    bpayload = '''
<HTML><head><meta http-equiv=3D"Content-Type" content=3D"text/html; charset=
=3Diso-8859-1"/></head><BODY><TABLE style=3D"PADDING-BOTTOM: 0px; TEXT-TRAN=
SFORM: none; BACKGROUND-COLOR: rgb(255,255,255); TEXT-INDENT: 0px; MARGIN: =
0px auto; PADDING-LEFT: 0px; PADDING-RIGHT: 0px; DISPLAY: table; BORDER-COL=
LAPSE: separate; FONT: 12px/16px Arial, sans-serif; WHITE-SPACE: normal; LE=
TTER-SPACING: normal; COLOR: rgb(51,51,51); WORD-SPACING: 0px; PADDING-TOP:=
 0px; -webkit-text-stroke-width: 0px" id=3Dyiv1765527594container cellSpaci=
ng=3D0 cellPadding=3D0 width=3D640>
'''
    result = quopri.decodestring(bpayload)
    print(result)


def split_file(file_path, dst_path=""):
    if os.path.exists(file_path):
        (base_path, file_name) = os.path.split(file_path)
        state = 0
        index = 0
        result = []
        line = ''
        line_index = 0
        with open(file_path, mode='r', encoding='utf-8') as f:
            while True:
                try:
                    line_index += 1
                    line = f.readline()
                except Exception as e:
                    print(e)
                    print("Line error: {}\nLast Line:".format(line_index, line))
                    break
                if line == "":
                    break
                if state == 0:
                    if len(line) > 2 and line.endswith("--\n"):
                        state = 1
                    elif line == "\n":
                        state = 2
                    result.append(line)
                elif state == 1:
                    if line == '\n':
                        state = 2
                    result.append(line)
                elif state == 2:
                    if not line == '\n':
                        state = 0
                    if line.startswith("From "):  # this line is just for performance
                        m = re.match(r"From .+@.+ \d{2}:\d{2}:\d{2} \d{4}", line)
                        if m:
                            index += 1
                            eml_path = "{}/{}-{:04d}.eml".format(dst_path, file_name, index)
                            with open(eml_path, 'w') as eml_file:
                                eml_file.writelines(result)
                                result = []
                                print("save eml to {}".format(eml_path))
                        else:
                            print("WARNING:{}".format(line))
                    result.append(line)


if __name__ == "__main__":
    # split_file("data/jose_phishing/phishing-2015.txt", "./data/output")
    # split_file("data/jose_phishing/phishing-2016.txt", "./data/output")
    # split_file("data/jose_phishing/phishing-2017.txt", "./data/output")
    # split_file("data/jose_phishing/phishing-2018.txt", "./data/output")
    # split_file("data/jose_phishing/phishing-2019.txt", "./data/output")
    # split_file("data/jose_phishing/phishing0.mbox", "./data/output")
    # split_file("data/jose_phishing/phishing1.mbox", "./data/output")
    # split_file("data/jose_phishing/phishing2.mbox", "./data/output")
    # split_file("data/jose_phishing/phishing3.mbox", "./data/output")
    # split_file("data/jose_phishing/phishing4.mbox", "./data/output")
    split_file("data/jose_phishing/phishing5.mbox", "./data/output")
