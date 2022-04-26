import sys

def remove_special_tags():
    file = "./MuCGEC_CGED_Dev.para"
    src_file = "./MuCGEC_CGED_Dev.src"
    tgt_file = "./MuCGEC_CGED_Dev.tgt"

    with open(file, "r", encoding="utf-8") as f:
        with open(src_file, "w", encoding="utf-8") as o1:
            with open(tgt_file, "w", encoding="utf-8") as o2:
                for line in f:
                    li = line.rstrip("\n").split("\t")[1:]
                    src = li[0]
                    for tgt in li[1:]:
                        if tgt == "无法标注" or tgt == "没有错误":
                            tgt = src
                        o1.write(src + "\n")
                        o2.write(tgt + "\n")

def get_erroneous_pair():
    src_file = "./train_data/lang8+hsk/train.src"
    tgt_file = "./train_data/lang8+hsk/train.tgt"
    with open(src_file, "r", encoding="utf-8") as f1:
        with open(tgt_file, "r", encoding="utf-8") as f2:
            with open(src_file + "_only_erroneous", "w", encoding="utf-8") as o1:
                with open(tgt_file + "_only_erroneous", "w", encoding="utf-8") as o2:
                    src_lines = f1.readlines()
                    tgt_lines = f2.readlines()
                    for src_line, tgt_line in zip(src_lines, tgt_lines):
                        if src_line != tgt_line:
                            o1.write(src_line + "\n")
                            o2.write(tgt_line + "\n")