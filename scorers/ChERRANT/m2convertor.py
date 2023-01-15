# -*- coding:UTF-8 -*-
# @Author: Xuezhi Fang
# @Date: 2020-06-19
# @Email: jasonfang3900@gmail.com

import argparse
import re


class M2Processor():
    def __init__(self, src_sent, edit_lines):
        self.src_sent = src_sent
        self.edit_lines = edit_lines
        self.edits = {}
        self.trg_sents = []
        
    def conv_edit(self, line):
        line = line.strip().split("|||")
        edit_span = line[0].split(" ")
        edit_span = (int(edit_span[0]), int(edit_span[1]))
        edit_res = line[2]
        editor = line[-1]
        if edit_span[0] == -1:
            return None
        if edit_span[0] == edit_span[1]:
            edit_tag = "ADD"
        elif edit_res == "-NONE-" or edit_res == "":
            edit_tag = "DEL"
        else:
            edit_tag = "REP"
        return editor, edit_tag, edit_span, edit_res
    
    def get_edits(self):
        for line in self.edit_lines:
            if line:
                edit_item = self.conv_edit(line)
                if not edit_item:
                    continue
                editor, edit_tag, edit_span, edit_res = edit_item
                if editor not in self.edits:
                    self.edits[editor] = []
                self.edits[editor].append({"span": edit_span, "op": edit_tag, "res": edit_res})
                
    def get_para(self):
        self.get_edits()
        if self.edits:
            for editor in self.edits:
                sent = self.src_sent.split(" ")
                for edit_item in self.edits[editor]:
                    edit_span, edit_tag, trg_tokens = edit_item["span"], edit_item["op"], edit_item["res"]
                    if edit_tag == "DEL":
                        sent[edit_span[0]:edit_span[1]] = [" " for _ in range(edit_span[1] - edit_span[0])]
                    else:
                        if edit_tag == "ADD":
                            if edit_span[0] != 0:
                                sent[edit_span[0]-1] += " " + trg_tokens
                            else:
                                sent[edit_span[0]] = trg_tokens + " " + sent[edit_span[0]]
                        elif edit_tag == "REP":
                            src_tokens_len = len(sent[edit_span[0]:edit_span[1]])
                            sent[edit_span[0]:edit_span[1]] = [trg_tokens] + [" " for _ in range(src_tokens_len-1)]
                sent = " ".join(sent).strip()
                res_sent = re.sub(" +", " ", sent)
                self.trg_sents.append(res_sent)
            return self.trg_sents
        else:
            return [self.src_sent]

    
def read_file():
    src_sent = None
    edit_lines = []
    with open(args.f, "r", encoding="utf8") as fr:
        for line in fr:
            if line:
                line = line.strip()
                if line.startswith("S "):
                    src_sent = line.replace("S ", "", 1)
                elif line.startswith("A "):
                    edit_lines.append(line.replace("A ", "", 1))
                elif line == "":
                    yield src_sent, edit_lines
                    edit_lines.clear()


def main():
    counter = 0
    fw_trg = open(args.o, "w", encoding="utf8")
    for src_sent, edit_lines in read_file():
        counter += 1
        m2_item = M2Processor(src_sent, edit_lines)
        trg_sents = m2_item.get_para()
        prefix_counter = 0
        fw_trg.write(trg_sents[0]+"\n")
    fw_trg.close()
 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="m2 file")
    parser.add_argument("-o", help="output file")
    args = parser.parse_args()
    main()
