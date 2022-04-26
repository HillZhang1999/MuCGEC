import argparse
from collections import Counter
from modules.classifier import check_spell_error
from tqdm import tqdm

def parse_m2(filename):
    """解析m2格式文件

    Arguments:
        filename -- 文件名
    """
    sources = []
    edits = []
    with open(filename, "r") as f:
        chunk = []
        for line in f:
            if line == "\n":
                sources.append(chunk[0])
                edit_list = []
                for s in chunk[2:]:
                    if s[0] != "A": break
                    edit_list.append(s)
                edits.append(edit_list)
                chunk = []
            else:
                chunk.append(line.rstrip("\n"))
        if chunk:
            sources.append(chunk[0])
            edit_list = []
            for s in chunk[2:]:
                if s[0] != "A": break
                edit_list.append(s)
            edits.append(edit_list)
    return sources, edits
        

def validate(edits):
    edits_with_pos = []
    for edit, times in edits:
        _, ss, se = edit.split("|||")[0].split(" ")
        ss, se = int(ss), int(se)
        edits_with_pos.append((ss, se, edit, times))
    edits_with_pos.sort(key=lambda x: (x[0], -times))  # 按照起始位置从小到大排序，起始位置相同，按照编辑出现次数从大到小排序
    final_edits = [edits_with_pos[0][2]]
    for i in range(1, len(edits_with_pos)):
        if edits_with_pos[i][0] < edits_with_pos[i-1][1]:  # 有重叠span
            edits_with_pos[i] = edits_with_pos[i-1]  # 后续的span和前一个span比较
            continue
        if edits_with_pos[i][0] == edits_with_pos[i-1][0] and edits_with_pos[i][1] == edits_with_pos[i-1][1]:
            edits_with_pos[i] = edits_with_pos[i-1]  # 后续的span和前一个span比较
            continue
        final_edits.append(edits_with_pos[i][-2])
    final_final_edits = []
    for e in final_edits:
        if len(final_final_edits) == 0 or e != final_final_edits[-1]:
            final_final_edits.append(e)
    return final_final_edits


def main(args):
    total_edits = []
    for f in args.result_path:
        sources, edits = parse_m2(f)
        total_edits.append(edits)
    with open(args.output_path, "w", encoding="utf-8") as o:
        for i in tqdm(range(len(sources))):
            src = sources[i]
            src_tokens = src.split(" ")[1:]
            edit_candidates = []
            for edits in total_edits:
                edit_candidates.extend(edits[i])
            final_edits = []
            c = Counter(edit_candidates)
            if c["A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0"] > (6 - args.threshold):  # 没有错误
                out = src + "\n" + "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" + "\n\n"
                o.write(out)
                continue
            for k, v in c.items():
                if v >= args.threshold:
                    if k != "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0":
                        final_edits.append((k, v))
                if "|||W|||" in k and v >= args.threshold - 1:  # 词序错误特殊阈值
                    final_edits.append((k, v))
                if "|||S|||" in k and v >= args.threshold - 1:  # 拼写错误特殊阈值
                    _, ss, se = k.split("|||")[0].split(" ")
                    src_span = "".join(src_tokens[int(ss): int(se)])
                    tgt_span = k.split("|||")[2].replace(" ", "")
                    if check_spell_error(src_span, tgt_span):
                        final_edits.append((k, v))
            if final_edits:
                final_edits = validate(final_edits)
                out = src + "\n" + "\n".join(final_edits) + "\n\n"
            else:
                out = src + "\n" + "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0" + "\n\n"
            o.write(out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path',
                        help='Path to the result file.', nargs='+',
                        required=True)
    parser.add_argument('--output_path',
                        help='Path to the output file.',
                        required=True)
    parser.add_argument('-T', '--threshold',
                        help='Threshold.',
                        type=int,
                        default=2)
    args = parser.parse_args()
    main(args)