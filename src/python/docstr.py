#!/usr/bin/env python3

import textwrap
import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input RST file")
    parser.add_argument("output", help="Output header file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    name = None
    buf = []
    strings = {}

    def do_append():
        if not name:
            return
        while len(buf) > 0 and len(buf[0]) == 0:
            del buf[0]
        while len(buf) > 0 and len(buf[-1]) == 0:
            del buf[-1]
        buf2 = [b.strip() if len(b.strip()) == 0 else b for b in buf]
        strings[name] = "\n".join(buf2)

    for line in lines:
        line = line.rstrip()
        if line.startswith(".."):
            if line.startswith(".. topic:: "):
                do_append()
                buf = []
                name = line[11:]
        else:
            buf.append(line)

    do_append()

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(
            f"// This file is automatically generated from {args.input}, please do not edit.\n\n"
        )

        f.write("#if defined(__GNUC__)\n")
        f.write("#  pragma GCC diagnostic push\n")
        f.write('#  pragma GCC diagnostic ignored "-Wunused-variable"\n')
        f.write("#endif\n\n")

        for k, v in strings.items():
            f.write(f'static const char *doc_{k} = R"(\n')
            f.write(textwrap.dedent(v).strip())
            f.write(')";\n\n')


if __name__ == "__main__":
    main()
