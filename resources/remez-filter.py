#!/usr/bin/env python3

import re
import sys

try:
    while True:
        s = input()
        s = re.sub(r"e([+-][0-9]*)`38", r"`38*10^\1", s)
        s = re.sub(r"e([+-][0-9]*)", r"*10^\1", s)
        sys.stdout.write(s)
        sys.stdout.write('\n')
        sys.stdout.flush()
except EOFError:
    pass
