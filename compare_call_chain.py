#!/usr/bin/env python3

import sys

def read_file_lines(file_path):
    with open(file_path, 'r') as f:
        return {line.strip() for line in f if line.strip()}

set1 = read_file_lines(sys.argv[1])
set2 = read_file_lines(sys.argv[2])

only_in_file1 = set1 - set2
only_in_file2 = set2 - set1

if only_in_file1:
    print(">"*20 + f"only in {sys.argv[1]}:" + "<"*20)
    for line in sorted(only_in_file1):
        print(line)

if only_in_file2:
    print(">"*20 + f"only in {sys.argv[2]}:" + "<"*20)
    for line in sorted(only_in_file2):
        print(line)