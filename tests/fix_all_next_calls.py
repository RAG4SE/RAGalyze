#!/usr/bin/env python3

import re

def fix_all_incomplete_next_calls():
    with open('test_fetch_member_var_query.py', 'r') as f:
        lines = f.readlines()

    fixed_lines = []
    for line in lines:
        # Check if line has incomplete next() call
        if re.search(r'next\(\(m for m in result if m\[\"name\"\] == \"[^\"]+\"\)\)\s*$', line):
            # Add the missing ), None
            line = line.replace(')', '), None)')
        elif re.search(r'next\(\(m for m in result if m\[\"name\"\] == \"[^\"]+\"\)\)', line):
            # This is a complete call, no change needed
            pass
        fixed_lines.append(line)

    with open('test_fetch_member_var_query.py', 'w') as f:
        f.writelines(fixed_lines)

    print("Fixed all incomplete next() calls")

if __name__ == "__main__":
    fix_all_incomplete_next_calls()