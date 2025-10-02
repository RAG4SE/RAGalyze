#!/usr/bin/env python3

# Script to fix all incomplete next() calls in the test file

import re

def fix_next_calls():
    with open('test_fetch_member_var_query.py', 'r') as f:
        content = f.read()

    # Pattern to match incomplete next() calls
    pattern = r'next\(\(m for m in result if m\[\"name\"\] == \"([^\"]+)\"\)\)'

    def replace_func(match):
        var_name = match.group(1)
        return f'next((m for m in result if m["name"] == "{var_name}"), None)'

    content = re.sub(pattern, replace_func, content)

    with open('test_fetch_member_var_query.py', 'w') as f:
        f.write(content)

    print("Fixed all next() calls")

if __name__ == "__main__":
    fix_next_calls()