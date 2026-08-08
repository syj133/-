#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验优秀论文目录下所有文件（PDF/DOC/DOCX 魔数与大小），打印异常文件。"""

import os
import sys


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "优秀论文"
    n = 0
    total = 0
    bad = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f == "_下载报告.txt" or f.lower().endswith((".md", ".txt")):
                continue
            p = os.path.join(dirpath, f)
            n += 1
            size = os.path.getsize(p)
            total += size
            with open(p, "rb") as fh:
                head = fh.read(8)
            low = f.lower()
            ok = False
            if size == 0:
                ok = False
            elif low.endswith(".pdf"):
                ok = head.startswith(b"%PDF")
            elif low.endswith((".doc", ".docx")):
                ok = head.startswith(b"PK") or head.startswith(b"\xd0\xcf\x11\xe0")
            else:
                ok = True
            if not ok:
                bad.append(p)
    print(f"files={n} total_MB={round(total / 1048576, 1)} bad={len(bad)}")
    for b in bad:
        print("BAD:", b)


if __name__ == "__main__":
    main()
