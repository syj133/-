#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""并发下载数学建模优秀论文（GitHub raw / Gitee raw / 任意 URL 列表）。

用法：
  python download_papers.py github <owner/repo> <branch> <输出目录> [--include 子串]... [--workers N] [--limit N]
  python download_papers.py urllist <url列表.tsv> <输出目录> [--workers N]

url 列表文件每行：<完整URL>\t<相对输出路径>
"""

import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def http_get(url, timeout=90, retries=3):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


_BAD_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_segment(seg):
    return _BAD_CHARS.sub("_", seg)


def looks_ok(path):
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    if size == 0:
        return False
    with open(path, "rb") as f:
        head = f.read(8)
    low = path.lower()
    if low.endswith(".pdf"):
        return head.startswith(b"%PDF")
    if low.endswith((".doc", ".docx")):
        return head.startswith(b"PK") or head.startswith(b"\xd0\xcf\x11\xe0")
    return True


def download_one(item):
    url, dest = item
    try:
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    except OSError:
        pass
    if looks_ok(dest):
        return ("skip", url, dest, "")
    try:
        data = http_get(url)
        with open(dest, "wb") as f:
            f.write(data)
        if looks_ok(dest):
            return ("ok", url, dest, "")
        return ("bad", url, dest, "魔数校验失败")
    except Exception as e:  # noqa: BLE001
        return ("fail", url, dest, str(e))


def run_downloads(items, args, label):
    total = len(items)
    print(f"[{label}] 待下载 {total} 个文件，workers={args.workers}")
    stats = {"ok": 0, "skip": 0, "bad": 0, "fail": 0}
    problems = []
    if total == 0:
        print("[%s] 无匹配文件" % label)
        return stats, problems
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, (status, url, dest, err) in enumerate(ex.map(download_one, items), 1):
            stats[status] = stats.get(status, 0) + 1
            if status in ("bad", "fail"):
                problems.append((status, url, dest, err))
            if i % 50 == 0 or i == total:
                print(f"[{label}] 进度 {i}/{total}  ok={stats['ok']} skip={stats['skip']} "
                      f"bad={stats['bad']} fail={stats['fail']}")
    return stats, problems


def cmd_github(args):
    api = f"https://api.github.com/repos/{args.repo}/git/trees/{args.branch}?recursive=1"
    data = json.loads(http_get(api).decode("utf-8"))
    tree = data.get("tree")
    if tree is None:
        raise RuntimeError("GitHub API 返回异常: " + json.dumps(data, ensure_ascii=False)[:500])
    items = []
    docs = 0
    for t in tree:
        if t.get("type") != "blob":
            continue
        p = t["path"]
        low = p.lower()
        if not low.endswith((".pdf", ".doc", ".docx")):
            continue
        docs += 1
        if args.include and not any(s in p for s in args.include):
            continue
        if args.match_regex and not re.search(args.match_regex, p):
            continue
        if args.exclude_regex and re.search(args.exclude_regex, p):
            continue
        rel = "/".join(sanitize_segment(s) for s in p.split("/"))
        url = f"https://raw.githubusercontent.com/{args.repo}/{args.branch}/{encode_path(p)}"
        dest = os.path.join(args.outdir, rel)
        items.append((url, dest))
    if args.limit:
        items = items[: args.limit]
    stats, problems = run_downloads(items, args, args.repo)
    write_report(args.outdir, stats, problems)


def cmd_urllist(args):
    items = []
    with open(args.listfile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            url, rel = parts[0], parts[1]
            dest = os.path.join(args.outdir, *[sanitize_segment(s) for s in rel.split("/")])
            items.append((url, dest))
    stats, problems = run_downloads(items, args, "urllist")
    write_report(args.outdir, stats, problems)


def encode_path(path):
    return "/".join(urllib.parse.quote(s, safe="") for s in path.split("/"))


def write_report(outdir, stats, problems):
    report = os.path.join(outdir, "_下载报告.txt")
    with open(report, "w", encoding="utf-8") as f:
        f.write("ok=%d skip=%d bad=%d fail=%d\n" % (
            stats.get("ok", 0), stats.get("skip", 0),
            stats.get("bad", 0), stats.get("fail", 0)))
        for status, url, dest, err in problems:
            f.write(f"[{status}] {url} -> {dest}  {err}\n")
    print("报告已写入:", report)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("github")
    g.add_argument("repo")
    g.add_argument("branch")
    g.add_argument("outdir")
    g.add_argument("--include", action="append", default=[])
    g.add_argument("--match-regex", default="")
    g.add_argument("--exclude-regex", default="")
    g.add_argument("--workers", type=int, default=8)
    g.add_argument("--limit", type=int, default=0)
    u = sub.add_parser("urllist")
    u.add_argument("listfile")
    u.add_argument("outdir")
    u.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "github":
        cmd_github(args)
    else:
        cmd_urllist(args)


if __name__ == "__main__":
    sys.exit(main())
