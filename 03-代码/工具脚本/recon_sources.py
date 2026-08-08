#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""侦察其他数学建模论文来源（GitHub 仓库搜索 + 指定仓库文件树）。"""

import json
import time
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def search_repos(query):
    url = "https://api.github.com/search/repositories?q=" + urllib.parse.quote(query) + "&per_page=15"
    try:
        d = json.loads(get(url))
        print(f"== search: {query}  total={d.get('total_count')}")
        for it in d.get("items", []):
            desc = (it.get("description") or "").replace("\n", " ")[:90]
            print(f"  {it['full_name']} | {it['default_branch']} | {it['size'] // 1024}MB | {desc}")
    except Exception as e:  # noqa: BLE001
        print(f"ERR search {query}: {e}")


def tree(repo, branch, show_paths=False, match=None, limit=100):
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    try:
        d = json.loads(get(url))
        blobs = [
            t["path"]
            for t in d.get("tree", [])
            if t.get("type") == "blob" and t["path"].lower().endswith((".pdf", ".doc", ".docx"))
        ]
        print(f"== tree {repo}@{branch}: docs={len(blobs)}")
        top = {}
        for p in blobs:
            seg = p.split("/")[0]
            top[seg] = top.get(seg, 0) + 1
        print("  top-level:", dict(sorted(top.items())))
        if match:
            hits = [p for p in blobs if match.lower() in p.lower()]
            print(f"  match '{match}': {len(hits)} docs")
            for p in hits[:limit]:
                print("  " + p)
        elif show_paths:
            for p in blobs[:limit]:
                print("  " + p)
    except Exception as e:  # noqa: BLE001
        print(f"ERR tree {repo}: {e}")


def breakdown(repo, branch, prefix):
    url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
    try:
        d = json.loads(get(url))
        blobs = [
            t["path"]
            for t in d.get("tree", [])
            if t.get("type") == "blob"
            and t["path"].startswith(prefix)
            and t["path"].lower().endswith((".pdf", ".doc", ".docx"))
        ]
        sub = {}
        for p in blobs:
            rest = p[len(prefix):].lstrip("/")
            seg = rest.split("/")[0]
            sub[seg] = sub.get(seg, 0) + 1
        print(f"== breakdown {repo}: {prefix}  docs={len(blobs)}")
        print("  ", dict(sorted(sub.items())))
    except Exception as e:  # noqa: BLE001
        print(f"ERR breakdown {repo}: {e}")


def main():
    for q in ["MCM-ICM", "数学建模论文", "数学建模优秀论文", "数学建模竞赛论文"]:
        search_repos(q)
    tree("personqianduixue/Math_Model", "master", match="2023")
    tree("yan-fanyu/CUMCM-Paper-And-SourceCode", "main", show_paths=True)
    tree("dick20/MCM-ICM", "master", show_paths=True, limit=150)
    breakdown("personqianduixue/Math_Model", "master", "2-1国赛题目+论文")


if __name__ == "__main__":
    main()
