# -*- coding: utf-8 -*-
"""Download specific GitHub repos (or selected files) via jsDelivr CDN.

Usage:
    python fetch_jsdelivr_repo.py user/repo branch dest_dir [--only-prefix path_prefix ...]
"""
import json
import os
import subprocess
import sys
import urllib.parse
import urllib.request


def list_files(repo, branch):
    url = f"https://data.jsdelivr.com/v1/packages/gh/{repo}@{branch}?structure=flat"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    return [
        (f["name"].lstrip("/"), f.get("size", 0)) for f in data.get("files", [])
    ]


def download_curl(repo, branch, rel, expected_size, dest_dir):
    url = "https://gcore.jsdelivr.net/gh/{repo}@{branch}/{path}".format(
        repo=repo, branch=branch, path=urllib.parse.quote(rel)
    )
    target = os.path.join(dest_dir, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if os.path.exists(target) and os.path.getsize(target) == expected_size:
        return "skip"
    for attempt in range(3):
        proc = subprocess.run(
            [
                "curl.exe", "-s", "-L", "--max-time", "90",
                "-o", target, url,
            ],
            capture_output=True,
        )
        if os.path.exists(target) and os.path.getsize(target) == expected_size:
            return "ok"
    return "fail"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    repo, branch, dest_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    prefixes = sys.argv[4:]
    names = list_files(repo, branch)
    print(f"repo={repo} branch={branch} files={len(names)}", flush=True)
    ok = skip = fail = 0
    for name, size in names:
        if prefixes and not any(name.startswith(p) for p in prefixes):
            continue
        status = download_curl(repo, branch, name, size, dest_dir)
        if status == "skip":
            skip += 1
        elif status == "ok":
            ok += 1
        else:
            fail += 1
        if (ok + skip + fail) % 10 == 0 or fail:
            print(f"  progress: ok={ok} skip={skip} fail={fail} last={name}", flush=True)
    print(f"RESULT downloaded={ok} skipped={skip} failed={fail}", flush=True)


if __name__ == "__main__":
    main()
