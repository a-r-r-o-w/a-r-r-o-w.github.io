import datetime
import os
import re
import subprocess
import sys
import zoneinfo

TIMEZONE = zoneinfo.ZoneInfo("America/Los_Angeles")
PATTERN = re.compile(r"^origin/publish/(\d{4}-\d{2}-\d{2})-.+")


def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout.strip()


def main():
    today = datetime.datetime.now(TIMEZONE).date().isoformat()
    git("fetch", "origin", "--prune")
    branches = git("branch", "-r", "--format=%(refname:short)").split()
    due = sorted(b for b in branches if (m := PATTERN.match(b)) and m.group(1) <= today)

    merged, failed = [], []
    for branch in due:
        try:
            git("merge", "--no-ff", branch, "-m", f"publish: {branch.removeprefix('origin/')}")
            merged.append(branch)
        except subprocess.CalledProcessError as e:
            subprocess.run(["git", "merge", "--abort"])
            failed.append(branch)
            print(f"::error::could not merge {branch}: {(e.stdout + e.stderr).strip()}")

    if merged:
        git("push", "origin", "HEAD:main")
        for branch in merged:
            git("push", "origin", "--delete", branch.removeprefix("origin/"))
            print(f"published {branch}")
    else:
        print("nothing to publish")

    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"merged={'true' if merged else 'false'}\n")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
