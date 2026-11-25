"""Publish only site/uav-sim-lab to the authenticated user's GitHub Pages repo.

Uses Git Credential Manager in memory. Does not print or persist credentials.
Existing front-page files are never modified. No front page is designed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.error import HTTPError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never")


def credentials():
    result = subprocess.run(
        ["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
        text=True, capture_output=True, env=ENV, check=True,
    )
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if not fields.get("password"):
        raise RuntimeError("Sign into GitHub through Git Credential Manager first")
    return fields["password"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publish", action="store_true", help="Create/update and publish the lab")
    args = parser.parse_args()
    token = credentials()

    def api(path, method="GET", data=None):
        headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json", "User-Agent": "UAV-Lab-Publisher", "X-GitHub-Api-Version": "2022-11-28"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request("https://api.github.com" + path, method=method, headers=headers, data=json.dumps(data).encode() if data is not None else None)
        try:
            with urlopen(request, timeout=40) as response:
                content = response.read()
                return json.loads(content) if content else {}
        except HTTPError as exc:
            if exc.code == 404:
                return None
            # Avoid displaying headers or any credential-bearing request object.
            raise RuntimeError(f"GitHub {method} {path} failed with HTTP {exc.code}: {exc.read().decode()[:700]}") from None

    user = api("/user")
    owner = user["login"]
    name = owner + ".github.io"
    repo = owner + "/" + name
    metadata = api("/repos/" + repo)
    if args.publish:
        if metadata is None:
            metadata = api("/user/repos", "POST", {"name": name, "description": "Personal project pages. UAV simulation lab at /uav-sim-lab/.", "private": False, "auto_init": False})
            print("Created repository:", metadata["html_url"], flush=True)
        branch = metadata.get("default_branch", "main")
        with tempfile.TemporaryDirectory(prefix="uav-pages-publish-") as temp:
            checkout = Path(temp) / "pages"
            heads = subprocess.check_output(["git", "ls-remote", "--heads", metadata["clone_url"], "refs/heads/" + branch], env=ENV).strip()
            if heads:
                subprocess.run(["git", "clone", "--depth", "1", "--branch", branch, metadata["clone_url"], str(checkout)], check=True, env=ENV, capture_output=True)
            else:
                checkout.mkdir()
                subprocess.run(["git", "init", "-b", branch], cwd=checkout, check=True, env=ENV, capture_output=True)
                subprocess.run(["git", "remote", "add", "origin", metadata["clone_url"]], cwd=checkout, check=True, env=ENV)
            shutil.copytree(ROOT / "site" / "uav-sim-lab", checkout / "uav-sim-lab", dirs_exist_ok=True)
            if not (checkout / ".nojekyll").exists():
                (checkout / ".nojekyll").write_text("", encoding="utf-8")
            subprocess.run(["git", "config", "user.name", user.get("name") or owner], cwd=checkout, check=True)
            subprocess.run(["git", "config", "user.email", f"{user['id']}+{owner}@users.noreply.github.com"], cwd=checkout, check=True)
            subprocess.run(["git", "add", "--", "uav-sim-lab", ".nojekyll"], cwd=checkout, check=True)
            changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=checkout).returncode != 0
            if changed:
                subprocess.run(["git", "commit", "-m", "Build UAV simulation lab with GitHub Pages interface"], cwd=checkout, check=True, capture_output=True)
                subprocess.run(["git", "push", "-u", "origin", branch], cwd=checkout, check=True, env=ENV, capture_output=True)
                print("Published source:", repo, flush=True)
            else:
                print("Published source is already current.", flush=True)
        pages = api("/repos/" + repo + "/pages")
        if pages is None:
            pages = api("/repos/" + repo + "/pages", "POST", {"build_type": "legacy", "source": {"branch": branch, "path": "/"}})
            print("Enabled GitHub Pages.", flush=True)
        elif pages.get("source") != {"branch": branch, "path": "/"} or pages.get("build_type") == "workflow":
            raise RuntimeError("Existing Pages configuration differs; source was pushed but publication settings were preserved")
    pages = api("/repos/" + repo + "/pages") if metadata else None
    latest = api("/repos/" + repo + "/pages/builds/latest") if pages else None
    print(json.dumps({"repository": repo, "url": f"https://{owner.lower()}.github.io/uav-sim-lab/", "pages_status": pages.get("status") if pages else None, "build_status": latest.get("status") if latest else None, "commit": latest.get("commit") if latest else None, "error": latest.get("error") if latest else None}, indent=2), flush=True)


if __name__ == "__main__":
    main()
