"""Exercise N1X source upgrades against a disposable checkout of the local pin."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile

WINDOWS = Path(__file__).resolve().parent
POLICY_PATCH = WINDOWS / "ds41-cache-policies.patch"
HEADERS = ("ds41_cache_policy.hpp", "ds41_expert_read.hpp", "ds41_expert_upload.hpp")


def git(repo, *args):
    return subprocess.check_output(["git", "-C", repo, *args])


def snapshot(repo):
    return {path.relative_to(repo): hashlib.sha256(path.read_bytes()).digest()
            for path in repo.rglob("*") if ".git" not in path.relative_to(repo).parts and path.is_file()}


def check_upgrade(source, test, crlf):
    git(test, "init", "--quiet")
    git(test, "config", "core.autocrlf", str(crlf).lower())
    # Share the pin's objects, never its worktree, refs or index.
    objects = git(source, "rev-parse", "--path-format=absolute", "--git-path", "objects").decode().strip()
    (test / ".git/objects/info/alternates").write_bytes((Path(objects).as_posix() + "\n").encode())
    pin = git(source, "rev-parse", "HEAD").decode().strip()
    git(test, "update-ref", "HEAD", pin)
    paths = {"README.md"}
    for patch in WINDOWS.glob("ds41-*.patch"):
        paths.update(row.split("\t", 2)[2] for row in git(test, "apply", "--numstat", patch).decode().splitlines())
    git(test, "read-tree", pin)
    git(test, "checkout-index", "--", *sorted(paths))

    shell = shutil.which("pwsh") or "powershell"
    def update(error=None, script_dir=WINDOWS):
        result = subprocess.run([shell, "-NoProfile", "-File", script_dir / "update-llama-source.ps1",
                                 "-Src", test], capture_output=True, text=True, errors="replace")
        if error:
            assert result.returncode and error in result.stderr, (result.returncode, result.stderr)
        else:
            assert result.returncode == 0, result.stderr

    update()
    latest = snapshot(test)
    git(test, "apply", "--reverse", "--check", POLICY_PATCH)
    for header in HEADERS:
        assert (test / "src" / header).read_bytes() == (WINDOWS / header).read_bytes()

    # Recreate the five-patch checkout produced before header receipts.
    git(test, "apply", "--reverse", POLICY_PATCH)
    (test / "src" / HEADERS[0]).unlink()
    receipt = test / ".git/ds4-n1x-headers"
    receipt.unlink()
    readme = test / "README.md"
    with readme.open("ab") as f:
        f.write(b"\nlocal staged edit\n")
    git(test, "add", "README.md")
    with readme.open("ab") as f:
        f.write(b"local unstaged edit\n")
    notes = test / "local-notes.txt"
    notes.write_bytes(b"untracked local work\n")
    original_index = (test / ".git/index").read_bytes()
    before = snapshot(test)

    for relative in ("src/llama-moe-cache.cpp", "src/" + HEADERS[1], "src/" + HEADERS[0]):
        path = test / relative
        saved = path.read_bytes() if path.exists() else None
        path.write_bytes((saved or b"") + b"\n// local source edit\n")
        conflict = snapshot(test)
        update("Unrecognized changes")
        assert snapshot(test) == conflict
        assert (test / ".git/index").read_bytes() == original_index
        assert not receipt.exists()
        if saved is None:
            path.unlink()
        else:
            path.write_bytes(saved)

    update()
    latest[Path("README.md")] = before[Path("README.md")]
    latest[Path("local-notes.txt")] = before[Path("local-notes.txt")]
    assert snapshot(test) == latest
    update()
    assert snapshot(test) == latest
    assert (test / ".git/index").read_bytes() == original_index

    # A later build can replace a previously installed header even when its
    # previous revision existed only in a developer's working copy.
    local_build = test.parent / ("build-" + str(crlf))
    local_build.mkdir()
    for file in [WINDOWS / "update-llama-source.ps1", *WINDOWS.glob("ds41-*.patch"),
                 *(WINDOWS / header for header in HEADERS)]:
        shutil.copyfile(file, local_build / file.name)
    git(local_build, "init", "--quiet")
    git(local_build, "add", *HEADERS)
    git(local_build, "-c", "user.name=Upgrade test", "-c", "user.email=test@localhost",
        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "Original N1X headers")
    receipt.unlink()
    for version in (b"// local revision one\n", b"// local revision two\n"):
        header = local_build / HEADERS[1]
        with header.open("ab") as f:
            f.write(version)
        update(script_dir=local_build)
        assert (test / "src" / header.name).read_bytes() == header.read_bytes()
    with (test / "src" / HEADERS[1]).open("ab") as f:
        f.write(b"// edited after install\n")
    conflict = snapshot(test)
    record = receipt.read_bytes()
    update("Unrecognized changes", script_dir=local_build)
    assert snapshot(test) == conflict and receipt.read_bytes() == record
    assert (test / ".git/index").read_bytes() == original_index
    print(f"core.autocrlf={crlf}: upgrade, conflicts, header revisions and index preservation passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="local pinned llama.cpp checkout")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        for crlf in (True, False):
            test = Path(tmp) / str(crlf)
            test.mkdir()
            check_upgrade(args.source.resolve(), test, crlf)
