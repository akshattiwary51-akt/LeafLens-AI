import subprocess
from pathlib import Path
import re


REPO_DIR = Path("data/raw/PlantDoc-Dataset")

OUTPUT_DIR = Path(
    "data/recovered_test"
)


def safe_filename(filename):
    """
    Convert a Git filename into a Windows-safe filename.
    """
    filename = filename.replace("?", "_")
    filename = filename.replace(":", "_")
    filename = filename.replace("*", "_")
    filename = filename.replace('"', "_")
    filename = filename.replace("<", "_")
    filename = filename.replace(">", "_")
    filename = filename.replace("|", "_")

    return filename


def main():

    print("=" * 70)
    print("Recovering Windows-incompatible PlantDoc test files")
    print("=" * 70)

    if not REPO_DIR.exists():
        raise FileNotFoundError(
            f"Repository not found: {REPO_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ---------------------------------------------------------
    # Get all test paths tracked by Git
    # ---------------------------------------------------------

    result = subprocess.run(
        ["git", "ls-files", "test"],
        cwd=REPO_DIR,
        capture_output=True,
        text=True,
        check=True
    )

    git_files = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]

    print(f"Git-tracked test files: {len(git_files)}")

    recovered = []

    for git_path in git_files:

        # Windows cannot represent filenames containing '?'
        if "?" not in git_path:
            continue

        path_parts = Path(git_path).parts

        if len(path_parts) < 3:
            continue

        class_name = path_parts[1]
        original_filename = path_parts[-1]

        new_filename = safe_filename(
            original_filename
        )

        output_path = (
            OUTPUT_DIR
            / class_name
            / new_filename
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # -----------------------------------------------------
        # Extract blob directly from Git
        # -----------------------------------------------------

        print()
        print(f"Recovering:")
        print(f"  Git path : {git_path}")
        print(f"  New path : {output_path}")

        blob = subprocess.run(
            [
                "git",
                "show",
                f"HEAD:{git_path}"
            ],
            cwd=REPO_DIR,
            capture_output=True,
            check=True
        )

        output_path.write_bytes(
            blob.stdout
        )

        recovered.append({
            "git_path": git_path,
            "output_path": str(output_path),
            "class": class_name
        })

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("RECOVERY COMPLETE")
    print("=" * 70)

    print(
        f"Recovered files: {len(recovered)}"
    )

    for item in recovered:
        print(
            f"[{item['class']}] "
            f"{item['output_path']}"
        )

    print()
    print(
        f"Recovery directory: {OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()