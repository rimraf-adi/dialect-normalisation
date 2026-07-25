import os
import sys
import tarfile
import time
from pathlib import Path


def extract_tar_gz_files(target_dir: Path):
    # Find all .tar.gz and .tgz files in the target directory
    archive_files = sorted(
        [
            f
            for f in target_dir.iterdir()
            if f.is_file() and (f.name.endswith(".tar.gz") or f.name.endswith(".tgz"))
        ]
    )

    if not archive_files:
        print(f"No .tar.gz or .tgz files found in {target_dir.resolve()}")
        return

    print(f"Found {len(archive_files)} archive file(s) to extract in {target_dir.resolve()}:\n")
    for idx, archive_path in enumerate(archive_files, 1):
        file_size_mb = archive_path.stat().st_size / (1024 * 1024)
        print(f"[{idx}/{len(archive_files)}] Extracting: {archive_path.name} ({file_size_mb:.2f} MB)...")
        
        start_time = time.time()
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                # Support Python 3.12+ safe extraction filter if available
                if hasattr(tarfile, "data_filter"):
                    tar.extractall(path=target_dir, filter="data")
                else:
                    tar.extractall(path=target_dir)
            
            elapsed = time.time() - start_time
            print(f"    Done in {elapsed:.2f}s.\n")
        except Exception as e:
            print(f"    Error extracting {archive_path.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    cwd = Path.cwd()
    extract_tar_gz_files(cwd)
