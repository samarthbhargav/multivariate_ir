import argparse
import shutil
import os
from pathlib import Path
from tqdm import tqdm


def walk_through_files(path, include_extensions):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if any([filename.endswith(_) for _ in include_extensions]):
                yield dirpath, full_path
            else:
                print(f"skipping {full_path}")
                continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser("snellius_state_models.py")
    parser.add_argument("--input_dir", help="path to input dir", required=True)
    parser.add_argument("--output_dir", help="path to staging dir", required=True)
    args = parser.parse_args()
    # gather files to copy over
    print(f"gathering files form {args.input_dir}")
    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    INCLUDE_EXT = {".run", ".json"}
    DO_NOT_OVERWRITE = True

    files_to_transfer = list(walk_through_files(inp, include_extensions=INCLUDE_EXT))

    for dir_path, file_path in tqdm(files_to_transfer, desc="transferring files"):
        path_pref_removed = file_path[len(os.path.abspath(args.input_dir)) + 1:]
        dest_path = out / path_pref_removed
        if os.path.exists(dest_path) and DO_NOT_OVERWRITE:
            print(f"file {dest_path} already exists!")
            continue

        print(f"transferring: {file_path} to {dest_path}")
        # os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        # shutil.copy(file_path, dest_path)
