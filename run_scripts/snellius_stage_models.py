import argparse
import shutil
import os
from pathlib import Path
from tqdm import tqdm


def walk_through_files(path, exclude_fnames, exclude_extensions, exclude_folder_prefixes):
    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs = dirpath.split("/")
        skip = False
        for dir_part in dirs:
            if any([dir_part.startswith(_) for _ in exclude_folder_prefixes]):
                skip = True
                break
        if skip:
            print(f"skipping {dirpath}")
            continue
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if filename in exclude_fnames:
                print(f"skipping {full_path}")
                continue
            if any([filename.endswith(_) for _ in exclude_extensions]):
                print(f"skipping {full_path}")
                continue

            yield full_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser("snellius_state_models.py")
    parser.add_argument("--input_dir", help="path to input dir", required=True)
    parser.add_argument("--output_dir", help="path to staging dir", required=True)
    args = parser.parse_args()
    # gather files to copy over
    print(f"gathering files form {args.input_dir}")
    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    EXCL_FILES = {"eval_result.json", "trainer_state.csv", ""}
    EXCL_EXT = {".pkl", ".log", ".run", ".zip"}
    EXCL_FOLDERS_PREFIX = {"checkpoint", "runs", "mean", "qpp", "original", "updated", "rep"}
    files_to_transfer = list(walk_through_files(inp, exclude_fnames=EXCL_FILES, exclude_extensions=EXCL_EXT,
                                                exclude_folder_prefixes=EXCL_FOLDERS_PREFIX))

    for file in tqdm(files_to_transfer, desc="transferring files"):
        print(f"transferring {file}")
