import argparse
import shutil
import os
from pathlib import Path
from tqdm import tqdm


def walk_through_files(path, fnames_to_include):
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            full_path = os.path.join(dirpath, filename)
            if any([filename == _ for _ in fnames_to_include]):
                yield dirpath, full_path
            else:
                print(f"skipping {full_path}")
                continue


RES_FILES = {
    "bier_test_cqadupstack-android.json",
    "bier_test_cqadupstack-english.json",
    "bier_test_cqadupstack-gaming.json",
    "bier_test_cqadupstack-gis.json",
    "bier_test_cqadupstack-mathematica.json",
    "bier_test_cqadupstack-physics.json",
    "bier_test_cqadupstack-programmers.json",
    "bier_test_cqadupstack-stats.json",
    "bier_test_cqadupstack-tex.json",
    "bier_test_cqadupstack-unix.json",
    "bier_test_cqadupstack-webmasters.json",
    "bier_test_cqadupstack-wordpress.json",
    "bier_test_fiqa.json",
    "bier_test_trec-covid.json",
    "dev_msmarco-passage.json",
    "dev_scifact.json",
    "dl19.json",
    "dl20.json",
}

QPP_FILES = {
    "bier_cqadupstack-android-corpus.txt", "bier_cqadupstack-android_test.txt",
    "bier_cqadupstack-english-corpus.txt", "bier_cqadupstack-english_test.txt",
    "bier_cqadupstack-gaming-corpus.txt", "bier_cqadupstack-gaming_test.txt",
    "bier_cqadupstack-gis-corpus.txt", "bier_cqadupstack-gis_test.txt", "bier_cqadupstack-physics-corpus.txt",
    "bier_cqadupstack-physics_test.txt", "bier_cqadupstack-programmers-corpus.txt",
    "bier_cqadupstack-programmers_test.txt", "bier_cqadupstack-stats-corpus.txt",
    "bier_cqadupstack-stats_test.txt", "bier_cqadupstack-tex-corpus.txt", "bier_cqadupstack-tex_test.txt",
    "bier_cqadupstack-unix-corpus.txt", "bier_cqadupstack-unix_test.txt",
    "bier_cqadupstack-webmasters-corpus.txt", "bier_cqadupstack-webmasters_test.txt",
    "bier_cqadupstack-wordpress-corpus.txt", "bier_cqadupstack-wordpress_test.txt", "bier_fiqa-corpus.txt",
    "bier_fiqa_test.txt", "bier_trec-covid-corpus.txt", "bier_trec-covid_test.txt", "corpus_scifact.txt",
    "dev_scifact.txt", "dl19_msmarco-passage.txt", "dl20_msmarco-passage.txt", "msmarco-dev.txt",
    "msmarco-dl19.txt", "msmarco-dl20.txt", "msmarco-med-corpus-0.1.txt", "msmarco-med-corpus-0.5.txt",
    "msmarco-med-corpus-1.0.txt", "msmarco-med-corpus.txt", "msmarco-perturbed-0.1-train.txt",
    "msmarco-perturbed-0.1-validation.txt", "msmarco-perturbed-0.5-train.txt",
    "msmarco-perturbed-0.5-validation.txt", "msmarco-perturbed-1.0-train.txt",
    "msmarco-perturbed-1.0-validation.txt", "msmarco-train.txt", "msmarco-validation.txt"
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser("snellius_state_models.py")
    parser.add_argument("--input_dir", help="path to input dir", required=True)
    parser.add_argument("--output_dir", help="path to staging dir", required=True)

    args = parser.parse_args()
    # gather files to copy over
    print(f"gathering files form {args.input_dir}")
    inp = Path(args.input_dir)
    out = Path(args.output_dir)

    files_to_transfer = list(walk_through_files(inp, fnames_to_include=QPP_FILES.union(RES_FILES)))
    for dir_path, file_path in tqdm(files_to_transfer, desc="transferring files"):
        path_pref_removed = file_path[len(os.path.abspath(args.input_dir)) + 1:]
        dest_dir = out / "_".join(path_pref_removed[:-1].split("/"))
        os.makedirs(os.path.dirname(dest_dir), exist_ok=True)
        dest_path = dest_dir / file_path.split("/")[-1]
        print(f"transferring: {file_path} to {dest_path}")
        # shutil.copy(file_path, dest_path)
