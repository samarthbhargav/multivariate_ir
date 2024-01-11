import argparse
import json

import os

from collections import OrderedDict
from typing import List, Dict

import numpy as np

import copy


def prod_dict(hyp_dict) -> List[Dict]:
    # single element values
    fixed_values = {k: v for (k, v) in hyp_dict.items() if not isinstance(v, list)}

    # grid values
    varying_values = {k: v for (k, v) in hyp_dict.items() if k not in fixed_values}

    hyperparameters = OrderedDict(
        sorted(varying_values.items(), key=lambda _: _[0]))
    keys = list(hyperparameters.keys())

    indices = [len(values) for (arg, values) in hyperparameters.items()]
    choices = []
    for idx_choice in np.ndindex(*indices):
        # copy over the fixed values
        one_choice = copy.deepcopy(fixed_values)
        # pick current values
        for arg_idx, (arg, val_idx) in enumerate(zip(keys, idx_choice)):
            one_choice[arg] = hyperparameters[arg][val_idx]
        choices.append(one_choice)

    return choices


def make_cmd(params: Dict) -> str:
    cmd = ""
    for p, v in params.items():
        if isinstance(v, bool):
            # only add if it's True
            if v:
                cmd += f"--{p}  "
        else:
            cmd += f"--{p} {v} "

    return " " + cmd


def make_params_file_from_commands(dest, script_name, commands, job_template, job_config):
    with open(job_template, "r") as reader:
        job_template = ""
        for line in reader:
            job_template += line

    job_file = dest + ".job"
    hparams_file = dest + ".params"

    n_jobs = 0
    with open(hparams_file, "w") as writer:
        for v in commands:
            writer.write(v + '\n')
            n_jobs += 1

    print(f"N Jobs: {n_jobs}")
    final_cmd = f"{script_name} $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)"

    with open(job_file, "w") as writer:
        job = job_template.format(n_jobs=n_jobs, hyperparams_file=hparams_file, cmd=final_cmd, **job_config)
        writer.write(job)


def make_commands(args, exp_root_folder, exp_name):
    with open(args.model_config) as reader:
        model_params = json.load(reader)
    commands = []

    for i, m in enumerate(prod_dict(model_params)):
        model_root = os.path.join(exp_root_folder, exp_name, str(i))

        assert "output_dir" not in m
        m["output_dir"] = model_root
        commands.append(make_cmd(m))

    return commands


def make_params_file(args, exp_root_folder, exp_name):
    with open(args.job_config) as reader:
        job_config = json.load(reader)
    commands = make_commands(args, exp_root_folder=exp_root_folder, exp_name=exp_name)

    print(f'Generated {len(commands)} experiments:')

    with open(args.job_template, "r") as reader:
        job_template = ""
        for line in reader:
            job_template += line

    job_file = args.dest + ".job"
    hparams_file = args.dest + ".params"

    n_jobs = 0
    with open(hparams_file, "w") as writer:
        for v in commands:
            writer.write(v + '\n')
            n_jobs += 1

    print(f"N Jobs: {n_jobs}")
    final_cmd = f"{args.cmd} $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)"

    with open(job_file, "w") as writer:
        job = job_template.format(n_jobs=n_jobs, hyperparams_file=hparams_file, cmd=final_cmd, **job_config)
        writer.write(job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("params",
                                     description="Utility for generating job scripts")

    parser.add_argument("--model_config", help="(json) location of model config", required=True)
    parser.add_argument("--job_config", help="(json) location of job config", required=True)

    parser.add_argument("--dest", help="location to dump script + hparams", required=True)

    parser.add_argument("--job_template", help="location of job template", default="./templates/job_template.sh")
    parser.add_argument("--exp_root_folder", required=True)
    parser.add_argument("--exp_name", required=True)
    parser.add_argument("--cmd", help="script name / path", required=True)

    args = parser.parse_args()

    make_params_file(args, args.exp_root_folder, args.exp_name)
