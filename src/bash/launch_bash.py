import argparse
import subprocess
import json
import os

def launch_slurm_job(script_path, env):
    command = ['sbatch', script_path]
    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode == 0:
        job_id = extract_job_id(stdout)
        print(stdout.decode("utf-8"))
        print(f'Successfully submitted SLURM job with ID {job_id}')
    else:
        print(f'Error submitting SLURM job: {stderr.decode("utf-8")}')


def extract_job_id(sbatch_output):
    output_lines = sbatch_output.decode("utf-8").split('\n')
    # The last line of the sbatch output contains the job ID
    job_id_line = output_lines[-2]
    job_id = job_id_line.split()[-1]
    return job_id


# Configuration file
parser = argparse.ArgumentParser(description="Configuration File")

parser.add_argument(

    "-e", "--experiment_config",
    help="Experiment Running config, select from a set of possible solutions: LoCo validation strategy, 5 fold cross validation, Hold-out",
    type=str,
    default='5', choices=['5', '10', 'L', 'H']
)
parser.add_argument(
    "-d", "--database",
    help="Select the type of modality: AFC, MC, JSRT",
    type=str,
    default='MC', choices=['AFC', 'AFC_death', 'CoCross', 'MC', 'CAR', 'NY', 'NY_icu', 'NY_all', 'NY_all_icu']

)



parser.add_argument(
    "-c", "--config_file",
    help="Configuration file",
    type=str,
    default='VANILLA', choices=['VANILLA', 'FSL', 'FCSL', 'FEXT', 'ML'],
)
parser.add_argument("--model_names", help="model_name",
    default=["biomedclip"])



args = parser.parse_args()

if __name__ == "__main__":


    if args.config_file == 'VANILLA':
        file_config = './configs/bash_experiments/multiple_experiments_run.json'
    elif args.config_file == 'FEXT':
        file_config = './configs/bash_experiments/features_extraction_run.json'
    elif args.config_file == 'FSL':
        file_config = './configs/bash_experiments/multiple_experiments_run_FSL.json'
    elif args.config_file == 'FCSL':
        file_config = './configs/bash_experiments/multiple_experiments_run_CFSL.json'
    else:
        raise ValueError('Config file not found')
    with open(file_config, 'r') as data_file:
        json_data = data_file.read()


    experiment_list = json.loads(json_data)
    print(experiment_list)
    processes = []  # List to store the subprocess instances

    # validation name change:
    dictionary_name_change = {'5': '5fold', 'L': 'loCo', 'H': 'hold-out', '10': '10fold'}

    for i, exp_config in enumerate(experiment_list):
        # pick model



        if exp_config['models'] == 'all':
            args.model_names = \
            ["biomedclip",
             "pubmedclip",
             "clip_large",
             "medclip_resnet50",
             "medclip_vision",
             "resnet_50",
             "resnet_18",
             "vitb14_pretrain",
             "vits14_pretrain",
             "vitl14_pretrain",
             "dense121"
            ]
        elif exp_config['models'] == 'single':
            args.model_names = ["resnet_50"]
        elif exp_config['models'] == 'two':
            args.model_names = ["vitl14_pretrain", "medclip_vision"]
        elif exp_config['models'] == 'debug':
            args.model_names = ["medclip_resnet50"]

        elif exp_config['models'] == 'peft_models':
            args.model_names = \
            [
             "vitb14_pretrain",
             "vits14_pretrain",
             "vitl14_pretrain",
             "biomedclip",
             "medclip_vision",
             "pubmedclip",
             "clip_large",
            ]

        elif exp_config['models'] == 'peft_models_minus_MEDCLIP':
            args.model_names = \
            [
             "vitb14_pretrain",
             "vits14_pretrain",
             "vitl14_pretrain",
             "biomedclip",
             "pubmedclip",
             "clip_large",
            ]

        elif exp_config['models'] == 'CNN':
            args.model_names = \
            ["resnet_50",
             "resnet_18",
             "dense121",
             "medclip_resnet50",
            ]



        else:
            args.model_names = eval(exp_config['models'])
        print('exp ', exp_config, 'models', args.model_names)


        for model in args.model_names:


            os.environ["model_name"] = str(model)

            os.environ["database"] = str(args.database)

            os.environ["paths"] = str(exp_config['paths'])

            os.environ["validation_strategy"] = str(dictionary_name_change[args.experiment_config])

            os.environ["experiment"] = str(exp_config['experiment'])


            print(
                f"Launching job for model {model} with experiment {exp_config['experiment']} and validation strategy {args.experiment_config}"
            )
            bash_file = {'linear': 'src/bash/linear.sh', 'extractor': 'src/bash/extractor.sh', 'ml': 'src/bash/ml.sh'}
            launch_slurm_job(bash_file['extractor' if 'FEXT' in args.config_file else 'ml' if 'ML' in args.config_file else 'linear'], os.environ)
