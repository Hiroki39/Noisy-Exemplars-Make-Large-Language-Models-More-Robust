# Noisy Exemplars Make Large Language Models More Robust: A Domain-Agnostic Behavioral Analysis

## Run Trials

`python evaluate.py --model <model_name> --dataset <dataset_name> --prompt <prompt method name> --shot <# shots> --perturb <perturbation type name> --perturb_exemplar <True/False> --dev/--no-dev`

- `model`: `gptturbo` (recommended) or `gpt3`
- `dataset`: currently, only `gsm8k` is supported
- `prompt`: `cot`, `0cot` or `ltm`
- `shot`: `1`, `2`, `4` or `8`
- `perturb`: `synonym`, `repetition`, `shortcut`, `typo`, or `None`
- `perturb_exemplar`: `True` or `False`, indicating whether to apply the perturbation on exemplar questions
- `dev`: `True` or `False`, indicating whether to use 5-example mini dataset for debugging or not

Before running the code, please make sure you create a `.env` file in the root directory and add the following line:

`OPENAI_API_KEY=<your openai api key>`

After the program completes running, the log file name of the trial along with the hyperparameters will be recorded in `log_files.csv`

## Performance Evaluation

`python compute_accuracy.py`

Upon completion, the program should create a new file called `log_files_with_accuracy.csv` which add `accuracy` column to the original `log_files.csv` 

## Visualization

`python generate_plots.py`

Plots are generated under `/images` directory

## Clean Up Log Files

`python clean_logs.py`

This will remove all log files that are not in `log_files.csv` from the `logs` directory