# DS-GA-3001-003-Final-Project

## Usage

`python evaluate.py --model <model_name> --dataset <dataset_name> --prompt <prompt method name> --shot <# shots> --perturb <perturbation type name> --perturb_exemplar <True/False> --dev/--no-dev`

- `model`: `gptturbo` (recommended) or `gpt3`
- `dataset`: currently, only `gsm8k` is supported
- `prompt`: `cot`, `0cot` or `ltm`
- `shot`: `1`, `2`, `4` or `8`
- `perturb`: `synonym`, `repetition`, `shortcut`, `typo`, or `None`
- `perturb_exemplar`: `True` or `False`, indicating whether to apply the perturbation on the exemplars
- `dev`: `True` or `False`, indicating whether to use 5-example mini dataset for debugging or not

Before running the code, please make sure you create a `.env` file in the root directory and add the following line:

`OPENAI_API_KEY=<your openai api key>`

## Visualization

`python generate_plots.py`

Plots are generated under `/images` directory
