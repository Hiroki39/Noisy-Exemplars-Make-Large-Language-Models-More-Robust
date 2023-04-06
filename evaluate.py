import argparse
import openai
from datasets import load_dataset
from utils import evaluate_openai
from uuid import uuid4
from dotenv import load_dotenv
import os
import csv


def conduct_test(model, dataset_name, prompt, perturb, perturb_exemplar):

    run_id = str(uuid4())
    # Load the GSM8K dataset from Hugging Face
    dataset = load_dataset(dataset_name, "main")

    if model == 'gpt3' or model == 'gptturbo':
        # Set up the OpenAI API client
        openai.api_key = os.getenv('OPENAI_API_KEY')
        evaluate_openai(run_id, model, dataset, prompt,
                        perturb, perturb_exemplar)

    else:
        pass

    with open('log_files.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([run_id, model, dataset_name,
                        prompt, perturb, perturb_exemplar])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, default='gpt3')
    parser.add_argument('--dataset', type=str, required=True, default='gsm8k')
    parser.add_argument('--prompt', type=str, required=True, default='cot')
    parser.add_argument('--perturb', type=str, required=False)
    parser.add_argument('--perturb_exemplar',
                        action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    load_dotenv()

    conduct_test(args.model, args.dataset, args.prompt,
                 args.perturb, args.perturb_exemplar)
