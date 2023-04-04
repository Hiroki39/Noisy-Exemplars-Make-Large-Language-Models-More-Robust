import openai
import re
import json
from tqdm import tqdm
from datasets import concatenate_datasets


def generate_exemplar(exemp_dict, prompt, perturb, perturb_exemplar):
    if prompt == 'cot':
        # Generate a response to a prompt
        exemp_question = exemp_dict['question']
        exemp_answer = exemp_dict['answer']

        # remove <<...>> pattern from answer with regex non greedy
        exemp_answer = re.sub(r'<<.*?>>', '', exemp_answer)
        exemp_answer = re.sub(
            r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", exemp_answer)
        exemp_answer = " ".join(exemp_answer.split())

        exemplar = "End your response with 'The answer is <answer>.'\n\nQ: " + \
            exemp_question + "\nA: " + exemp_answer

        return exemplar

    else:
        pass


def build_record(sample, result):

    record = {}
    record['question'] = sample['question']

    record['answer'] = re.sub(
        r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", re.sub(r'<<.*?>>', '', sample['answer']))
    record['numeric_answer'] = re.search(
        r"#### (\-?[0-9\.\,]+)", sample['answer']).group(1)

    if result['model'] == 'text-davinci-003':
        record['response'] = result['choices'][0]['text']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['choices'][0]['text'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None
        record['tokens'] = result['choices'][0]['logprobs']['tokens']
        record['logprobs'] = result['choices'][0]['logprobs']['token_logprobs']
    else:
        record['response'] = result['choices'][0]['message']['content']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['choices'][0]['message']['content'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None

    return record


def evaluate_openai(run_id, model, dataset, prompt, perturb, perturb_exemplar):
    with open(f'logs/{run_id}.jsonl', 'w') as f:
        if prompt == 'cot':
            exemp_dict = dataset["train"][0]

            exemplar = generate_exemplar(
                exemp_dict, prompt, perturb, perturb_exemplar)

            # merge train and test datasets and remove sample for exemplar
            # modified_ds = concatenate_datasets([dataset["train"].select(range(1, len(dataset["train"]))), dataset["test"]])
            modified_ds = concatenate_datasets([dataset["train"].select(
                range(1, len(dataset["train"]))), dataset["test"]])

            engine = 'text-davinci-003' if model == 'gpt3' else 'gpt-3.5-turbo'

            for sample in tqdm(modified_ds):
                prompt = exemplar + "\n\nQ: " + sample['question'] + "\nA:"
                result = generate_response(prompt, engine).to_dict()

                record = build_record(sample, result)
                f.write(json.dumps(record) + '\n')
        else:
            pass

# Define a function to generate responses using GPT-3


def generate_response(prompt, engine):
    if engine == 'text-davinci-003':
        while True:
            try:
                response = openai.Completion.create(
                    engine=engine,
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=1,
                )
            except openai.error.APIError or OSError:
                continue
            break
    else:
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
            except openai.error.APIError or OSError:
                continue
            break
    return response
