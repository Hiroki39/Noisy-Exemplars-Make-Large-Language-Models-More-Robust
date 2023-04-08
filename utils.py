import openai
import re
import json
from tqdm import tqdm
from datasets import concatenate_datasets
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
import time
import random
from openai.error import APIError, APIConnectionError, RateLimitError, Timeout

random.seed(42)


def generate_exemplar(exemp_dict, prompt, perturb):
    if prompt == 'cot':
        # Generate a response to a prompt
        exemp_question = perturb_question(exemp_dict, perturb)
        exemp_answer = exemp_dict['answer']

        # remove <<...>> pattern from answer with regex non greedy
        exemp_answer = re.sub(r'<<.*?>>', '', exemp_answer)
        exemp_answer = re.sub(
            r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", exemp_answer)
        exemp_answer = " ".join(exemp_answer.split("\n"))

        exemplar = "Q: " + exemp_question + "\nA: " + exemp_answer

    elif prompt == '0cot':

        exemplar = ""

    else:
        pass

    return exemplar


def generate_prompt(question, exemplar, prompt):

    instr = "End your response with 'The answer is <answer>.'"

    if prompt == 'cot':
        prompt_text = instr + "\n\n" + exemplar + \
            "\n\nQ: " + question + "\nA:"

    elif prompt == '0cot':
        prompt_text = instr + "\n\nQ: " + \
            question + "\nA: Let's think step by step: "

    else:
        pass

    return prompt_text


def perturb_question(sample, perturb):
    if perturb is None:
        return sample["question"]
    elif perturb == "shortcut":
        sample_answer = sample["answer"]

        # remove <<...>> pattern from answer with regex non greedy
        sample_answer = re.sub(r'<<.*?>>', '', sample_answer)
        sample_answer = re.sub(
            r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", sample_answer)

        first_step = sample_answer.split("\n")[0]

        # insert first step before the last sentence of the question
        sents = sent_tokenize(sample['question'])
        sents.insert(len(sents) - 1, first_step)
        return " ".join(sents)
    elif perturb == "typo":

        tokens = word_tokenize(sample['question'])
        for i, token in enumerate(tokens):
            if len(token) > 1 and not token.isnumeric():
                # introduce a typo with probability 0.1
                if random.random() < 0.1:
                    typo_ind = random.randint(0, len(token) - 2)
                    tokens[i] = token[:typo_ind] + \
                        token[typo_ind + 1] + token[typo_ind] + \
                        token[typo_ind + 2:]

        detokenizer = TreebankWordDetokenizer()
        return detokenizer.detokenize(tokens)


def build_record(sample, result, perturb):

    record = {}
    record['question'] = sample['question']

    if perturb is not None:
        record['perturbed_question'] = perturb_question(sample, perturb)

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

    elif result['model'].startswith('gpt-3.5-turbo'):
        record['response'] = result['choices'][0]['message']['content']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['choices'][0]['message']['content'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None

    elif result['model'] == 'ul2':
        record['response'] = result['response']
        try:
            record['numeric_response'] = re.search(
                r'The answer is (.*?)\.', result['response'], re.IGNORECASE).group(1)
        except AttributeError:
            record['numeric_response'] = None

    return record


def fetch_model_and_tokenizer(model_name):
    if model_name == 'ul2':
        model_file = T5ForConditionalGeneration.from_pretrained(
            "google/flan-ul2", device_map="auto", load_in_8bit=True)
        model_tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

        return model_file, model_tokenizer
    else:
        return None, None


def evaluate_openai(run_id, model_name, dataset, prompt, perturb, perturb_exemplar):

    with open(f'logs/{run_id}.jsonl', 'w') as f:

        exemp_dict = dataset["train"][0]

        # generate exemplar
        exemplar_perturb = perturb if perturb_exemplar else None
        exemplar = generate_exemplar(exemp_dict, prompt, exemplar_perturb)

        # merge train and test datasets and remove sample for exemplar
        modified_ds = concatenate_datasets([dataset["train"].select(
            range(1, len(dataset["train"]))), dataset["test"]])
        # modified_ds = dataset["train"].select(range(1, 1690))

        model_file, model_tokenizer = fetch_model_and_tokenizer(model_name)

        for sample in tqdm(modified_ds):

            # generate question text
            question = perturb_question(sample, perturb)
            # generate prompt text
            prompt_text = generate_prompt(question, exemplar, prompt)
            # get response
            result = generate_response(
                prompt_text, model_name, model_file, model_tokenizer)

            record = build_record(sample, result, perturb)

            f.write(json.dumps(record) + '\n')

# Function to interact with the model and generate a response


def generate_response(prompt, model_name, model_file, model_tokenizer):
    if model_name == 'gpt3':
        while True:
            try:
                response = openai.Completion.create(
                    engine='text-davinci-003',
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    logprobs=1,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout):
                time.sleep(1)
                continue
            break
    elif model_name == 'gptturbo':
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-3.5-turbo',
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                ).to_dict()
            except (APIError, OSError, APIConnectionError, RateLimitError, Timeout):
                time.sleep(1)
                continue
            break
    elif model_name == 'ul2':
        inputs = model_tokenizer(
            prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model_file.generate(inputs, max_length=300, do_sample=True,
                                      top_p=1, temperature=0, num_return_sequences=1)
        answer_text = model_tokenizer.decode(
            outputs[0], skip_special_tokens=True)

        response = {
            'model': 'ul2',
            'response': answer_text,
        }

    return response
