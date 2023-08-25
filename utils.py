import re
import json
import nltk
from tqdm import tqdm
from datasets import concatenate_datasets
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import wordnet
from transformers import T5ForConditionalGeneration, AutoTokenizer
import time
import random

random.seed(42)


def generate_exemplar(exemp_ds, prompt, perturb, perturb_exemplar):
    if prompt != '0cot':

        ratio = len(exemp_ds) // perturb_exemplar if perturb_exemplar > 0 else len(exemp_ds)

        exemplar = ""

        for i in range(len(exemp_ds)):

            next_perturb = perturb if i % ratio == 0 else None

            # Generate a response to a prompt
            exemp_question = perturb_question(exemp_ds[i], next_perturb)
            exemp_answer = exemp_ds[i]['answer']

            # remove <<...>> pattern from answer with regex non greedy
            exemp_answer = re.sub(r'<<.*?>>', '', exemp_answer)
            exemp_answer = re.sub(
                r"#### (\-?[0-9\.\,]+)", r"The answer is \1.", exemp_answer)
            exemp_answer = " ".join(exemp_answer.split("\n"))

            exemplar += "Q: " + exemp_question + "\nA: " + exemp_answer + "\n\n"

    elif prompt == '0cot':

        exemplar = ""

    else:
        pass

    return exemplar


def generate_prompt(question, exemplar, prompt):

    instr = "End your response with 'The answer is <answer>.'"

    prompt_text = instr + "\n\n" + exemplar + \
        "Q: " + question + "\nA:"

    if prompt == '0cot':
        prompt_text += " Let's think step by step:"

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

    elif perturb == "repetition":

        sents = sent_tokenize(sample['question'])

        if len(sents) > 1:
            # randomly select a sentence to repeat
            rep_sent = random.choice(sents[:-1])
            # insert repeated sentence to the sentence before the last sentence
            sents.insert(len(sents) - 1, rep_sent)

        return " ".join(sents)

    elif perturb == "synonym":

        tokens = word_tokenize(sample['question'])
        pos_tags = nltk.pos_tag(tokens)

        # replace nouns and verbs with synonyms
        for i, (token, pos_tag) in enumerate(pos_tags):
            if pos_tag.startswith('N') or pos_tag.startswith('V'):

                if pos_tag.startswith('N'):
                    synsets = wordnet.synsets(token, pos=wordnet.NOUN)
                else:
                    synsets = wordnet.synsets(token, pos=wordnet.VERB)

                if len(synsets) > 0:
                    # get set of synonyms
                    syn_tokens = set()
                    for syn in synsets:
                        for lemma in syn.lemmas():
                            # remove underscore
                            lemma = lemma.name().replace("_", " ")
                            syn_tokens.add(lemma)

                    # remove original token from set of synonyms
                    syn_tokens.discard(token)

                    if len(syn_tokens) > 0:

                        if random.random() < 0.2:
                            # randomly select a synonym
                            syn_token = random.choice(list(syn_tokens))
                            tokens[i] = syn_token

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


def evaluate_openai(run_id, model_name, dataset, prompt, shots, perturb, perturb_exemplar, dev):

    if not dev:
        f = open(f'logs/{run_id}.jsonl', 'w')

    exemp_ds = dataset["train"].select(range(shots))

    # generate exemplar
    exemplar = generate_exemplar(exemp_ds, prompt, perturb, perturb_exemplar)

    if not dev:
        # merge train and test datasets and remove sample for exemplar
        # modified_ds = concatenate_datasets([dataset["train"].select(
        #     range(shots, len(dataset["train"]))), dataset["test"]])
        modified_ds = dataset["test"]
    else:
        modified_ds = dataset["test"].select(range(5))

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

        if not dev:
            f.write(json.dumps(record) + '\n')
        else:
            print(prompt_text)
            print(record)

    if not dev:
        f.close()


def print_model_inputs(run_id, model_name, dataset, prompt, shots, perturb, perturb_exemplar, dev, output_filename):
    if not dev:
        f = open(f'logs/{run_id}.jsonl', 'w')

    exemp_ds = dataset["train"].select(range(shots))

    # generate exemplar
    exemplar = generate_exemplar(exemp_ds, prompt, perturb, perturb_exemplar)

    if not dev:
        # merge train and test datasets and remove sample for exemplar
        # modified_ds = concatenate_datasets([dataset["train"].select(
        #     range(shots, len(dataset["train"]))), dataset["test"]])
        modified_ds = dataset["test"]
    else:
        modified_ds = dataset["test"].select(range(5))

    prompts = []
    for sample in tqdm(modified_ds):
        # generate question text
        question = perturb_question(sample, perturb)
        # generate prompt text
        prompt_text = generate_prompt(question, exemplar, prompt)
        prompts.append(prompt_text)

    with open(output_filename, 'w') as fout:
        json.dump(prompts, fout, indent='')

    if not dev:
        f.close()


# Function to interact with the model and generate a response
def generate_response(prompt, model_name, model_file, model_tokenizer):
    if model_name == 'gpt3':
        import openai
        from openai.error import APIError, APIConnectionError, RateLimitError, Timeout
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
        import openai
        from openai.error import APIError, APIConnectionError, RateLimitError, Timeout
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
