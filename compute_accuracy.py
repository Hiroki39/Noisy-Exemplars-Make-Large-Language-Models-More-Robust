import pandas as pd
import json

df = pd.read_csv('log_files.csv', names=[
                 'run_id', 'model', 'dataset', 'prompt', 'shots', 'perturb', 'perturb_exemplar'])


def get_accuracy(row):

    cnt = 0
    correct = 0

    run_id = row['run_id']

    with open(f'logs/{run_id}.jsonl', 'r') as f:
        for rec in f:
            record = json.loads(rec)

            if record['numeric_response'] is not None and record['numeric_answer'] in record['numeric_response']:
                correct += 1

            cnt += 1

    return correct / cnt


df["accuracy"] = df.apply(get_accuracy, axis=1)

df.to_csv('log_files_with_accuracy.csv', index=False)
