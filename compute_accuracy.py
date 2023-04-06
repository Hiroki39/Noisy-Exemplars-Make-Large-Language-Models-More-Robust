import pandas as pd
import json

df = pd.read_csv('log_files.csv', names=[
                 'run_id', 'model', 'dataset', 'prompt', 'perturb', 'perturb_exemplar'])

for index, row in df.iterrows():

    cnt = 0
    correct = 0

    run_id = row['run_id']

    with open(f'logs/{run_id}.jsonl', 'r') as f:
        for rec in f:
            record = json.loads(rec)

            if record['numeric_response'] is not None and record['numeric_answer'] in record['numeric_response']:
                correct += 1

            cnt += 1

        print(row)
        print(f'Accuracy: {correct/cnt}')
