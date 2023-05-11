import pandas as pd
import os

df = pd.read_csv('log_files.csv', names=[
                 'run_id', 'model', 'dataset', 'prompt', 'shots', 'perturb', 'perturb_exemplar'])

# iterate over filenames in logs folder
for filename in os.listdir('logs'):
    # if filename is not in the run_id column of the dataframe
    if filename[:-6] not in df['run_id'].values:
        # delete the file
        os.remove(f'logs/{filename}')
