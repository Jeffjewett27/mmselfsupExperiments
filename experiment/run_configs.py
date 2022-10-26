import pandas as pd
import os
import subprocess

execution = 'execution/smm2_test_1.sh'
rawconfigs = pd.read_csv('experiment/configurations.csv', index_col=0)
shardMag = 1
shardIdx = 0
configs = rawconfigs.iloc[shardIdx::shardMag]
epochs = 5
max_errors = 1
hpo_step = 1

# print(configs)

errors = 0
for index, row in configs.iterrows():
    if row['trained'] == 'Trained':
        print(f'pretrain{index} already trained')
        continue
    config = row.to_dict()
    config = {c.upper():str(config[c]) for c in config}
    config['TRIAL'] = f'pretrain{hpo_step}-{index}'
    config['EPOCHS'] = str(epochs)
    print(config)

    environ=os.environ.copy()
    environ.update(config)
    complete = subprocess.run(['bash', execution], env=environ)
    print(complete)
    if complete.returncode == 0:
        rawconfigs.at[index, 'trained'] = 'Trained'
        rawconfigs.to_csv('experiment/configurations.csv')
    else:
        rawconfigs.at[index, 'trained'] = 'Error'
        rawconfigs.to_csv('experiment/configurations.csv')
        errors += 1
        if errors >= max_errors:
            print('Too many errors. Killing experiment')
            break