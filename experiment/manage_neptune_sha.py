import neptune.new as neptune
import os
import pandas as pd
import time
import subprocess

FAILURE_TIMEOUT = 60 #sleep 1 minutes before checking for new trial

def get_project():
    return neptune.get_project(name=os.getenv('NEPTUNE_PROJECT'))

def get_runs(project):
    runs = project.fetch_runs_table(owner=os.getenv('NEPTUNE_PROFILE'), columns=['sys/name', 'train/loss', 'sys/tags']).to_pandas()
    runs[['stage','trial']] = runs['sys/name'].str.split('-',expand=True)
    runs = runs.dropna()
    runs['stage'] = runs['stage'].str.replace(r'\D+', '', regex=True).astype(int)
    runs['trial'] = runs['trial'].astype(int)
    runs['finished'] = runs['sys/tags'].str.contains('trained')
    return runs

def get_next_trial(runs, stage, finished=False):
    if finished:
        runs = runs[runs.finished == True]
        print('lenruns', len(runs))
    trials = runs[runs.stage == stage].sort_values(by='trial')['trial']
    prevtrial = -1
    for t in trials:
        if t - prevtrial > 1:
            break
        prevtrial = t
    return prevtrial + 1

def get_sha_config(shaconfig):
    return pd.read_csv(shaconfig)

def get_trial_configs(trialconfig):
    return pd.read_csv(trialconfig)

def get_next_stage(runs, sha):
    sha = sha.sort_values(by='stage')
    for _, stage in sha.iterrows():
        trial = get_next_trial(runs, stage.stage, finished=True)
        if trial < stage.trials:
            return stage
    return None

def run_trial(config, stage, trial):
    config = config.to_dict()
    config = {c.upper():str(config[c]) for c in config}
    config['TRIAL'] = f'pretrain{stage.stage}-{trial}'
    config['EPOCHS'] = str(stage.epochs)
    print('RUNNING TRIAL FOR CONFIG', config)

    run = neptune.init_run(
        name=config['TRIAL'],
        custom_run_id=config['TRIAL'],
        tags=[f'pretrain{stage.stage}']
    )
    run['params'] = config

    environ=os.environ.copy()
    environ.update(config)
    complete = subprocess.run(['bash', execution], env=environ)
    if complete.returncode == 0:
        run['sys/tags'].add('trained')
        # if 'failed' in run['sys/tags']:
        #     run['sys/tags'].remove('failed')
    else:
        run['sys/tags'].add('failed')
    run.stop()
    

trialconfigs = get_trial_configs('experiment/configurations.csv')
sha = get_sha_config('experiment/pretrain_small_sha.csv')
project = get_project()
execution = 'execution/smm2_test_1.sh'

while True:
    runs = get_runs(project)
    stage = get_next_stage(runs, sha)
    trial = get_next_trial(runs, stage.stage)
    if trial < stage.trials:
        # train on the trial
        print(f'TRAINING ON STAGE {stage.stage} AND TRIAL {trial}')

        if stage.stage > 1:
            stagetrials = runs[runs.stage==stage.stage-1].sort_values(by='train/loss').head(stage.trials)
            selected = stagetrials.iloc[trial].trial
        else:
            selected = trial
        config = trialconfigs.iloc[selected]
        run_trial(config, stage, trial)
    else:
        # wait a timeout period before checking for a new available trial
        print(f'NO TRIAL AVAILABLE FOR STAGE {stage.stage}. WILL TRY AGAIN IN {FAILURE_TIMEOUT/60} MINUTES')
        time.sleep(FAILURE_TIMEOUT)