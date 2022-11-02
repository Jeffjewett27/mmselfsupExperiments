import neptune.new as neptune
import os
import pandas as pd
import time
import subprocess
from datetime import datetime, timedelta

FAILURE_TIMEOUT = 60 #sleep 1 minutes before checking for new trial
MODIFICATION_TIMEOUT = 300 #wait 5 minutes before modifying unfinished run

def get_project():
    return neptune.get_project(name=os.getenv('NEPTUNE_PROJECT'))

def get_runs(project):
    runs = project.fetch_runs_table(owner=os.getenv('NEPTUNE_PROFILE'), columns=[
            'sys/name', 
            'sys/state', 
            'train/loss', 
            'sys/tags', 
            'sys/modification_time',
            'meta/stage',
            'meta/trial',
            'meta/index',
            'meta/status'
        ]).to_pandas()
    if 'meta/stage' not in runs.columns:
        runs['meta/stage'] = -1
    if 'meta/trial' not in runs.columns:
        runs['meta/trial'] = -1
    if 'meta/index' not in runs.columns:
        runs['meta/index'] = -1
    if 'meta/status' not in runs.columns:
        runs['meta/status'] = 'untrained'

    runs['meta/stage'] = runs['meta/stage'].astype('Int64')
    runs['meta/trial'] = runs['meta/trial'].astype('Int64')
    runs['meta/index'] = runs['meta/index'].astype('Int64')
    runs[['stage','trial']] = runs['sys/name'].str.split('-',expand=True)
    runs = runs.dropna()
    runs['stage'] = runs['stage'].str.replace(r'\D+', '', regex=True).astype(int)
    runs['trial'] = runs['trial'].astype(int)
    runs['finished'] = runs['sys/tags'].str.contains('trained')
    return runs

def get_next_trial(runs, stage, finished=False):
    if finished:
        runs = runs[runs['meta/status'] == 'trained']
        print('lenruns', len(runs))
    else:
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(seconds=MODIFICATION_TIMEOUT)).tz_localize('utc')
        runs = runs[(runs['sys/modification_time'] > cutoff) | (runs['meta/status'] == 'trained')]
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

def run_trial(config, stage, trial, stageIndex):
    name = f'{prefix}{stage.stage}-{stageIndex}'

    config = config.to_dict()
    config = {c.upper():str(config[c]) for c in config}
    config['TRIAL'] = name
    config['EPOCHS'] = str(stage.epochs)
    print('RUNNING TRIAL FOR CONFIG', config)

    meta = dict(
        name=name,
        stage=stage.stage,
        trial=trial,
        index=stageIndex,
        epochs=stage.epochs,
        status='training'
    )

    run = neptune.init_run(
        name=meta['name'],
        custom_run_id=meta['name'],
        tags=tags
    )
    run['params'] = config
    run['meta'] = meta

    environ=os.environ.copy()
    environ.update(config)
    complete = subprocess.run(['bash', execution], env=environ)
    if complete.returncode == 0:
        run['meta/status'] = 'trained'
    else:
        run['meta/status'] = 'failed'
    run.stop()
    

trialconfigs = get_trial_configs('experiment/configurations.csv')
sha = get_sha_config('experiment/pretrain_small_sha_v2.csv')
project = get_project()
execution = 'execution/smm2_test_1.sh'
prefix = 'presmall'
tags = ['pretrain']

runs = get_runs(project)

while True:
    runs = get_runs(project)
    stage = get_next_stage(runs, sha)
    stageIndex = get_next_trial(runs, stage.stage)
    if stageIndex < stage.trials:
        # train on the trial

        if stage.stage > 1:
            stagetrials = runs[runs.stage==stage.stage-1].sort_values(by='train/loss').head(stage.trials)
            selected = stagetrials.iloc[stageIndex]['meta/trial']
        else:
            selected = stageIndex
        print(f'TRAINING ON STAGE {stage.stage}-{stageIndex} AND TRIAL {selected}')
        config = trialconfigs.iloc[selected]
        run_trial(config, stage, selected, stageIndex)
    else:
        # wait a timeout period before checking for a new available trial
        print(f'NO TRIAL AVAILABLE FOR STAGE {stage.stage}. WILL TRY AGAIN IN {FAILURE_TIMEOUT/60} MINUTES')
        time.sleep(FAILURE_TIMEOUT)