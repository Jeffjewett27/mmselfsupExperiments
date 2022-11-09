import argparse
import shutil
import neptune.new as neptune
import os
import pandas as pd
import time
import subprocess
from datetime import datetime, timedelta

FAILURE_TIMEOUT = 60 #sleep 1 minutes before checking for new trial
MODIFICATION_TIMEOUT = 300 #wait 5 minutes before modifying unfinished run
EXPERIMENT='meta/experiment'
STAGE='meta/stage'
TRIAL='meta/trial'
INDEX='meta/index'
STATUS='meta/status'

STATUS_TRAINED='trained'
STATUS_UNTRAINED='untrained'
STATUS_FAILED='failed'

def get_project():
    return neptune.get_project(name=os.getenv('NEPTUNE_PROJECT'))

def get_runs(project, experiment):
    runs = project.fetch_runs_table(owner=os.getenv('NEPTUNE_PROFILE'), columns=[
            'sys/name', 
            'sys/state', 
            'train/loss',
            'sys/modification_time',
            'sys/tags',
            EXPERIMENT,
            STAGE,
            TRIAL,
            INDEX,
            STATUS
        ]).to_pandas()
    if STAGE not in runs.columns:
        runs[STAGE] = -1
    if TRIAL not in runs.columns:
        runs[TRIAL] = -1
    if INDEX not in runs.columns:
        runs[TRIAL] = -1
    if STATUS not in runs.columns:
        runs[STATUS] = STATUS_UNTRAINED
    if EXPERIMENT not in runs.columns:
        runs[EXPERIMENT] = ''

    runs[STAGE] = runs[STAGE].astype('Int64')
    runs[TRIAL] = runs[TRIAL].astype('Int64')
    runs[INDEX] = runs[INDEX].astype('Int64')
    runs = runs.dropna()
    runs = runs[runs[EXPERIMENT]==experiment]
    runs['override'] = runs['sys/tags'].str.contains('override')
    return runs

def get_next_trial(runs, stage, finished=False):
    if finished:
        runs = runs[runs[STATUS] == STATUS_TRAINED]
    else:
        cutoff = pd.Timestamp(datetime.utcnow() - timedelta(seconds=MODIFICATION_TIMEOUT)).tz_localize('utc')
        runs = runs[(runs['sys/modification_time'] > cutoff) | (runs[STATUS] == STATUS_TRAINED) | runs['override']]
    trials = runs[runs[STAGE] == stage].sort_values(by=INDEX)[[TRIAL,INDEX]]
    prevtrial = -1
    for _, row in trials.iterrows():
        if row[INDEX] - prevtrial > 1:
            break
        prevtrial = row[INDEX]
    return prevtrial + 1

def get_sha_config(shaconfig):
    return pd.read_csv(shaconfig)

def get_trial_configs(trialconfig):
    return pd.read_csv(trialconfig)

def get_next_stage(runs, sha):
    sha = sha.sort_values(by='stage')
    for _, stage in sha.iterrows():
        trial = get_next_trial(runs, stage.stage, finished=False)
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
        status='training',
        experiment=prefix
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
        run[STATUS] = STATUS_TRAINED
        if not stage.save:
            workdir = f'work_dirs/pretrain/{meta["name"]}'
            shutil.rmtree(workdir)
    else:
        run[STATUS] = STATUS_FAILED
    run.stop()
    
parser = argparse.ArgumentParser(description='Run SHA on trials')
parser.add_argument('--config', '-c', metavar='C', type=str,
    help='the trial configurations file')
parser.add_argument('--sha', '-s', metavar='S', type=str,
        help='the SHA configurations file')
parser.add_argument('--execution', '-e', metavar='E', type=str,
    help='the script to execute on each trial')
parser.add_argument('--prefix', '-p', metavar='P', type=str,
    help='the prefix for the name, eg prefix2-4'),
parser.add_argument('--tags', '-t', metavar='T', type=str, nargs='*',
        help='tags to add to neptune')
args = parser.parse_args()

trialconfigs = get_trial_configs(args.config)
sha = get_sha_config(args.sha)
project = get_project()
execution = args.execution
prefix = args.prefix
tags = args.tags

# runs = get_runs(project, prefix)

while True:
    runs = get_runs(project, prefix)
    stage = get_next_stage(runs, sha)
    stageIndex = get_next_trial(runs, stage.stage)
    if stageIndex is None:
        break
    if stageIndex < stage.trials:
        # train on the trial

        if stage.stage > 1:
            stagetrials = runs[runs[STAGE]==stage.stage-1].sort_values(by='train/loss').head(stage.trials)
            selected = stagetrials.iloc[stageIndex][TRIAL]
            # trialconfigs.iloc[stagetrials['trial']].reset_index().drop(['idx','index'], axis=1).reset_index().to_csv('experiment/mediumtrials.csv', index=False)
            # break
        else:
            selected = stageIndex
        print(f'TRAINING ON STAGE {stage.stage}-{stageIndex} AND TRIAL {selected}')
        config = trialconfigs.iloc[selected]
        run_trial(config, stage, selected, stageIndex)
    else:
        # wait a timeout period before checking for a new available trial
        print(f'NO TRIAL AVAILABLE FOR STAGE {stage.stage}. WILL TRY AGAIN IN {FAILURE_TIMEOUT/60} MINUTES')
        time.sleep(FAILURE_TIMEOUT)