import os
import sys

env = os.environ
init = env['INITIALIZATION'] if 'INITIALIZATION' in env else 'Pretrained'
if init == 'Xavier':
    initCfg=dict(type='Xavier', distribution='uniform')
elif init == 'Kaiming':
    initCfg=dict(type='Kaiming', distribution='uniform')
else:
    initCfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')

nperiods = 20
restartDecay = float(env['RESTART_DECAY']) if 'RESTART_DECAY' in env else 1
cosperiod = [i*int(float(env['COSINE_PERIOD'])) for i in range(1,nperiods+1)]
restarts = [restartDecay**i for i in range(0,nperiods)]
cosrestart = dict(policy='CosineRestart', min_lr=0., periods=cosperiod, restart_weights=restarts)
schedule = cosrestart

warmupEpochs = int(float(env['WARMUP_EPOCHS'])) if 'WARMUP_EPOCHS' in env else 0
if warmupEpochs > 0:
    schedule['warmup'] = 'linear'
    schedule['warmup_iters'] = warmupEpochs
    schedule['warmup_ratio'] = 1e-4
    schedule['warmup_by_epoch'] = True

epochs = int(float(env['EPOCHS']))
lr = float(env['LEARNING_RATE'])
decay = float(env['DECAY'])
momentum = float(env['MOMENTUM'])

trial = env['TRIAL']
lossLambda = float(env['LOSS_LAMBDA'])