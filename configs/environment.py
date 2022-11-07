import os
import sys

env = os.environ
init = env['INITIALIZATION']

nperiods = 20
cosperiod = [i*int(env['COSINE_PERIOD']) for i in range(1,nperiods+1)]
cosrestart = dict(policy='CosineRestart', min_lr=0., periods=cosperiod, restart_weights=[1]*nperiods)
schedule = cosrestart

epochs = int(env['EPOCHS'])
lr = float(env['LEARNING_RATE'])
decay = float(env['DECAY'])
momentum = float(env['MOMENTUM'])

trial = env['TRIAL']
lossLambda = float(env['LOSS_LAMBDA'])