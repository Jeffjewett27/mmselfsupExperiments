import optuna
from optuna.distributions import CategoricalDistribution, LogUniformDistribution, UniformDistribution
import pandas as pd

TOTAL_TRIALS = 24

STUDY = optuna.create_study()
DISTRIBUTIONS = {
    "learning_rate": LogUniformDistribution(0.0001, 0.1),
    "momentum": CategoricalDistribution([0.86, 0.88, 0.9, 0.92]),
    "decay": LogUniformDistribution(0.0001, 0.01),
    "step_period": CategoricalDistribution([6, 8, 10, 12, 14]),
    "warmup_ratio": LogUniformDistribution(0.0003, 0.001),
    "warmup_iters": CategoricalDistribution([500, 1000, 1500, 2000])
}

trials = [STUDY.ask(DISTRIBUTIONS).params for _ in range(TOTAL_TRIALS)]
df = pd.DataFrame(trials).reset_index().rename(columns={'index': 'trial'})
print(df)
df.to_csv('experiment/configs/dettrials.csv', index=False)