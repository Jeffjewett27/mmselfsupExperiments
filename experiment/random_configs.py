import optuna
from optuna.distributions import CategoricalDistribution, LogUniformDistribution, UniformDistribution
import pandas as pd

TOTAL_TRIALS = 24

STUDY = optuna.create_study()
DISTRIBUTIONS = {
    "learning_rate": LogUniformDistribution(0.0001, 0.1),
    "momentum": CategoricalDistribution([0.8, 0.9, 0.95, 0.99]),
    "decay": LogUniformDistribution(0.0001, 0.01),
    "step_period": CategoricalDistribution([10, 15, 20, 30]),
    "restart_decay": CategoricalDistribution([0.8, 0.9, 0.95, 1]),
    "warmup_epochs": CategoricalDistribution([1,3,5,7])
}

trials = [STUDY.ask(DISTRIBUTIONS).params for _ in range(TOTAL_TRIALS)]
df = pd.DataFrame(trials).reset_index().rename(columns={'index': 'trial'})
print(df)
df.to_csv('experiment/configs/dettrials.csv', index=False)