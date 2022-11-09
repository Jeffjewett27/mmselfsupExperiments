import optuna
from optuna.distributions import CategoricalDistribution, LogUniformDistribution, UniformDistribution
import pandas as pd

TOTAL_TRIALS = 72

STUDY = optuna.create_study()
DISTRIBUTIONS = {
    "learning_rate": LogUniformDistribution(0.0001, 0.1),
    "momentum": CategoricalDistribution([0.8, 0.9, 0.95, 0.99]),
    "decay": LogUniformDistribution(0.0001, 0.01),
    "cosine_period": CategoricalDistribution([10, 15, 20, 30]),
    "initialization": CategoricalDistribution(['Xavier', 'Kaiming', 'Pretrained', 'Pretrained']),
    "loss_lambda": UniformDistribution(0.5, 0.75)
}

trials = [STUDY.ask(DISTRIBUTIONS).params for _ in range(TOTAL_TRIALS)]
df = pd.DataFrame(trials).reset_index().rename(columns={'index': 'trial'})
print(df)
df.to_csv('experiment/configs/smalltrials2.csv', index=False)