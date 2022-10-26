import optuna
from optuna.distributions import CategoricalDistribution, LogUniformDistribution
import pandas as pd

TOTAL_TRIALS = 72

STUDY = optuna.create_study()
DISTRIBUTIONS = {
    "backbone": CategoricalDistribution(["resnet18", "resnet34"]),
    "learning_rate": LogUniformDistribution(0.0001, 0.1),
    "momentum": CategoricalDistribution([0.8, 0.9, 0.95, 0.99]),
    "decay": LogUniformDistribution(0.0001, 0.01),
    "scheduler": CategoricalDistribution(['CosineRestart', 'step']),
    "cosine_period": CategoricalDistribution([5, 10, 15, 20]),
    "step_period": CategoricalDistribution([10, 20, 30, 40]),
    "initialization": CategoricalDistribution(['Xavier', 'Kaiming'])
}

trials = [STUDY.ask(DISTRIBUTIONS).params for _ in range(TOTAL_TRIALS)]
df = pd.DataFrame(trials)
df['trained'] = 'Untrained'
print(df)
df.to_csv('experiment/configurations.csv')