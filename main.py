from abunet_trainer import ABUNetTrainer
from abunet_predictor import ABUNetPredictor


def main():
    for risk in [1, 2]:
        Trainer = ABUNetTrainer(risk=risk)
        Trainer.train_valid()

        Predictor = ABUNetPredictor(risk=risk)
        Predictor.predict()