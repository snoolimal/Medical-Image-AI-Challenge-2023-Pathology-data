from abunet_trainer import ABUNetTrainer
from abunet_predictor import ABUNetPredictor


def abunet_main():
    for risk in [1, 2]:
        Trainer = ABUNetTrainer(risk=risk)
        Predictor = ABUNetPredictor(risk=risk)
        Trainer.train_valid()
        Predictor.predict()


abunet_main()