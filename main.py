from preprocessing.removing_background import BGRemover
from preprocessing.patch_extraction import PatchExtractor
from preprocessing.patch_conversion import PatchConverter
from resnet.trainer import RNTrainer
from resnet.predictor import RNPredictor
from unetvit.trainer import UVTrainer
from unetvit.predictor import UVPredictor
from utils.utils import seed_everything


def main(preprocessing=False):
    seed_everything()

    if preprocessing:
        bg_remover = BGRemover()
        bg_remover.remove_background('train')
        bg_remover.remove_background('test')

        patch_extractor = PatchExtractor()
        patch_extractor.extract_save_patch('train')
        patch_extractor.extract_save_patch('test')

        patch_converter = PatchConverter()
        patch_converter.rescale_save_patch('train')
        patch_converter.rescale_save_patch('test')

    for risk in [0, 1, 2]:
        resnet_trainer = RNTrainer(risk)
        resnet_trainer.train_valid()
        resnet_predictor = RNPredictor(risk)
        resnet_predictor.predict()

    for risk in [1, 2]:
        uv_trainer = UVTrainer(risk)
        uv_trainer.train_valid()
        uv_predictor = UVPredictor(risk)
        uv_predictor.predict()


if __name__ == '__main__':
    main()
