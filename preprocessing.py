from _remove_background import BGRemover
from _patch_extraction import PatchExtractor
from _patch_conversion import PatchConverter
from _utils import get_patch_metadata


# BGRemover = BGRemover()
# BGRemover.save_rmbg_slides('train')
# BGRemover.save_rmbg_slides('test')
#
# PatchExtractor = PatchExtractor()
# PatchExtractor.save_patch('train')
# PatchExtractor.save_patch('test')
#
# PatchConverter = PatchConverter()
# PatchConverter.save_rescaled_patch('train')
# PatchConverter.save_rescaled_patch('test')

get_patch_metadata('train')
get_patch_metadata('test')
