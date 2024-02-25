**현재 작업 중!**

### File Tree
```
Vault
├── main.py
├── preprocessing
│   ├── patch_conversion.py
│   ├── patch_extraction.py
│   └── removing_background.py
├── resnet
│   ├── predictions
│   ├── weights
│   ├── dataset.py
│   ├── model.py
│   ├── predictor.py
│   └── trainer.py
├── unetvit
│   ├── predictions
│   ├── weights
│   ├── dataset.py
│   ├── model.py
│   ├── predictor.py
│   └── trainer.py
├── utils
│   ├── config.py
│   └── utils.py
└── dataset
    ├── train
    │   ├── train
    │   │   ├── train_0001.png
    │   │   ├── ...
    │   │   └── train_xxxx.png
    │   ├── train_bg_rm
    │   │   ├── train_0001.png
    │   │   ├── ...
    │   │   └── train_xxxx.png
    │   ├── train_patch
    │   │   ├── train_0001_1.png
    │   │   ├── ...
    │   │   ├── train_0001_5.png
    │   │   ├── train_0002_1.png
    │   │   ├── ...
    │   │   └── train_xxxx_5.png
    │   └── train_patch_rs
    │       ├── train_0001_1.npy
    │       ├── ...
    │       ├── train_0001_5.npy
    │       ├── train_0002_1.npy
    │       ├── ...
    │       └── train_xxxx_5.npy
    ├── test
    │   ├── test
    │   ├── test_bg_rm
    │   ├── test_patch
    │   └── test_patch_rs
    ├── train_dataset.csv   
    └── test_dataset.csv

```