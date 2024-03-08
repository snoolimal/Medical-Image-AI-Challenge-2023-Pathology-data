## Medical Image AI Challenge 2023: Pathology data

This repository contains the final submission of team "달려달려" in the **Medical Image AI Challenge 2023: Pathology data** hosted by Seoul National University Hospital. <br>
All results in this repository are coded by [soonoolimal](https://github.com/snoolimal) and [jjunstone7](https://github.com/jjunstone7).

- **Competition**: [Medical Image AI Challenge 2023: Pathology data](https://maic.or.kr/competitions/28/infomation) 
- **Host**: Seoul National University Hospital
- **Objective**: Prediction of melanoma recurrence using whole slide image and tabular information
- **Result**: Reached top 5 private score out of 62 teams, advanced to the final presentation stage but failed to win awards
- **Reproduction**: Clone this repository and execute `main.py`
- **Presentation: [Here](https://1drv.ms/b/s!AsCGiwhl8mgznYFQe2yFCCkXjXlwAw?e=OzGmiG)**

cf. The datasets used cannot be publicly available due to the regulations.

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

### Logs and References
- [Notion Workspace](https://www.notion.so/7b1514b752be49fab6b7390126c70565)
- [more?](https://dgist.edwith.org/medical-20200327/joinLectures/30437)
