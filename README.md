# CLSPRec

This is the pytorch implementation of paper "CLSPRec: Contrastive Learning of Long and Short-term Preferences for Next
POI Recommendation"

![model](model.png)

## Installation

```
pip install -r requirements.txt
```

## Valid Requirements

```
torch==2.0.1
numpy==1.24.3
pandas==2.0.2
Pillow==9.4.0
python-dateutil==2.8.2
pytz==2023.3
six==1.16.0
torchvision==0.15.2
typing_extensions==4.5.0
```

## Train

- Modify the configuration in settings.py

- Train and evaluate the model using python `main.py`.

- The training and evaluation results will be stored in `result` folder.

## Cite Our Paper

### CIKM2023


    @inproceedings{CLSPRec2023,
        title = {{CLSPRec}: Contrastive Learning of Long and Short-term Preferences for Next {POI} Recommendation},
        doi = {10.1145/3583780.3614813},
        shorttitle = {{CLSPRec}},
        pages = {473--482},
        booktitle = {Proceedings of the 32nd {ACM} International Conference on Information and Knowledge Management, {CIKM} 2023, Birmingham, United Kingdom, October 21-25, 2023},
        publisher = {{ACM}},
        author = {Duan, Chenghua and Fan, Wei and Zhou, Wei and Liu, Hu and Wen, Junhao},
        editor = {Frommholz, Ingo and Hopfgartner, Frank and Lee, Mark and Oakes, Michael and Lalmas, Mounia and Zhang, Min and Santos, Rodrygo L. T.},
        date = {2023}
    }
