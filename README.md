# IndexRL

### Setup
Step 1: Install the requirements with:
```
pip install -r requirements.txt
```

Step 2: Download the dataset directly with:
```
gdown 1RRKWENclBWFpq-7jr5JzhKiIE-mGYmLy
```
or download from the Google Drive link:
https://drive.google.com/file/d/1RRKWENclBWFpq-7jr5JzhKiIE-mGYmLy/view?usp=sharing

Step 3: Extract dataset
```
unzip dataset.zip -d dataset/
```

#### (Optional) Clone CAINet
```
git clone https://github.com/YingLv1106/CAINet.git
```

### Training Index Generation
To generate indices for a given dataset
```
python indexrl_main.py -dn <dataset_name>
```
Usage help:
```
usage: indexrl_main.py [-h] [-dd DATA_DIR] [-o INDEXRL_OUT_DIR] [-a {gpt,lstm}] [-dn DATASET_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -dd DATA_DIR, --data_dir DATA_DIR
                        Directory with the entire training set (default: dataset/train/)
  -o INDEXRL_OUT_DIR, --indexrl_out_dir INDEXRL_OUT_DIR
                        Directory to save all outputs of the training (models, logs, and cache) (default:
                        indexrl_out/)
  -a {gpt,lstm}, --arch {gpt,lstm}
                        Agent model architecture (default: gpt)
  -dn DATASET_NAME, --dataset_name DATASET_NAME
                        Name of the dataset to find indices for (default: None)
```

### Training Segmentation Model
To train all model variations:
```
sh train.sh
```

To train all model variations for a given mode:
```
python train.py -m <mode>
```

Usage help:
```
usage: train.py [-h] [-m MODE] [-md MODEL_DIR] [-np NEPTUNE_PROJECT] [-nt NEPTUNE_TOKEN]

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Choose mode to update the dataset (baseline, concat, concat_multi, replace, replace_multi)
                        (default: baseline)
  -md MODEL_DIR, --model_dir MODEL_DIR
                        Directory to output the models. Subdirectories will be created for each mode. (default: models-train/)
  -np NEPTUNE_PROJECT, --neptune_project NEPTUNE_PROJECT
                        Name of your neptune project (default: )
  -nt NEPTUNE_TOKEN, --neptune_token NEPTUNE_TOKEN
                        Neptune API token (default: )
```

### Evaluating Segmentation Model
To evaluate all model variations:
```
sh evaluate.sh
```

To evaluate all model variations for a given mode:
```
python evaluate.py -m <mode>
```

Usage help:
```
usage: evaluate.py [-h] [-m MODE] [-md MODEL_DIR] [-np NEPTUNE_PROJECT] [-nt NEPTUNE_TOKEN]

optional arguments:
  -h, --help            show this help message and exit
  -m MODE, --mode MODE  Choose mode to update the dataset (baseline, concat, concat_multi, replace, replace_multi)
                        (default: baseline)
  -md MODEL_DIR, --model_dir MODEL_DIR
                        Directory to output the models. Subdirectories will be created for each mode. (default: models/)
```
