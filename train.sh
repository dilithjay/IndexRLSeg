#!/bin/bash

python train.py -m baseline
python train.py -m concat
python train.py -m concat_multi
python train.py -m replace
python train.py -m replace_multi
