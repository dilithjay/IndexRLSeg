#!/bin/bash

python evaluate.py -m baseline
python evaluate.py -m concat
python evaluate.py -m concat_multi
python evaluate.py -m replace
python evaluate.py -m replace_multi
python evaluate.py -m best
python evaluate.py -m nonminified
python evaluate.py -m train_size
