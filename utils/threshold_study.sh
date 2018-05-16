#!/bin/sh

python main.py bc3/full --type email --model regression_br --threshold 0.1
python main.py bc3/full --type email --model regression_br --threshold 0.3
python main.py bc3/full --type email --model regression_br --threshold 0.5
python main.py bc3/full --type email --model regression_br --threshold 0.7
python main.py bc3/full --type email --model regression_br --threshold 0.9

