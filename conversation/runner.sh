#!/bin/sh

python main.py gnue/small --type chat --model naivebayes
python main.py gnue/small --type chat --model decisiontree
python main.py gnue/small --type chat --model perceptron
