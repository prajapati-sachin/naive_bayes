export PYTHONPATH="/home/sachin/libsvm-3.23/python:${PYTHONPATH}"
#!/bin/bash

echo "Running first"
./run.sh 1 ../p1/train.json ../p1/test.json d > d.txt
echo "Running second"
./run.sh 1 ../p1/train.json ../p1/test.json f > f.txt
echo "Running third"
./run.sh 1 ../p1/train_full.json ../p1/test.json g > g.txt