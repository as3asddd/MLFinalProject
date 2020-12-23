#!/bin/bash

bash train.sh && \
bash infer.sh && \
bash gen_result.sh > result.csv
