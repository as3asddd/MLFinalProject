# MLFinalProject
ML for Cyber Sec; CSAW-HackML-2020

**Authors:**  
Tieliu Zhou, tz1430, N14973953;

Juntao Jiang, jj2910, N10249972;

Kangning Li, kl3596, N12602103;

Jiazhen Han, jh6419, N19591264;

## Repo Architecture:

**Models:** fixed models;

**Src:** the raw code and help scripts;

## How to Run Code:

**For approach1(without PCA src/Fine-Pruning Approach):** There is only a colab file with code and description inside.

**For approach2(use PCA src/Fine-Pruning-PCA-Approach):** There are train.sh and infer.sh to do the training part and inference part. Please make sure the data is in the right directory(MLFinalProject/src/Fine-Pruning-PCA-Approach/data).

**For single image:** Please use the eval script and put the test image into the data floder(MLFinalProject/data).

**For evaluating the model, execute eval.py by running:**  
python3 eval_*.py </data directory>  
E.g  
python3 eval_anonymous_1.py data/test_image.png
