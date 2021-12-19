# Chowder

## What is this ?
Code for a submission to the data challenge https://challengedata.ens.fr/participants/challenges/18/ contains:

## What's in here ?

* **chowder.ipynb** which implements, trains and tests the original method from the paper, it uses:
  * **utils.py** containing the batch generation functions
  *  **MILpooling.py** containing the implementation of the min-max pooling layer

* **proposed_step1_supervised-pretrain-MLP.ipynb** builds a MLP on top of the Resnet50 features and trains in with full supervision on the tile-level annotations
  * model is saved as **base_model.h5**

* **proposed_step2_MIL-finetune.ipynb** fine-tunes the aforementioned MLP with the image-level anntations in a MIL fashion such that the maximum scoring tile in an image would give the image score (respecting the naive MIL assumptions that: 1) if an instance is positive the whole bag is positive, 2) negative bags contain only negative instances)

## Did this work ?

|       Method       	|  5-fold CV AUC  	| Test AUC 	| Leaderboard Position 	|
|:------------------:	|:---------------:	|:--------:	|:--------------------:	|
|       Chowder      	| 0.892 +/- 0.019 	|   0.78   	|          #66         	|
|       MaxPool      	| 0.853 +/- 0.099 	|   0.83   	|          #18         	|
| Pretrained MaxPool 	|                 	|   0.86   	|          #8          	|
|   Supervised MLP   	| 0.954 +/- 0.002 	|   N.A.   	|         N.A.         	|
