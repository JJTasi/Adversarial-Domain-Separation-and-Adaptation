## Adversarial Domain Separation and Adaptation
In this project, we implement Adversarial Domain Separation and Adaptation on sentiment classification for Amazon Reviews.

<img src="figs/ADSA.png" width="50%">

## Setting
- Hardware:
	- CPU:Intel Core i7-4930k @3.40GHz
	- RAM: 32GB DDR3-1600
	- GPU: NVIDIA TITAN X 6GB RAM

- Tensorflow: 0.12

- Dataset:
	- Amazon Review: [ link ](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/) (Please download the processed_acl.tar.gz)


## Result
- Classification Accuracies (\%) for adaptation among different domains

<img src="figs/table.PNG">

-Visualization of shared features by t-SNE.(RED: source domain , BLUE: target domain)

|<img src="figs/shared_feature_ADSA_books_to_dvd.png" width="40%">|<img src="figs/shared_feature_ADSA_books_to_electronics.png" width="40%">
|:---------------------------------------------------------------:|:------------------------------------------------------------------------:
Books to DVDs                                                     |Books to Electronics

|<img src="figs/shared_feature_ADSA_books_to_kitchen.png" width="40%">|<img src="figs/shared_feature_ADSA_dvd_to_books.png" width="40%">
|:---------------------------------------------------------------:|:---------------------------------------------------------------------:
Books to Kitchen                                                  |DVDs to Books

|<img src="figs/shared_feature_ADSA_dvd_to_electronics.png" width="40%">|<img src="figs/shared_feature_ADSA_dvd_to_kitchen.png" width="40%">
|:---------------------------------------------------------------:|:---------------------------------------------------------------------:
DVDs to Electronics                                               |DVDs to Kitchen

|<img src="figs/shared_feature_ADSA_electronics_to_books.png" width="40%">|<img src="figs/shared_feature_ADSA_electronics_to_dvd.png" width="40%">
|:---------------------------------------------------------------:|:---------------------------------------------------------------------:
Electronics to Books                                              |electronics to DVDs

|<img src="figs/shared_feature_ADSA_electronics_to_kitchen.png" width="40%">|<img src="figs/shared_feature_ADSA_kitchen_to_books.png" width="40%">
|:---------------------------------------------------------------:|:---------------------------------------------------------------------:
Electronics to Kitchen                                            |Kitchen to Books

|<img src="figs/shared_feature_ADSA_kitchen_to_dvd.png" width="40%">|<img src="figs/shared_feature_ADSA_kitchen_to_electronics.png" width="40%">
|:---------------------------------------------------------------:|:---------------------------------------------------------------------:
Kitchen to DVDs                                                   |Kitchen to Electronics



