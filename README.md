# Skeleton-Based-Generation-Model
The Code for "A Skeleton-Based Model for Promoting Coherence Among Sentences in Narrative Story Generation" 
## Requirements
* ubuntu 16.04
* python 3.5
* tensorflow 1.4.1
* nltk 3.2.5
## Data Source
* [visual storytelling dataset](http://visionandlanguage.net/VIST/dataset.html)
* [compression dataset](https://github.com/google-research-datasets/sentence-compression/tree/master/data)

## The Processed Data
* Compression dataset:
Since the original dataset is too large, we only use a subset of this dataset. The processed data can be found at data/trainfeature02.json, data/testfeature02.json, data/validfeature02.json.

* Storytelling dataset:
The dataset is listed at data/story/train_process.txt, data/story/valid_process.txt, data/story/test_process.txt.

## Method Details
 1. First, we pre-train a sentence compression module. 
 2. Second, we use the pre-trained compression module to extract skeletons for storytelling dataset. The feature files for extracting skeleton are listed at data/story/train_sc.txt, data/story/valid_sc.txt, data/story/test_sc.txt. The extracted skeleton files are listed at data/0/train_skeleton.txt, data/0/valid_skeleton.txt, data/0/test_skeleton.txt.
 3. Third, we use the extracted skeletons to train the input-to-skeleton module and the skeleton-to-sentence module.
 4. Finally, we connect all modules by reinforcement learning. 
 
## Run
```bash
CUDA_VISIBLE_DEVICES=2 nohup bash run_train.sh > log_train.txt &
CUDA_VISIBLE_DEVICES=2 nohup bash run_test.sh > log_test.txt &
```
## Cite
To use this code, please cite the following paper:<br><br>
Jingjing Xu, Yi Zhang, Qi Zeng, Xuancheng Ren, Xiaoyan Cai, Xu Sun.
A Skeleton-Based Model for Promoting Coherence Among Sentences in Narrative Story Generation. EMNLP 2018.

bibtext:
```
@inproceedings{Skeleton-Based-Generation-Model,
  author    = {Jingjing Xu and Yi Zhang and Qi Zeng and Xuancheng Ren and Xiaoyan Cai and Xu Sun},
  title     = {A Skeleton-Based Model for Promoting Coherence Among Sentences in Narrative Story Generation},
  booktitle = {EMNLP},
  year      = {2018}
}
```
