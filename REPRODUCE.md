# Steps to Reproduce the Experiments in the Paper

## Setup 
```
# After Cloning Repo and cd'ing into repo
conda create --name multivariate_ir python=3.8
conda activate multivariate_ir
conda install faiss-gpu pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip install -e .
pip install accelerate -U && pip install pytrec_eval ir_datasets

```



## Baseline models

### TAS-B (0Shot, pre-trained)

We used the [DistilBERT TAS-B pretrained on MS-MARCO](https://huggingface.co/sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco) for 
our zero-shot run. 


#### Hyperparam search
None

#### Obtaining run
Execute the following command to obtain the trained model:

```
sh run_scripts/tasb.sh
``` 


### DR Model

#### Hyperparam search

#### Training

#### Obtaining run

### TAS-B

#### Hyperparam search

TODO
#### Obtaining run




