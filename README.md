# Multivariate Dense Retrieval: A Reproducibility Study under a Memory-limited Setup

by **Georgios Sidiropoulos*** and **Samarth Bhargav***, **Panagiotis Eustratiadis**, **Evangelos Kanoulas**

**equal contribution*

Accepted Paper at **TMLR**

<details>
<summary><b>Abstract</b></summary>
The current paradigm in dense retrieval is to represent queries and passages as low-dimensional real-valued vectors 
using neural language models, and then compute query-passage similarity as the dot product of these vector 
representations. A limitation of this approach is that these learned representations cannot capture or express 
uncertainty. At the same time, information retrieval over large corpora contains several sources of uncertainty, 
such as misspelled or ambiguous text. Consequently, retrieval methods that incorporate uncertainty estimation are more 
likely to generalize well to such data distribution shifts.
The multivariate representation learning (MRL) framework proposed by Zamani & Bendersky (2023) is the first method 
that works in the direction of modeling uncertainty in dense retrieval. This framework represents queries and passages 
as multivariate normal distributions and computes query-passage similarity as the negative Kullback-Leibler (KL) 
divergence between these distributions. Furthermore, MRL formulates KL divergence as a dot product, allowing for 
efficient first-stage retrieval using standard maximum inner product search.

In this paper, we attempt to reproduce MRL under memory constraints (e.g., an academic computational budget). 
In particular, we focus on a memory-limited, single GPU setup. We find that the original work (i) introduces a 
typographical/mathematical error early in the formulation of the method that propagates to the rest of the original
paper's mathematical formulations, and (ii) does not fully specify certain important design choices that can strongly 
influence performance. In light of the aforementioned, we address the mathematical error and make some reasonable 
design choices when important details are unspecified. Additionally, we expand on the results from the original paper
with a thorough ablation study which provides more insight into the impact of the framework's different components. 
While we confirm that MRL can have state-of-the-art performance, we could not reproduce the results reported in the 
original paper or uncover the reported trends against the baselines under a memory-limited setup that facilitates fair 
comparisons of MRL against its baselines. Our analysis offers insights as to why that is the case. Most importantly,
our empirical results suggest that the variance definition in MRL does not consistently capture uncertainty.
</details>

**PDF**: https://openreview.net/pdf?id=wF3ZtSlOcT

This repository contains code to reproduce our reproducibility study on [Multivariate Representation Learning 
for IR](https://arxiv.org/abs/2304.14522). The code is built upon [Tevatron](https://github.com/texttron/tevatron), a simple and efficient toolkit for training 
and running dense retrievers with deep language models.


**Detailed instructions to reproduce the set of experiments described in our paper are outlined in [REPRODUCE.md](REPRODUCE.md).**


**Cite our paper**
```
TODO
```