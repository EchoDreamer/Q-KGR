# Question-guided Knowledge Graph Re-scoring and Injection for Knowledge Graph Question Answering
Figure 3, from the paper, is presented in the Qualitative Analysis section (6.1). 

Figure3_nodes is an extended version of Figure 3, containing more nodes. This expanded graph demonstrates that the edge scoring method predicts higher scores for question-relevant edges, rather than the high-relevance edges connecting to more question-and-answer nodes.

Question:
There is an ancient invention still used in some parts of the world today that allows people to see through walls. Fans is it.
Options:
(A) Fans     (B) window    (C) electric socket  
(D) talk      (E) kaleidoscope

Model Prediction: 
w/o QG-KGR:  (E) kaleidoscope        Ours:  (B) window



## Install 🛠️

1. Clone this repository and navigate to Q-KGR folder

```bash
git clone https://github.com/EchoDreamer/Q-KGR.git
cd Q-KGR
```

2. Install packages

```bash
conda create -n graphllm python=3.8
conda activate graphllm
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
cd transformers-4.33.3
pip install -e . 
bash setup.sh
```

