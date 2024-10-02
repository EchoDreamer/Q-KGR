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

\begin{table*}[!ht]
    \centering
    \scalebox{0.73}{
    \begin{tabular}{llcccccc}
    \toprule
        LLM & Method & 0 & 10\%  & 20\% &  30\%  & 40\% & 50\%  \\ 
        \midrule
        \multirow{2}{*}{\textbf{FLAN-T5-XL}} 
        & w/o Q-KGR & 73.73 & 73.30 & 72.86 & 71.53 & 69.25 & 65.45  \\ 
        & Ours & 78.43 (+4.7) & 78.44 (+5.14)  & 78.30 (+5.44)  & 78.14 (+6.61) &  77.54 (+8.29)  & 76.28 (+10.83) \\ 
        \midrule
        \multirow{2}{*}{\textbf{FLAN-T5-XXL}} 
        & w/o Q-KGR & 79.61 & 79.53 & 78.70 & 76.01 & 74.31 & 72.27  \\ 
        & Ours & 80.98(+1.37) & 80.93(+1.40) & 80.56(+1.86) & 79.63 (+3.62) & 78.52 (+4.21) & 77.89 (+5.62)  \\ 
        \bottomrule
    \end{tabular}
    }
    \caption{Performances by adjusting the number of distractor nodes on the OBQA datasets.}
    \label{tab:noisy}
\end{table*}
