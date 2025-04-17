---
title: LLM Evaluations
date: 2024-07-07
math: true
---

## Metrics

- Question Answering
  - [Accuracy](#accuracy)
  - [Exact Match](#em)
  - [Precision](#precision)
  - [Recall](#recall)
  - [F-Measure](#f-measure)
- Text Generation Tasks
  - [Perplexity](#perplexity)
  - [BLEU](#bleu)
  - [ROUGE](#rouge)
  - [METEOR](#meteor)
  - [BERTScore](#bertscore)
  - [BLEURT](#bleurt)
- Document Retrieval Rank
  - [MRR](#mrr)
  - [NDCG](#ndcg)

### Accuracy

Accuracy is the proportion of correct predictions among the total. Mathematically, it is defined by,

$$
\mathrm{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}
$$

### EM

Exact Match (EM) is the percentage of predications that exact match the reference. Mathematically, it is defined by,

$$
\mathrm{Match} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}
$$

### Precision

Precision (also known as positive predictive value) is the fraction of relevant instances among of the prediction to be relevant, and thus it gives the proportion of correct **prediction (e.g., classification)**. Mathematically, it is defined by,

$$
\mathrm{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}
$$

#### Top-K Results

Precision is also applied to evaluate recommendation systems. By examining top-k relevant instances, precision is defined accordingly,

$$
\text{Precision}@k = \frac{\text{TP}_\text{top-k results}}{\text{TP}_\text{top-k results}+\text{FP}_\text{top-k results}}
$$

### Recall

Recall (also known as sensitivity) is the fraction of relevant instances among of the all relevant instances, and thus it gives the number of actual relevant that are **retrieved**. Mathematically, it is defined by,

$$
\mathrm{Recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

The difference between precision and recall can be distinguished by the fact that precision calculates the accuracy of truly predicted or classified instances among all retrieved instances, whereas recall calculates the accuracy of truly predicted or classified instances among all relevant instances.

> ![[confusion_matrix.png|300]]![[Pasted image 20240715155713.png|350]]
> An illustration of confusion matrix

### F-Measure

The $\textbf{F}_1$ score is the harmonic mean of the precision and recall, by taking both of which into consideration, it gives a fair representation of the model's accuracy despite the class imbalance. Mathematically, it is defined by,

$$
\mathbf{F}_1 = 2\frac{\mathrm{precision}\cdot\mathrm{recall}}{\mathrm{precision}+\mathrm{recall}} = \frac{2\mathrm{tp}}{2\mathrm{tp}+\mathrm{fp}+\mathrm{fn}}
$$

A more general F score, $\mathbf{F}_\beta$, uses a positive real factor $\beta$, where $\beta$ is chosen such that recall is considered $\beta$ times as important as precision, i.e.,

$$
\mathbf{F}_\beta = \frac{(1+\beta^2) \cdot R \cdot P}{R + \beta^2 \cdot P} = \frac{(1+\beta^2)\cdot\mathrm{tp}}{(1+\beta^2)\cdot\mathrm{tp}+\mathrm{fp}+\beta^2\cdot\mathrm{fn}}
$$

Commonly used $\mathbf{F}_2$ refers to the F-measure when $\beta=2$.

### Perplexity

Perplexity (PPL) is a measure of the likelihood of a model to generate the input text sequence. In other words, it describes how the model is confused by the real-world data compared to the development data. **It evaluates how well the model predicts the Next Word or Character based on the context provided by previous words or characters.** Note, PPL is not well defined for masked language models (MLM). Mathematically, it is a negative log probability of current item given all previous items and is defined by,

$$
\mathrm{PPL(X)} = \exp\left\{-\frac{1}{t}\sum_i^t\log p_\theta(x_i 
\vert x_{\lt i})\right\}
$$

A model of perplexity being 1 means that it is $100\%$ confident about the next token predicted or generated. The increasing PPL suggests that the model have PPL many options at each time steps on average, thus showing the level of uncertainty.

#### Implementation

```python
import torch

max_length = model.config.n_positions
stride = 512  # sliding window length
nlls = []     # list of negative log likelihood
prev = 0
for pos in range(0, seq_len, stride):
    end = min(pos + max_length, seq_len)
    trg_len = end - prev
    input_ids = encodings.input_ids[:, pos:end]
    tgt_ids = input_ids.clone()
    tgt_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=tgt_ids)
        neg_log_likelihood = outputs.loss
    nlls.append(neg_log_likelihood)
    prev = end
    if end == seq_len:
        break
ppl = torch.exp(torch.stack(nlls),.mean())
```

### BLEU

The Bilingual Evaluation Understudy (BLEU) measures similarity (or quality) between the machine-translated text and human-labeled reference based on n-gram overlap. BLEU often evaluates the model in machine translation tasks.

#### Modified N-gram Precision

The original n-gram precision exhaustively find all the matches between the reference texts and the candidate texts. However, for some systems that can over-generate "reasonable" words, it become problematic. Consider an example,

```
Candidate    the the the the the the the.
Reference 1: The cat is  on  the mat.
```

which is evaluated with standard unigram precision of $7/7$.

Therefore, n-gram precision shall be modified based on the intuition: "a reference word should be considered exhausted after a matching candidate word is identified."[^1] The modified n-gram precision is thus computed by the clipped n-gram counts for all the candidate sentences over the total number of candidate n-grams in the test corpus,

$$
p_n = \frac{\sum_{C\in\{Candidates\}}\sum_{gram_n\in C}Count_{clip}(gram_n)}{\sum_{C^\prime\in\{Candidates\}}\sum_{gram_n^\prime\in C^\prime}Count_{clip}(gram_n^\prime)}
$$

The modified n-gram precision now become $2/7$ (only $1^{st}$ "The" and $5^{th}$ "the" are accounted as matching) for the previous example.

#### Sentence Brevity Penalty

Brevity penalty is a penalty term to penalize the translation system for producing very short translations and yet still exhibit promising behavior. Instead, in order to be evaluated to high score, the length of candidate translation $\mathbf{c}$ must match that of reference translation $\mathbf{r}$. Note that modified n-gram precision has already penalized candidate translations which are longer than references. It is defined by,

$$
BP =
\begin{cases}
1\ & \text{if } c > r\\
e^{(1-r/c)}\ & \text{otherwise}
\end{cases}
$$

Finally,

$$
\text{BLEU}= BP \cdot \exp\left(\sum_{n=1}^N \text{w}_n \log p_n\right)
$$

where $\text{w}_n$ is a positive weight summing to one.

The ranking behavior emerges in the logarithm domain,

$$
\log \text{BLEU} = \min(1-\frac{r}{c}, 0)+\sum_{n=1}^N\text{w}_n\log p_n.
$$

It can be seen that BLEU ranges from 0 to 1. BLEU reaches its maximum value $1$ when two conditions are satisfied: 1) both candidate translation and reference translation are the exact match in length as in no brevity penalty; 2) modified n-gram precisions $p_n$ are the exact and thus $p_n = 1$. The conditions implies that a translation task can attain the score 1 only if both of candidate and reference are identical.

> Self-BLEU is a variant of BLEU for evaluation of intrinsic BLEU scores in a system. It is computed by first calculate the BLEU scores by choosing one sentence in the set of generated sentences as hypothesis and the others as reference, and then taking an average of BLEU scores overall.

[^1]: [BLUE: a Method of Automatic Evaluation of Machine Translation](https://dl.acm.org/doi/pdf/10.3115/1073083.1073135)

### ROUGE

The Recall-Oriented Understudy for Gisting Evaluation (ROUGE) measures the similarity between machine-generated text and human-generated summaries based on n-gram recall and precision. ROUGE evaluates the model in text summarization tasks and sometimes machine translation tasks.

ROGUE has four different measures based on different statistical basis: ROUGE-N, ROUGE-L, ROUGE-W, and ROUGE-S.

#### ROUGE-N: N-gram Co-Occurrence Statistics

ROUGE-N is an n-gram recall between a candidate summary and a set of reference summaries. ROUGE-N is computed by,

$$
\text{ROUGE-N} = \frac{\sum_{S\in\{Reference\}}\sum_{gram_n\in S}Count_{match}(gram_n)}{\sum_{S\in\{Reference\}}\sum_{gram_n\in S}Count(gram_n)}
$$

where $n$ refers to $n$-gram, and $Count_{match}(gram_n)$ is the maximum number of $n$-grams co-occurring in a candidate summary and a set of reference summaries. Simply put, it is the recall computed by the number of overlapping n-gram tokens between a candidate summary and a reference summary divided by the total number of tokens in reference.

The original paper instantiated ROUGE-N as a recall-based measure. Nowadays, ROUGE-N has been extended to F-measure-based measure.

$$
\begin{align}
R &= \frac{\sum_{S\in\{Reference\}}\sum_{gram_n\in S}Count_{match}(gram_n)}{\sum_{S\in\{Reference\}}\sum_{gram_n\in S}Count(gram_n)}\\[2ex]
P &=\frac{\sum_{S\in\{Reference\}}\sum_{gram_n\in S}Count_{match}(gram_n)}{\sum_{S\in\{Candidate\}}\sum_{gram_n\in S}Count(gram_n)}\\[2ex]
F_n&=\frac{(1+\beta^2)RP}{R + \beta^2P}
\end{align}
$$

where n-gram based F-measure, $F_n$, is ROUGE-N (hereafter the other ROUGE measures, and $\beta$ is normally set to 1.

When there are multiple references, we compute pairwise summary level ROUGE-N between a single candidate summary $s$ and every reference $r_i$ in the reference set. Then the final ROUGE-N score for multiple references is defined by,

$$
\text{ROUGE-N}_{multi}=\arg\max_i\text{ROUGE-N}(r_i, s)
$$

#### ROUGE-L: Longest Common Subsequence

Compared to ROUGE-N, ROUGE-L replaces n-gram based statistics with longest common subsequence. Supposing we have two summaries, a reference summary sentence $X$ of length $m$ and a candidate summary sentence $Y$ of length $n$, then we have recall and precision measures given by,

$$
\begin{align}
R_{lcs} &= \frac{LCS(X,Y)}{m}\\[2ex]
P_{lcs} &= \frac{LCS(X,Y)}{n}.
\end{align}
$$

The LCS-based F-measure to thus calculated by,

$$
F_{lcs} = \frac{(1+\beta^2)R_{lcs}P_{lcs}}{R_{lcs}+\beta^2P_{lcs}}
$$

where $LCS(X,Y)$ is the length of a longest common subsequence of $X$ and $Y$, and $\beta = P_{lcs}/R_{lcs}$. And F measure $F_{lcs}$ is called ROUGE-L.

#### ROUGE-W: Weighted Longest Common Subsequence

$$
\begin{align}
R_{wlcs}&=f^{-1}\left(\frac{WLCS(X,Y)}{f(m)}\right)\\[2ex]
P_{wlcs}&=f^{-1}\left(\frac{WLCS(X,Y)}{f(n)}\right)\\[2ex]
F_{wlcs}&=\frac{(1+\beta^2)R_{wlcs}P_{wlcs}}{R_{wlcs}+\beta^2P_{wlcs}}
\end{align}
$$

where $f$ is the weighting function satisfying the property, $f(x+y) > f(x)+f(y)$ for any positive integers $x$ and $y$. And $WLCS(X,Y)$ score is computed following dynamic programming introduced in the paper.

#### ROUGE-S: Skip-Bigram Co-Occurrence Statistics

$$
\begin{align}
R_{skip2}&=\frac{SKIP2(X,Y)}{C(m,2)}\\[2ex]
P_{skip2}&=\frac{SKIP2(X,Y)}{C(n,2)}\\[2ex]
F_{skip2}&=\frac{(1+\beta^2)R_{skip2}P_{skip2}}{R_{skip2}+\beta^2P_{skip2}}
\end{align}
$$

where $SKIP2(X,Y)$ is the number of skip-bigram matches between $X$ and $Y$, and $\beta$ is the relative importance of $P_{skip2}$ and $R_{skip2}$, and $C$ is the combination function counting the total number of bigrams.

> [ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013.pdf)

#### Implementation

```python
# Refer to Google's Python ROUGE Implementation
# https://github.com/google-research/google-research/tree/master/rouge/rouge_scorer.py)

def score(target: str, prediction: str, rouge_type: str):
    scores = []
    target_tokens = _tokenizer.tokenize(target)
    prediction_tokens = _tokenizer.tokenize(prediction)

    if rouge_type == "rougeL":
        scores = _score_lcs(target_tokens, prediction_tokens)

    elif rouge_type == "rougeLsum":
        def get_sents(text):
            sents = nltk.sent_tokenize(text)
            return [x for x in sents if len(x)]
        target_tokens_list = [_tokenizer.tokenize(s) for s in get_sents(target)]
        prediction_tokens_list = [_tokenizer.tokenize(s) for s in get_sents(prediction)]
        scores = _summary_level_lcs(target_tokens_list,
                                    prediction_tokens_list)

    elif re.match(r"rouge[0-9]$", rouge_type):
        n = int(rouge_type)[5:]
        target_ngrams = _create_ngrams(target_tokens, n)
        prediction_ngrams = _create_ngrams(prediction_tokens, n)
        scores = _score_ngrams(target_ngrams, prediction_ngrams)
    return scores

def _score_ngrams(target_ngrams: Dict[str, int], 
                  prediction_ngrams: Dict[str, int]):
    intersection_ngrams_count = 0
    for ngrams in target_ngrams:
        intersection_ngrams_count += min(target_ngrams[ngrams], 
                                         prediction_ngrams[ngrams])
        target_ngrams_count = sum(target_ngrams.values())
        predcition_ngrams_count = sum(prediction_ngrams.values())
        precision = intersection_ngrams_count / max(prediciton_ngrams_count, 1)
        recall = intersection_ngrams_count / max(target_ngrams_count, 1)
        return 2 * precision * recall / (precision + recall)
    
```

> [!warning] Limitations
> N-gram based metrics have two common drawbacks, including poor awareness of paraphrase where semantically correct expressions are penalized due to different form word choice, and failed capture for long-range dependencies. Overall, they do not correlate with human judgements.

### METEOR

Metric for Evaluation for Translation with Explicit Ordering (METEOR) was designed to explicitly address weaknesses in BLEU such as the lack of recall, use of high order n-grams, lacking of explicit word-matching, and use of geometric averaging of n-grams.

BLEU counts the number of exact matches for unigrams where *running* and *run* do not match, even though they share the same stem. On the contrary, METEOR has a "porter module" which takes stemmed unigrams into consideration. Other than that, there is also a "synonymy module". It identifies two synonymous unigrams and considers them a match. Same for paraphrase. Such matching modules start unigrams process off exact match like BLEU does. When there is non-exact match, the process fallback to other modules mentioned above.

METEOR allows relaxed matches by introducing external paraphrase resources. However, it results in a inevitable correlation between the quality of external resources and the evaluation results. Not to mention the difficulty of human annotating for multi-lingual tasks.

### BERTScore

BERTScore alleviate the limitation of N-gram based metrics with an eye toward learned dense token representations over conventional N-grams. As the computation of the recall, precision, and f-measure shares a lot in common, we will briefly introduce the recall metric $R_\text{BERT}$ to explain the idea.

The computation is composed by four steps: Contextual Embedding, Pairwise Cosine Similarity, Maximum Similarity, and Optional Importance Weighting.

#### Contextual Embedding

Given a tokenized reference sentence $x = \left<x_1,\cdots,x_k\right>$ and a tokenized candidate $\hat{x} = \left<\hat{x}_1,\cdots,\hat{x}_m\right>$, we generate contextual embeddings with BERT-based models for them, resulting in two sequence of vectors, $\mathbf{X}=\left<\mathbf{x}_1,\cdots,\mathbf{x}_k\right>$ and $\mathbf{\hat{X}}\left<\mathbf{\hat{x}}_1,\cdots,\mathbf{\hat{x}}_m\right>$. Except, unknown tokens are split into several commonly observed sequence of characters rather than original `<UKN>` token.

Authors recommend generating contextual embedding with a 24-layer RoBERTa model fine-tuned on the MNLI dataset, as well as using domain- or language-specified contextual embedding when possible (i.e., using BERT-based model pre-trained with Chinese corpus for evaluating Chinese tasks). Note that the model is only fine-tuned to obtain the optimal hyper-parameters - specifically the number of layers - instead of re-parameterization.

#### Pairwise Cosine Similarity

We group each token representation $\mathbf{x}_i$ obtained from the reference embedding $\mathbf{X}$ and $\mathbf{\hat{x}}_j$ obtained from the candidate embedding $\mathbf{\hat{X}}$ into pairs $\left(\mathbf{x}_i,\mathbf{\hat{x}}_j\right)$, where $i \le k$ and $j \le m$. The cosine similarity is thus computed by,

$$
CosSimilarity(\mathbf{x}_i,\mathbf{\hat{x}}_j)=\frac{\mathbf{x}_i^\top\mathbf{\hat{x}_j}}{||\mathbf{x}_i||\ ||\mathbf{\hat{x}}_j||}
$$
where we can reduce the calculation by pre-normalizing both vectors, so that the denominator is simplified to the constant $1$. And then, we calculate cosine similarity for every pair $\left(\mathbf{x}_i,\mathbf{\hat{x}}_j\right)$.

#### Maximum Similarity

The resulting similarity matrix has the dimension $\mathcal{R}^{k\times m}$, with $k$ rows and $m$ columns representing similarity between each token in the reference sentence and the candidate sentence respectively. Afterwards, we find the maximum similarity across each row, as in searching for the most similar token in the candidate sentence given by a particular token in the reference sentence. Hence, the recall of the BERTScore is defined by,

$$
R_\text{BERT}=\frac{1}{k}\sum_{x_i \in x}\max_{\hat{x}_j \in \hat{x}}\mathbf{x}_i^\top \mathbf{\hat{x}}_j.
$$

Accordingly, precision and $\mathbf{F}_1$ for BERTScore is defined by,

$$
\begin{align}
P_\text{BERT}&=\frac{1}{m}\sum_{\hat{x}_j \in \hat{x}}\max_{x_i \in x}\mathbf{x}_i^\top \mathbf{\hat{x}}_j,\\[2ex]
\text{F-1}_{BERT}&=2\frac{R_\text{BERT}\cdot P_\text{BERT}}{R_\text{BERT} + P_\text{BERT}}.
\end{align}
$$

> [!tip]
> A small tip of enhancing the understanding of calculation for recall and precision of BERTScore is to locate the maximum similarity horizontally and vertically. Since recall computes the accuracy among relevant instances (i.e., reference), we want to scan the matrix horizontally, resulting the most similar candidate token to each reference token. And precision computes the accuracy among retrieved/predicted instances (i.e., candidate), we want to scan the matrix vertically, resulting the most similar reference token to each candidate token.

#### Importance Weighting

The paper also provides an importance weighting method based on inverse documentation frequency ($\text{idf}$) scores computed from the test corpus. Remember that $\text{idf}$ score measures the reciprocal rarity a term across the corpus. It is used to penalize the common words across documents. Given $D$ reference sentences $\{x^{(d)}\}_{d=1}^D$, the $\text{idf}$ score of a token $w$ is

$$
\text{idf}(w) = -\log\frac{1}{D}\sum_{i=1}^D\mathbb{I}\left[w \in x^{(i)}\right]\ ,
$$

where $\mathbb{I}[\cdot]$ denotes an indicator function. A straightforward version is $-\log(\text{the number of token appeared in references}/D)$. The recall with $\text{idf}$ weight is then given by,

$$
R_\text{BERT}=\frac{\sum_{x_i \in x}\text{idf}(x_i)\max_{\hat{x}_j \in \hat{x}}\mathbf{x}_i^\top \mathbf{\hat{x}}_j}{\sum_{x_i \in x}\text{idf}(x_i)}\ .
$$

After experiments carried out by authors, it is shown that "microsoft/debertaxlarge-mnli" is the best model correlating with human evaluation.

### BLEURT

Both of BLEU and ROUGE relying on N-gram overlap (even ROUGE-L cannot be aware of paraphrase), as they are sensitive to lexical variation. As a result, "they cannot appropriately reward semantic or syntactic variations of a given reference."[^2]

BLEURT mitigate the problem by pre-training a fully learned text generation metric on large amounts of synthetic data based on BERT.

[^2]: [BLEURT: Learning Robust Metrics for Text Generation](https://aclanthology.org/2020.acl-main.704)

### MRR

Mean Reciprocal Rank (MRR) evaluates the efficiency of a ranking system that shows the first relevant instance in the top-K results. Mathematically, it is defined by,

$$
\text{MRR}@K=\frac{1}{U}\sum_{u=1}^U \frac{1}{\mathrm{rank}_u@K}
$$

where $U$ denotes the total number of users or queries in the evaluated dataset, and $\text{rank}_u$ denotes the rank position of the first relevant document for the $u$th query (i.e., $\text{rank}_u = \text{find}(query[u], response[u])$) among top-k retrieved documents.

The metric intuitively reflects how efficient an information retrieval system is performed. As the metric increases approaching to 1, the system is evaluated to be of the most efficiency because the first retrieved instance is precisely relevant. On the other hand, the result indicates that the system is incapable of retrieving relevant instances among $k$ instances.

### NDCG

Normalized Discounted Cumulative Gain (DCG) measures the effectiveness of search engine algorithms in the information retrieval field, by normalizing DCG. Mathematically, they are defined by,

$$
\begin{align*}
&\mathrm{CG}_p = \sum_{i=1}^p rel_i\\[2ex]
&\mathrm{DCG}_p = \frac{CG_p}{\sum_{i=1}^p\log_2(i+1)} =\sum_{i=1}^p\frac{rel_i}{\log_2(i+1)} \approx \sum_{i=1}^p\frac{2^{rel_i}-1}{\log_2(i+1)}\\[2ex]
&\mathrm{IDCG}_p=\sum_{i=1}^{\vert REL_p \vert}\frac{2^{rel_i}-1}{\log_2(i+1)}\\[2ex]
&\mathrm{nDCG}_p=\frac{\mathrm{DCG}_p}{\mathrm{IDCG}_p}
\end{align*}
$$

where $rel_i$ denotes the graded relevance of the result at position $i$, and $REL_p$ denotes the list of relevant documents in the corpus up to position $p$.

[Normalized Discounted Cumulative Gain (NDCG) explained](https://www.evidentlyai.com/ranking-metrics/ndcg-metric#how-to-compute-ndcg)

## Benchmarks

This note summarized popular benchmarks for the following natural language processing tasks,

- Natural Language Inference (NLI)
- Sentiment
- Paraphrase
- Coreference
- Translation
- Summarization
- [Word Embedding](#word-embedding)
- [General Language Understand Evaluation](#general-language-understand-evaluation)
- [Common Sense Reasoning](#common-sense-reasoning)
- [Closed-book Question Answering](#question-answering)
- [Reading Comprehension](#reading-comprehension)
- [Mathematical Reasoning](#mathematical-reasoning)
- [Code Generation](#code-generation)
- [Massive Multitask Language Understanding](#massive-multitask-language-understanding)

### Word Embedding

- [Word-in-Context (WiC)](https://pilehvar.github.io/wic/) is a benchmark for the evaluation of context-sensitive word embeddings. Meanwhile, the dataset can also be regarded as an application of Word Sense Disambiguation.
- [Massive Text Embedding Benchmark (MTEB)](https://github.com/embeddings-benchmark/mteb) is a massive benchmark for evaluation the performance of text embedding models on diverse embedding tasks.
- [ConceptNet](https://conceptnet.io) is a semantic network used to create word embeddings. The word embeddings can be multilingual, aligned across languages and to avoid representing harmful information.

### General Language Understand

- [General Language Understanding Evaluation (GLUE)](https://gluebenchmark.com/) is a benchmark for training, evaluating and analyzing natural language understanding systems, consisting of a benchmark dataset, a diagnostic dataset and a public leaderboard.
- [SuperGLUE](https://super.gluebenchmark.com/) is a benchmark styled after GLUE with a series of new functionalities, including
  - more difficult language understanding tasks,
  - software toolkits (evaluation tools),
  - and a public leaderboard.
- [SuperCLUE](https://arxiv.org/pdf/2307.15020) is a comprehensive Chinese LLM benchmark, including three sub-components: rating by actual users in a LLM battle platform (CArena), answering open-ended with single and multiple-turn dialogues (OPEN), and answering closed-ended questions.

### Commonsense Reasoning

- [Social Interaction QA (SIQA)](https://leaderboard.allenai.org/socialiqa/submissions/get-started) is a question answering benchmark for testing social commonsense intelligence, containing 38,000 5-way annotation (context-question-answerA-answerB-answerC) examples. The model is scored and evaluated based on prediction of the correct answer among answerA, answerB and answerC.
- [OpenBookQA](https://allenai.org/data/open-book-qa) is a question answering dataset, containing 5,957 multiple-choice elementary-level science questions. Whilst the dataset provides with a few number of core concepts (specifically, 1,326 core science facts), answering such a dataset correctly still requires some extensive common knowledge off-books. Interestingly, questions modeled by OpenBookQA is not solvable by retrieval-based algorithms nor word co-occurrence algorithms.
- ⭐ [HellaSwag](https://rowanzellers.com/hellaswag/)
- [Winograd Schema Challenge (WSC)](https://winogrande.allenai.org/) is a benchmark for commonsense reasoning, containing 273 expert-crafted pronoun resolution problems. Similar to OpenBookQA, questions modeled by WSC is not solvable by statistical approaches.
- ARC easy and challenge

### Question Answering

> Sometimes the task Question Answering is also referred as World Knowledge

- [Natural Questions](https://ai.google.com/research/NaturalQuestions) contains 30K training examples with single annotations, 7K validating examples with 5-way annotations and another 7K testing examples with 5-way annotations.
- [TriviaQA](https://nlp.cs.washington.edu/triviaqa/) contains over 650K question-answer-evidence triplets and 95K question-answer pairs.
- [BoolQ](https://github.com/google-research-datasets/boolean-questions?tab=readme-ov-file) is a question answering dataset for yes/no questions with 15,942 3-way annotation (i.e., a triplet of question-passage-answer format) examples.

### Reading Comprehension

- [RACE](https://www.cs.cmu.edu/~glai1/data/race/) is a large-scale reading comprehension dataset with 28,000 passages and nearly 100,000 questions collected from English examinations in China.
  - RACE has significantly longer context than other popular reading comprehension datasets.
- [Stanford Question Answering  Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles.
  - SQuAD v2.0 contains unanswerable questions that may need to be classified in the first place.

### Mathematical Reasoning

- MATH (Hendrycks et al., 2021)
- GSM8k

### Code Generation

- HumanEval
- MBPP

### Massive Multitask Language Understanding

> MMLU is referred as an aggregated benchmark as in its philosophy of multitask.

- ⭐ [Massive Multitask Language Understand Benchmark (MMLU)](https://arxiv.org/pdf/2009.03300) measures a language model's multitask accuracy, consisting 15908 multiple-choice questions across 4 super-categories, including Humanities, Social Science, STEM, and Others. There are 57 tasks at varying levels of difficulty in total.
- [Big Bench Hard]
- [AGI Eval]

### LLM Community Comparison

- ⭐ [LMSYS ChatBot Arena](https://chat.lmsys.org/?leaderboard)is a battle arena for LLMs, in which a user will chat with two anonymous chat LLMs. When getting responses from both, a user can vote for desirable models. The platform then rates two models based on the Elo Rating System.