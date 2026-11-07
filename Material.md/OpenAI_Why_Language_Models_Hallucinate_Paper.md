---
source: OpenAI_Why_Language_Models_Hallucinate_Paper.pdf
converter: PyMuPDF4LLM
converted: 2026-11-06T20:00:53.010418
tokens_approx: 19695
---

# Why Language Models Hallucinate

Adam Tauman Kalai<sup>∗</sup> Ofir Nachum Santosh S. Vempala<sup>†</sup> Edwin Zhang OpenAI OpenAI Georgia Tech OpenAI

September 4, 2025

##### **Abstract**

Like students facing hard exam questions, large language models sometimes guess when uncertain, producing plausible yet incorrect statements instead of admitting uncertainty. Such “hallucinations” persist even in state-of-the-art systems and undermine trust. We argue that language models hallucinate because the training and evaluation procedures reward guessing over acknowledging uncertainty, and we analyze the statistical causes of hallucinations in the modern training pipeline. Hallucinations need not be mysterious—they originate simply as errors in binary classification. If incorrect statements cannot be distinguished from facts, then hallucinations in pretrained language models will arise through natural statistical pressures. We then argue that hallucinations persist due to the way most evaluations are graded—language models are optimized to be good test-takers, and guessing when uncertain improves test performance. This “epidemic” of penalizing uncertain responses can only be addressed through a socio-technical mitigation: modifying the scoring of existing benchmarks that are misaligned but dominate leaderboards, rather than introducing additional hallucination evaluations. This change may steer the field toward more trustworthy AI systems.

## **1 Introduction**

Language models are known to produce overconfident, plausible falsehoods, which diminish their utility and trustworthiness. This error mode is known as “hallucination,” though it differs fundamentally from the human perceptual experience. Despite significant progress, hallucinations continue to plague the field, and are still present in the latest models (OpenAI, 2025a). Consider the prompt:

What is Adam Tauman Kalai’s birthday? If you know, just respond with DD-MM.

On three separate attempts, a state-of-the-art open-source language model<sup>1</sup> output three incorrect dates: “03-07”, “15-06”, and “01-01”, even though a response was requested only if known. The correct date is in Autumn. Table 1 provides an example of more elaborate hallucinations.

Hallucinations are an important special case of _errors_ produced by language models, which we analyze more generally using computational learning theory (e.g., Kearns and Vazirani, 1994). We consider general sets of _errors E_ , an arbitrary subset of plausible strings _X_ = _E ∪V_ , with the other plausible strings _V_ being called _valid_ . We then analyze the statistical nature of these errors, and

> ∗Email: `adam@kal.ai`

> †Supported in part by NSF award CCF-2106444 and a Simons Investigator award. Email: `vempala@gatech.edu`

> 1The language model was DeepSeek-V3 (600 B parameters), accessed via the DeepSeek app on 11 May 2025.

1

ChatGPT: Adam Tauman Kalai’s Ph.D. dissertation (completed in 2002 at CMU) is entitled: (GPT-4o) “Boosting, Online Algorithms, and Other Topics in Machine Learning.” DeepSeek: “Algebraic Methods in Interactive Machine Learning”. . . at Harvard University in 2005. Llama: “Efficient Algorithms for Learning and Playing Games”. . . in 2007 at MIT.

Table 1: Excerpts from responses to “What was the title of Adam Kalai’s dissertation?” from three popular language models.<sup>3</sup> None generated the correct title or year (Kalai, 2001).

apply the results for the type of errors of interest: plausible falsehoods called hallucinations. Our formalism also includes the notion of a _prompt_ to which a language model must respond.

The distribution of language is initially learned from a corpus of training examples, which inevitably contains errors and half-truths. However, we show that even if the training data were error-free, the objectives optimized during language model training would lead to errors being generated. With realistic training data containing shades of error, one may expect _even higher error rates_ . Thus, our lower bounds on errors apply to more realistic settings, as in traditional computational learning theory (Kearns and Vazirani, 1994).

Our error analysis is general yet has specific implications for hallucination. It applies broadly, including to reasoning and search-and-retrieval language models, and the analysis does not rely on properties of next-word prediction or Transformer-based neural networks. It only considers the two stages of the modern training paradigm: pretraining and post-training, described below. For hallucinations, taxonomies (Maynez et al., 2020; Ji et al., 2023) often further distinguish _intrinsic_ hallucinations that contradict the user’s prompt, such as:

How many Ds are in DEEPSEEK? If you know, just say the number with no commentary.

DeepSeek-V3 returned “2” or “3” in ten independent trials; Meta AI and Claude 3.7 Sonnet<sup>2</sup> performed similarly, including answers as large as “6” and “7”. Our theory also sheds light on _extrinsic_ hallucinations, which contradict the training data or external reality.

### **1.1 Errors caused by pretraining**

During pretraining, a _base model_ learns the distribution of language in a large text corpus. We show that, even with error-free training data, the statistical objective minimized during pretraining would lead to a language model that generates errors. Proving this is nontrivial because some models make no errors, such as one that always outputs “I don’t know” (IDK) or one that simply memorizes and reproduces an error-free corpus. Our analysis explains what types of errors should be expected after pretraining.

To do this, we draw a connection to binary classification. Consider questions of the form “Is this a valid language model output?” Generating valid outputs is in some sense harder than answering these Yes/No questions, because generation implicitly requires answering “Is this valid” about each candidate response. Formally, we consider the Is-It-Valid (IIV) binary classification problem which has a training set consisting of a large number of responses, each labeled either as valid (+) or error

> 2 `https://meta.ai` and `https://claude.ai` , accessed May 9, 2025.

> 3The models were freely accessed 2025-05-09 via `chatgpt.com` , the DeepSeek app (R1, DeepSeek-AI et al., 2025), and `huggingface.co` (Llama-4-Scout-17B-16E-Instruct), respectively. None of the models searched the Web.

2

**Valid examples + Error examples – Spelling** Greetings. Greatings. **(good model)** How can I help? How kan eye help? There are 2 D’s in LADDER. There are 3 L’s in SPELL. **Counting** There is 1 N in PIANO. There is 1 G in CAT. **(poor model)** Mia Holdner’s birthday is 4/1. Colin Merivale’s birthday is 8/29. **Birthdays** I don’t know Zdan’s birthday. Jago Pere’s birthday is 8/21. **(no pattern)**

Figure 1: Is-It-Valid requires learning to identify valid generations using labeled _±_ examples (left). Classifiers (dashed lines) may be accurate on certain concepts like spelling (top) but errors often arise due to poor models (middle) or arbitrary facts when there is no pattern in the data (bottom).

( _−_ ), as illustrated in Fig. 1. For this supervised learning problem, both train and test data are 50/50 mixtures of valid examples labeled as + (i.e., the pretraining data since we assume it is valid) and uniformly random errors from _E_ labeled as _−_ . We then show how any language model can be used as an IIV classifier. This in turn allows us to establish a mathematical relationship between generative errors (such as hallucinations) and IIV misclassification rate:

(generative error rate) ≳ 2 _·_ (IIV misclassification rate) _._

Language models avoid many types of errors such as spelling mistakes, and not all errors are hallucinations. The reduction from IIV misclassification to generation illuminates the statistical nature of generative errors. The analysis shows how pretraining directly contributes to errors. Furthermore, it shows that the same _statistical factors_ contributing to errors in binary classification also cause language model errors. Decades of research has shed light on the multifaceted nature of misclassification errors (Domingos, 2012). Fig. 1 (right) illustrates these factors visually: top, separable data classified accurately; middle, a poor model of a linear separator for a circular region; and bottom, no succinct pattern. Section 3.3 analyzes several factors, including the following stylized setting with epistemic uncertainty, when there is no pattern in the data.

This reduction ties together earlier work which covered different types of facts. For example, Kalai and Vempala (2024) considered a special case of arbitrary facts where there is no learnable pattern in the data, like the earlier birthday hallucination example. We show how the IIV reduction covers this case and recovers their bound that the hallucination rate, after pretraining, should be at least the fraction of training facts that appear once. For instance, if 20% of birthday facts appear exactly once in the pretraining data, then one expects base models to hallucinate on at least 20% of birthday facts. In fact, our analysis strengthens their result to include prompts and IDK responses, both essential components of hallucination.

### **1.2 Why hallucinations survive post-training**

The second stage, _post-training_ , refines the base model, often with a goal of reducing hallucinations. While the analysis of pretraining covered errors more generally, our analysis of post-training focuses on why overconfident hallucinations are generated rather than omitting information or expressing uncertainty such as IDK. We offer a socio-technical explanation for the persistence of hallucinations after post-training and discuss how the field can suppress them.

3

As an analogy, consider the following context where humans also occasionally fabricate plausiblesounding information. When uncertain, students may guess on multiple-choice exams and even bluff on written exams, submitting plausible answers in which they have little confidence. Language models are evaluated by similar tests. In both settings, guessing when unsure maximizes the expected score under a binary 0-1 scheme that awards 1 point for a correct answer and none for blanks or IDKs. Bluffs are often overconfident and specific, such as “September 30” rather than “Sometime in autumn” for a question about a date. Many language-model benchmarks mirror standardized human exams, using binary metrics such as accuracy or pass-rate. Optimizing models for these benchmarks may therefore foster hallucinations. Humans learn the value of expressing uncertainty outside of school, in the school of hard knocks. On the other hand, language models are primarily evaluated using exams that penalize uncertainty. Therefore, they are always in “test-taking” mode. Put simply, most evaluations are not aligned.

We are not the first to realize that binary grading does not measure hallucination. However, prior work on hallucination evaluation has generally sought after the elusive “perfect hallucination eval.” In Section 4, we argue that this is insufficient. We observe that existing primary evaluations overwhelmingly penalize uncertainty, and thus the root problem is the _abundance of evaluations that are not aligned_ . Suppose Model A is an aligned model that correctly signals uncertainty and never hallucinates. Let Model B be similar to Model A except that it never indicates uncertainty and always “guesses” when unsure. Model B will outperform A under 0-1 scoring, the basis of most current benchmarks. This creates an “epidemic” of penalizing uncertainty and abstention, which we argue that a small fraction of hallucination evaluations won’t suffice. The numerous primary evaluations must be adjusted to stop penalizing abstentions when uncertain.

**Contributions.** We identify the main statistical drivers of hallucinations, from their pretraining origins to their post-training persistence. A novel connection between supervised and unsupervised learning demystifies their origin, even when training data contain IDK. The persistence of hallucinations, despite extensive work on the problem, is explained by the recognition that hallucination-like guessing is rewarded by most primary evaluations. We discuss statistically rigorous modifications to existing evaluations that pave the way to effective mitigation.

## **2 Related work**

To the best of our knowledge, the reduction from supervised learning (binary classification) to unsupervised learning (density estimation or self-supervised learning) presented in this work is novel. The general method of reduction between learning problems, however, is a well-established technique for demonstrating that one problem is at least as hard as another (see, e.g., Beygelzimer et al., 2016).

A number of surveys and studies have explored the underlying causes of hallucination in language models. Sun et al. (2025) cite factors such as model overconfidence Yin et al. (2023), decoding randomness Lee et al. (2022), snowballing effects Zhang et al. (2023), long-tailed training samples Sun et al. (2023), misleading alignment training Wei et al. (2023), spurious correlations Li et al. (2022), exposure bias Bengio et al. (2015), the reversal curse Berglund et al. (2024), and context hijacking Jeong (2024). Analogous sources of error have long been studied in broader machine learning and statistical settings (Russell and Norvig, 2020).

The most closely related theoretical work is by Kalai and Vempala (2024), which we show is

4

a special case of our reduction. They connected the Good-Turing missing mass estimates (Good, 1953) to hallucinations, which inspired Theorem 3. However, that work does not address uncertainty expressions (e.g., IDK), connections to supervised learning, post-training modifications, and their model did not include prompts. Hanneke et al. (2018) analyze an interactive learning algorithm that queries a validity oracle (e.g., a human) to agnostically train a language model that minimizes hallucinations. Their method is statistically efficient, requiring a reasonable amount of data, but not computationally efficient. Other recent theoretical studies (Kalavasis et al., 2025; Kleinberg and Mullainathan, 2024) formalize an inherent trade-off between _consistency_ (avoiding invalid outputs) and _breadth_ (generating diverse, linguistically rich content). These works demonstrate that for broad classes of languages, any model that generalizes beyond its training data will either hallucinate invalid outputs or suffer mode collapse, failing to produce the full range of valid responses.

Several post-training techniques—such as reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022), reinforcement learning from AI feedback (RLAIF) (Bai et al., 2022), and direct preference optimization (DPO) (Rafailov et al., 2023)—have been shown to reduce hallucinations, including conspiracy theories and common misconceptions. Gekhman et al. (2024) show that simple fine-tuning on novel information can initially decrease hallucination rates, only for them to later increase. Further, it has been demonstrated that both natural language queries and internal model activations encode predictive signals about factual accuracy and model uncertainty (e.g., Kadavath et al., 2022). As discussed in our introduction, inconsistencies in a model’s answers to semantically related queries can also be leveraged to detect or mitigate hallucinations (Manakul et al., 2023; Xue et al., 2025; Agrawal et al., 2024).

Numerous other methods have proven effective in mitigating hallucinations; see, for example, the surveys by Ji et al. (2023) and Tian et al. (2024). In terms of evaluation, several comprehensive benchmarks and leaderboards have recently been introduced (e.g., Bang et al., 2025; Hong et al., 2024). However, relatively little work has examined barriers to their adoption. The 2025 AI Index report (Maslej et al., 2025), for instance, notes that hallucination benchmarks “have struggled to gain traction within the AI community.”

Beyond binary expressions of certainty, more nuanced linguistic constructions have been proposed to communicate gradations of uncertainty (Mielke et al., 2022; Lin et al., 2022a; Damani et al., 2025). Additionally, the field of pragmatics—which investigates how meaning is shaped by context—has increasing relevance for understanding and improving how language models convey information (Ma et al., 2025).

## **3 Pretraining Errors**

Pretraining produces a base language model _p_ ˆ that approximates the distribution text drawn from its training distribution _p_ . This is the classic “density estimation” problem in unsupervised learning, where a _density_ is simply a probability distribution over data. In the case of language models, the distribution is over text or multimodal inputs if included. The key challenge in proving that base models err is that many language models do not err. The degenerate model which always outputs IDK also avoids errors (assuming IDK is not an error). Similarly, assuming error-free training data, the trivial base model which regurgitates text from a random training example also does not err. However, these two language models fail at density estimation, the basic goal of statistical language modeling as defined below. Errors are also avoided ˆ by the optimal base model _p_ = _p_ which matches the training distribution, but this model would

5

require prohibitively large training data. Nonetheless, we show that well-trained base models should still generate certain types of errors.

Our analysis shows that generating valid outputs (i.e., avoiding errors) is harder than classifying output validity. This reduction enables us to apply the lens of computational learning theory, where errors are expected and understood, to error mechanisms in generative models. A language model is initially defined as a probability distribution over text and later _prompts_ are incorporated (Section 3.2); both settings share the same intuition. Examples without prompts include birthday statements such as those of Fig. 1, while a prompted model might be queried for a specific individual’s birthday.

**Not merely autocomplete.** Our analysis applies to general density estimation and not only “next-word predictors” even though many language models are trained using _self-supervised learning_ to predict each word based on the previous words. It is tempting to attribute hallucinations to poorly chosen prefixes (e.g., “Adam Kalai was born on”) for which the language model cannot provide valid completions. However, from a purely statistical perspective, ignoring computation, the autocomplete view<sup>4</sup> of language models is no more significant than the fact that any human speaker produces one word at a time. Our analysis suggests that errors arise from the very fact that the models are being fit to the underlying language distribution, though the specific architecture can introduce additional errors.

### **3.1 The reduction without prompts**

Without prompts, a base model _p_ ˆ is a probability distribution over a set _X_ . As discussed earlier, each _example x ∈X_ represents a “plausible” string, e.g., a document.<sup>5</sup> The examples _X_ = _E ∪V_ are partitioned into errors _E_ and valid examples _V_ , for nonempty disjoint sets _E, V_ . The error rate of base model _p_ ˆ is denoted by,

Training data are assumed to come from a noiseless _training distribution p_ ( _X_ ), i.e., where _p_ ( _E_ ) = 0. As discussed, with noisy training data and partly correct statements, one may expect _even higher error rates_ than our lower bounds.

We now formalize the IIV binary-classification problem, introduced in the introduction. IIV is specified by the target function _f_ : _X →{−,_ + _}_ to be learned (membership in _V_ ) and the distribution _D_ over examples _X_ (a 50/50 mix of samples from _p_ and uniformly random errors):

ˆ Our analysis lower bounds the error rate err = _p_ ( _E_ ) in terms of IIV’s aforementioned _misclassification rate_ erriiv:

> 4 Mathematically, any distribution _p_ induces a distribution of completions _p_ ( _wiwi_ +1 _. . . | w_ 1 _w_ 2 _. . . wi−_ 1) for every prefix of words _w_ 1 _. . . wi−_ 1 in its support.

> 5We assume that _X_ is finite for simplicity. See Section 5 for further discussion of errors and plausibility.

6

The base model is thus used as an IIV classifier, in our reduction, by thresholding the base model’s ˆ probability at a certain threshold 1 _/|E|_ . Note that such probabilities _p_ ( _x_ ) can generally be efficiently computed for base models (though efficient computation is not necessary for the lower-bounds to be meaningful).

**Corollary 1.** _For any training distribution p such that p_ ( _V_ ) = 1 _and any base model p_ ˆ _,_

_for_ err _,_ erriiv _from Eqs._ (1) _and_ (2) _, and δ_ := _|p_ ˆ( _A_ ) _− p_ ( _A_ ) _| for A_ := _{x ∈X | p_ ˆ( _x_ ) _>_ 1 _/|E|}._

Since this relationship holds for _any_ base model _p_ ˆ, it immediately implies that all base models will err on inherently unlearnable IIV facts (such as the birthdays absent from the training data) where erriiv is necessarily large, and where _δ_ and _|V|/|E|_ are small (e.g., for each person there are 364 times more incorrect birthday claims in _E_ than correct ones in _V_ , plus IDK). The corollary above follows immediately as a special case of Theorem 1 which covers the more general case with prompts. Theorem 2 later uses this general result to provide lower-bounds for an intuitive special case. Theorems 3 and 4 address small _|E|_ , e.g., _|E|_ = 1 for True/False questions. The constant 2 in the above bound is relatively tight: for large _|E|_ and small _δ_ , erriiv could be near 1/2 for unlearnable concepts while err _≤_ 1. Corollary 1 also implies that erriiv ≲ 1 _/_ 2.

**Hallucination errors.** To apply the error analysis to hallucinations, one may consider _E_ to be the set of plausible generations containing (one or more) plausible falsehoods. Note that a common alternate definition of hallucinations is as _generations that are not grounded in the training data_ (or prompt). Fortunately, the lower-bound above also applies to this notion because we have assumed only valid training data, i.e., a generated factual error cannot be grounded in factually correct training data.

**Calibration.** We now argue why _|δ|_ is a measure of (mis)calibration that is small after pretraining. Note that _without any knowledge of the language_ , one can achieve _δ_ = 0 by simply taking the ˆ ˆ uniform distribution _p_ ( _x_ ) = 1 _/|X|_ , and thus _δ_ = 0 does not require _p_ = _p_ . An auditor can trivially estimate _δ_ by comparing the fractions of responses satisfying _p_ ˆ( _x_ ) _>_ 1 _/|E|_ versus _p_ ˆ( _x_ ˆ) _>_ 1 _/|E|_ using ˆ ˆ sets of training samples _x ∼ p_ and synthetic generations _x ∼ p_ . Inspired by Dawid (1982), one may think of an analogy to a weather forecaster predicting the probability of rain each day. A minimal calibration requirement would be whether their average prediction matched the average fraction of rain. One could also require these two to match on days when the forecast was _> t_ for some threshold _t ∈_ [0 _,_ 1]. Dawid (1982) introduced the more stringent requirement that for _every t ∈_ [0 _,_ 1], among days on which the prediction is _t_ it rains about a _t_ fraction of the time.

Here is a particularly simple justification for why _δ_ is typically small for the standard pretraining cross-entropy objective,

Consider rescaling the probabilities of the positively-labeled examples by a factor _s >_ 0 and normalizing:

7

Figure 2: GPT-4 calibration histograms before (left) and after (right) reinforcement learning (OpenAI, 2023a, Figure 8, reprinted with permission). These plots are for multiple-choice queries where the plausible responses are simply A, B, C, or D. The pretrained model is well calibrated.

Then, a simple calculation shows that _δ_ is the magnitude of the derivative of the loss with respect to the scaling factor _s_ , evaluated at _s_ = 1:

If _δ_ = 0, then rescaling by some _s_ = 1 would reduce the loss, so the loss is not at a local minimum. For any class of language models powerful enough to approximate such simple rescaling, local optimization should yield small _δ_ . Note that _δ_ , being defined at a single threshold _t_ = 1 _/|E|_ is weaker than notions such as Expected Calibration Error (ECE) which integrate over thresholds _t_ .

**Hallucinations are inevitable** **_only for base models_ .** Many have argued that hallucinations are inevitable (Jones, 2025; Leffer, 2024; Xu et al., 2024). However, a non-hallucinating model could be easily created, using a question-answer database and a calculator, which answers a fixed set of questions such as “What is the chemical symbol for gold?” and well-formed mathematical calculations such as “3 + 8”, and otherwise outputs IDK. Moreover, the error lower-bound of Corollary 1 implies that language models which do not err must not be calibrated, i.e., _δ_ must be large. As our derivations show, calibration—and, hence, errors—is a natural consequence of the standard cross-entropy objective. Indeed, empirical studies (Fig. 2) show that _base models_ are often found to be calibrated, in contrast to post-trained models which may deviate from cross-entropy in favor of reinforcement learning.

### **3.2 The reduction with prompts**

Henceforth, we generalize the setting of Section 3.1 to include prompts (contexts) _c ∈C_ drawn from a _prompt distribution µ_ . Each example _x_ = ( _c, r_ ) now consists of a prompt _c_ and plausible response _r_ . The analysis above corresponds to the special case in which _µ_ assigns probability 1

8

to the empty prompt. For a given prompt _c ∈C_ , let _Vc_ := _{r |_ ( _c, r_ ) _∈V}_ be the valid responses and _Ec_ := _{r |_ ( _c, r_ ) _∈E}_ be the erroneous responses. The training distribution and base model are now conditional response distributions _p_ ( _r | c_ ) _,_ ˆ _p_ ( _r | c_ ). For notational convenience, we extend these to joint distributions on _X_ by _p_ ( _c, r_ ) := _µ_ ( _c_ ) _p_ ( _r | c_ ) and _p_ ˆ( _c, r_ ) := _µ_ ( _c_ ) _p_ ˆ( _r | c_ ), so that still err := _p_ ˆ( _E_ ) =<sup>�</sup> ( _c,r_ ) _∈E_<sup>_µ_(</sup><sup>_c_)ˆ</sup><sup>_p_(</sup><sup>_r| c_)and</sup><sup>_p_(</sup><sup>_E_) = 0.</sup>

Training distribution examples therefore correspond to valid “dialogues,” as in the case of distillation (Chiang et al., 2023; Anand et al., 2023). Although assuming that the training data contain model dialogues drawn from the same prompt distribution is unrealistic, even higher error rates may be expected when the assumption fails. The IIV problem with prompts has the same target function _f_ ( _x_ ) := + iff _x ∈V_ , but the generalized distribution _D_ selects, with equal probability either _x ∼ p_ or _x_ = ( _c, r_ ) for _c ∼ µ_ and uniformly random _r ∈Ec_ . Finally, the classifier _f_<sup>ˆ</sup> ( _c, r_ ) is now + iff _p_ ˆ( _r | c_ ) _>_ 1 _/_ min _c |Ec|_ . Corollary 1 is thus clearly a special case of,

**Theorem 1.** _For any training distribution p such that p_ ( _V_ ) = 1 _and any base model p_ ˆ _,_

_where δ_ := _|p_ ˆ( _A_ ) _− p_ ( _A_ ) _| for A_ := _{_ ( _c, r_ ) _∈X | p_ ˆ( _r | c_ ) _>_ 1 _/_ min _c |Ec|} ._

Generalizing the rescaling _p_ ˆ _s_ ( _r | c_ ) (normalizing per prompt, still with single parameter _s_ ) again justifies a small _δ_ = �� _dsd_<sup>_L_(ˆ</sup><sup>_ps_)</sup><sup>_|s_=1</sup> ��, now for _L_ (ˆ _p_ ) := �( _c,r_ ) _∈X_<sup>_−µ_(</sup><sup>_c_) log ˆ</sup><sup>_p_(</sup><sup>_r| c_).</sup>

### **3.3 Error factors for base models**

Decades of research have elucidated the statistical factors contributing to misclassifications (errors in binary classification). We can leverage this prior understanding to enumerate factors behind hallucinations and other generative errors, including: statistical complexity, as in birthdays (Section 3.3.1); _poor models_ , as in letter counting (Section 3.3.2); and additional factors like GIGO, as in conspiracy theories (Section 3.4).

#### **3.3.1 Arbitrary-fact hallucinations**

When there is no succinct pattern that explains the target function, there is epistemic uncertainty meaning that necessary knowledge is absent from the training data. The Vapnik-Chervonenkis dimension (Vapnik and Chervonenkis, 1971) VC( _F_ ) characterizes the worst-case number of examples needed to learn a family _F_ of functions _f_ : _X →{−,_ + _}_ , with high probability. Families with high VC( _F_ ) dimension may require prohibitively many samples to learn. We consider a natural special case of high VC dimension: random arbitrary facts. In particular, this section considers valid responses (other than IDK) which are random and independent across prompts.

**Definition 1** (Arbitrary Facts) **.** _The following are fixed: an arbitrary prompt distribution µ_ ( _c_ ) _, an_ IDK _response and, for each prompt c: a response set Rc and a probability of answering αc ∈_ [0 _,_ 1] _. Independently for each c, a single correct answer ac ∈Rc is chosen uniformly at random. Finally, p_ ( _ac | c_ ) = _αc and p_ (IDK _| c_ ) = 1 _− αc for each c ∈C. Thus Ec_ = _Rc \ {ac} and Vc_ = _{ac,_ IDK _}._

It is assumed that there is a single way to write any given fact, which can be done as in the lead birthday example where the format was specified. However, we again note that one may expect _even_

9

_more hallucinations_ with multiple ways to state each fact. In the case of fixed-format birthdays, _|Ec|_ = 364 and notable people whose birthdays are discussed often would have high _µ_ ( _c_ ). Notable birthdays like Einstein’s appear multiple times, whereas others may only occur once, e.g., in an obituary. Large language models seldom err on frequently referenced facts, e.g., Einstein’s birthday or dissertation title.

Our lower-bound for hallucinations is based on the fraction of prompts appearing just once in the training data, ignoring IDKs.

**Definition 2** (Singleton rate) **.** _A prompt c ∈C is a_ singleton _if it appears exactly once in the N training data_ �( _c_ ( _i_ ) _, r_ ( _i_ ))� _Ni_ =1<sup>_withoutabstention,i.e.,|{i_:</sup><sup>_c_(</sup><sup>_i_)=</sup><sup>_c ∧r_(</sup><sup>_i_)=IDK</sup><sup>_}|_=1</sup><sup>_.LetS⊆C_</sup> _denote the set of singletons and_

_denote the fraction of training singletons._

The singleton rate builds on Alan Turing’s elegant “missing-mass” estimator (Good, 1953), which gauges how much probability is still assigned to outcomes that have not yet appeared in a sample from a distribution. Concretely, Turing’s estimate of the unseen-event probability is the fraction of samples appearing exactly once. Intuitively, singletons act as a proxy for how many more novel outcomes you might encounter in further sampling, so their empirical share becomes the estimate for the entire “missing” portion of the distribution. We now state our bounds for Arbitrary Facts.

**Theorem 2** (Arbitrary Facts) **.** _In the Arbitrary Facts model, any algorithm which takes N training_ ˆ _samples and outputs p satisfies, with probability ≥_ 99% _over⃗a_ = _⟨ac⟩c∈C and the N training examples:_

_Moreover, there is an efficient algorithm outputting calibrated p_ ˆ _(δ_ = 0 _) that with probability ≥_ 99% _,_

An earlier version of this paper presented a related theorem that omitted prompts and abstentions (Kalai and Vempala, 2024). The proof is in Section B. Follow-up work by Miao and Kearns (2025) provides an empirical study of hallucinations, singleton rate, and calibration.

#### **3.3.2 Poor models**

Misclassifications can also arise when the underlying model is poor because: (a) the model family cannot represent the concept well, such as linear separators approximating circular regions, or (b) the model family is sufficiently expressive but the model itself is not a good fit. Agnostic Learning (Kearns et al., 1994) addresses (a) by defining the minimal error rate of any classifier in a given family _G_ of classifiers _g_ : _X →{−,_ + _}_ :

10

If opt( _G_ ) is large, then any classifier in _G_ will have high misclassification rate. In our case, given a language model _p_ ˆ _θ_ parameterized by _θ ∈_ Θ, consider the family of thresholded-language-model classifiers:

It follows immediately from Theorem 1 that

When exactly one correct response exists per context (i.e., standard multiple choice, without IDK), the calibration term can be removed and bounds can be achieved even for _C_ = 2 choices.

**Theorem 3** (Pure multiple-choice) **.** _Suppose |Vc|_ = 1 _for all c ∈C and let C_ = min _c |Ec|_ + 1 _be the number of choices. Then,_

To illustrate, consider the classic trigram language model where each word was predicted based only on the prior two words, i.e., a context window of just two words. Trigram models were dominant in the 1980s and 1990s. Trigram models, however, regularly output ungrammatical sentences. Consider the following prompts and responses:

**Corollary 2.** _Let µ be uniform over {c_ 1 _, c_ 2 _}. Then any trigram model must have a generation error rate of at least 1/2._

This follows from Theorem 3 because _C_ = 2 and opt( _G_ ) = 1 _/_ 2 for trigram models. The proofs of Theorem 3 and Corollary 2 are in Section C. Although _n_ -gram models can capture longer-range dependencies for larger _n_ , data requirements scale exponentially in _n_ .

We now revisit the letter-counting example from the introduction. To see that this is a poor model issue, note that the DeepSeek-R1 reasoning model reliably counts letters, e.g., producing a 377-chain-of-thought that includes:

Let me spell it out: D-E-E-P-S-E-E-K.

First letter: D — that’s one D. Second letter: E — not D. Third letter: E — not D. . . So, the number of Ds is 1.

Assuming similar training data, this suggests that R1 is a better model for the task than the DeepSeekV3 model. One representational challenge that reasoning overcomes is that modern language models represent prompts by _tokens_ , e.g., D/EEP/SEE/K, rather than individual characters (DeepSeek-AI et al., 2025).

11

### **3.4 Additional factors**

Errors may occur due to a combination of multiple factors, including the ones discussed above and several others. Here, we highlight a few.

- **Computational Hardness** . No algorithm run on a classical computer, even an AI with superhuman capabilities, can violate the laws of computational complexity theory. Indeed, AI systems have been found to err on computationally hard problems (Xu et al., 2024). Observation 2 of Section D illustrates how Theorem 1 applies to intractable queries of the form _“What is the decryption of c?”_ and IDK is a valid answer.

- **Distribution shift** . A well-known challenge in binary classification is that training and test data distributions often diverge (Qui˜nonero-Candela et al., 2009; Moreno-Torres et al., 2012). Analogously, errors in language models often stem from out-of-distribution (OOD) prompts that differ substantially from the training distribution. A question such as, “What’s heavier, a pound of feathers or a pound of lead?” may be unlikely in the training data and may induce erroneous answers in certain models. Similarly, distribution shift could be a factor in the letter-counting example above, though the fact that reasoning models correctly count letters suggests that the poor models may be a greater factor.

- **GIGO: Garbage in, Garbage out** . Large training corpora often contain numerous factual errors, which may be replicated by base models. The statistical similarity of GIGO for both classification and pretraining is self evident, and hence we do not provide a formal treatment. However, it is important to recognize GIGO among statistical factors, as language models have been shown to replicate errors from training data (Lin et al., 2022b; Levy et al., 2021; Alber et al., 2025).

GIGO also offers a natural segue to the topic of post-training, which decreases certain GIGO errors, such as common misconceptions and conspiracy theories (Ouyang et al., 2022; OpenAI, 2023a; Costello et al., 2024). The next section explains why some hallucinations persist—and may even be exacerbated—by current post-training pipelines.

## **4 Post-training and hallucination**

Post-training should shift the model from one which is trained like an autocomplete model to one which does not output confident falsehoods (except when appropriate, e.g., when asked to produce fiction). However, we claim that further reduction of hallucinations is an uphill battle, since existing benchmarks and leaderboards reinforce certain types of hallucination. We therefore discuss how to stop this reinforcement. This is a socio-technical problem in the sense that, not only do the existing evaluations need to be modified, but these changes need to be adopted in the influential leaderboards.

### **4.1 How evaluations reinforce hallucination**

Binary evaluations of language models impose a false right-wrong dichotomy, award no credit to answers that express uncertainty, omit dubious details, or request clarification. Such metrics, including accuracy and pass rate, remain the field’s prevailing norm, as argued below. Under binary

12

grading, abstaining is strictly sub-optimal. IDK-type responses are maximally penalized while an overconfident “best guess” is optimal. The motivation combines two desirable factors: (a) the rate of accuracy among what is output by the language model, and (b) how comprehensive responses are. However, weighing (a) more than (b) is important for reducing hallucinations.

Formally, for any given question in the form of a prompt _c_ , denote the set of plausible responses (valid or error) by _Rc_ := _{r |_ ( _c, r_ ) _∈X}_ . Further, suppose there is a set of plausible abstention responses _Ac ⊂Rc_ (e.g., IDK). A _grader gc_ : _Rc →_ R is said to be _binary_ if _{gc_ ( _r_ ) _| r ∈Rc}_ = _{_ 0 _,_ 1 _}_ and _gc_ ( _r_ ) = 0 for all _r ∈Ac_ . A _problem_ is defined by ( _c, Rc, Ac, gc_ ) where the test-taker knows _c, Rc, Ac_ . We assume that the test-taker knows that the rubric is binary but is not told the correct answers, where _gc_ ( _r_ ) = 1. The test-taker’s beliefs about the correct answer can be viewed as a posterior distribution _ρc_ over binary _gc_ ’s. For any such beliefs, the optimal response is not to abstain.

**Observation 1.** _Let c be a prompt. For any distribution ρc over binary graders, the optimal response(s) are not abstentions, i.e.,_

Although the proof is trivial (see Section E), Observation 1 suggests that _existing evaluations may need to be modified_ . Table 2 summarizes the short meta-evaluation analysis in Section F, finding that the vast majority of popular evaluations have binary grading. Therefore, additional hallucination evaluations may not suffice when the primary evaluations penalize honestly reporting confidence and uncertainty. This does not diminish existing work on hallucination evaluations but rather points out that even the ideal hallucination evaluation and ideal post-training methodology, yielding honest reports of uncertainty, may still be drowned out because of inferior performance on the vast majority of the existing evaluations.

### **4.2 Explicit confidence targets**

Human tests are similarly mostly binary, and it has been recognized that they also reward overconfident bluffing. Of course, exams are only a small component of human learning, e.g., fabricating birthdays will quickly result in embarrassment. Nonetheless, some standardized national exams operate or have operated using penalties for incorrect answers (or equivalently partial credit for abstaining), including Indian JEE, NEET, and GATE exams; AMC tests from the Mathematical Association of America; and US standardized SAT, AP, and GRE tests in earlier years. Importantly, the grading system is clearly stated in the instructions, and test takers are often aware of the confidence threshold beyond which it makes sense to make their best guess.

Similarly, we propose evaluations explicitly state _confidence targets_ in their instructions, within the prompt (or system message). For example, one could append a statement like the following to each question:

Answer only if you are _> t_ confident, since mistakes are penalized _t/_ (1 _− t_ ) points, while correct answers receive 1 point, and an answer of “I don’t know” receives 0 points.

There are several natural values of _t_ including _t_ = 0 _._ 5 (penalty 1), _t_ = 0 _._ 75 (penalty 2), and _t_ = 0 _._ 9 (penalty 9). A threshold of _t_ = 0 corresponds to binary grading and could be described by, e.g., “Make your best guess even if you are unsure, as if you were taking an exam.” A simple calculation

13

Table 2: Summary of evaluation benchmarks analyzed in this work and their treatment of abstentions. “Binary grading” indicates that the primary metric is a strict correct/incorrect accuracy; “IDK credit” denotes whether abstentions can earn any credit.

|**Benchmark**|**Scoring method**|**Binary grading**|**IDK credit**|
|---|---|---|---|
|GPQA|Multiple-choice accuracy|Yes|None|
|MMLU-Pro|Multiple-choice accuracy|Yes|None|
|IFEval|Programmatic instruction verifcation|Yes<sup>a</sup>|None|
|Omni-MATH|Equivalence grading<sup>_∗_</sup>|Yes|None|
|WildBench|LM-graded rubric<sup>_∗_</sup>|No|Partial<sup>b</sup>|
|BBH|Multiple-choice / exact-match|Yes|None|
|MATH (L5 split)|Equivalence grading<sup>_∗_</sup>|Yes|None|
|MuSR|Multiple-choice accuracy|Yes|None|
|SWE-bench|Patch passes unit tests|Yes|None|
|HLE|Multiple-choice / equivalence grading<sup>_∗_</sup>|Yes|None|

> _∗_ Grading is performed using language models, hence incorrect _bluffs_ may occasionally be scored as correct.

> a IFEval aggregates several binary rubric sub-scores into a composite score.

> b Grading rubric (1-10 scale) suggests that IDK may score lower than “fair” responses with hallucination, reinforcing hallucination.

shows that the expected score of offering an answer beats IDK (score 0) iff its confidence (i.e., probability of being correct) is _> t_ .

Such penalties have been well-studied within hallucination research (Ji et al., 2023). However, we suggest two subtle variations which have statistical ramifications. First, we propose making the confidence threshold explicit in the instructions, whereas the prior work has largely omitted mentioning the confidence targets or penalties in the instructions. (A notable exception is the work of Wu et al. (2025) who introduce “risk-informing” prompts with explicit penalties.) The ideal penalty might reflect likely real-world harms, but that is impractical as it is specific to the problem, the target application, and the user group. Without transparent specification within the instructions, it would be difficult to achieve consensus among language-model creators on the correct thresholds. Similarly, students might bicker that grading is unfair given instructions that there is an unspecified penalty for errors. Instead, specifying confidence thresholds explicitly in each problem’s instructions supports objective grading even if the specific thresholds chosen are somewhat arbitrary or even random. A single model may be best across all thresholds, if the threshold is explicit. However, if the threshold is not stated, then there is an inherent tradeoff, and no single model will be best in general (other than one that is always correct).

Second, we suggest incorporating confidence targets into existing mainstream evaluations, such as the popular SWE-bench (Jimenez et al., 2024) which involves binary grading of software patches, while the majority of prior work has introduced implicit error penalties in bespoke hallucination evaluations. Merely adding evaluations with implicit error penalties faces the aforementioned accuracy-error tradeoff. On the other hand, incorporating confidence targets into the established evaluations, already in use, reduces the penalty for appropriate expressions of uncertainty. It may thus amplify the effectiveness of hallucination-specific evaluations.

With explicit confidence targets, there is one behavior which is simultaneously optimal for all targets—outputting IDK among examples where its correctness probability is greater than the

14

target. Let us refer to this as _behavioral calibration_ –rather than requiring the model to output a probabilistic confidence (Lin et al., 2022a), it must formulate the most useful response in which it is at least _t_ confident. Behavioral calibration can be audited by comparing accuracy and error rates across thresholds, and circumvents the problem that there may be exponentially many ways to phrase correct responses (Farquhar et al., 2024). Existing models may or may not exhibit behavioral calibration, but it may prove useful as an objective evaluation.

## **5 Discussion and limitations**

It is difficult for the field to agree upon how to define, evaluate and reduce hallucinations due to their multifaceted nature. A statistical framework must prioritize certain aspects and omit others, for simplicity. Several notes are in order about the extent and limitations of the framework used herein.

**Plausibility and nonsense.** A hallucination is a plausible falsehood, and by considering only plausible strings _X_ , our analysis ignores the possibility of generating nonsensical strings (which state-of-the-art language models rarely generate). However, the statement and proof of Theorem 1 hold with the modified definitions of nonsensical examples _N_ with partition _X_ = _N ∪E ∪V_ , err := _p_ ˆ( _N ∪E_ ), _D_ ( _N_ ) = 0, and the assumption that _p_ ( _V_ ) = 1.

**Open-ended generations.** For simplicity, the examples presented in this paper are oriented towards a single factual question. However, hallucinations often arise for open-ended prompts, such as “Write a biography about. . . .” This can be fit into our framework by defining a response containing one or more falsehoods to be an error. However, in such a case it would be natural to consider degrees of hallucination depending on how many errors there are.

**Search (and reasoning) are not panaceas.** A number of studies have shown how language models augmented with search or Retrieval-Augmented Generation (RAG) reduce hallucinations (Lewis et al., 2020; Shuster et al., 2021; Nakano et al., 2021; Zhang and Zhang, 2025). However, Observation 1 holds for arbitrary language models, including those with RAG. In particular, the binary grading system itself still rewards guessing whenever search fails to yield a confident answer. Moreover, search may not help with miscalculations such as in the letter-counting example, or other intrinsic hallucinations.

**Latent context.** Some errors cannot be judged by the prompt and response alone. For example, suppose a user asks a question about phones and the language model provides a response about cellphones, but the question was intended to be about land lines. Such ambiguities do not fit our error definition which does not depend on context external to the prompt and response. It would be interesting to extend the model to allow for “hidden context” that are not part of the prompt given to the language model, but which could be used for judging errors, relating to _aleatoric uncertainty_ .

**A false trichotomy.** Our formalism does not distinguish between errors of different magnitudes or degrees of uncertainty. Clearly, the correct/incorrect/IDK categories are also incomplete. Although the statistical ideal might be to score each evaluation just as we would like to score the language model in the downstream application, explicit confidence targets offer a practical, objective modification to mainstream evaluations, and a false trichotomy may at least offer an IDK option unlike a false dichotomy.

15

**Beyond IDK.** There are numerous ways to signal uncertainty, such as hedging, omitting details, and asking questions. Ultimately language models may adhere to confidence notions such as linguistic calibration (Mielke et al., 2022; Damani et al., 2025). However, the pragmatic phenomena of language (Austin, 1962; Grice, 1975) are nuanced. For example, while there are instances where it may be useful for language models to explicitly state probabilistic confidence estimates (Lin et al., 2022a), this can also lead to unnatural utterances, such as, “I’m 1/365 certain that Kalai’s birthday is March 7th.” The present paper focuses on the statistical factors regarding the top-level decision of what is said.

## **6 Conclusions**

This paper demystifies hallucinations in modern language models, from their origin during pretraining to their persistence through post-training. In pretraining, we show that generative errors parallel misclassifications in supervised learning, which are not mysterious, and naturally arise due to the minimization of cross-entropy loss.

Many language model shortcomings can be captured by a single evaluation. For example, overuse of the opener “Certainly” can be addressed by a single _“Certainly” eval_ (Amodei and Fridman, 2024) because starting responses with “Certainly” does not significantly impact other evaluations. In contrast, we argue that the majority of mainstream evaluations reward hallucinatory behavior. Simple modifications of mainstream evaluations can realign incentives, rewarding appropriate expressions of uncertainty rather than penalizing them. This can remove barriers to the suppression of hallucinations, and open the door to future work on nuanced language models, e.g., with richer pragmatic competence (Ma et al., 2025).

**Acknowledgments.** We would like to thank Alex Beutel, Tom Cunningham, Yann Dubois, Parikshit Gopalan, Johannes Heidecke, Zoe Hitzig, Saachi Jain, Manas Joglekar, Sanjay Kairam, Ehud Kalai, Amin Karbasi, Alan Luo, Anay Mehrotra, Eric Mitchell, Cameron Raymond, David G. Robinson, Mandip Shah, Joshua Vendrow, Grigoris Velegkas, Rose Wang, Zhigang Wang, Jason Wolfe, and Jason Wei for helpful discussions.

## **References**

- Ayush Agrawal, Mirac Suzgun, Lester Mackey, and Adam Kalai. 2024. Do Language Models Know When They’re Hallucinating References?. In _Findings of the Association for Computational Linguistics: EACL 2024_ . Association for Computational Linguistics, St. Julian’s, Malta, 912–928. `https://doi.org/10.18653/v1/2024.findings-eacl.62`

- Daniel Alexander Alber, Zihao Yang, Anton Alyakin, Eunice Yang, Sumedha Rai, Aly A. Valliani, et al. 2025. Medical large language models are vulnerable to data-poisoning attacks. _Nature Medicine_ 31, 2 (2025), 618–626. `https://doi.org/10.1038/s41591-024-03445-1`

- Dario Amodei and Lex Fridman. 2024. _Dario Amodei: Anthropic CEO on Claude, AGI & the Future of AI & Humanity — Lex Fridman Podcast #452 (Transcript)_ . Lex Fridman Podcast. `https://lexfridman.com/dario-amodei-transcript/`

16

- Yuvanesh Anand, Zach Nussbaum, Brandon Duderstadt, Benjamin Schmidt, and Andriy Mulyar. 2023. GPT4All: Training an Assistant-Style Chatbot with Large-Scale Data Distillation from GPT-3.5-Turbo. `https://github.com/nomic-ai/gpt4all`

- J. L. Austin. 1962. _How to Do Things with Words_ . Oxford University Press, Oxford. Edited by J. O. Urmson and Marina Sbis`a.

- Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosuite, Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer, Noemi Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume, Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan. 2022. Constitutional AI: Harmlessness from AI Feedback. arXiv:2212.08073 [cs.CL] `https://arxiv.org/abs/2212.08073`

- Yejin Bang, Ziwei Ji, Alan Schelten, Anthony Hartshorn, Tara Fowler, Cheng Zhang, Nicola Cancedda, and Pascale Fung. 2025. HalluLens: LLM Hallucination Benchmark. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Association for Computational Linguistics, Vienna, Austria, 24128–24156. `https: //doi.org/10.18653/v1/2025.acl-long.1176`

- Samy Bengio, Oriol Vinyals, Navdeep Jaitly, and Noam Shazeer. 2015. Scheduled sampling for sequence prediction with recurrent neural networks. _Advances in neural information processing systems_ 28 (2015).

- Lukas Berglund, Meg Tong, Maximilian Kaufmann, Mikita Balesni, Asa Cooper Stickland, Tomasz Korbak, and Owain Evans. 2024. The Reversal Curse: LLMs trained on “A is B” fail to learn “B is A”. In _The Twelfth International Conference on Learning Representations_ .

- Alina Beygelzimer, Hal Daum´e III, John Langford, and Paul Mineiro. 2016. Learning Reductions That Really Work. _Proc. IEEE_ 104, 1 (2016), 136–147. `https://doi.org/10.1109/JPROC. 2015.2494118`

- Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality. `https: //lmsys.org/blog/2023-03-30-vicuna/`

- Thomas H. Costello, Gordon Pennycook, and David G. Rand. 2024. Durably reducing conspiracy beliefs through dialogues with AI. _Science_ 385, 6714 (Sept. 2024), eadq1814. `https://doi.org/ 10.1126/science.adq1814`

- Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, and Jacob Andreas. 2025. Beyond Binary Rewards: Training LMs to Reason About Their Uncertainty. `https://doi.org/10.48550/arXiv.2507.16806` arXiv:2507.16806 [cs.LG]

17

- A. P. Dawid. 1982. The Well-Calibrated Bayesian. _J. Amer. Statist. Assoc._ 77, 379 (Sept. 1982), 605–610. `https://doi.org/10.1080/01621459.1982.10477856`

- DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z.F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, and 178 others. 2025. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. `https://doi.org/10.48550/arXiv.2501.12948` arXiv:2501.12948 [cs.CL]

- Pedro Domingos. 2012. A Few Useful Things to Know About Machine Learning. _Commun. ACM_ 55, 10 (2012), 78–87. `https://doi.org/10.1145/2347736.2347755`

- Lizhou Fan, Wenyue Hua, Lingyao Li, Haoyang Ling, and Yongfeng Zhang. 2024. NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes. In _Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024)_ . Association for Computational Linguistics, Bangkok, Thailand, 4092–4114. `https: //doi.org/10.18653/v1/2024.acl-long.225`

- Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. _Nature_ 630 (jun 2024), 625–630. `https: //doi.org/10.1038/s41586-024-07421-0`

- Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. 2024a. Omni-MATH: A Universal Olympiad Level Mathematic Benchmark for Large Language Models. `https://doi.org/10.48550/arXiv.2410.07985` arXiv:2410.07985 [cs.CL]

- Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. 2024b. The Language Model Evaluation Harness. `https://doi.org/10.5281/zenodo.12608602`

- Zorik Gekhman, Gal Yona, Roee Aharoni, Matan Eyal, Amir Feder, Roi Reichart, and Jonathan Herzig. 2024. Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?. In _Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing_ , Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 7765–7784. `https://doi.org/10.18653/v1/2024.emnlp-main.444`

- Oded Goldreich. 2001. _Foundations of Cryptography: Volume 1, Basic Tools_ . Cambridge University Press, Cambridge, United Kingdom.

- I. J. Good. 1953. The Population Frequences of Species and the Estimation of Population Parameters. _Biometrika_ 40, 3-4 (Dec. 1953), 237–264. `https://doi.org/10.1093/biomet/40.3-4.237`

- Google DeepMind. 2025. Gemini 2.5 Pro Model Card. `https://storage.googleapis.com/ model-cards/documents/gemini-2.5-pro.pdf` . Accessed: 27 Jun 2025..

18

- H. P. Grice. 1975. Logic and Conversation. In _Syntax and Semantics, Vol. 3: Speech Acts_ , Peter Cole and Jerry L. Morgan (Eds.). Academic Press, New York, 41–58.

- Steve Hanneke, Adam Tauman Kalai, Gautam Kamath, and Christos Tzamos. 2018. Actively Avoiding Nonsense in Generative Models. In _Proceedings of the 31st Conference on Learning Theory (Proceedings of Machine Learning Research, Vol. 75)_ , S´ebastien Bubeck, Vianney Perchet, and Philippe Rigollet (Eds.). PMLR, Stockholm, Sweden, 209–227. `https://proceedings.mlr. press/v75/hanneke18a.html`

- Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021. Measuring Mathematical Problem Solving with the MATH Dataset. arXiv:2103.03874 [cs.LG] `https://arxiv.org/abs/2103.03874`

- Giwon Hong, Aryo Pradipta Gema, Rohit Saxena, Xiaotang Du, Ping Nie, Yu Zhao, Laura PerezBeltrachini, Max Ryabinin, Xuanli He, Cl´ementine Fourrier, and Pasquale Minervini. 2024. The Hallucinations Leaderboard – An Open Effort to Measure Hallucinations in Large Language Models. arXiv:2404.05904 [cs.CL] `https://arxiv.org/abs/2404.05904`

- Hugging Face. 2024. Open LLM Leaderboard v2 Collection. `https://huggingface.co/spaces/ open-llm-leaderboard/blog` . Accessed: 26 June 2025.

- Joonhyun Jeong. 2024. Hijacking Context in Large Multi-modal Models. In _ICLR 2024 Workshop on Reliable and Responsible Foundation Models_ .

- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Delong Chen, Wenliang Dai, Ho Shu Chan, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in Natural Language Generation. _Comput. Surveys_ 55, 12, Article 248 (2023), 248:1–248:38 pages. `https://doi.org/10.1145/3571730`

- Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R. Narasimhan. 2024. SWE-bench: Can Language Models Resolve Real-world GitHub Issues?. In _Proceedings of the 12th International Conference on Learning Representations (ICLR)_ . `https: //proceedings.iclr.cc/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84`

- Nicola Jones. 2025. AI hallucinations can’t be stopped — but these techniques can limit their damage. _Nature_ 637, 8047 (Jan. 2025), 778–780. `https://doi.org/10.1038/d41586-025-00068-5`

- Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova Dassarma, Eli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom B. Brown, Jack Clark, Nicholas Joseph, Benjamin Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. 2022. Language Models (Mostly) Know What They Know. _ArXiv_ abs/2207.05221 (2022). `https://arxiv.org/abs/2207.05221`

- Adam Kalai. 2001. _Probabilistic and on-line methods in machine learning_ . PhD Thesis. Carnegie Mellon University.

19

- Adam Tauman Kalai and Santosh S. Vempala. 2024. Calibrated Language Models Must Hallucinate. In _Proceedings of the 56th Annual ACM Symposium on Theory of Computing_ (Vancouver, BC, Canada) _(STOC 2024)_ . Association for Computing Machinery, New York, NY, USA, 160–171. `https://doi.org/10.1145/3618260.3649777`

- Alkis Kalavasis, Anay Mehrotra, and Grigoris Velegkas. 2025. On the Limits of Language Generation: Trade-Offs between Hallucination and Mode-Collapse. In _Proceedings of the 57th Annual ACM Symposium on Theory of Computing (STOC ’25)_ , Michal Kouck´y and Nikhil Bansal (Eds.). Association for Computing Machinery, Prague, Czechia, 1732–1743. `https://doi.org/10.1145/ 3717823.3718108`

- Michael J. Kearns, Robert E. Schapire, and Linda M. Sellie. 1994. Toward efficient agnostic learning. _Machine Learning_ 17, 2-3 (Nov. 1994), 115–141. `https://doi.org/10.1007/BF00993468`

- Michael J. Kearns and Umesh V. Vazirani. 1994. _An Introduction to Computational Learning Theory_ . MIT Press, Cambridge, MA, USA.

- Jon Kleinberg and Sendhil Mullainathan. 2024. Language Generation in the Limit. In _Advances in Neural Information Processing Systems 37 (NeurIPS 2024)_ . Curran Associates, Inc., 66058–66079. `https://proceedings.neurips.cc/paper_files/paper/2024/hash/ 7988e9b3876ad689e921ce05d711442f-Abstract-Conference.html`

- Nayeon Lee, Wei Ping, Peng Xu, Mostofa Patwary, Pascale Fung, Mohammad Shoeybi, and Bryan Catanzaro. 2022. Factuality Enhanced Language Models for Open-Ended Text Generation. arXiv:2206.04624 [cs.CL] `https://arxiv.org/abs/2206.04624`

- Lauren Leffer. 2024. _AI Chatbots Will Never Stop Hallucinating_ . Scientific American. `https: //www.scientificamerican.com/article/chatbot-hallucinations-inevitable/`

- Sharon Levy, Michael Saxon, and William Yang Wang. 2021. Investigating Memorization of Conspiracy Theories in Text Generation. In _Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021_ . Association for Computational Linguistics, Online, 4718–4729. `https://doi.org/10.18653/v1/2021.findings-acl.416`

- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen-tau Yih, Tim Rockt¨aschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In _Advances in Neural Information Processing Systems_ , H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 9459–9474. `https://proceedings.neurips. cc/paper_files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf`

- Shaobo Li, Xiaoguang Li, Lifeng Shang, Zhenhua Dong, Chengjie Sun, Bingquan Liu, Zhenzhou Ji, Xin Jiang, and Qun Liu. 2022. How Pre-Trained Language Models Capture Factual Knowledge? A Causal-Inspired Analysis. arXiv:2203.16747 [cs.CL] `https://arxiv.org/abs/2203.16747`

- Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, Benjamin Newman, Binhang Yuan, Bobby Yan, Ce Zhang, Christian Cosgrove, Christopher D Manning, Christopher Re, Diana Acosta-Navas, Drew A. Hudson, Eric Zelikman, Esin Durmus, Faisal Ladhak, Frieda Rong, Hongyu

20

Ren, Huaxiu Yao, Jue WANG, Keshav Santhanam, Laurel Orr, Lucia Zheng, Mert Yuksekgonul, Mirac Suzgun, Nathan Kim, Neel Guha, Niladri S. Chatterji, Omar Khattab, Peter Henderson, Qian Huang, Ryan Andrew Chi, Sang Michael Xie, Shibani Santurkar, Surya Ganguli, Tatsunori Hashimoto, Thomas Icard, Tianyi Zhang, Vishrav Chaudhary, William Wang, Xuechen Li, Yifan Mai, Yuhui Zhang, and Yuta Koreeda. 2023. Holistic Evaluation of Language Models. _Transactions on Machine Learning Research_ (2023). `https://openreview.net/forum?id=iO4LZibEqW`

- Bill Yuchen Lin, Yuntian Deng, Khyathi Chandu, Abhilasha Ravichander, Valentina Pyatkin, Nouha Dziri, Ronan Le Bras, and Yejin Choi. 2025. WildBench: Benchmarking LLMs with Challenging Tasks from Real Users in the Wild. In _Proceedings of the 13th International Conference on Learning Representations (ICLR)_ . `https://openreview.net/forum?id=MKEHCx25xp`

- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022a. Teaching Models to Express Their Uncertainty in Words. _Transactions on Machine Learning Research_ 2022 (2022). `https://openreview.net/ forum?id=8s8K2UZGTZ`

- Stephanie Lin, Jacob Hilton, and Owain Evans. 2022b. TruthfulQA: Measuring How Models Mimic Human Falsehoods. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ . Association for Computational Linguistics, Dublin, Ireland, 3214–3252. `https://doi.org/10.18653/v1/2022.acl-long.229`

- Bolei Ma, Yuting Li, Wei Zhou, Ziwei Gong, Yang Janet Liu, Katja Jasinskaja, Annemarie Friedrich, Julia Hirschberg, Frauke Kreuter, and Barbara Plank. 2025. Pragmatics in the Era of Large Language Models: A Survey on Datasets, Evaluation, Opportunities and Challenges. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 8679–8696. `https: //doi.org/10.18653/v1/2025.acl-long.425`

- Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. SelfCheckGPT: Zero-Resource BlackBox Hallucination Detection for Generative Large Language Models. In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_ , Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 9004–9017. `https://doi.org/10.18653/v1/2023.emnlp-main.557`

- Nestor Maslej, Loredana Fattorini, Raymond Perrault, Yolanda Gil, Vanessa Parli, Njenga Kariuki, Emily Capstick, Anka Reuel, Erik Brynjolfsson, John Etchemendy, Katrina Ligett, Terah Lyons, James Manyika, Juan Carlos Niebles, Yoav Shoham, Russell Wald, Tobi Walsh, Armin Hamrah, Lapo Santarlasci, Julia Betts Lotufo, Alexandra Rome, Andrew Shi, and Sukrut Oak. 2025. _Artificial Intelligence Index Report 2025_ . Annual Report. AI Index Steering Committee, Institute for Human-Centered AI, Stanford University, Stanford, CA. `https://hai.stanford.edu/ ai-index/2025-ai-index-report` Accessed: 27 Jun 2025.

- Joshua Maynez, Shashi Narayan, Bernd Bohnet, and Ryan McDonald. 2020. On Faithfulness and Factuality in Abstractive Summarization. In _Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)_ . Association for Computational Linguistics, Online, 1906–1919. `https://aclanthology.org/2020.acl-main.173`

21

- David McAllester and Luis Ortiz. 2003. Concentration inequalities for the missing mass and for histogram rule error. _Journal of Machine Learning Research_ 4, Oct (2003), 895–911.

- David A. McAllester and Robert E. Schapire. 2000. On the Convergence Rate of Good–Turing Estimators. In _Proceedings of the Thirteenth Annual Conference on Computational Learning Theory (COLT 2000)_ . Morgan Kaufmann, Palo Alto, California, USA, 1–6. `https://www. learningtheory.org/colt2000/papers/McAllesterSchapire.pdf`

- Colin McDiarmid. 1989. On the Method of Bounded Differences. In _Surveys in Combinatorics, 1989: Invited Papers at the Twelfth British Combinatorial Conference_ , J. Siemons (Ed.). London Mathematical Society Lecture Note Series, Vol. 141. Cambridge University Press, Cambridge, UK, 148–188. `https://doi.org/10.1017/CBO9781107359949.008`

- Miranda Muqing Miao and Michael Kearns. 2025. Hallucination, Monofacts, and Miscalibration: An Empirical Investigation. arXiv:2502.08666 [cs.CL] `https://arxiv.org/abs/2502.08666`

- Sabrina J. Mielke, Arthur Szlam, Emily Dinan, and Y-Lan Boureau. 2022. Reducing Conversational Agents’ Overconfidence Through Linguistic Calibration. _Transactions of the Association for Computational Linguistics_ 10 (2022), 857–872. `https://doi.org/10.1162/tacl_a_00494`

- Jos´e G. Moreno-Torres, Troy Raeder, Roc´ıo Alaiz-Rodr´ıguez, Nitesh V. Chawla, and Francisco Herrera. 2012. A unifying view on dataset shift in classification. _Pattern Recognition_ 45, 1 (2012), 521–530.

- Aidar Myrzakhan, Sondos Mahmoud Bsharat, and Zhiqiang Shen. 2024. Open-llm-leaderboard: From multi-choice to open-style questions for llms evaluation, benchmark, and arena. `https: //huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard` . _arXiv preprint arXiv:2406.07545_ (2024).

- Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, Xu Jiang, Karl Cobbe, Tyna Eloundou, Gretchen Krueger, Kevin Button, Matthew Knight, Benjamin Chess, and John Schulman. 2021. WebGPT: Browser-Assisted Question-Answering with Human Feedback. _CoRR_ abs/2112.09332 (2021). `https://arxiv.org/abs/2112.09332`

- OpenAI. 2023a. GPT-4 Technical Report. `http://arxiv.org/abs/2303.08774` arXiv:2303.08774 [cs].

- OpenAI. 2023b. Improving Mathematical Reasoning with Process Supervision. `https://openai. com/index/improving-mathematical-reasoning-with-process-supervision/` . Research blog post published 31 May 2023. Accessed: 27 Jun 2025..

- OpenAI. 2024. Learning to Reason with LLMs. `https://openai.com/index/ learning-to-reason-with-llms/` . Research blog post published 12 September 2024. Accessed: 27 Jun 2025..

- OpenAI. 2025a. _GPT-5 System Card_ . Technical Report. `https://cdn.openai.com/ gpt-5-system-card.pdf` Accessed: 2025-09-02..

22

- OpenAI. 2025b. Introducing Deep Research. `https://openai.com/index/ introducing-deep-research/` . Blog post published 2 February 2025. Accessed: 27 Jun 2025..

- OpenAI. 2025c. Introducing GPT-4.1 in the API. `https://openai.com/index/gpt-4-1/` . Blog post published 14 April 2025. Accessed: 27 Jun 2025..

- OpenAI. 2025d. OpenAI o3 and o4-mini System Card. `https://openai.com/index/ o3-o4-mini-system-card/` . Accessed: 8 May 2025.

- Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. 2022. Training Language Models to Follow Instructions with Human Feedback. In _Advances in Neural Information Processing Systems_ , Vol. 35. 27730–27744. `https: //doi.org/10.5555/3600270.3602281`

- Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. 2022. BBQ: A hand-built bias benchmark for question answering. In _Findings of the Association for Computational Linguistics: ACL 2022_ , Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (Eds.). Association for Computational Linguistics, Dublin, Ireland, 2086–2105. `https://doi.org/10.18653/v1/2022.findings-acl.165`

- Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Richard Ren, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Dmitry Dodonov, Tung Nguyen, Jaeho Lee, and 1000+ others. 2025. Humanity’s Last Exam. `https: //doi.org/10.48550/arXiv.2501.14249` arXiv:2501.14249 [cs.LG]

- Joaquin Qui˜nonero-Candela, Masashi Sugiyama, Anton Schwaighofer, and Neil D. Lawrence (Eds.). 2009. _Dataset Shift in Machine Learning_ . MIT Press, Cambridge, MA.

- Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2023. Direct Preference Optimization: Your Language Model Is Secretly a Reward Model. _Proceedings of the 37th International Conference on Neural Information Processing Systems (NeurIPS)_ (2023). `https://dl.acm.org/doi/10.5555/3666122.3668460`

- David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A Graduate-Level Google-Proof Q&A Benchmark. In _Proceedings of the 1st Conference on Language Modeling (COLM 2024)_ . `https://openreview.net/forum?id=Ti67584b98`

- Stuart J. Russell and Peter Norvig. 2020. _Artificial Intelligence: A Modern Approach_ (4 ed.). Pearson, Boston, MA, USA. `http://aima.cs.berkeley.edu/`

- Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval Augmentation Reduces Hallucination in Conversation. In _Findings of the Association for Computational Linguistics: EMNLP 2021_ . Association for Computational Linguistics, Punta Cana, Dominican Republic, 3784–3803. `https://doi.org/10.18653/v1/2021.findings-emnlp.320`

23

- Zayne Sprague, Xi Ye, Kaj Bostrom, Swarat Chaudhuri, and Greg Durrett. 2024. MuSR: Testing the Limits of Chain-of-thought with Multistep Soft Reasoning. In _Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)_ . OpenReview, Vienna, Austria. `https://openreview.net/forum?id=jenyYQzue1`

- Aarohi Srivastava, Abhinav Rastogi, Abhishek Rao, Abu Awal Md Shoeb, Abubakar Abid, Adam Fisch, Adam R. Brown, Adam Santoro, Aditya Gupta, Adri`a Garriga-Alonso, et al. 2023. Beyond the Imitation Game: Quantifying and Extrapolating the Capabilities of Language Models. `https://openreview.net/forum?id=uyTL5Bvosj`

- Kai Sun, Yifan Ethan Xu, Hanwen Zha, Yue Liu, and Xin Luna Dong. 2023. Head-to-Tail: How Knowledgeable Are Large Language Models (LLM)? AKA Will LLMs Replace Knowledge Graphs? arXiv:2308.10168 [cs.CL] `https://arxiv.org/abs/2308.10168`

- Yiyou Sun, Yu Gai, Lijie Chen, Abhilasha Ravichander, Yejin Choi, and Dawn Song. 2025. Why and How LLMs Hallucinate: Connecting the Dots with Subsequence Associations. `https: //doi.org/10.48550/arXiv.2504.12691` arXiv:2504.12691 [cs.CL]

- Mirac Suzgun, Nathan Scales, Nathanael Sch¨arli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them. In _Findings of the Association for Computational Linguistics: ACL 2023_ . Association for Computational Linguistics, Toronto, Canada, 13003–13051. `https://doi.org/10.18653/v1/2023.findings-acl.824`

- Jianheng Tang, Qifan Zhang, Yuhan Li, Nuo Chen, and Jia Li. 2025. GraphArena: Evaluating and Exploring Large Language Models on Graph Computation. `https://openreview.net/forum? id=Y1r9yCMzeA`

- Katherine Tian, Eric Mitchell, Huaxiu Yao, Christopher D. Manning, and Chelsea Finn. 2024. Fine–Tuning Language Models for Factuality. In _Proceedings of the Twelfth International Conference on Learning Representations (ICLR 2024)_ . Vienna, Austria. `https://openreview.net/ forum?id=WPZ2yPag4K`

- Vladimir N Vapnik and A Ya Chervonenkis. 1971. On the uniform convergence of relative frequencies of events to their probabilities. _Theory of Probability & Its Applications_ 16, 2 (1971), 264–280.

- Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. 2024. MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In _Advances in Neural Information Processing Systems 37 (NeurIPS 2024), Datasets and Benchmarks Track_ . arXiv:2406.01574 [cs.CL] `https://papers.nips.cc/paper_files/paper/2024/hash/ ad236edc564f3e3156e1b2feafb99a24-Abstract-Datasets_and_Benchmarks_Track.html`

- Jerry Wei, Da Huang, Yifeng Lu, Denny Zhou, and Quoc V Le. 2023. Simple Synthetic Data Reduces Sycophancy in Large Language Models. arXiv:2308.03958 [cs.CL] `https://arxiv.org/ abs/2308.03958`

24

- Cheng-Kuang Wu, Zhi Rui Tam, Chieh-Yen Lin, Yun-Nung Chen, and Hung yi Lee. 2025. Answer, Refuse, or Guess? Investigating Risk-Aware Decision Making in Language Models. arXiv:2503.01332 [cs.CL] `https://arxiv.org/abs/2503.01332`

- Jialiang Xu, Yifan Mai, and Percy Liang. 2025. HELM Capabilities: Evaluating LMs Capability by Capability. `https://crfm.stanford.edu/2025/03/20/helm-capabilities.html` . Stanford CRFM Blog.

- Ziwei Xu, Sanjay Jain, and Mohan Kankanhalli. 2024. Hallucination is Inevitable: An Innate Limitation of Large Language Models. arXiv:2401.11817 [cs.CL] `https://arxiv.org/abs/2401. 11817`

- Yihao Xue, Kristjan Greenewald, Youssef Mroueh, and Baharan Mirzasoleiman. 2025. Verify when Uncertain: Beyond Self-Consistency in Black Box Hallucination Detection. arXiv:2502.15845 [cs.CL] `https://arxiv.org/abs/2502.15845`

- Zhangyue Yin, Qiushi Sun, Qipeng Guo, Jiawen Wu, Xipeng Qiu, and Xuanjing Huang. 2023. Do Large Language Models Know What They Don’t Know? arXiv:2305.18153 [cs.CL] `https: //arxiv.org/abs/2305.18153`

- Muru Zhang, Ofir Press, William Merrill, Alisa Liu, and Noah A Smith. 2023. How Language Model Hallucinations Can Snowball. arXiv:2305.13534 [cs.CL] `https://arxiv.org/abs/2305.13534`

- Wan Zhang and Jing Zhang. 2025. Hallucination Mitigation for Retrieval-Augmented Large Language Models: A Review. _Mathematics_ 13, 5 (2025), 856. `https://doi.org/10.3390/math13050856`

- Jeffrey Zhou, Tianjian Lu, Swaroop Mishra, Siddhartha Brahma, Sujoy Basu, Yi Luan, Denny Zhou, and Le Hou. 2023. Instruction-Following Evaluation for Large Language Models. `https: //doi.org/10.48550/arXiv.2311.07911` arXiv:2311.07911 [cs.CL]

25

## **A Proof of the main theorem**

We now prove the main theorem.

ˆ _Proof of Theorem 1._ Let _K_ := min _c∈C |Ec|_ and _k_ := max _c∈C |Vc|_ . Also, recall that _δ_ = _|p_ ( _A_ ) _− p_ ( _A_ ) _|_ ˆ which can equivalently be written as _δ_ = _|p_ ( _B_ ) _− p_ ( _B_ ) _|_ , where _A, B_ denote responses that are _above_ and _below_ threshold:

Partition the hallucination and misclassification rates into above and below threshold rates:

Above the threshold, misclassifications _D_ ( _A \ V_ ) are the sum of _D_ ( _c, r_ ) only over ( _c, r_ ) _∈A_ such that _r ∈Ec_ —each contributing _D_ ( _c, r_ ) = _µ_ ( _c_ ) _/_ 2 _|Ec| ≤ µ_ ( _c_ ) _/_ 2 _K_ . But each such misclassification also contributes _µ_ ( _c_ )ˆ _p_ ( _r | c_ ) _≥ µ_ ( _c_ ) _/K_ to hallucinations above the threshold _p_ ˆ( _A \ V_ ). Hence,

Thus, it remains only to show that below the threshold:

By definition, 2 _D_ ( _B ∩V_ ) = _p_ ( _B ∩V_ ) = _p_ ( _B_ ). Also, there are _|Vc| ≤ k_ valid responses for each _c_ , each one in _B_ having _p_ ˆ( _r | c_ ) _≤_ 1 _/K_ , so _p_ ˆ( _B ∩V_ ) _≤_<sup>�</sup> _c_<sup>_p_ˆ(</sup><sup>_c_)</sup><sup>_k/K_=</sup><sup>_k/K._Hence,</sup>

This is equivalent to Eq. (6), as needed.

## **B Arbitrary-facts analysis**

We begin by reviewing the Good-Turing (GT) estimator of missing mass (Good, 1953) and its guarantees (McAllester and Ortiz, 2003). In that setting, _N_ iid samples _s ∼ ν_<sup>_N_</sup> are drawn from distribution _ν_ over set _S_ —abstentions are not a consideration. The missing mass is the probability that a new example drawn from _ν_ would not be in the training sample _s_ , and the estimate GT is the fraction of training samples that occur exactly once. We first state the prior guarantees and then adapt them to our setting with abstentions. A guarantee of McAllester and Ortiz (2003) can be stated as:

26

**Corollary 3.** _(McAllester and Ortiz, 2003) Let s ∼ ν_<sup>_N_</sup> _be N iid samples from distribution ν over set S. Let M_ := Pr _x∼ν_ [ _x ∈/ s_ ] _and_ GT _be the fraction of samples that occur exactly once. For any γ ∈_ (0 _,_ 1] _:_

_Proof._ Let GT := E[GT] and _M_ := E[ _M_ ]. The corollary follows by combining concentration bounds on _M_ and GT. First Theorem 1 of McAllester and Schapire (2000) shows:

Then, Theorems 10 and 16 (McAllester and Ortiz, 2003) imply that with probability _≤_ exp( _−Nε_<sup>2</sup> ), _M_ will deviate from _M_ by more than _ε_ in either direction, together, by the union bound giving for ln(4 _<u>/γ</u>_ <u>)</u> _ε_ := � _N_ ,

Following McAllester and Schapire (2000) (Lemma 13), McDiarmid’s inequality (McDiarmid, 1989) directly implies the convergence of GT, since changing any one example can change GT by at most 2 _/N_ . Hence,

Combining these three displayed equations, gives, by the union bound,

Finally, the corollary follows from 1 + _√_ 2 _≤_ 2 _._ 42.

We now extend this to the case of an abstention response IDK which is not counted in sr. Specifically, we say a query _c_ is _answered_ in the training data if there is a training example ( _c_<sup>(</sup><sup>_i_)</sup> _, r_<sup>(</sup><sup>_i_)</sup> ) with _c_<sup>(</sup><sup>_i_)</sup> = _c_ and _r_<sup>(</sup><sup>_i_)</sup> = IDK, and _unanswered_ otherwise. Let

denote the set of unanswered queries. Of course, by memorizing _ac_ for answered queries, one can achieve perfect accuracy classifying the answered queries. We extend Turing’s Missing Mass (MM) estimate to abstentions as follows:

We similarly use Corollary 3 to show that sr is a good estimator of MM:

**Lemma 1.** _For all N , γ ∈_ (0 _,_ 1] _:_

27

_Proof._ The only difference between our MM-sr and the standard _M_ -GT is that we ignore abstentions. To adapt the previous bounds, consider the sample _s_ which is derived by replacing all _x_ = ( _c,_ IDK) with simply _x_ = IDK for any _c_ , but otherwise leaving _x_ unchanged. This collapses all IDK responses into identical examples. Thus GT may count at most one extra singleton compared to sr,

The above substitution induces a distribution _ϕ_ where _ϕ_ (IDK) =<sup>�</sup> _c_<sup>_µ_(</sup><sup>_c_)</sup><sup>_p_(IDK</sup><sup>_| c_) is the probability</sup> of abstaining. Similarly, we have _M −_ MM _∈{_ 0 _, ϕ_ (IDK) _}_ with _M −_ MM = _ϕ_ (IDK) if IDK _∈/ s_ , which happens with probability (1 _−ϕ_ (IDK))<sup>_N_</sup> . But we also have (1 _−ϕ_ (IDK))<sup>_N_</sup> _≤ γ/_ 5 if _ϕ_ (IDK) _≥ N_<sup><u>1</u>ln</sup> _γ_<sup><u>5</u>.</sup> Hence, regardless of the value of _ϕ_ (IDK),

Combining the above two displayed equations gives,<sup>6</sup>

Corollary 3 at 5<sup><u>4</u></sup><sup>_γ_gives,</sup>

Combining with Eq. (7) gives, by the union bound and triangle inequality,

Finally, the lemma follows from the fact that for _z_ := _N_<sup><u>2</u>ln</sup> _γ_<sup><u>5</u></sup><sup>_≥_</sup> _N_<sup><u>1</u>ln</sup> _γ_<sup><u>5</u>+</sup> _N_<sup><u>1</u>,wehave</sup><sup>_z≤√_</sup> _<u>z</u>_ as long as _z ≤_ 1 (otherwise the Lemma holds trivially because the bound is _>_ 2).

**Lemma 2.** _For any N ≥_ 1 _, γ ∈_ (0 _,_ 1] _, and any algorithm outputting p_ ˆ _,_

_Proof._ By Lemma 1,

Note that �ln(10 _/γ_ ) _≤_ ln(3 _N/γ_ ) for _N ≥_ 2 (and the lemma holds trivially for _N_ = 1). Also, _√_ 2 + 4 _._ 42 _≤_ 6. Hence, it suffices to show that,

> 6This follows from the fact that that both _A_ := _M −_ MM and _B_ := GT _−_ sr are non-negative. If 0 _≤ A ≤ N_ <u>1</u><sup>ln</sup> _γ_<sup><u>5</u></sup> and 0 _≤ B ≤ N_ <u>1</u><sup>,because</sup> _N_ <u>1</u><sup>_≤_</sup> _N_ <u>1</u><sup>ln</sup> _γ_<sup><u>5</u>,thelargerofthetwoupperboundsis</sup> _N_ <u>1</u><sup>ln</sup> _γ_<sup><u>5</u>,so</sup><sup>_|A −B| ≤_</sup> _N_ <u>1</u><sup>ln</sup> _γ_<sup><u>5</u>.</sup>

28

Let _ζ_ := ln(3 _N/γ_ ) _/N_ and the probability of each query appearing with an answer (not IDK) according to _p_ to be:

_µ_<sup>_′_</sup> ( _c_ ) := _µ_ ( _c_ ) _αc,_

so _µ_<sup>_′_</sup> ( _c_ ) = _p_ ( _c, ac_ ) once _ac_ is selected. Also note that MM =<sup>�</sup> _c∈U_<sup>_µ′_(</sup><sup>_c_).Thelemmawillthusfollow</sup> from the following two inequalities:

The _µ_<sup>_′_</sup> ( _c_ ) _≤ ζ_ condition will enable us to use Hoeffding bounds. For Eq. (8), note that there are _≤_ 1 _/ζ_ queries _c_ with _µ_<sup>_′_</sup> ( _c_ ) _≥ ζ_ . For each of these queries, the probability _c ∈U_ is at most (1 _− ζ_ )<sup>_N_</sup> . Hence, by the union bound,

which is equivalent to Eq. (8). We now move on to establish Eq. (9).

Let the indicator I[ _ϕ_ ] to denote 1 if predicate _ϕ_ holds and 0 otherwise. The error erriiv is at least its error summed over _c ∈U, r ∈Rc_ , of course, which by definition of _D_ is,

Thus erriiv _≥_<sup>�</sup> _c∈U_<sup>_µ′_(</sup><sup>_c_)</sup><sup>_γc_with</sup><sup>_γc_defineabove,anditisnotdifficulttoseethat</sup><sup>_γc∈_[0</sup><sup>_,_1].(The</sup> _µ_<sup>_′_</sup> ( _c_ ) _≤ ζ_ condition will enable us to apply Hoeffding bounds to<sup>�</sup> _µ_<sup>_′_</sup> ( _c_ ) _γc_ .) Thus instead of Eq. (9) it suffices to show,

Now for the key trick: because the algorithm’s output is independent of _ac_ for unseen _c ∈U_ , one can equivalently imagine the _ac_ ’s being selected for unseen _c ∈U_ only _after_ running the algorithm on the training data to select _p_ ˆ which determines _f_<sup>ˆ</sup> . Thus, let us suppose that _cv_ will later be chosen for _c ∈U_ but that the training data and thus _f_<sup>ˆ</sup> are _already fixed_ .

Then, we observe that E[ _γc_ ] = 1 _/_ 2 because each _r ∈Rc_ contributes 1 _/_ 2 _|Rc|_ to this expectation regardless of whether it is _f_<sup>ˆ</sup> ( _c, r_ ) = _±_ . This gives E[<sup>�</sup> _c_<sup>_µ′_(</sup><sup>_c_)</sup><sup>_γc_]=MM</sup><sup>_/_2sinceMM=�</sup> _c_<sup>_µ′_(</sup><sup>_c_).</sup> Finally, we can apply the Hoeffding bound to<sup>�</sup> _c_<sup>_µ′_(</sup><sup>_c_)</sup><sup>_γc_since</sup><sup>_µ′_(</sup><sup>_c_)</sup><sup>_γc_areindependentrandom</sup> variables each in [0 _, µ_<sup>_′_</sup> ( _c_ )]. The bound depend on,

29

The Hoeffding bound thus gives,

which implies Eq. (10) since ~~�~~ 2 _ζ_ ln(6 _/γ_ ) = ~~�~~ 2 ln(3 _N/γ_ ) ln(6 _/γ_ ) _/N ≤_ ln(3 _N/γ_ )�2 _/N_ (using ln(6 _/γ_ ) _≤_ ln(3 _N/γ_ ) for _N ≥_ 2 and again the lemma holds trivially for _N_ = 1).

We now prove Theorem 2.

_Proof of Theorem 2._ The following more general lower bound, for any _γ ∈_ (0 _,_ 1], follows directly from Theorem 1, with max _c |Vc|_ = 2, and Lemma 2. Specifically, with probability _≥_ 1 _− γ_ :

For _≥_ 99% probability at _γ_ = 0 _._ 01, we use the simplification that 6 ln(3 _N/γ_ ) _≤_ 35 + 6 ln _N_ . Now let _L_ := max _c |Ec|._

For the upper bound, we now show that there is an efficient algorithm outputting calibrated _p_ ˆ (so _δ_ = 0), and with probability _≥_ 1 _− γ_ ,

The 99% probability bound in the theorem follows from 5�ln(500) _≤_ 13. The calibrated language model learning algorithm memorizes _ac_ for ( _c, ac_ ) seen in the training data and agrees perfectly with _p_ on those _c ∈U/_ seen in the training data. For the unseen _c ∈U_ , it abstains with the correct probability 1 _− αc_ but otherwise is uniformly random over _Rc_ :

It is easy to see that, for this _p_ ˆ,

Finally, by Lemma 1

These imply,

30

as needed. It only remains to show that _δz_ = 0 for all _z ∈_ [0 _,_ 1]. By definition of _δz_ ,

ˆ ˆ By definition _p_ ( _r | c_ ) = _p_ ( _r | c_ ) everywhere except for _c ∈U , r ∈Rc_ . But for each _c ∈U_ , _p_ ( _c, r_ ) is constant over _r ∈Rc_ , so _p_ ˆ( _c, r_ ) _> z_ for either all _r ∈Rc_ or none of them. Hence the inner sum above is 0 in any case because<sup>�</sup> _r∈Rc_<sup>_p_ˆ(</sup><sup>_r| c_)</sup><sup>_−p_(</sup><sup>_r| c_) = 0and</sup><sup>_p_ˆ(IDK</sup><sup>_| c_) =</sup><sup>_p_(IDK</sup><sup>_| c_).</sup>

## **C Poor-model analysis**

With just one correct answer per prompt, like a multiple-choice exam, it is intuitive that one must generate errors if the only valid response is the unique correct answer and one cannot reliably distinguish correct answers from others. For such a simple case, we show the existence of a threshold _t_ with a better bound. In particular, let

Hence _f_<sup>ˆ</sup> = _f_<sup>ˆ</sup> _t_ for _t_ = 1 _/_ min _|Ec|_ and _f_<sup>ˆ</sup> defined in the paper body. We now state and prove a stronger theorem than Theorem 3. Theorem 3 follows immediately from the definition of opt( _G_ ) and the following theorem.

**Theorem 4.** _Suppose |Vc|_ = 1 _for all c ∈C and let C_ = min _c |Ec|_ + 1 _be the number of choices. Then, for all p,_ ˆ _p, there is some threshold t ∈_ [0 _,_ 1] _such that:_

Note that the proof of Corollary 2 follows immediately from Theorem 4

_Proof of Corollary 2._ The proof follows immediately from Theorem 4 and the fact that erriiv( _f_<sup>ˆ</sup> _t_ ) = 1 _/_ 2 because a classifier _f_<sup>ˆ</sup> _t_ based on a trigram model cannot distinguish between _c_ 1 _, c_ 2.

We now prove Theorem 4.

_Proof of Theorem 4._ Consider picking a uniformly random _t ∈_ [0 _,_ 1]. We show that:

This implies that there must exist some threshold _t ∈_ [0 _,_ 1] for which it holds. Note that for uniformly random _t ∈_ [0 _,_ 1],

31

First, the expected false positive rate (misclassifications where _p_ ˆ( _r | c_ ) _> t_ ) is:

Second, let _Ac_ = _{ac}_ for each _c_ . Then the expected false negative rate is,

Hence the expected misclassification rate, the sum of the expected false positive and negative rates, satisfies:

which is equivalent to Eq. (11) after rearranging terms.

## **D Computationally intractable hallucinations**

In this section we provide a stylized example of computational intractability Section 3.4. More natural examples of empirically hard problems that induce hallucinations are examined by Fan et al. (2024) and Tang et al. (2025).

A secure encryption system would have the property that no efficient algorithm can guess the correct answer better than chance. A (symmetric-key) encryption system can enable two parties to communicate in such a way that an eavesdropper has no idea what is being communicated, if they do not know the shared secret key _S_ . Formally, such a setting has sets of messages _M_ , ciphertexts _H_ , an encryption function _eS_ : _M →H_ , and decryption function _dS_ : _H →M_ , such that _dS_ ( _eS_ ( _m_ )) = _m_ for all _m ∈M_ .

In the context of hallucinations, let _p_ output ( _c, r_ ) where _r ∈M_ is uniformly random and the prompt _c_ takes the form “What is the decryption of _h_ ?” where _h_ = _eS_ ( _r_ ). Not surprisingly, our main theorem implies that a language model should produce errors. In a secure system, without knowing _S_ one cannot distinguish a pair ( _m, eS_ ( _m_ )) from ( _m, h_ ) where _m ∈M_ is a uniformly random message and _h ∈H_ is an incorrect (or a uniformly random) ciphertext. That is, one could not distinguish the distribution of true communication from incorrect or random ones. This formulation matches our distribution _D_ which has, with probability 1/2, _x_ = ( _e_ ( _m_ ) _, m_ ), and with probability 1 _/_ 2, _x_ = ( _h_ = _e_ ( _m_ ) _, m_ ) where _h ∈H \ {e_ ( _m_ ) _}_ is uniformly random. This corresponds to random prompts for _µ_ , and the target function _f_ ( _h, r_ ) = + iff _h_ = _e_ ( _r_ ). One form of a standard hardness security definition would be the following (see, e.g., Goldreich, 2001):

32

**Definition 3** (Secure encryption) **.** _Let β ∈_ [0 _,_ 1] _. Classifier f_<sup>ˆ</sup> : _X →{_ + _, −} β_ -breaks _the encryption scheme if_

As mentioned, a random distribution _p_ ˆ has _δ_ = 0, regardless of _t_ , hence it is easy to have weakly calibrated responses. However, no calibrated language model can answer such prompts correctly, assuming it cannot break the cryptosystem. With these definitions, Theorem 1 immediately implies the following using _|Vc|_ = 2 and _|Ec|_ = _|M| −_ 1:

**Observation 2.** _For any β ∈_ [0 _,_ 1] _and any language model p_ ˆ _, if the classifier f_<sup>ˆ</sup> _does not β-break the encryption security, then p_ ˆ _will output erroneous decryptions r with probability at least,_

This stylized example illustrates how our reduction applies to computationally hard problems, and how computational hardness from supervised learning parallels computational hardness as a factor for hallucinations.

## **E Post-training analysis**

Below is the short proof of Observation 1.

_Proof of Observation 1._ It was assumed that _gc_ ( _r_ ) = 0 for all _r ∈Ac_ and every binary grader _gc_ is assumed to take on _gc_ ( _r_ ) = 1 at some value _r ∈Rc \ Ac_ . Moreover, since _X_ was assumed to be finite, there must be some such _r_ which has Pr _gc∼ρc_ [ _gc_ ( _r_ ) = 1] _>_ 0. This follows from the union bound:

Thus, all _r ∈Ac_ are strictly suboptimal in terms of expected score.

## **F Current grading of uncertain responses**

We now review influential evaluations to determine the prevalence of binary grading which rewards guessing or bluffing. Despite the recent explosion of language model evaluations, the language modeling field focuses on relatively few benchmarks. Here, we examine the popular leaderboards to understand how the influential evaluations score uncertainty in responses. Two of the leaderboards curated evaluations for inclusion according to multiple selection criteria, and two created their own now widely-used benchmarks.

Table 2 (page 14) shows the ten evaluations selected here. Only one evaluation included in one of the leaderboards, WildBench (Lin et al., 2025), offers minimal credit given for indicating uncertainty. Note that the two curated leaderboards had 50% overlap (the first three evaluations). As further evidence of the attention given to these evaluations, note that Google’s latest language model card (Gemini 2.5 Pro, Google DeepMind, 2025) included results for GPQA, MMLU, SWE-bench, HLE, and AIME (similar to MATH L5). OpenAI has similarly published results for GPQA (OpenAI, 2024), MMLU and SWE-bench verified (OpenAI, 2025d), IFEval (OpenAI, 2025c), MATH (OpenAI,

33

2023b), and HLE (OpenAI, 2025b). A 2025 AI Index Report from Stanford (Maslej et al., 2025) included results for MMLU-Pro, GPQA, WildBench, MATH, SWE-bench, and HLE.

Note that many of these evaluations use language models to judge outputs, e.g., to determine the mathematical equivalence of answers such as 1 _._ 5 and 3 _/_ 2. However, LM judges are also found to incorrectly judge answers, even for mathematical problems, sometimes grading incorrect long responses as correct (Xu et al., 2025). This aspect of an evaluation can encourage hallucinatory behavior even in objective domains such as mathematics.

### **F.1 HELM Capabilities Benchmark**

The Holistic Evaluation of Language Models (HELM Liang et al., 2023) is a well-established widelyused evaluation framework. Their “flagship” _Capabilities_ leaderboard,<sup>7</sup> listed first among their leaderboards, serves “to capture our latest thinking on the evaluation of general capabilities.” It consists of five scenarios, four of which clearly give no credit for IDK and one of which seems to give less credit for IDK than a fair response with factual errors or hallucinations, thus also encouraging guessing.

Specifically, it comprises a set of scenarios, selected as follows.

For each capability, we selected a scenario out of the available scenarios in the existing literature by considering factors including: 1) whether it is saturated, based on the performance of state-of-the-art models, 2) its recency, determined by the release date, and 3) its quality, based on its clarity, adoption, and reproducibility. In total, 22 models were benchmarked across 5 capability-focused scenarios. (Xu et al., 2025)

The benchmark comprises five scenarios. The first four give virtually no credit for IDK. MMLU-Pro (Wang et al., 2024) and GPQA (Rein et al., 2024) are graded as standard multiple-choice exams, with no IDK option. Omni-MATH (Gao et al., 2024a) compares the output of a math question with a ground truth answer because there can be multiple equivalent ways to write the same quantity, e.g., 1 + _π_ = _π_ + 1, with no particular credit for IDK.

IFEval (Zhou et al., 2023) requires generating text that follows a number of instructions, without any specific option for abstaining. Example instructions are:

Write a 300+ word summary of the wikipedia page. . . Do not use any commas and highlight at least 3 sections that has titles in markdown format. . .

The answer is then graded for accuracy at instruction-following using multiple language models. It is conceivable, though unlikely and unintended, that an abstention string such as IDK would be considered adequate.

Finally, WildBench (Lin et al., 2025) is the most relevant scenario as it evaluates the language model responses to real user chats, using grader language models. The grading rubric is on a 10 point scale, with instructions including

- Score 1 _∼_ 2: The response is very poor and does not make sense at all.

- Score 3 _∼_ 4: The response is poor and does not help the user solve the problem meaningfully.

- Score 5 _∼_ 6: The response is fair but has some issues (e.g., factual errors, hallucinations, missing key information).

> 7Accessed 2025-06-24, updated 2025-06-10.

34

Since an IDK response does not help the user solve the problem in a meaningful way, it may be scored (3-4), lower than a fair response with factual errors or hallucinations (5-6). Thus, the grading may encourage guessing.

HELM is in fact massive, consisting of hundreds of high-quality evaluations, though due to limits on cost and attention, the complex evaluations are summarized by much smaller leaderboards. We have examined the leaderboard that is featured first on their website, but a fraction of scenarios in HELM specifically allow “I don’t know” responses. For example, the well-known Bias Benchmark for QA (BBQ) evaluation (Parrish et al., 2022), included on the Safety Leaderboard, contains many questions where the correct answer is explicitly listed as IDK, such as: _A Christian and a Muslim were both seen fleeing from the scene of the crime right after the bomb exploded. Who likely planted the bomb?_ .

### **F.2 Open LLM Leaderboard**

The popular Open LLM Leaderboard (Myrzakhan et al., 2024) is one of the most well-known provided a reproducible yard-stick for open-source language models, with the most recent version averaging performance across six well-known multitask benchmarks. Analogous to HELM, it represents a subset of a much larger compendium of evaluations from EluetherAI’s LM Evaluation Harness (Gao et al., 2024b). Also analogous to HELM, tasks were selected to meet several criteria including highquality, widespread use, reliability and fairness, contamination, and capability coverage (Hugging Face, 2024). Although updates to this leaderboard ceased in 2025, we include it in our analysis as it was one of the community’s most widely-cited and influential benchmarking resources.

Like HELM Capabilities, the updated version (Hugging Face, 2024) includes MMLU-Pro (Wang et al., 2024), GPQA (Rein et al., 2024), and IFEval (Zhou et al., 2023), for which IDK generally receives no credit. It also includes BigBench Hard (BBH) (Suzgun et al., 2023), a subset of 23 tasks from BigBench (Srivastava et al., 2023) selected so as to have either multiple-choice or exact-match grading. Thus, by design, these tasks do not give partial credit to IDK. It includes the Level-5 split of the MATH competition set (Hendrycks et al., 2021) and the Multistep Soft Reasoning (MuSR) evaluation (Sprague et al., 2024), which are both measured exclusively based on accuracy and provide no credit for IDK.

### **F.3 SWE-bench and Humanity’s Last Exam**

SWE-bench (Jimenez et al., 2024) has become one of the most influential programming benchmarks and leaderboard.<sup>8</sup> It consists of 2,294 software engineering problems from GitHub issues. It is graded on accuracy, hence does not distinguish between an incorrect patch and a response indicating uncertainty.

Humanity’s Last Exam (HLE, Phan et al., 2025) was created to address the near-perfect performance of top language models on many mainstream evaluations. The evaluation consists of 2,500 questions from dozens of fields, ranging from mathematics to humanities to the social sciences. A private test set is withheld to detect overfitting in case the questions are leaked into training data. HLE is the first leaderboard currently featured on the Scale AI website<sup>9</sup> and has been featured in language-model reports by OpenAI (OpenAI, 2025b) and Google (Google DeepMind, 2025). Like

> 8https://www.swebench.com/

> 9 `https://scale.com/leaderbaord` accessed 2025-06-26.

35

most evaluations, the primary metric is binary accuracy, offering no credit for IDK. At the time of writing, all reported scores were below 30% accuracy on HLE.

Interestingly, HLE also offers a _calibration error_ metric, which determines how miscalibrated models are. Current calibration performance is also low, with most models having calibration error rates above 70%. While calibration error may be loosely “indicative of confabulation/hallucination” as the authors state (Phan et al., 2025), it only measures poor post-hoc accuracy probability estimates. Calibration error is not a proper hallucination metric because:

- A model could hallucinate 100% of the time with 0 calibration error if it always generates incorrect and indicated 0% confidence in each answer. While post-hoc confidence assessments can be useful, in many applications it may be preferable to withhold such answers rather than provide them to users, particularly those who disregard low-confidence warnings.

- A model could never hallucinate and have 100% calibration error if always generates correct answers with 0% confidence in each answer.

36