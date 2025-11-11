---
{
  "title": "AI World Models (Keyon Vafa)",
  "authors": ["Sasha Rush", "Keyon Vafa"],
  "url": "https://www.youtube.com/watch?v=hguIUmMsvA4",
  "date": "2025-09-07",
  "tags": ["deep-learing", "transformer", "world-models"]
}
---

functionalities we may want from an AI system:
- synthesize concepts
- applying concepts to new domains (few-shot learning, in-context learning)
- reasoning (Gemini 2.5 IMO winning gold example)
- creativity (text, video models, etc.)

all of these can be performed by a model that has learned the "correct" world model.

What does it mean to have a world model?
- what about benchmarks?
  - benchmarks/exams are usually used to check the understanding of humans
  - but, this requires strong assumptions that an LLM learns like human-like ways
  - GPT 5 does great on AIME competition, but thinks `4.11 > 4.9`
  - analogy: it's like evaluating a vision model with an eye exam

one of the early tests was "Othello" (similar to Go).

```
- - - - - -  | - - - - - - | - - - - - -
- - 1 0 - -  | - - 1 0 - - | - 0 0 0 - -
- - 0 1 - -  | - 1 1 1 - - | - 1 1 1 - -
- - - - - -  | - - - - - - | - - - - - -
```

- transformer trained on sequence of games. transformer never sees the "true world" (othello board).
- it only sees sequences of moves

question: world model recovery possible? can transformer uncover the implicit rules and understanding of the othello board?

two kinds of world models:
- testing for a world model on a single task (todo read: Justin Chen, Jon Kleinberg, Ashesh Rambachan and Sendhil Mullainathan [NeurIPS 2024])
- testing for a  world model across many tasks (todo read: Peter Chang, Ashesh Rambachan and Sendhil Mullainathan [ICML 2025])

testbed: manhattan road taxi dataset

transformer trained on sequences of taxi rides (pick up, drop off, time, directions):

```
7283 1932 SW SW SW NE SE N N ... end
2919 4885 SW SW NE NE N S ... end
```

Training objective: predict the next token of each sequence (like language model training). evaluate model's ability to generate new rides:

```
510 3982 <generate> end
```

The model looked good
  - `> 99.9%` of proposed turns are legal
  - model finds valid routes between new points 98% of the time

has the model discovered the world model for manhattan?

taxi traversals obey a deterministic finite automaton (DFA)
  - states: each intersection in Manhattan
  - transition rules: legal turns at each intersection and where they take you

definition: a generative model recovers a DFA if every sequence in generates is valid in the DFA (and vice-versa)

result: if a model always predicts legal next-tokens, it recovered the DFA

suggests a test: measure how often a model's predicted tokens are valid.
  - but problem: cumulative connect 4 example. a very simple model can get 99% accuracy for large n=1000. many states have same possible next tokens.

- perfect next-token prediction implies world model recovery.
- near perfect next-token prediction doesn't mean you're close to the true world model.

single next-tokens aren't enough to differentiate states.
- (todo: read) Myhill-Nerode theorem (1975): for every pair of states, there is some `k` where k-next tokens differentiate states

new metrics motivated by going beyond next-token prediction:
- compression: if two sequences lead to the same state, a model shouldn't distinguish their continuations of any length
- distinction: if two sequences lead to distinct states, a model should distinguish their length-k continuations

three kinds of training data:
- shortest paths of actual rides (120M tokens)
- perturbed shortest paths (e.g. traffic) (1.7B tokens)
- random paths (4.7B tokens)

this results in all models have next-token accuracy (`>99.9%`). but their compression/distinction precision and recall is `~0`. A true model should have `~1`.

why should we care about world models?
- the model can find shortest paths
- because: not having the right world model means it could do badly on different but related tasks (adding detours but fail to re-route)

attempt to visualize the implicit world model (map of Manhattan):
  - equivalent to graph reconstruction
  - generate sequences of taxi traversal from transformer
  - assume model knows locations of intersections (very generously)
  - what roads must exist for generated sequences to be valid?

- sanity check 1: generate data from true world model (reconstructed map is true map)
- sanity check 2: generate data from true model but add noise to match transformer error rate (reconstructed map is imperfect but largely sensible)
- now, reconstruct transformer's map (assumes many roads exist that don't e.g. flyovers; this despite having been generous to the model like mapping correct physical locations, minimizing wrong roads/flyovers)

while these definitions and tests are specific to today's generative models, we've been here before:
- Rashomon effect: two models can achieve similar performance in dramatically different ways (Breiman 2001; D'Amour et al., 2020; Black et al. 2022).
- here: a m odel can achieve near-perfect prediction without recovering structure

but what if the model gets perfect predictions? is that always good enough?

**Foundation model** -> adaptation -> tasks

something that provides a good enough base structure to solve new tasks

tasks: question answering, sentiment analysis, information extraction, image captioning, object recognition, instruction following, etc.

no free lunch theorem for learning algorithms: every foundation model has inductive bias toward some set of functions (todo read: Wolpert and Macready, 1997)

world model: restriction over functions described by a state-space

goal: test if a foundation model's inductive bias is towards a given world model

inductive bias probe: test how a foundation model behaves when it is adapted to small amounts of data
- step 1: fit foundattion model to synthetic datasets and extract learned functions
- step 2: compare learned functions to the given world model

example: lattice (1d state tracking). good inductive bias for small states but worsen quickly.

example: foundation model of planetary orbits:
- like kepler, it makes good predictions
- but has it learned newtonian mechanics?
- inductive bias on new tasks isn't toward newtonian mechanics
- similar predictions for orbits with different states; different predictions for orbits with similar states
- the laws recovered via symbolic regression to estimate implied force law are incorrect and changes based on which galaxy it is applied to; this is not just with domain-specific transformer. LLMs, trained on a lot of newtonian mechanics, struggle too

so, what are inductive biases toward?
- possibility: models conflate sequences that have similar legal next-tokens, even if those sequences represent different states
  - example: two distinct othello boards can have the same allowed set of legal next-tokens
  - general pattern: foundation model only recovers "enough of" the board to calculate legal next moves

related ideas:
- mechanistic interpretability
- analyzing theoretical capabilities of architectures
- world models in reinforcement learning

so far, we've taken a functional approach: evaluate models by their functional performance (todo read: Toshniwal et al., 2021; Patel and Pavlick, 2022; Treutlein et al., 2024; Yan et al., 2024)

Mechanistic approach: evaluate a model's inner workings.

Mechanistic Interpretability: Tools for understanding the internal mechanisms of neural networks
- goal: improving or aligning model performance (e.g. steering its behavior in some way; example: Anthropic's Golden Gate Bridge)
- many interesting results adapting MI techniques to study world models (todo read: Abdou, 2021; Li, 2021; Gurnee and Tegmark, 2023; Li, 2023; Nanda, 2023; Nikankin, 2024; Spies, 2024; Feng, 2024; Li, 2025)

Comprehensive mechanistic understanding would make it easy to evaluate if models understand the world. How feasible is comprehensive understanding?

todo read: "The Dark Matter of Neural Networks" by Chris Olah.

"If you're aiming to explain 99.9% of a model's performance, there's probably going to be a long tail of random crap you need to care about" - Neel Nanda (Google DeepMind)

todo read: "Emergent world representations: exploring a sequence model trained on a synthetic task" - Kenneth Li et al. (Harvard) - uncovers evidence of an emergent nonlinear internal representation of the board state

todo read: "Actually, Othello-GPT has a linear emergent world representation" - Neel Nanda

todo read: "OthelloGPT learned a bag of heuristics" - jylin04, JackS, Adam Karvonen, Can (AI Alignment Forum)

related idea: use of world models in RL: predictive models of an environment's dynamics (todo read: Ha and Schmidhuber, 2018; Hafner, 2019; Guan, 2023; Genie 3 team, 2025; and many more)
- world models in RL are trained on state explicitly
- Goal isn't recovering structure; it's making better predictions or improving an agent's planning capabilities

So, we've seen:
- generative models can do amazing things with incoherent world models
- but, it makes them fragile for other tasks
- where to go from here?
  - accept the fact that our world models are imperfect.
  - one approach: zoom in and evaluate models based on how people use them (todo read: Lee, 2023; Collins, 2024; Chiang, 2024; Ibrahim, 2024; Vafa, 2024; Bean, 2025; Chang, 2025)
  - also work on improving architectures (state-space models seem to have better inductive biases than transformers)
    - neuro-symbolic models can combine neural and formal reasoning (todo read: Lake, 2015; Ellis 2020; Wong 2023, Wong 2025)
  - new training procedures
    - next-token prediction creates unwanted heuristics (McCoy 2023; Bachmann and Nagarajan 2024)
  - alternative ideas:
    - moving beyond next-token prediction
    - incorporating human feedback to improve world models
    - causal representation learning (todo read: Arjovsky 2019; Scholkopf 2021; Ahuja 2022; von Kugelgen 2024)
  - many promising ways to improve world models; evaluation metrics will help get us there

Links to papers mentioned (by Keyon in youtube comment):
- Bubeck et al. (2023): Sparks of Artificial General Intelligence: Early experiments with GPT-4 
- Hendrycks et al. (2020): Measuring Massive Multitask Language Understanding 
- Bowman and Dahl (2021): What Will it Take to Fix Benchmarking in Natural Language Understanding? 
- Mitchell (2021): Why AI is Harder Than We Think
- Raji et al. (2021): AI and the Everything in the Whole Wide World Benchmark 
- Mancoridis et al. (2025): Potemkin Understanding in Large Language Models 
- Toshniwal et al. (2021): Chess as a Testbed for Language Model State Tracking 
- Li et al. (2021): Implicit Representations of Meaning in Neural Language Models 
- Patel and Pavlick (2021): Mapping Language Models to Grounded Conceptual Spaces 
- Kim and Schuster (2023): Entity Tracking in Language Models 
- Li et al. (2023): Emergent World Representations: Exploring a Sequence Model Trained on a Synthetic Task 
- Vafa et al. (2024): Evaluating the World Model Implicit in a Generative Model 
- Vafa et al. (2025): What Has a Foundation Model Found? Using Inductive Bias to Probe for World Models 
- Breiman (2001): Statistical Modeling: The Two Cultures 
- D'Amour et al. (2022): Underspecification Presents Challenges for Credibility in Modern Machine Learning 
- Black et al. (2022): Model Multiplicity: Opportunities, Concerns, and Solutions 
- Bommasani et al. (2021): On the Opportunities and Risks of Foundation Models 
- Treutlein et al. (2024): Connecting the Dots: LLMs can Infer and Verbalize Latent Structure from Disparate Training Data 
- Yan et al. (2024): Inconsistency of LLMs in Molecular Representations 
- Nanda et al. (2023): Progress measures for grokking via mechanistic interpretability 
- Abdou et al. (2021): Can Language Models Encode Perceptual Structure Without Grounding? A Case Study in Color 
- Gurnee and Tegmark (2023): Language Models Represent Space and Time 
- Nanda et al. (2023): Emergent Linear Representations in World Models of Self-Supervised Sequence Models 
- Nikankin et al. (2024): Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics
- Spies et al. (2024): Transformers Use Causal World Models in Maze-Solving Tasks 
- Feng et al. (2024): Monitoring Latent World States in Language Models with Propositional Probes 
- Li et al. (2025): (How) Do Language Models Track State?
- Suzgun et al. (2018): On Evaluating the Generalization of LSTM Models in Formal Languages 
- Bhattamishra et al. (2020): On the Ability and Limitations of Transformers to Recognize Formal Languages
- Liu et al. (2022): Transformers Learn Shortcuts to Automata
- Merrill and Sabharwal (2023): The Parallelism Tradeoff: Limitations of Log-Precision Transformers 
- Merrill et al. (2024): The Illusion of State in State-Space Models 
- Ha and Schmidhuber (2018): World Models 
- Hafner et al. (2019): Dream to Control: Learning Behaviors by Latent Imagination 
- Guan et al. (2023): Leveraging Pre-trained Large Language Models to Construct and Utilize World Models for Model-based Task Planning 
- Genie 3 team (2025): Genie 3: A new frontier for world models 
- Lee et al. (2023): Evaluating Human-Language Model Interaction 
- Collins et al. (2024): Building Machines that Learn and Think with People 
- Chiang et al. (2024): Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference 
- Vafa et al. (2024): Do Large Language Models Perform the Way People Expect? Measuring the Human Generalization Function 
- Bean et al. (2025): Clinical knowledge in LLMs does not translate to human interactions 
- Chang et al. (2025): ChatBench: From Static Benchmarks to Human-AI Evaluation  
- Lake et al. (2015): Human-level concept learning through probabilistic program induction 
- Ellis et al. (2020): DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning
- Wong et al. (2023): From Word Models to World Models: Translating from Natural Language to the Probabilistic Language of Thought
- Wong et al. (2025): Modeling Open-World Cognition as On-Demand Synthesis of Probabilistic Models 
- Arjovsky et al. (2019): Invariant Risk Minimization 
- Schölkopf et al. (2021): Towards Causal Representation Learning 
- Ahuja et al. (2022): Interventional Causal Representation Learning 
- von Kügelgen (2024): Identifiable Causal Representation Learning: Unsupervised, Multi-View, and Multi-Environment
