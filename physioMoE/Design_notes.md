# physioMoE

This is a markdown file to explain the thinking process of making physioMoE which stands for physiological Mixture of Experts inspired by the [Mixtral of Experts paper](https://arxiv.org/abs/2401.04088). The goal of this paper was to decrease the resources needed to process a prompt by decreaseing the number of active parameters but I guess I would help the performance in our case. 
## Architecture
![](mixture_of_experts_architecture%201.png)
The intuition behind this is that hopefully with using multiple neural networks (Experts) the model can learn from different datasets reliable patterns. The embedding are routed the convient expert using a router, a neural network conditionned by the embedding of a text describing the task and other additionnal contextual details about the enviromenet of performance.
### Ablation studies
Such an architecture require making choice in several technically places. One hardly finds the optimal setup from the first try so here a couple of choices to try:
#### Resampling
NormWear resampled every signal to 65Hz. try techniques to make the model more robust against sampling rates and admits everything
#### Fusion mechanism of signal embeddings:
- averaging them
- CNN 
- CLS-fusion mechanism
#### Router network: 
try different architectures
#### Text encoder:
- try different Text encoders
- try fine-tune (fully or partially) a model to encode the task this may lead the attention mechanism to focus on terms that are relevant to the cognitive workload
- try an encoder that is from an LLM that plays in different enviroment
- try different fusion methods for the embedding.
#### Text of the context
The main question here is how will the prompt to the text encoder be because the prompt can influence massively the encoding, here are a few way of prompting to try
- unified standard text for all the datasets 
- free text to decribe the task
- the influence of text augmentation

I have an thought here: Prompt engineering is specific to the model used. In fact, every big AI company like OpenAI, Claude,... provide a guide for prompt engineering of there model. so when using the text encoder we should look for the guide if provided otherwise it think it would helpful to follow the method these company did to come up with their guide which will be helpful in the case of trying to fine tune our model.
## Datasets
the datasets are going to be concatenated, segmented and preprocessed with the same pipline as normWear.
### Dataset split
- classical train test split
- Leave one participant out: it is better, I think to leave a subject out for testing for every dataset.
### Dataset content
the dataset contains signals for each participant during a task with certain level of complexity along side with this a text describing the task, and any other contextual info.

Exemple (TODO)

## Training
There are two methods here either 
- train the whole network at once
- train each expert on a dataset then freezing it and train the router network
For the validation set there are two options:
- contains a participant never existed in the training 
- or it is the classic validation set where everything is mixed
For the trainning process, I want to try autoResearch to train the model with an LLM.
## Evaluation
either with a LOPO style or classic 