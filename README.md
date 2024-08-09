# shakespeare-GPT

## Introduction
Welcome to this repository, an extension of the engaging lecture by Andrej Karpathy. This project dives deep into the transformer architecture, as explained in the groundbreaking paper "Attention Is All You Need". Together, we will implement the topics into real code and construct our version of a GPT that might produce nonsensical output but in a style reminiscent of Shakespeare's works.

## Starting Point
Our journey starts with Andrej providing a starting point for building a GPT from scratch. We use a compact dataset, which is an amalgamation of all of Shakespeare's works, and a similarly modest number of parameters (approximately 370K were used in training).

The starting point is illustrated below:

![Transformer - model architecture](./transformer.png)

**Note**: This transformer is purely a decoding transformer, as depicted by the large 'X' canceling out the encoding section. It relies on self-attention rather than cross-attention, rendering the second attention layer redundant (indicated by the small 'X').

## My Continuation of Work
The subsequent work conducted in the `improved-gpt.py` file continues Andrej's original endeavors. This progression wouldn't have been achievable without a solid understanding of the topics discussed in the initial models. My understanding can additionally be demonstrated by the extensive comments detailing key concepts and flow of information/data.

The main continuation of Andrej's work is my introduction of cyclical learning rates to attone for the guess work previously surrounding selecting the most optimal learning rate for our model. Learning rate is one of the, if not the most, important hyperpatameters to tune and the basis for the logic applied can be found in the following article: 'https://ieeexplore.ieee.org/document/7926641'.

### Scheduler
* Scales learning rate based on position relative to the local minima approached in gradient descent
  * This decreased loss rates by a mean factor of "___" after 100 trials holding all other hyperparameters equal

## Skills and Lessons Acquired
Through this project, the key skills and lessons I've learned are:
* Tokenization
* Constructing a transformer from scratch
* PyTorch
* Model training using AdamW
* Model validation
* Self-attention
* Autoregressive language modeling

## Next Steps
The stopping point for our GPT is the pre-training stage where we have essentially a document finisher that is unaligned to any system of reward and cannot properly fulfill the request of a prompt. To truly resemble the dev process (on a minute scale) of a product such as ChatGPT 3.0 there are a handful of steps still left to complete. Namely, aggregation of prompt -> response context, creation of a reward model and response ranking, and lastly a PPO model to fine-tune our GPT based on scores from the reward model. Continuation of the work conducted in this repository might attempt some of these steps to further the quality and parallels to industry norms. For now, this serves as a great introduction to the logic behind transformers and some of the necessary topics needed to understand in order to interact and improve them.