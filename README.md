# GPT-2 From Scratch

![GPT-2 Architecture](https://github.com/user-attachments/assets/31074003-b747-4e59-8f30-d4c63c2e8fd0)

## Model Architecture

### Model Config:
- Embedding Dimensions: 768
- Vocabulary Size: 50,257
- Sequence Length: 1,024
- Attention Heads: 8
- Decoder Blocks: 12
- Dropout: 0.1

### Architecture Overview

The GPT-2 model is based on the transformer architecture, specifically designed for natural language processing tasks. Key components include:

1. **Positional Encoding**: Helps the model understand the order of words in a sequence.
2. **Multi-Head Attention**: Allows the model to focus on different parts of the input simultaneously.
3. **Feed-Forward Networks**: Applies non-linear transformations to the input data.
4. **Layer Normalization**: Stabilizes and accelerates the training process.

## Implementation Details

This project implements the GPT-2 model from scratch, providing a deep understanding of its inner workings. The implementation closely follows the original architecture while offering customization options.

## Resources

- Andreij Karpathy Lectures - https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&pp=iAQB
- Sebastian Raschka - https://github.com/rasbt/LLMs-from-scratch
- Original GPT-2 Paper - https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
