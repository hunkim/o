# o: Step-by-Step Reasoning Demos

This repository contains toy demos for step-by-step reasoning, inspired by:
- STaR paper https://arxiv.org/abs/2203.14465
- OpenAI's o1
- Reflection techniques
- SkunkworksAI/reasoning-0.01 dataset
- Implementation from [bklieger-groq/g1](https://github.com/bklieger-groq/g1)
- And many more

## Quick Demo
- toy-o1: https://toy-o1.streamlit.app/
- toy-o2: https://toy-o2.streamlit.app/

<img width="1319" alt="image" src="https://github.com/user-attachments/assets/50f01e84-c0b5-454b-b0da-f807955c9119">

## Overview

We present two approaches:
1. **o1**: Fixed reasoning based on the SkunkworksAI/reasoning-0.01 dataset. You can also leverage web search.
2. **o2**: Relies on the LLM's basic planning skills (code mostly reused from bklieger-groq/g1).

Both approaches have their pros and cons, providing interesting comparisons.

We use [Solar-Pro Preview](https://huggingface.co/upstage/solar-pro-preview-instruct) as the base LLM, but you can try others using Langchain.
## Running Locally
1. Clone this repository
2. Add 'UPSTAGE_API_KEY' to your environment variables (use .streamlit config or .env)
3. Run `make o1` or `make o2`

## Changing Base LLMs
Replace the following code with your preferred LLM:

```python
langchain_upstage import ChatUpstage as Chat
llm = Chat(model="solar-pro")
```

## Limitations
Solar-Pro is only 22B size mode with 4K context windows. https://huggingface.co/upstage/solar-pro-preview-instruct

## Contributions
comments, pull request are always welcome
