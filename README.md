# Benchmarking LLM Response Coherence Across Paraphrased Prompts

## Research Objective

Evaluate the coherence and stability of LLM responses when prompts are paraphrased or structurally modified.

## Models Tested

GPT  
Gemini  
DeepSeek  

## Prompt Dataset

470 prompts covering three reasoning dimensions:

- Instructability
- Consistency under paraphrasing
- Multi-step planning

## Evaluation Metrics

Jaccard Similarity  
Constraint Violation Rate  
Count Agreement Error Rate  
Structural Drift Rate  

## Repository Structure

data/
Prompt dataset

results/
Model output CSV files

scripts/
Model querying scripts and evaluation pipeline

analysis/
Interpretation of benchmark results
