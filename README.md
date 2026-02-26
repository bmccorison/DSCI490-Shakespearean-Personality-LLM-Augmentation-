# DSCI490 - Shakespearean Personality LLM Augmentation

This project aims to develop and evaluate methods for persona-consistent AI, testing whether LLMs can truly encapsulate true personal identities. Language Models will be created for Shakespeare’s works of “Hamlet” and “Macbeth”, creating models for four total characters and analyzing performance by test questions and two-way/four-way conversations.

## Project Structure

- `benchmarking/`: notebooks and notes for benchmark development and evaluation workflows.
- `data/`: source text data used for training and experimentation.
- `docs/`: citations, roadmap, and supporting project documentation.
- `misc/`: random scripts and notebooks for tests, such as response-generation and model-size testing.
- `models/`: saved LoRA adapter checkpoints and fine-tuned model artifacts.
- `pipeline/`: reusable internal pipeline library modules (ingestion, RAG, and generation).
- `training/`: training notebooks and logs for LoRA fine-tuning runs.
- `requirements.txt`: Python dependency list for the project environment.
