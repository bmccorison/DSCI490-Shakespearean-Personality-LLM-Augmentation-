# Initial Planning

These ideas will be converted into different milestones and GitHub issues from CI. Temporary file, and all everything is currently very tentative.

## Phase 1 - Background Learning on LLMs (Week 1)

- Understand how the HuggingFace API works (specifically with text-to-text transformer models)
- Know the basics on transformer models/LLMs (baseline, will learn more as we go)
- Test open source models on personal and HPCL hardware
- Build diagrams of the final pipeline (training, reinforcement, prompt engineering, RAG, etc.(**TBD**)). Should roughly understand what all of these do.
- Choose appropriate baseline open-weight models
  - Comprehensive enough to understand stuff, but not too big to where we can't run it
  - It would probably be best to aim for one smaller one (more for testing/dev) and a bigger one (prod), then we could also compare them at the end.
- Read the *Hamlet* and *Macbeth* works

## Phase 2 - Data Aquisition, Exploration, Pre-processing, Benchmark Development (Weeks 2-4)

- Find the Shakespeare texts *Hamlet* and *Macbeth* (some format like csv or txt would be preferred)
- Write a parser to clearly seperate/distinguish dialouge based on speaker
- Examine and plot data structure, determine cleaning steps
- Clean data (stuff like whitespaces and things that the parser didn't catch. **TBD** once texts are found)
- Search for any additional helpful context for the model (this could be moved to phase 3 probably)
- **Benchmark Development** (The big one)
  - **TBD** - Determining this will be a hard task, and initial benchmark development may bleed into phase 3. Would probably be best if it consisted of both training and testing benchmarks.

## Phase 3 - Initial Character-Conditioned Model Development (Weeks 4-7)

- Create a basic testing pipeline for evaluating correctness (ie if it actually works, like a staging environment)
  - For example, a CLI tool that would ask for a prompt and return the response. This could be expanded to support multiple characters in later phases.
- Load in the initial small transformer model and test basic functionality
- Make some sort of "model history" functionality for rollbacks (in case we accidentally poison the model at any point)
- Begin basic training pipeline
  - Feed in dialouge from the plays into a LoRA/some adapter (? Could also do full fine-tuning, but i'm not sure)
  - Feed in any other relevant context (same way as above)
  - Data-side shaping (? Not too familiar with this)
- Finish initial benchmarking functions
- Begin development of reinforcement training pipeline (Could be used to dynamically "teach" the model to pass some training benchmarks)
- Initial prompt-engineering (instruction fine-tuning), integrate into testing & reinforcement pipeline
- Implement RAG functionality, feeding in context at runtime
- Implement lightweight "runtime steering", such as Logit biasing, decoding, or self-refleciton (? Not too familiar with this)

## Phase 4 - Continue Character-Condition Model Development (Weeks 8-9)

- Continue optimizing initial character model
- Expand to different characters
- Tune benchmarks
- **TBD**
- Begin development/research into making conversational LLM tools (conversing with each other)

## Phase 5 - Conversation Experiments (Weeks 10-13)

- Finalize benchmarking for two-way and four-way conversations
- Develop basic CLI tooling to monitor/log conversations (expand upon testing tooling)
- Develop front-end interface & backend endpoints to view conversations (JJ can do this mwahahaha)
- Refine prompts based on experiments
- Write scripts to generate visualizations and performance metrics based on dialouge
- **TBD**
- Begin comparing performances of small vs. large models

## Phase 6 - Reporting (Weeks 14-15)

- Capture and visualize final relevant benchmarks (small vs. large & conversational)
- Write final report/presentation
- Deploy frontend interface for demo purposes (optional, will require compute resources (aka money))
