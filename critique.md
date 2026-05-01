# Approach Critique

APPROACH.md explains an approach to shrinking LLMs while retaining "useful" capabilities. INTERLEVEAVED_TRAINING.md explains an approach to add new capabilities to a model, alternating "shrinking" phases to "growing phases", and AUTONOMOUS_AGENT.md describes an approach for continuous learning.

Perform a review of the approach, the goal is to find and enhance "good points" while identifying potential criticalities.

Some points to address (but more are welcome):

- How does the neuron swapping and compacting in few layers actually helps computation?
- What exactly is swapped? Does it require to alter the structure of the LLM, or it's compatible with the structure, just a reshuffle?
- Regarding the interleaved training and continuous learning, what kind of "harness" might be needed to control the "temperature" that activates / deactivates less used neuron layers, and keep track of the activations? The idea is to have a "dynamic" size model where some layers might even be stored on disk and dynamically added or removed to the running model (maybe restarting it). Can this approach be added on top of current agentic harnesses and frameworks like pytorch, or does it require completely reviewing the fundamental structure of the frameworks? The point is understanding whether this tecnique can be already tested on real world models

Be specific and detailed in performing the technical analysis, explain the technicalities in detail, describing exactly which parts of the model will be addressed. If you need, pick a given architecture to perform this analysis, e.g. gemma4 26b Moe, or a Qwen smaller model (e.g. 8b).


