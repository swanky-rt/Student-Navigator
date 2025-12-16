import time
import torch
import numpy as np
from context_agent_classifier import ContextAgentClassifier

# Setup
model = ContextAgentClassifier()
# Random weights are ok for speed testing
model.eval() 

# Dummy input
text = "I need to transfer $500 to my checking account."

print("Running benchmark...")
latencies = []

# Warmup (GPUs/CPUs need a few runs to wake up)
for _ in range(10):
    model.predict(text, {0: "A", 1: "B"})

# Actual Test (100 runs)
for _ in range(100):
    start_time = time.time()
    model.predict(text, {0: "A", 1: "B"})
    end_time = time.time()
    latencies.append((end_time - start_time) * 1000) # Converting to ms

avg_time = np.mean(latencies)
print(f"\n--- RESULTS ---")
print(f"Average Latency: {avg_time:.2f} ms")
print(f"Min Latency: {np.min(latencies):.2f} ms")
print(f"Max Latency: {np.max(latencies):.2f} ms")