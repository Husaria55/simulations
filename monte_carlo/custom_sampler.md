# Advanced Stochastic Modeling: Implementing Custom Samplers in RocketPy

**Instructor's Note:** *While standard distributions (Normal, Uniform) cover 90% of engineering use cases, the remaining 10%—where data is bimodal, skewed, or correlated—often holds the critical failure modes. This guide moves beyond standard initialization to give you complete control over how uncertainty is injected into your simulation.*

---

## 1. The Backbone: `CustomSampler` Class
RocketPy standardizes uncertainty injection via the `CustomSampler` abstract base class. To define your own probability logic, you must inherit from this class.

### Required Architecture
Your custom class **must** implement two specific methods to interface correctly with RocketPy’s stochastic engines (like `StochasticSolidMotor` or `StochasticEnvironment`).

| Method | Signature | Purpose | Expert Context |
| :--- | :--- | :--- | :--- |
| **Sample** | `sample(self, n_samples)` | Returns a list of `n` samples. | This is the core logic. It doesn't matter *how* you generate the numbers (math formula, look-up table, API call), as long as a list of floats returns. |
| **Reset Seed** | `reset_seed(self, seed)` | Resets the RNG state. | **Critical for Monte Carlo.** If running parallel simulations, failing to implement this correctly will cause every thread to generate the *exact same* "random" numbers, rendering your statistical analysis void. |

#### Basic Template
```python
from rocketpy import CustomSampler
import numpy as np

class MySampler(CustomSampler):
    def sample(self, n_samples=1):
        # Your custom logic here
        return [ ... ] 
        
    def reset_seed(self, seed=None):
        # Ensure independence across runs
        np.random.default_rng(seed)