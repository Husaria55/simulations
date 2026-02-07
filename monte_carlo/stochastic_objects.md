# RocketPy: Working with Stochastic Objects

Stochastic objects allow you to model real-world uncertainties in rocket simulations by extending deterministic models (like `Environment`, `SolidMotor`, or `Rocket`). Instead of fixed values, parameters are treated as random variables to create more realistic simulations.

---

## 1. Core Concepts
* **Deterministic vs. Stochastic:** A deterministic class (e.g., `SolidMotor`) uses fixed values. Its stochastic counterpart (e.g., `StochasticSolidMotor`) assigns probability distributions to those values.
* **The Workflow:** 1. Define a standard deterministic object.
    2. Pass it into a Stochastic object to define uncertainties.
    3. Call `.create_object()` to generate a randomly sampled version of the base object.

---

## 2. Argument Categories
When initializing a Stochastic class, arguments fall into three categories:

| Category | Description | Requirement |
| :--- | :--- | :--- |
| **Deterministic Object** | The base model instance (e.g., `solid_motor=motor`). | **Mandatory** |
| **Optional Arguments** | Parameters from the base class you wish to vary. | Optional |
| **Additional Arguments** | Parameters unique to stochastic models (e.g., drag multiplication factors). | Optional |

> **Note:** If an optional argument is omitted, the simulation uses the fixed (nominal) value from the deterministic object.

---

## 3. Specifying Uncertainties
You can define how a parameter varies using several input formats:

* **Single Value:** Interpreted as the **Standard Deviation**. Uses a Normal distribution; the nominal value is pulled from the deterministic object.
* **Tuple `(nominal, std_dev)`:** Explicitly sets both; uses a Normal distribution.
* **Tuple `(nominal, std_dev, "type")`:** Sets nominal, standard deviation, and a specific distribution (e.g., `"uniform"`, `"binomial"`, `"poisson"`, `"exponential"`, etc.).
* **Tuple `(std_dev, "type")`:** Sets standard deviation and type; pulls the nominal value from the base object.
* **List `[val1, val2, ...]`:** Randomly chooses one value from the list for each simulation (cannot use standard deviations here).
* **`CustomSampler` Object:** For advanced users requiring full control over sample generation.


---

##