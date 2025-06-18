## SwarmGPU Code Guide

*A C# / TorchSharp CUDA implementation of Particle Swarm Optimisation (PSO)*

---

### 1. What the project does - in one paragraph

`SwarmGPU` is a lightweight optimisation library that runs **Particle Swarm Optimisation** entirely on the GPU via **TorchSharp-CUDA**.  The core class, `SwarmGPU`, manages a population of `Particle` objects, iteratively updates their positions/velocities, and returns the best‐found solution to a user–supplied objective function.  The accompanying **MSTest** project demonstrates ten analytic objective functions (in `ObjectiveFunctions.cs`) and validates convergence with unit tests in `OptimizationTests.cs`.

---

### 2. Solution structure

| Project          | Purpose                                                         |
| ---------------- | --------------------------------------------------------------- |
| **SwarmGPU**     | Library – PSO engine (`Particle.cs`, `SwarmGPU.cs`)             |
| **SwarmGPUTest** | MSTest project – sample objective functions + convergence tests |
| **SwarmGPU.sln** | Top-level Visual Studio solution                                |

---

### 3. Prerequisites

| Requirement                                  | Notes                                                                                            |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **.NET 9.0 SDK**                             | Declared in both `.csproj` files.                                                                |
| **CUDA-capable GPU** + drivers               | TorchSharp will silently fall back to CPU if no GPU is found, but the library is tuned for CUDA. |
| **NuGet:** `TorchSharp-cuda-windows 0.105.0` | Pulled automatically by `dotnet restore`.                                                        |

---

### 4. Building & running tests

```bash
# Clone and restore packages
git clone https://github.com/<you>/SwarmGPU.git
cd SwarmGPU/src
dotnet restore

# Build both projects
dotnet build -c Release

# Run MSTest suite (SwarmGPUTest)
dotnet test
```

All ten tests in **`OptimizationTests.cs`** should pass; each asserts that PSO finds the known minimum of the provided analytic function within `0.01` units.

---

### 5. Quick start – using the library in your own code

```csharp
using SwarmGPU;
using TorchSharp;
using static TorchSharp.torch;

// (1) Define an objective function that follows TorchSharp
//     memory-management rules (see §7 for details).
Tensor Sphere(Tensor x)
{
    using var scope = torch.NewDisposeScope();
    Tensor value = torch.sum(x.pow(2));      // f(x) = Σ x_i²
    return value.MoveToOuterDisposeScope();  // <- critical
}

int dims = 5;
Tensor min = torch.full(new long[]{dims}, -5.0f, ScalarType.Float32);
Tensor max = torch.full(new long[]{dims},  5.0f, ScalarType.Float32);

using var pso = new SwarmGPU.SwarmGPU(
    numberOfParticles: 50,
    dimensions:        dims,
    minBounds:         min,
    maxBounds:         max,
    omega:             0.5f,   // inertia weight
    phiP:              1.8f,   // cognitive coefficient
    phiG:              1.8f,   // social coefficient
    maxIterations:     500);

(var bestPos, var bestVal) = pso.Optimize(Sphere);

Console.WriteLine($"Best value  : {bestVal.item<float>()}");
Console.WriteLine($"Best vector : {bestPos}");

// Dispose tensors you own
bestPos.Dispose();
bestVal.Dispose();
min.Dispose();
max.Dispose();
```

---

### 6. API reference (the public surface)

| Member                                                                                                                                        | Summary                                                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `SwarmGPU(int numberOfParticles, int dimensions, Tensor minBounds, Tensor maxBounds, float omega, float phiP, float phiG, int maxIterations)` | Constructs and **clones** the bound tensors so you can dispose your copies immediately. |
| `(Tensor bestPosition, Tensor bestValue) Optimize(Func<Tensor, Tensor> objective)`                                                            | Runs PSO, returns **clones** of the global-best tensors (caller must dispose).          |
| `void Dispose()`                                                                                                                              | Releases all internally-held tensors & particles.                                       |
| `Particle` class (internal users rarely interact with it)                                                                                     | Holds `Position`, `Velocity`, personal best tensors; implements `IDisposable`.          |

---

### 7. Writing safe objective functions (TorchSharp disposal rules)

`SwarmGPU.Optimize` expects your delegate to **return a tensor that *you* create and then “move to the caller”**.  The pattern is always:

```csharp
Tensor MyFunc(Tensor x)
{
    using var scope = torch.NewDisposeScope();  // own all intermediates
    Tensor result = /* ... maths ... */;
    return result.MoveToOuterDisposeScope();    // hand ownership to SwarmGPU
}
```

Common pitfalls:

* **Forgetting `MoveToOuterDisposeScope()`** ⇒ the tensor is destroyed at the end of the scope and PSO throws `Tensor invalid -- empty handle`.
* **Returning a view of `x` without cloning** ⇒ PSO will later mutate `x`; your objective value becomes undefined.

---

### 8. Parameter tuning cheatsheet

| Symbol | In code             | Typical range                        | Effect                                                       |
| ------ | ------------------- | ------------------------------------ | ------------------------------------------------------------ |
| ω      | `omega`             | 0.3 – 0.9                            | Inertia; higher keeps momentum, lower promotes exploitation. |
| φᵖ     | `phiP`              | 1.0 – 2.5                            | Cognitive weight (particle’s own best).                      |
| φᵍ     | `phiG`              | 1.0 – 2.5                            | Social weight (global best).                                 |
| *N*    | `numberOfParticles` | 20 – 100 (depends on dimensionality) | Swarm size.                                                  |
| *iter* | `maxIterations`     | 200 – 2000                           | More iterations → better accuracy but longer run.            |

The defaults in the tests (`N = 50`, `ω = 0.5`, `φᵖ = φᵍ = 1.8`) work well for most smooth functions in up to \~10 dimensions.

---

### 9. Extending the library

* **Custom objective functions** – inherit `ObjectiveFunctions.FunctionData` only if you want to plug into the test harness; otherwise any `Func<Tensor, Tensor>` works.
* **Different topologies** – the current update rule is global-best PSO. To implement local-best or ring topologies, modify `SwarmGPU.Optimize` where `_globalBestPosition` is referenced.
* **Constraint handling** – positions are clamped to `[minBounds,maxBounds]`.  Replace `torch.clamp` with custom repair logic if you need penalty methods or reflecting bounds.

---

### 10. Unit-test framework walkthrough

1. `ObjectiveFunctions.GetTestFunctions()` builds ten **FunctionData** instances with known minima.
2. `OptimizationTests` uses `DynamicData` to feed each into PSO.
3. Assertions ensure:

   * Returned value ≈ expected value (±0.01).
   * Each coordinate of returned position ≈ expected vector (±0.01).
4. Every tensor produced by the test is explicitly disposed – examples of best practice.

Use the test project as a template for your own benchmark suites.

---

### 11. Troubleshooting

| Symptom                                | Likely cause / fix                                                                          |
| -------------------------------------- | ------------------------------------------------------------------------------------------- |
| `Tensor invalid -- empty handle`       | Objective function forgot `MoveToOuterDisposeScope()` or disposed a tensor still in use.    |
| Convergence stalls at `float.MaxValue` | Objective returned `+∞` or `NaN`; check bounds and maths.                                   |
| Very slow on GPU                       | Ensure CUDA toolkit + matching driver is installed; otherwise TorchSharp falls back to CPU. |
| Memory leak warning                    | Forgot to `Dispose()` tensors returned by `Optimize`.                                       |

---

### 12. Performance tips

* Compile in **Release** and target **x64**.
* Keep tensors on the same device – do **not** move them back to CPU inside the objective.
* Use **single-precision (`float32`)** unless your problem needs higher precision.
* Reduce allocations inside the objective by re-using pre-constructed constant tensors (see `ObjectiveFunctions` pattern).

---

Happy swarming!

</br>
Copyright [TranscendAI.tech](https://TranscendAI.tech) 2025.<br>
Authored by Warren Harding. AI assisted.</br>
