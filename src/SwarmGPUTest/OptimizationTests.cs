//Copyright Warren Harding 2025.
using Microsoft.VisualStudio.TestTools.UnitTesting;
using SwarmGPU;
using TorchSharp;
using static TorchSharp.torch;
using System.Collections.Generic;

namespace SwarmGPUTest
{
    [TestClass]
    public sealed class OptimizationTests
    {
        private const float Tolerance = 0.01f; // Tolerance for float comparisons

        public static IEnumerable<object[]> GetOptimizationTestData()
        {
            List<ObjectiveFunctions.FunctionData> functions = ObjectiveFunctions.GetTestFunctions();
            foreach (ObjectiveFunctions.FunctionData funcData in functions)
            {
                yield return new object[] { funcData };
            }
        }

        [DataTestMethod]
        [DynamicData(nameof(GetOptimizationTestData), DynamicDataSourceType.Method)]
        public void TestSwarmOptimization(ObjectiveFunctions.FunctionData functionData)
        {
            // Ensure the FunctionData object and its internal Tensors are disposed after use.
            using (functionData) 
            {
                // Test data for optimization parameters
                int numberOfParticles = 50; 
                float omega = 0.5f;
                float phiP = 1.8f;
                float phiG = 1.8f;
                int maxIterations = 1000; 

                // Initialize SwarmGPU optimizer. It clones the bounds tensors itself.
                using (SwarmGPU.SwarmGPU swarmOptimizer = new SwarmGPU.SwarmGPU(
                    numberOfParticles,
                    functionData.Dimensions,
                    functionData.MinBounds,
                    functionData.MaxBounds,
                    omega,
                    phiP,
                    phiG,
                    maxIterations))
                {
                    // Perform optimization and get results. The returned Tensors are new and must be disposed.
                    // The functionData.Evaluate method is now called via a lambda.
                    (Tensor bestPosition, Tensor bestValue) = swarmOptimizer.Optimize(x => functionData.Evaluate(x));

                    // Assertions
                    Assert.IsNotNull(bestPosition, $"Best position from {functionData.Name} should not be null.");
                    Assert.IsNotNull(bestValue, $"Best value from {functionData.Name} should not be null.");

                    // Validate dimensions of the returned position tensor.
                    Assert.AreEqual(functionData.Dimensions, bestPosition.shape[0], $"Dimensions mismatch for {functionData.Name} BestPosition.");

                    // Check if the found best value is within tolerance of the expected minimum value.
                    float actualBestValue = bestValue.item<float>();
                    Assert.AreEqual(functionData.ExpectedMinimumValue, actualBestValue, Tolerance, $"Best value for {functionData.Name} is not within tolerance.");

                    // Check if each component of the found best position is within tolerance of the expected position.
                    for (int i = 0; i < functionData.Dimensions; i++)
                    {
                        Assert.AreEqual(functionData.ExpectedMinimumPosition[i].item<float>(), bestPosition[i].item<float>(), Tolerance,
                                        $"Position component {i} for {functionData.Name} is not within tolerance.");
                    }

                    // Dispose the result tensors from Optimize method, as per contract.
                    bestPosition.Dispose();
                    bestValue.Dispose();
                }
            }
        }
    }
}