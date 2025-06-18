//Copyright Warren Harding 2025.
using System;
using System.Collections.Generic;
using TorchSharp;
using static TorchSharp.torch;

namespace SwarmGPUTest
{
    public static class ObjectiveFunctions
    {
        public abstract class FunctionData : IDisposable
        {
            public string Name { get; private set; }
            public int Dimensions { get; private set; }
            public Tensor MinBounds { get; private set; }
            public Tensor MaxBounds { get; private set; }
            public float ExpectedMinimumValue { get; private set; }
            public Tensor ExpectedMinimumPosition { get; private set; }

            // Cached constant tensors for objective function evaluation
            public Tensor? ShiftTensor { get; private set; }
            public Tensor? WeightsTensor { get; private set; }
            public Tensor? Coefficient1Tensor { get; private set; } // For functions like VarCoeffQuadratic3D
            public Tensor? Coefficient2Tensor { get; private set; } // For functions like VarCoeffQuadratic3D

            public FunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds,
                                float expectedMinimumValue, Tensor expectedMinimumPosition,
                                Tensor? shiftTensor = null, Tensor? weightsTensor = null,
                                Tensor? coefficient1Tensor = null, Tensor? coefficient2Tensor = null)
            {
                Name = name;
                Dimensions = dimensions;
                MinBounds = minBounds.clone(); // Clone to own the tensor
                MaxBounds = maxBounds.clone(); // Clone to own the tensor
                ExpectedMinimumValue = expectedMinimumValue;
                ExpectedMinimumPosition = expectedMinimumPosition.clone(); // Clone to own the tensor

                ShiftTensor = shiftTensor?.clone();
                WeightsTensor = weightsTensor?.clone();
                Coefficient1Tensor = coefficient1Tensor?.clone();
                Coefficient2Tensor = coefficient2Tensor?.clone();
            }

            public abstract Tensor Evaluate(Tensor x); // New abstract method

            private bool _disposed = false;

            public void Dispose()
            {
                Dispose(true);
                GC.SuppressFinalize(this);
            }

            protected virtual void Dispose(bool disposing)
            {
                if (!_disposed)
                {
                    if (disposing)
                    {
                        // Dispose owned tensors
                        MinBounds?.Dispose();
                        MaxBounds?.Dispose();
                        ExpectedMinimumPosition?.Dispose();
                        ShiftTensor?.Dispose();
                        WeightsTensor?.Dispose();
                        Coefficient1Tensor?.Dispose();
                        Coefficient2Tensor?.Dispose();
                    }
                    _disposed = true;
                }
            }

            // Finalizer in case Dispose is not called explicitly.
            ~FunctionData()
            {
                Dispose(false);
            }
        }

        // Region for new derived FunctionData classes
        public class ShiftedSphere2DFunctionData : FunctionData
        {
            public ShiftedSphere2DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(diff.pow(2)) + 0.1f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class ShiftedSphere3DFunctionData : FunctionData
        {
            public ShiftedSphere3DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(diff.pow(2)) + 10.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class WeightedSumSquares2DFunctionData : FunctionData
        {
            public WeightedSumSquares2DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor, Tensor? weightsTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor, weightsTensor: weightsTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(this.WeightsTensor! * diff.pow(2)) + 2.5f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class WeightedSumSquares4DFunctionData : FunctionData
        {
            public WeightedSumSquares4DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor, Tensor? weightsTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor, weightsTensor: weightsTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(this.WeightsTensor! * diff.pow(2)) + 1.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class PowerQuadratic2DFunctionData : FunctionData
        {
            public PowerQuadratic2DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shift1, Tensor? shift2)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shift1, weightsTensor: shift2) { } // Mapping shift1 to ShiftTensor, shift2 to WeightsTensor

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor term1 = (x_working[0] - this.ShiftTensor!.item<float>()).pow(2);
                    Tensor term2 = (x_working[1] - this.WeightsTensor!.item<float>()).pow(4);
                    Tensor result = (term1 + term2 + 0.2f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class ShiftedSphere5DFunctionData : FunctionData
        {
            public ShiftedSphere5DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(diff.pow(2)) + 5.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class VarCoeffQuadratic3DFunctionData : FunctionData
        {
            public VarCoeffQuadratic3DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? coefficient1Tensor, Tensor? coefficient2Tensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, coefficient1Tensor: coefficient1Tensor, coefficient2Tensor: coefficient2Tensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor term1 = (x_working[0] - 3.0f).pow(2);
                    Tensor term2 = (this.Coefficient1Tensor!.item<float>() * (x_working[1] + 2.0f).pow(2));
                    Tensor term3 = (this.Coefficient2Tensor!.item<float>() * (x_working[2] - 1.0f).pow(2));
                    Tensor result = (term1 + term2 + term3 + 15.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class HighOffsetQuadratic4DFunctionData : FunctionData
        {
            public HighOffsetQuadratic4DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(diff.pow(2)) + 100.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class SimpleShiftedQuad2DFunctionData : FunctionData
        {
            public SimpleShiftedQuad2DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(diff.pow(2)) + 7.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }

        public class ShiftedSphere6DFunctionData : FunctionData
        {
            public ShiftedSphere6DFunctionData(string name, int dimensions, Tensor minBounds, Tensor maxBounds, float expectedMinimumValue, Tensor expectedMinimumPosition, Tensor? shiftTensor)
                : base(name, dimensions, minBounds, maxBounds, expectedMinimumValue, expectedMinimumPosition, shiftTensor: shiftTensor) { }

            public override Tensor Evaluate(Tensor x)
            {
                using (System.IDisposable funcScope = torch.NewDisposeScope())
                {
                    Tensor x_working = x.clone();
                    Tensor diff = (x_working - this.ShiftTensor!);
                    Tensor result = (torch.sum(diff.pow(2)) + 25.0f);
                    return result.MoveToOuterDisposeScope();
                }
            }
        }
        // End region for new derived FunctionData classes

        public static List<FunctionData> GetTestFunctions()
        {
            List<FunctionData> testFunctions = new List<FunctionData>();

            // Function 1: Shifted Sphere (Dim=2)
            {
                int dim = 2;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 1.0f, 2.0f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { 1.0f, 2.0f }, ScalarType.Float32);

                testFunctions.Add(new ShiftedSphere2DFunctionData("ShiftedSphere2D", dim, minB, maxB, 0.1f, expectedPos, shiftTensor: initialShift));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
            }

            // Function 2: Shifted Sphere (Dim=3)
            {
                int dim = 3;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { -1.0f, 0.5f, 3.0f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { -1.0f, 0.5f, 3.0f }, ScalarType.Float32);

                testFunctions.Add(new ShiftedSphere3DFunctionData("ShiftedSphere3D", dim, minB, maxB, 10.0f, expectedPos, shiftTensor: initialShift));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
            }

            // Function 3: Weighted Sum of Squares (Dim=2)
            {
                int dim = 2;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 2.0f, -1.0f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { 2.0f, -1.0f }, ScalarType.Float32);
                Tensor initialWeights = torch.tensor(new float[] { 0.5f, 2.0f }, ScalarType.Float32);

                testFunctions.Add(new WeightedSumSquares2DFunctionData("WeightedSumSquares2D", dim, minB, maxB, 2.5f, expectedPos, shiftTensor: initialShift, weightsTensor: initialWeights));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
                initialWeights.Dispose();
            }

            // Function 4: Weighted Sum of Squares (Dim=4)
            {
                int dim = 4;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 0.5f, 1.5f, -0.5f, -1.5f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { 0.5f, 1.5f, -0.5f, -1.5f }, ScalarType.Float32);
                Tensor initialWeights = torch.tensor(new float[] { 0.1f, 0.2f, 0.3f, 0.4f }, ScalarType.Float32);

                testFunctions.Add(new WeightedSumSquares4DFunctionData("WeightedSumSquares4D", dim, minB, maxB, 1.0f, expectedPos, shiftTensor: initialShift, weightsTensor: initialWeights));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
                initialWeights.Dispose();
            }

            // Function 5: Quadratic with different powers (Dim=2)
            {
                int dim = 2;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 1.0f, -1.0f }, ScalarType.Float32);
                Tensor shift1 = torch.tensor(1.0f, ScalarType.Float32);
                Tensor shift2 = torch.tensor(-1.0f, ScalarType.Float32);

                testFunctions.Add(new PowerQuadratic2DFunctionData("PowerQuadratic2D", dim, minB, maxB, 0.2f, expectedPos, shift1: shift1, shift2: shift2));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                shift1.Dispose();
                shift2.Dispose();
            }

            // Function 6: Shifted Sphere (Dim=5)
            {
                int dim = 5;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f }, ScalarType.Float32);

                testFunctions.Add(new ShiftedSphere5DFunctionData("ShiftedSphere5D", dim, minB, maxB, 5.0f, expectedPos, shiftTensor: initialShift));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
            }

            // Function 7: Quadratic with varied coefficients and shifts (Dim=3)
            {
                int dim = 3;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 3.0f, -2.0f, 1.0f }, ScalarType.Float32);
                Tensor twoFloat = torch.tensor(2.0f, ScalarType.Float32);
                Tensor halfFloat = torch.tensor(0.5f, ScalarType.Float32);

                testFunctions.Add(new VarCoeffQuadratic3DFunctionData("VarCoeffQuadratic3D", dim, minB, maxB, 15.0f, expectedPos, coefficient1Tensor: twoFloat, coefficient2Tensor: halfFloat));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                twoFloat.Dispose();
                halfFloat.Dispose();
            }

            // Function 8: Quadratic sum with higher constant term (Dim=4)
            {
                int dim = 4;
                Tensor minB = torch.full(new long[] { dim }, -10.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 10.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { -2.0f, -2.0f, -2.0f, -2.0f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { -2.0f, -2.0f, -2.0f, -2.0f }, ScalarType.Float32);

                testFunctions.Add(new HighOffsetQuadratic4DFunctionData("HighOffsetQuadratic4D", dim, minB, maxB, 100.0f, expectedPos, shiftTensor: initialShift));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
            }

            // Function 9: Simple shifted quadratic (Dim=2)
            {
                int dim = 2;
                Tensor minB = torch.full(new long[] { dim }, -5.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 5.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 3.0f, 1.0f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { 3.0f, 1.0f }, ScalarType.Float32);

                testFunctions.Add(new SimpleShiftedQuad2DFunctionData("SimpleShiftedQuad2D", dim, minB, maxB, 7.0f, expectedPos, shiftTensor: initialShift));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
            }

            // Function 10: Shifted Sphere (Dim=6)
            {
                int dim = 6;
                Tensor minB = torch.full(new long[] { dim }, -10.0f, ScalarType.Float32);
                Tensor maxB = torch.full(new long[] { dim }, 10.0f, ScalarType.Float32);
                Tensor expectedPos = torch.tensor(new float[] { 1.5f, -0.5f, 2.5f, -1.5f, 0.5f, 3.5f }, ScalarType.Float32);
                Tensor initialShift = torch.tensor(new float[] { 1.5f, -0.5f, 2.5f, -1.5f, 0.5f, 3.5f }, ScalarType.Float32);

                testFunctions.Add(new ShiftedSphere6DFunctionData("ShiftedSphere6D", dim, minB, maxB, 25.0f, expectedPos, shiftTensor: initialShift));
                minB.Dispose();
                maxB.Dispose();
                expectedPos.Dispose();
                initialShift.Dispose();
            }

            return testFunctions;
        }
    }
}