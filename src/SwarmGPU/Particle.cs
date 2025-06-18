//Copyright Warren Harding 2025.
using System;
using TorchSharp;
using static TorchSharp.torch;

namespace SwarmGPU
{
    public class Particle : IDisposable
    {
        public Tensor Position { get; private set; }
        public Tensor Velocity { get; private set; }
        public Tensor PersonalBestPosition { get; private set; }
        public Tensor PersonalBestValue { get; private set; }

        public Particle(Tensor initialPosition, Tensor initialVelocity)
        {
            // Clone initial tensors as Particle takes ownership of these.
            Position = initialPosition.clone().MoveToOuterDisposeScope();
            Velocity = initialVelocity.clone().MoveToOuterDisposeScope();
            PersonalBestPosition = initialPosition.clone().MoveToOuterDisposeScope(); 
            PersonalBestValue = torch.full(new long[] { 1 }, float.MaxValue).to(ScalarType.Float32).MoveToOuterDisposeScope();
        }

        public void Update(Tensor newPosition, Tensor newVelocity)
        {
            // Dispose old tensors before assigning new ones to prevent memory leaks.
            Position?.Dispose();
            Velocity?.Dispose();
            
            // Clone new tensors as Particle takes ownership.
            Position = newPosition.clone().MoveToOuterDisposeScope(); 
            Velocity = newVelocity.clone().MoveToOuterDisposeScope(); 
        }

        public void SetPersonalBest(Tensor bestPosition, Tensor bestValue)
        {
            // Dispose old personal best tensors.
            PersonalBestPosition?.Dispose();
            PersonalBestValue?.Dispose();
            
            // Clone new personal best tensors as Particle takes ownership.
            PersonalBestPosition = bestPosition.clone().MoveToOuterDisposeScope(); 
            PersonalBestValue = bestValue.clone().MoveToOuterDisposeScope(); 
        }

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
                    // Dispose all managed (Tensor) resources.
                    Position?.Dispose();
                    Velocity?.Dispose();
                    PersonalBestPosition?.Dispose();
                    PersonalBestValue?.Dispose();
                }
                _disposed = true;
            }
        }

        // Finalizer in case Dispose is not called explicitly.
        ~Particle()
        {
            Dispose(false);
        }
    }
}