//Copyright Warren Harding 2025.
using System;
using System.Collections.Generic;
using static TorchSharp.torch;
using TorchSharp;

namespace SwarmGPU
{
    public class SwarmGPU : IDisposable
    {
        private readonly int _numberOfParticles;
        private readonly int _dimensions;
        private readonly Tensor _minBounds;
        private readonly Tensor _maxBounds;
        private readonly float _omega;
        private readonly float _phiP;
        private readonly float _phiG;
        private readonly int _maxIterations;
        private readonly List<Particle> _particles;

        private Tensor _globalBestPosition;
        private Tensor _globalBestValue;

        public SwarmGPU(int numberOfParticles, int dimensions,
                        Tensor minBounds, Tensor maxBounds,
                        float omega, float phiP, float phiG, int maxIterations)
        {
            _numberOfParticles = numberOfParticles;
            _dimensions = dimensions;

            // Clone bounds to take ownership and manage their disposal within this class
            _minBounds = minBounds.clone();
            _maxBounds = maxBounds.clone();

            _omega = omega;
            _phiP = phiP;
            _phiG = phiG;
            _maxIterations = maxIterations;

            // Initialize global bests directly in the constructor to satisfy non-nullable field warnings (CS8618).
            // These will be updated during the initial evaluation and optimization steps.
            _globalBestValue = torch.full(new long[] { 1 }, float.MaxValue).to(ScalarType.Float32);
            _globalBestPosition = torch.zeros(_dimensions, ScalarType.Float32);

            _particles = new List<Particle>();
            InitializeParticles(); // This method now only initializes the particles themselves.
        }

        private void InitializeParticles()
        {
            // _globalBestValue and _globalBestPosition are now initialized in the constructor.
            // This method now focuses solely on initializing individual particles.

            for (int i = 0; i < _numberOfParticles; i++)
            {
                // Create a dispose scope for tensors generated within this loop iteration.
                // Tensors passed to Particle constructor are cloned by Particle,
                // so the originals must be disposed by this scope after cloning.
                using (var tempParticleInitScope = NewDisposeScope())
                {
                    // Generate random position within bounds: Position = min + (rand * (max - min))
                    Tensor rand_val = torch.rand(_dimensions, ScalarType.Float32);
                    Tensor range = _maxBounds - _minBounds;
                    Tensor scaled_rand = rand_val * range;
                    // Initial position tensor. Its ownership remains with tempParticleInitScope
                    // until Particle clones it.
                    Tensor initialPosition = (_minBounds + scaled_rand);

                    // Velocities are typically initialized to zero for PSO.
                    // Initial velocity tensor. Its ownership remains with tempParticleInitScope.
                    Tensor initialVelocity = torch.zeros(_dimensions, ScalarType.Float32);

                    Particle newParticle = new Particle(initialPosition, initialVelocity);
                    _particles.Add(newParticle);

                    // Personal best initialized based on the particle's own initial position.
                    // The particle's Position property is already a clone managed by the Particle.
                    // The initial PersonalBestValue tensor creation must also be within the dispose scope.
                    Tensor initialPersonalBestValue = torch.full(new long[] { 1 }, float.MaxValue).to(ScalarType.Float32);
                    newParticle.SetPersonalBest(newParticle.Position, initialPersonalBestValue);
                } // tempParticleInitScope disposes initialPosition, initialVelocity, and initialPersonalBestValue here.
            }
        }

        /// <summary>
        /// Optimizes the objective function using the Particle Swarm Optimization algorithm.
        /// </summary>
        /// <param name="objectiveFunction">
        /// The function to minimize. It takes a <see cref="TorchSharp.torch.Tensor"/> as input (representing a particle's position)
        /// and returns a <see cref="TorchSharp.torch.Tensor"/> of shape [1] representing the objective value at that position.
        /// <para>
        /// IMPORTANT: To prevent "Tensor invalid -- empty handle" errors, the <paramref name="objectiveFunction"/>
        /// *must* follow TorchSharp's strict memory management rules for delegates:
        /// <list type="bullet">
        /// <item>Wrap all operations that create intermediate tensors (e.g., <c>clone()</c>, <c>pow</c>, <c>abs</c>, <c>sub</c>)
        /// within a <c>using (var scope = torch.NewDisposeScope()) { ... }</c> block.</item>
        /// <item>Before returning the final result <see cref="TorchSharp.torch.Tensor"/>,
        /// it *must* be moved to the outer dispose scope of the calling code by
        /// calling <c>tensor.MoveToOuterDisposeScope()</c> on it.
        /// Example: <c>return finalResult.MoveToOuterDisposeScope();</c></item>
        /// </list>
        /// </para>
        /// </param>
        /// <returns>A tuple containing the best position found as a <see cref="TorchSharp.torch.Tensor"/>
        /// and the corresponding best value as a <see cref="TorchSharp.torch.Tensor"/>.</returns>
        public (Tensor bestPosition, Tensor bestValue) Optimize(Func<Tensor, Tensor> objectiveFunction)
        {
            // Initial evaluation phase to set the first global best.
            // This loop ensures _globalBestPosition and _globalBestValue are set based on actual particle states.
            foreach (Particle particle in _particles)
            {
                // The tensor returned by objectiveFunction is now detached and owned by the caller,
                // so it can be directly consumed by the using statement for proper disposal.
                using (Tensor currentValue = objectiveFunction(particle.Position))
                {
                    if (currentValue.item<float>() < particle.PersonalBestValue.item<float>())
                    {
                        // particle.SetPersonalBest clones the input tensors, so 'currentValue' will be disposed by this scope.
                        particle.SetPersonalBest(particle.Position, currentValue);
                    }

                    if (particle.PersonalBestValue.item<float>() < _globalBestValue.item<float>())
                    {
                        _globalBestValue?.Dispose();
                        _globalBestPosition?.Dispose();

                        // Move the cloned tensors to the outer scope to prevent premature disposal.
                        _globalBestValue = particle.PersonalBestValue.clone().MoveToOuterDisposeScope();
                        _globalBestPosition = particle.PersonalBestPosition.clone().MoveToOuterDisposeScope();
                    }
                }
            }

            // Main optimization loop.
            for (int iter = 0; iter < _maxIterations; iter++)
            {
                // Create a dispose scope for all temporary tensors generated within this iteration.
                // All tensors created in this 'using' block that are not explicitly disposed or moved
                // to another scope will be disposed when the scope exits.
                using (var iterationScope = NewDisposeScope())
                {
                    foreach (Particle particle in _particles)
                    {
                        // Generate random numbers for cognitive (rP) and social (rG) components.
                        Tensor rP = torch.rand(_dimensions, ScalarType.Float32);
                        Tensor rG = torch.rand(_dimensions, ScalarType.Float32);

                        // Update velocity formula: v = w*v + phi_p*r_p*(p_best - x) + phi_g*r_g*(g_best - x)
                        Tensor cognitiveFactor = rP * _phiP;
                        Tensor socialFactor = rG * _phiG;

                        Tensor term1 = _omega * particle.Velocity;
                        Tensor term2 = cognitiveFactor * (particle.PersonalBestPosition - particle.Position);
                        Tensor term3 = socialFactor * (_globalBestPosition - particle.Position);

                        // All intermediate tensors created here are automatically registered to iterationScope.
                        // newVelocityToPass holds the final result of this calculation.
                        Tensor newVelocityToPass = (term1 + term2 + term3).to(ScalarType.Float32);
                        
                        // Update position formula: x = x + v
                        // newPositionIntermediate holds the sum before clamping.
                        Tensor newPositionIntermediate = (particle.Position + newVelocityToPass).to(ScalarType.Float32);

                        // Clamp position within defined bounds.
                        // newPositionToPass holds the final clamped position.
                        Tensor newPositionToPass = torch.clamp(newPositionIntermediate, _minBounds, _maxBounds);

                        // Update the particle's internal state. Particle.Update clones inputs.
                        // The original newPositionToPass and newVelocityToPass tensors remain in iterationScope.
                        particle.Update(newPositionToPass, newVelocityToPass);

                        // The tensor returned by objectiveFunction is now detached and owned by the caller,
                        // so it can be directly consumed by the using statement for proper disposal.
                        using (Tensor currentValue = objectiveFunction(particle.Position))
                        {
                            // Update particle's personal best if current value is better.
                            if (currentValue.item<float>() < particle.PersonalBestValue.item<float>())
                            {
                                // particle.SetPersonalBest clones the input tensors, so 'currentValue' will be disposed by this scope.
                                particle.SetPersonalBest(particle.Position, currentValue);
                            }

                            // Update global best if particle's personal best is better than current global best.
                            if (particle.PersonalBestValue.item<float>() < _globalBestValue.item<float>())
                            {
                                _globalBestValue?.Dispose();
                                _globalBestPosition?.Dispose();

                                // Move the cloned tensors to the outer scope to prevent premature disposal.
                                _globalBestValue = particle.PersonalBestValue.clone().MoveToOuterDisposeScope();
                                _globalBestPosition = particle.PersonalBestPosition.clone().MoveToOuterDisposeScope();
                            }
                        }
                    }
                } // iterationScope disposes all its temporary tensors (rP, rG, cognitiveFactor, socialFactor, term1, term2, term3, newVelocityToPass, newPositionIntermediate, newPositionToPass) here.
            }

            // Return cloned results, transferring ownership to the caller.
            // The caller is responsible for disposing these returned Tensors.
            return (_globalBestPosition.clone(), _globalBestValue.clone());
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
                    // Dispose tensors owned by the SwarmGPU class.
                    _globalBestPosition?.Dispose();
                    _globalBestValue?.Dispose();
                    _minBounds?.Dispose();
                    _maxBounds?.Dispose();

                    // Dispose all particles and clear the list.
                    foreach (Particle particle in _particles)
                    {
                        particle.Dispose();
                    }
                    _particles.Clear();
                }
                _disposed = true;
            }
        }

        // Finalizer in case Dispose is not called explicitly.
        ~SwarmGPU()
        {
            Dispose(false);
        }
    }
}