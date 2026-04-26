# SINGULARITY 🌌

An advanced, GPU-accelerated fluid dynamics cursor animation with gravitational particle systems, bloom post-processing, and chromatic aberration.

## Features
- **GPU Fluid Simulation**: Real-time Navier-Stokes fluid simulation computed entirely on the GPU via WebGL2 fragment shaders.
- **Particle Galaxy**: 3000+ particles with gravitational attraction/repulsion logic.
- **Post-Processing**: ACES tone mapping, bloom, and subtle chromatic aberration for a premium "Singularity" aesthetic.
- **Interactive**: 
  - **Move**: Disturb the fluid and attract particles.
  - **Click**: Trigger shockwave explosions and massive fluid splats.
  - **Hold**: Increase gravitational pull and fluid intensity.

## Technologies Used
- **WebGL 2.0**: For heavy lifting of fluid simulations and shaders.
- **Canvas 2D API**: For the particle overlay system.
- **Vanilla JavaScript**: No external dependencies.
- **CSS3**: For the premium UI and glassmorphism effects.

## How to Run
1. Clone the repository.
2. Open `index.html` in any modern web browser that supports WebGL 2.0.

## Implementation Details
The simulation uses a multi-pass shader approach:
1. **Advection**: Moving velocity and dye through the field.
2. **Divergence**: Calculating the "flow" in/out of cells.
3. **Pressure Solve**: 20-iteration Jacobi solver to satisfy the incompressibility constraint.
4. **Gradient Subtraction**: Correcting the velocity field.
5. **Post-Process**: Gaussian blur for bloom and composite for final rendering.

---
Created with ✨ by Antigravity
