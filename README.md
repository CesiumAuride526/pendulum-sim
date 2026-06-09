# pendulum-sim
Interactive N-pendulum simulator with real-time physics, motion trails, and
configurable UI. Built with Python, NumPy, Matplotlib.

License: MIT

Author: [CesiumAuride526]


Features
--------------------------------------------------------------------------------
- Supports 1 to 7 pendulums
- Adjust length and color for each pendulum independently
- Adjust gravity, initial angle, simulation speed
- Continuous motion trails (gradient-filled swept surfaces + edge lines)
- Dynamic auto-zoom (can be disabled)
- Customizable background color, grid color, grid step
- Save / load configuration to/from pendulum_config.json
- Keyboard shortcuts: Space = pause/resume, R = reset


Dependencies
--------------------------------------------------------------------------------
- Python 3.7 or higher
- NumPy
- Matplotlib


Installation & Usage
--------------------------------------------------------------------------------
1. Clone the repository (or download pendulum_sim.py directly)

2. Install dependencies
   pip install numpy matplotlib

3. Run the simulator
   python pendulum_sim.py


Controls
--------------------------------------------------------------------------------
The window is split into two parts:
- Left: animation area
- Right: control panel

Control panel sections:

  PENDULUM group:
    Count        - number of pendulums (1-7)
    L1..L7       - length of each pendulum (1-15)

  PHYSICS group:
    Gravity (g)  - gravitational acceleration (0.5-30)
    Init angle   - initial angle (-180° to 180°)

  DISPLAY group:
    Grid step    - grid spacing (1-10)
    Auto zoom    - auto-zoom toggle (checkbox)
    [X] Grid     - show/hide grid
    Trail len    - trail length (20-250 frames)
    BG color     - background color
    Grid color   - grid color

  SIMULATION group:
    Speed        - simulation speed multiplier (0.1-5.0)
    Reset        - reset pendulum state
    Save         - save current configuration to pendulum_config.json
    Pause        - pause / resume

Keyboard shortcuts:
  Space          - pause / resume
  R              - reset



Physics overview
--------------------------------------------------------------------------------
The equations of motion for an N-pendulum system are derived via Lagrangian
mechanics and integrated using 4th order Runge-Kutta (RK4).
Trails are rendered as swept quadrilaterals between consecutive frames; only
vertex data is updated each frame (no per-frame object creation) for high
performance.


License
--------------------------------------------------------------------------------
This project is licensed under the MIT License. See the LICENSE file.
