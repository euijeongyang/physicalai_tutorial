# MuJoCo Cart-Pole based on `cartpole_termp.pdf`

The project reproduces the **dominant-pole / Ackermann pole-placement state-feedback controller** from the uploaded 33-page undergraduate design report, then visualizes the physical plant in MuJoCo.

## Installation

```bash
python -m pip install mujoco numpy scipy matplotlib
```

## Run

From this directory:

```bash
python main.py --mode balance
python main.py --mode off
python main.py --mode step --duration 20
```

- `balance`: Start at 0.1 rad (5.7 deg) tilt, control upright equilibrium for 5 s.
- `off`: Same tilt without control for 5 s.
- `step`: 1 m cart position step, 20 s by default (report settling-time target).
- `--duration 5`: Override simulation duration in seconds.
- `--no-viewer`: Run without the interactive Viewer and only save the result graph.
- Terminal status prints every 0.5 s for balance/off, and every 2 s for step. Viewer remains open after simulation time expires.
- A PNG file with cart position, pendulum angle, and force is saved as `result_<mode>.png`.

## Mapping from report

Report variables: `[x, x_dot, phi, phi_dot]`. MuJoCo's pole joint uses axis `0 -1 0`, so positive `phi` leans in the -X direction. Control convention is `u = -K x + Nbar r`. The analytical model has `M=0.2 kg`, `m=0.5 kg`, `l=0.3 m`, `I=0.3 kg m^2`, `b=0.1 N s/m`, and `g=9.8 m/s^2`. The report's inertia is unusual for a visually slender pole, so `inertial` is specified explicitly; letting MuJoCo infer inertia from capsule geometry would change the plant and the correct gain.

The controller uses SciPy pole placement to reproduce the report's `K ~ [-0.0936, -0.5765, 18.9724, 8.8193]`, from poles `-0.23 +/- 0.2346j`, `-2.3`, `-2.53`. It includes viscous damping on the cart's slide joint but no pole damping, ground interaction, or extra actuator dynamics; these would change the source model.

## Notes / limitations

- The source report's results are linearized around the upright equilibrium. These values are intended for small deviations, not swing-up control.
- MuJoCo solves the nonlinear multibody dynamics. Some discrepancy with the report's MATLAB linear response is expected away from upright.
- The original report does not include a force saturation value; +/-30 N is included as a generous numerical/actuator safety bound. If the plotted force clips, comparison with the unconstrained report model is no longer exact.
- The visual rail and body geoms do not participate in collisions, ensuring the modeled plant behaves like the report's unconstrained cart/pendulum equations.
- This package has been statically checked and the linear state-space gain has been numerically verified; running MuJoCo end-to-end is not verified in the authoring environment.
