# Cascaded TD3 Quadcopter Trajectory Tracking

A cascaded deep reinforcement learning controller for quadcopter trajectory tracking. Two independent TD3 agents are trained in sequence — an inner **attitude controller** and an outer **position controller** — and composed at inference time so the position agent commands attitude setpoints that the attitude agent executes.

---
![Trajectory Tracking Video](examples/trajectory.gif)
![Trajectory Tracking Multi Start](examples/multi_start_comparison_8starts.png)

## Architecture

```
Trajectory planner
      │  pos_des, vel_des, acc_des
      ▼
┌─────────────────────┐     attitude setpoint [φ_des, θ_des, ψ_des]
│  Position Agent     │ ──────────────────────────────────────────►
│  TD3 · 18-dim obs   │                                           │
│  3-dim action       │                                           ▼
└─────────────────────┘                               ┌─────────────────────┐
                                                       │  Attitude Agent     │
                                                       │  TD3 · 9-dim obs    │
                                                       │  3-dim action       │
                                                       └─────────────────────┘
                                                                  │  [τx, τy, τz]
                                                                  ▼
                                                         system_solve() → Ft
                                                                  │
                                                                  ▼
                                                          RK4 dynamics (100 Hz)
```

The position agent runs at **100 Hz** in the outer loop. Its 3-dim body-acceleration output is passed to `system_solve()` which computes the required total thrust `Ft` and desired tilt angles. The attitude agent then tracks those angles and produces the three body torques.

---

## Observation and Action Spaces

**Position agent — observation (18-dim)**
```
[pos_error/MAX_POS (3), vel/MAX_VEL (3), att_rad (3),
 rates/10 (3), vel_des/MAX_VEL (3), acc_des/MAX_ACCEL (3)]
```

**Position agent — action (3-dim, in `[-1, 1]`)**
```
[ax, ay, az]  →  body accelerations scaled to ±MAX_BODY_ACCEL (15 m/s²)
```
Attitude setpoint clipped to **±35°** to keep the attitude agent in its trained distribution.

**Attitude agent — observation (9-dim)**
```
[φ, θ, ψ, p, q, r, φ_des, θ_des, ψ_des]
```

**Attitude agent — action (3-dim, in `[-MAX_TORQUE, +MAX_TORQUE]`)**
```
[τx, τy, τz]  ∈  [-1.962, +1.962] Nm
```

---

## Training

Training proceeds in three sequential stages totalling ~17 500 episodes:

### Stage 1 — Attitude Controller (3 000 episodes)
Trained in isolation on randomised attitude targets `±40°`. The widened range (vs the default ±30°) ensures the agent handles the large tilt commands produced by the position agent during far-start recoveries.

### Stage 2 — Position Controller (11 500 episodes)

Four-phase curriculum with the pre-trained attitude agent frozen as the inner loop:

| Phase | Trajectory | Max Steps | Episodes | Start radius |
|---|---|---|---|---|
| 1 | Hover | 500 | 2 000 | 0.5 m |
| 2 | Helix 25% | 1 250 | 2 000 | 1.0 m |
| 3 | Helix 50% | 2 500 | 2 000 | 2.0 m |
| 4 | Full helix | 5 000 | 4 000 | 3.0 m |

The start radius ramps linearly within each phase (curriculum over start distance). Wind disturbance is introduced from phase 3.

### Stage 3 — Joint Fine-tuning (optional, 500 episodes)
Both agents updated simultaneously at reduced learning rates (attitude ×0.1, position ×0.3).

**Run full training:**
```bash
cd agents_modified
python main.py
```
![Attitude Training](examples/attitude_training.png)
![Position Training](examples/position_training.png)

Weights are saved to `results/<run>/` as `.pth` checkpoints 

---

## Evaluation

```bash
# Full disturbance table (10 conditions)
python evaluate_performance_table.py \
    --cascaded_att results/<run>/attitude_controller_final.pth \
    --cascaded_pos results/<run>/position_controller_final.pth \
    --episodes 50

# MATLAB-faithful noise sweep (17 conditions: control noise, sensor noise, wind)
python evaluate_matlab_noise.py \
    --cascaded_att results/<run>/attitude_controller_final.pth \
    --cascaded_pos results/<run>/position_controller_final.pth \
    --episodes 30

# With motor-level saturation (saturate_controls)
python evaluate_matlab_noise.py \
    --cascaded_att results/<run>/attitude_best.pth \
    --cascaded_pos results/<run>/position_best.pth \
    --saturate --episodes 30

# Start-distance robustness sweep
python evaluate_performance_table.py \
    --cascaded_att results/<run>/attitude_best.pth \
    --cascaded_pos results/<run>/position_best.pth \
    --distance_test --episodes 20
```

---

## Key Hyperparameters

| Parameter | Attitude Agent | Position Agent |
|---|---|---|
| Observation dim | 9 | 18 |
| Action dim | 3 | 3 |
| Hidden dims | 128 | 256 |
| Buffer size | 200 000 | 300 000 |
| Batch size | 256 | 256 |
| Actor LR 
| Critic LR 
| Policy noise σ 
| Noise clip 
| Policy delay 
| γ 
| τ (Polyak) 

---

## Physical Constants

| Parameter | Value |
|---|---|
| Mass | 1.0 kg |
| Arm half-length | 0.2 m |
| Ixx / Iyy / Izz | 0.3 / 0.4 / 0.5 kg·m² |
| Max thrust 
| Max torque (roll/pitch) 
| Max torque (yaw) 
| Simulation step | 0.01 s (100 Hz) |
| Mission time | 50 s |

---

## Project Structure

```
agents_modified/
├── agents/
│   └── td3_agent.py              # TD3 with twin critics, delayed policy update
├── environments/
│   ├── attitude_env.py           # Inner-loop env: 9-dim obs, torque action
│   ├── position_env.py           # Outer-loop env: 18-dim obs, body-acc action
│   │                             #   includes system_solve() and inner-loop call
│   └── dynamics.py               # RK4 quadcopter dynamics (shared)
├── configs/
│   └── config.py                 # All hyperparameters, trajectory definition
├── training/
│   ├── attitude_trainer.py       # Stage 1 training loop
│   ├── position_trainer.py       # Stage 2 curriculum loop
│   └── fine_tuner.py             # Stage 3 joint fine-tuning
├── utils/
│   ├── evaluation.py             # Episode metrics (RMSE, convergence, crash rate)
│   ├── visualization.py          # Training curves and 3-D trajectory plots
│   └── control_allocation.py     # Motor thrust allocation utilities
├── main.py                       # Entry point (runs all three stages)
├── trajectory_visualization.py
└── physics_feasibility_analysis.py

evaluate_performance_table.py     # 10-condition disturbance table
all_test_results.py          # 17-condition  noise sweep
```

---

## Requirements

```
python >= 3.12
gym==0.26.2
matplotlib==3.10.8
numpy==2.4.3
scipy==1.17.1
torch==2.6.0+cu126
```
