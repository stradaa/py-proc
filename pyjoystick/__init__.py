"""pyjoystick - canonical joystick->cursor reconstruction shared across the pipeline.

Single source of truth for turning the raw joystick analog signal into the
normalized [0,1] cursor trajectory the Rust task renders. Both the metrics
layer (pyCheck) and the video replay (pyReplay) import from here so their
reconstructions cannot drift.

Supported ``control_mode`` values (must mirror rust/joystick_task/src/state.rs):
- "direct":     cursor = 0.5 + influence(joystick) * direct_range   (stateless)
- "cumulative": integrate influence(joystick) * cumulative_speed * dt
- "blend":      leaky integrator between the two, driven by
                ``position_velocity_blend`` in [0,1] (0 == direct, 1 == cumulative)
"""

from pyjoystick.cursor import apply_direction_influence, reconstruct_cursor

__all__ = ["apply_direction_influence", "reconstruct_cursor"]
