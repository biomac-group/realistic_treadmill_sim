"""Express an overground gait solution in the treadmill reference frame."""

from __future__ import annotations

from pathlib import Path

import cloudpickle
import jax.numpy as jnp


def convert_overground_result(source, destination, belt_speed, *, overwrite=False):
    """Convert pelvis progression to backward belt motion for an initial guess."""
    source = Path(source)
    destination = Path(destination)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Converted treadmill seed already exists: {destination}. "
            "Pass --overwrite to recreate it."
        )

    with source.open("rb") as handle:
        (states, globals_), info, settings = cloudpickle.load(handle)

    q = jnp.asarray(states.q)
    qd = jnp.asarray(states.qd)
    duration = float(jnp.asarray(globals_.dur))
    if duration <= 0:
        raise ValueError(f"Invalid overground duration {duration} in {source}")

    # Coordinate zero is horizontal pelvis translation in the gait2d models.
    progression = float(q[-1, 0] - q[0, 0])
    frame_speed = progression / duration
    time = jnp.linspace(0.0, duration, q.shape[0], dtype=q.dtype)
    q = q.at[:, 0].add(-frame_speed * time)
    qd = qd.at[:, 0].add(-frame_speed)

    target_belt_speed = -abs(float(belt_speed))
    converted_states = states.replace(
        q=q,
        qd=qd,
        gc_model=jnp.full(
            (q.shape[0], 2), target_belt_speed, dtype=q.dtype
        ),
    )
    converted_globals = globals_.replace(
        speed=jnp.asarray(0.0, dtype=jnp.asarray(globals_.speed).dtype)
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        cloudpickle.dump(
            ((converted_states, converted_globals), info, settings), handle
        )
    print(
        f"Converted overground gait to treadmill frame: {destination} "
        f"(pelvis frame speed {frame_speed:.6g} m/s, "
        f"belt speed {target_belt_speed:.6g} m/s)"
    )
    return destination
