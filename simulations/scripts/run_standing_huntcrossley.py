"""Build or solve the P1 Hunt-Crossley standing equilibrium problem."""

from __future__ import annotations

import argparse

import jax

from biosym.ocp.collocation import Collocation
from src.contact_huntcrossley import install as install_huntcrossley_contact


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    default="standing_files/standing_p1_6spheres.yaml",
)
parser.add_argument(
    "--validate-only",
    action="store_true",
    help="Compile and evaluate the initial problem without running IPOPT.",
)
parser.add_argument("--visualize", action="store_true")
args = parser.parse_args()

install_huntcrossley_contact()
problem = Collocation(args.config, force_rebuild=True)
if args.validate_only:
    objective = problem.objective.objfun(
        problem.initial_guess_states,
        problem.initial_guess_globals,
    )
    constraints = problem.constraints.confun(
        problem.initial_guess_states,
        problem.initial_guess_globals,
    )
    jax.block_until_ready(objective)
    jax.block_until_ready(constraints)
    print(
        f"Standing problem validated: objective={float(objective):.6g}, "
        f"constraints={constraints.size}."
    )
else:
    problem.solve(visualize=args.visualize)
