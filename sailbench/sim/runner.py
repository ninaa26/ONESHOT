"""Main simulation runner for SailBench."""

import sailbench.sim.sailboat_hub as boat
import time


def run():
    end_time = time.monotonic() + boat.simulation_cfg.get("t_final", 30.0)
    while time.monotonic() < end_time:
        

if __name__ == "__main__":
    main()