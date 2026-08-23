import json
from config import SimulationConfig
from simulation.engine import SimulationEngine


def main():
    engine=SimulationEngine(SimulationConfig())
    engine.run(); metrics=engine.save_results("results")
    print("Simulation complete\n"+json.dumps(metrics,indent=2))


if __name__ == "__main__": main()

