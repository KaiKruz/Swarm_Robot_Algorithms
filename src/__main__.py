### src/__main__.py

from simulation import Simulation

if __name__ == "__main__":
    simulation = Simulation()
    stats = simulation.run()
    print("Final Statistics:", stats)