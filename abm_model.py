import random

# ----------------------------
#  HUMAN AGENT
# ----------------------------
class HumanAgent:
    def __init__(self, unique_id):
        self.unique_id = unique_id
        self.state = "S"  # S, E, I, R

    def step(self, infection_prob):
        """Update agent state using SEIR transitions."""
        
        if self.state == "S":
            if random.random() < infection_prob:
                self.state = "E"
        
        elif self.state == "E":
            if random.random() < 0.10:   # Incubation → Infectious
                self.state = "I"
        
        elif self.state == "I":
            if random.random() < 0.05:   # Recovery
                self.state = "R"
        
        elif self.state == "R":
            if random.random() < 0.001:  # Loss of immunity
                self.state = "S"


# ----------------------------
#  MALARIA MODEL
# ----------------------------
class MalariaModel:
    def __init__(self, population_size):
        self.population_size = population_size
        self.agents = [HumanAgent(i) for i in range(population_size)]

        # Infect a few people at the beginning
        for agent in random.sample(self.agents, k=10):
            agent.state = "I"

    def compute_infection_probability(self, rain_norm, temp_norm):
        """
        Dynamic infection probability from rainfall & temperature.

        Formula:
            P = 0.03 × (1 + 1.5×rain + 0.5×temperature)
        """

        base_prob = 0.03
        return base_prob * (1 + 1.5 * rain_norm + 0.5 * temp_norm)

    def step(self, rain_norm, temp_norm):
        infection_prob = self.compute_infection_probability(rain_norm, temp_norm)

        for agent in self.agents:
            agent.step(infection_prob)

    def count_state(self, state):
        """Count S/E/I/R at each step."""
        return sum(1 for agent in self.agents if agent.state == state)
		