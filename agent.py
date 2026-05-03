"""
CultureAgent: Axelrod (1997) "The Dissemination of Culture".

- Each agent occupies one cell on a fixed grid (no movement)
- Its state is a vector of `F` cultural features, each taking an integer trait in [0, q-1].
- Interaction based on similarity = the active agent talks to a random neighbor with probability equal to the fraction of features they already share
- If they interact and are not yet identical, then they copy one of the neighbor's differing traits
"""

from __future__ import annotations
import numpy as np
from mesa.discrete_space import CellAgent


class CultureAgent(CellAgent):

    def __init__(self, model, n_features: int, n_traits: int):
        # CellAgent handles unique_id and registration with the model.
        super().__init__(model)
        self.n_features = n_features
        self.n_traits = n_traits
        # Initialization (CultureAgent) : F features, each drawn uniformly from {0,q-1}.
        # Use np.array for vector comparison, use int for hashing.
        self.culture = self.model.rng.integers(
            low=0, high=n_traits, size=n_features, dtype=np.int64
        )

    # Feature similarity 

    def similarity(self, other: "CultureAgent") -> float:
        # Features on which self and other have the same trait
        return float(np.mean(self.culture == other.culture))

    def differing_features(self, other: "CultureAgent") -> np.ndarray:
        # Features where self and other disagree
        return np.flatnonzero(self.culture != other.culture)

    # Axelrod two step interactions

    def step(self) -> None:
        """
        One activation event for this agent (the 'active site' in Axelrod).

        Step 1: pick one neighbor at random.
        Step 2: based on probability = similarity, interact. 
        Interaction means:
                the ACTIVE site copies one of the neighbor's traits on a
                feature where they currently disagree.

        Note from paper: the active site changes, not the neighbor.
        So that edge sites still have an equal chance of being a *target* of influence per event. *despite having fewer neighbor
        """

        # cell.neighborhood returns the set of neighboring cells
        # (Von Neumann or Moore depending on grid type chosen in the model as sensitivity analysis on range of interaction).
        neighbor_cells = list(self.cell.neighborhood)
        # Filter to cells that actually have an agent 
        neighbors = [c.agents[0] for c in neighbor_cells if len(c.agents) > 0]
        if not neighbors:
            return

        neighbor = self.model.rng.choice(neighbors)

        sim = self.similarity(neighbor)

        # If sim == 0 -> no interaction (incompatible).
        # If sim == 1 -> would interact but no differing features to copy.
        if sim == 0.0 or sim == 1.0:
            return

        # probability equal to their cultural similarity -> these two sites interact
        if self.model.rng.random() >= sim:
            return
        
        # Selecting random feature on which they differ
        diff = self.differing_features(neighbor)
        
        feature_to_copy = int(self.model.rng.choice(diff))
        # changing the active site's trait to the neighbor's trait
        # Based on paper: the active site changes, not the neighbor.
        self.culture[feature_to_copy] = neighbor.culture[feature_to_copy]
