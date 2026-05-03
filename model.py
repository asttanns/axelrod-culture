"""
CultureModel: Axelrod (1997) "The Dissemination of Culture"

- Agents are placed on a fixed 2D grid. 
- Each agent holds a culture vector of F features, each taking one of q possible integer traits. 
- At each event, one agent is selected at random, picks a random neighbor, and interacts with probability equal to their cultural similarity. 
- If they interact, the active agent copies one of the neighbor's differing traits.
- The model runs until stable: every neighbor pair has similarity in {0, 1}.
"""

from __future__ import annotations
import numpy as np
import mesa
from mesa import Model
from mesa.discrete_space import OrthogonalVonNeumannGrid, OrthogonalMooreGrid

from agent import CultureAgent

# DataCollector reporter functions

def _connected_components(model, edge_predicate) -> int:
    """
    Count connected components on the grid.
    Two adjacent cells are connected iff edge_predicate(agent_a, agent_b) is True. 
    """
    visited = set()
    components = 0
    for cell in model.grid.all_cells.cells:
        if cell.coordinate in visited:
            continue
        components += 1
        stack = [cell]
        while stack:
            c = stack.pop()
            if c.coordinate in visited:
                continue
            visited.add(c.coordinate)
            a = c.agents[0]
            for nb in c.neighborhood:
                if nb.coordinate in visited:
                    continue
                b = nb.agents[0]
                if edge_predicate(a, b):
                    stack.append(nb)
    return components


def count_regions(model) -> int:
    """
    Number of stable cultural REGIONS.
    A region = maximal set of contiguous cells with IDENTICAL culture.
    Axelrod's primary outcome measure (Table 2, Figure 1).
    """
    return _connected_components(
        model, lambda a, b: np.array_equal(a.culture, b.culture)
    )


def count_zones(model) -> int:
    """
    Number of cultural ZONES.
    A zone = maximal set of contiguous cells where each adjacent pair shares at least one feature (similarity > 0). 
    Zones settle faster than regions — when regions == zones, the model is stable.
    """
    return _connected_components(
        model, lambda a, b: np.any(a.culture == b.culture)
    )


def mean_neighbor_similarity(model) -> float:
    """
    Average cultural similarity over all unordered neighbor pairs.
    Should increase toward 1.0 as the model converges.
    """
    sims = []
    seen = set()
    for cell in model.grid.all_cells.cells:
        a = cell.agents[0]
        for nb in cell.neighborhood:
            key = tuple(sorted((cell.coordinate, nb.coordinate)))
            if key in seen:
                continue
            seen.add(key)
            b = nb.agents[0]
            sims.append(a.similarity(b))
    return float(np.mean(sims)) if sims else 0.0

# Model

class CultureModel(Model):
    """
    Axelrod (1997) cultural dissemination model.

    Parameters
    
    width, height : grid dimensions
    n_features    : number of cultural features (F)
    n_traits      : number of possible traits per feature (q)
    neighborhood  : "von_neumann" (4 neighbors) or "moore" (8 neighbors)
    seed          : random seed for reproducibility
    """

    def __init__(
        self,
        *,
        width: int = 10,
        height: int = 10,
        n_features: int = 5,
        n_traits: int = 5,
        neighborhood: str = "von_neumann",
        seed: int | None = None,
    ):
        # Seed both Mesa's internal RNG and numpy Generator
        np_rng = np.random.default_rng(seed)
        try:
            super().__init__(rng=np_rng)
        except TypeError:
            super().__init__(seed=seed)
        self.rng = np_rng

        self.width = width
        self.height = height
        self.n_features = n_features
        self.n_traits = n_traits

        # Grid: torus=False for Axelrod's hard-bordered map.
        # Von Neumann = 4 neighbors.
        # Moore = 8 neighbors (sensitivity test).
        if neighborhood == "moore":
            self.grid = OrthogonalMooreGrid(
                (width, height), torus=False, random=self.random
            )
        else:
            self.grid = OrthogonalVonNeumannGrid(
                (width, height), torus=False, random=self.random
            )

        # One agent per cell, randomly initialized culture vector
        for cell in self.grid.all_cells.cells:
            agent = CultureAgent(self, n_features=n_features, n_traits=n_traits)
            agent.cell = cell

        # events_per_step = W*H so one Mesa step ≈ one activation per site,
        # matching Axelrod's "Events/Site" time unit.
        self.events_per_step = width * height

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Regions": count_regions,
                "Zones": count_zones,
                "MeanSimilarity": mean_neighbor_similarity,
            }
        )
        self.datacollector.collect(self)
        self.running = True

    def is_stable(self) -> bool:
        """
        The model is stable when every neighbor pair has similarity in {0, 1} — either identical (can interact but nothing changes) or completely different (cannot interact at all).
        Early return on first unstable pair found.
        """
        for cell in self.grid.all_cells.cells:
            a = cell.agents[0]
            for nb in cell.neighborhood:
                b = nb.agents[0]
                s = a.similarity(b)
                if 0.0 < s < 1.0:
                    return False
        return True

    def step(self) -> None:
        """
        Run events_per_step random asynchronous activations.

        Each event: pick one agent uniformly at random (sampling WITH replacement). 
        """
        agents_list = list(self.agents)
        for _ in range(self.events_per_step):
            agent = self.rng.choice(agents_list)
            agent.step()

        self.datacollector.collect(self)

        if self.is_stable():
            self.running = False