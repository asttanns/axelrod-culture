# Axelrod (1997) — Cultural Dissemination in Mesa

Implementation of Robert Axelrod's "The Dissemination of Culture: A Model
with Local Convergence and Global Polarization" (*Journal of Conflict
Resolution* 41(2), 1997) using Mesa 3.x.

## Files

| File       | Role                                                         |
|------------|--------------------------------------------------------------|
| `agent.py` | `CultureAgent` — feature vector, similarity, copy-trait rule |
| `model.py` | `CultureModel` — grid, scheduling, region/zone counters      |
| `app.py`   | `SolaraViz` dashboard (grid + line plots + sliders)          |

## Running the Model

To run the model, execute the following command in your terminal:

```bash
solara run app.py
```

This will start the Mesa server and open the SolaraViz dashboard in your web browser.
