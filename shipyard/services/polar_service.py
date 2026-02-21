import numpy as np
import matplotlib.pyplot as plt

from sailbench.models.foil import Foil


def generate_and_plot(airfoil_name: str, re: float):
    params = {
        "airfoil_name": airfoil_name,
        "alpha_min": -20,
        "alpha_max": 20,
        "backend": "neuralfoil",
    }

    foil = Foil(params)

    alpha_rad = np.linspace(
        np.deg2rad(params["alpha_min"]),
        np.deg2rad(params["alpha_max"]),
        150,
    )

    alpha_deg = np.degrees(alpha_rad)

    # Vectorized NeuralFoil call
    result = foil.foil.get_aero_from_neuralfoil(
        alpha=alpha_deg,
        Re=re,
        mach=0.0,
    )

    cl = result["CL"]
    cd = result["CD"]

    # Plot polar
    plt.figure()
    plt.plot(cd, cl)
    plt.xlabel("CD")
    plt.ylabel("CL")
    plt.title(f"Polar: {airfoil_name} (Re={re:.1e})")
    plt.grid()
    plt.show()
