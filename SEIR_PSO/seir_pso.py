import numpy as np
import pandas as pd
from scipy.integrate import odeint
from pathlib import Path
import pyswarms as ps
import matplotlib.pyplot as plt


def beta_t(t, beta0, a1, phi1, a2, phi2, T):
    """Calcula beta(t) con oscilaciones."""
    return beta0 * np.exp(
        a1 * np.cos(2 * np.pi * t / T + phi1) +
        a2 * np.cos(4 * np.pi * t / T + phi2)
    )


def seir_ode(state, t, N, beta_params, sigma, gamma):
    S, E, I, R = state
    beta0, a1, phi1, a2, phi2, T = beta_params
    b = beta_t(t, beta0, a1, phi1, a2, phi2, T)
    dSdt = -b * S * I / N
    dEdt = b * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def simulate_seir(days, init, N, params):
    beta0, a1, phi1, a2, phi2, sigma, gamma, T = params
    beta_params = (beta0, a1, phi1, a2, phi2, T)
    sol = odeint(seir_ode, init, days, args=(N, beta_params, sigma, gamma))
    return sol


def objective_function(params, days, infected_obs, init, N):
    # params shape (n_particles, dim)
    n_particles = params.shape[0]
    errors = np.zeros(n_particles)
    for i, par in enumerate(params):
        sim = simulate_seir(days, init, N, par)
        I_sim = sim[:, 2]
        I_norm = I_sim / N
        err = np.mean((I_norm - infected_obs)**2)
        errors[i] = err
    return errors


def run_pso(days, infected, init, N, bounds, iters=100, particles=50):
    dim = bounds.shape[0]
    optimizer = ps.single.GlobalBestPSO(
        n_particles=particles,
        dimensions=dim,
        options={'c1': 0.5, 'c2': 0.3, 'w': 0.9},
        bounds=(bounds[:,0], bounds[:,1])
    )
    cost, best_pos = optimizer.optimize(
        objective_function, iters,
        days=days,
        infected_obs=infected,
        init=init,
        N=N
    )
    return best_pos, cost


def fit_intervals(days, infected, init, N, bounds, n_intervals=3):
    interval_len = len(days) // n_intervals
    params_list = []
    for i in range(n_intervals):
        start = i * interval_len
        end = (i + 1) * interval_len if i < n_intervals - 1 else len(days)
        d_seg = days[start:end]
        inf_seg = infected[start:end]
        best, _ = run_pso(d_seg, inf_seg, init, N, bounds, iters=50, particles=30)
        params_list.append(best)
    avg_params = np.mean(params_list, axis=0)
    # refine on full data
    best_global, _ = run_pso(days, infected, init, N, bounds, iters=100, particles=50)
    return best_global


def load_data(path):
    path = Path(path)
    if path.suffix == '.csv':
        df = pd.read_csv(path)
    elif path.suffix in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    else:
        raise ValueError('Formato de archivo no soportado')
    col_time = df.columns[0]
    col_inf = df.columns[1]
    days = df[col_time].to_numpy()
    infected = df[col_inf].to_numpy()
    infected_norm = infected / infected.max()
    return days, infected_norm, infected


def main(data_file):
    days, infected, infected_raw = load_data(data_file)
    N = 10000  # población total asumida
    init = (N-1, 1, 0, 0)  # condiciones iniciales simples

    # parámetros: beta0, a1, phi1, a2, phi2, sigma, gamma, T
    bounds = np.array([
        [0.1, 1.0],   # beta0
        [0.0, 1.0],   # a1
        [0.0, 2*np.pi],  # phi1
        [0.0, 1.0],   # a2
        [0.0, 2*np.pi],  # phi2
        [0.1, 0.5],   # sigma
        [0.05, 0.5],  # gamma
        [10.0, 60.0]  # T
    ])

    best_params = fit_intervals(days, infected, init, N, bounds)
    print('Mejores parámetros:')
    print(best_params)

    # Simular con los parámetros obtenidos para comparar con los datos
    sol = simulate_seir(days, init, N, best_params)
    infected_sim = sol[:, 2] / N

    plt.scatter(days, infected, label='infectados observados', color='black')
    plt.plot(days, infected_sim, 'r-', label='ajuste PSO')
    plt.xlabel('Tiempo')
    plt.ylabel('Fracción infectados')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ajuste de modelo SEIR con PSO')
    parser.add_argument('data', help='Archivo CSV o XLSX con columnas de tiempo e infectados')
    args = parser.parse_args()
    main(args.data)
