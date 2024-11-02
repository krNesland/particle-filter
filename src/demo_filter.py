import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go


class DataColumns:
    time = "TIME"
    theta_observation = "THETA_OBSERVATION"
    theta_mean = "THETA_MEAN"
    theta_std = "THETA_STD"


TY_PARTICLES = dict[int, np.ndarray]


class StateSpace:
    theta = 0  # Angle
    omega = 1  # Angular velocity


class Constants:
    L = 2.0
    g = 9.81
    dt = 0.1



################
# DATA


def get_data() -> pd.DataFrame:
    """
    Simulate pendulum data
    """
    time_steps = 200
    theta = np.zeros(time_steps)
    omega = np.zeros(time_steps)
    theta[0] = math.pi / 4  # initial angle
    omega[0] = 0  # initial angular velocity

    for t in range(1, time_steps):
        omega[t] = omega[t - 1] - (Constants.g / Constants.L) * np.sin(theta[t - 1]) * Constants.dt
        theta[t] = theta[t - 1] + omega[t] * Constants.dt

    data = pd.DataFrame({
        DataColumns.time: np.arange(time_steps) * Constants.dt,
        DataColumns.theta_observation: theta + np.random.normal(loc=0.0, scale=0.1, size=time_steps),
    })
    return data


################
# FILTER


class ParticleFilter:    
    def __init__(self, num_particles: int):
        self.num_particles = num_particles
        self.particles: TY_PARTICLES = {
            StateSpace.theta: (np.random.random(size=num_particles) - 0.5) * math.pi,
            StateSpace.omega: (np.random.random(size=num_particles) - 0.5) * 0.01,
        }
        self.weights = np.ones(shape=num_particles) / num_particles


    def predict(self):
        theta = self.particles[StateSpace.theta].copy()
        omega = self.particles[StateSpace.omega].copy()

        self.particles[StateSpace.theta] = theta + np.random.normal(loc=omega * Constants.dt, scale=0.05, size=self.num_particles)
        self.particles[StateSpace.omega] = omega + np.random.normal(loc=((-Constants.g * np.sin(theta)) / Constants.L) * Constants.dt, scale=0.01, size=self.num_particles)

    def update(self, measurement: float):
        """
        Only updating based on theta as this is the only dimension we measure.
        """
        self.weights *= np.exp(-0.5 * ((self.particles[StateSpace.theta] - measurement) / 0.1) ** 2)
        print(np.sum(self.weights))
        self.weights /= np.sum(self.weights)

    def resample(self):
        indices = np.random.choice(a=self.num_particles, size=self.num_particles, p=self.weights)
        self.particles[StateSpace.theta] = self.particles[StateSpace.theta][indices]
        self.particles[StateSpace.omega] = self.particles[StateSpace.omega][indices]

        self.weights = np.ones(shape=self.num_particles) / self.num_particles  # Reset weights (could potentially do this only occasionally)

    def estimate_mean(self) -> tuple[float, float]:
        return np.mean(self.particles[StateSpace.theta]).astype(dtype=float), np.mean(self.particles[StateSpace.omega]).astype(dtype=float)

    def estimate_std(self) -> tuple[float, float]:
        return np.std(self.particles[StateSpace.theta]).astype(dtype=float), np.std(self.particles[StateSpace.omega]).astype(dtype=float)

    def get_particles(self) -> TY_PARTICLES:
        return self.particles


def run_filter(data: pd.DataFrame, num_particles: int = 1000) -> pd.DataFrame:
    filter = ParticleFilter(num_particles)
    thetas_mean: list[float] = []
    thetas_std: list[float] = []

    for i, row in data.iterrows():
        theta_mean, theta_std = filter.estimate_mean()[0], filter.estimate_std()[0]

        thetas_mean.append(theta_mean)
        thetas_std.append(theta_std)

        print(i, theta_mean, theta_std)

        filter.predict()
        filter.update(measurement=row[DataColumns.theta_observation])
        filter.resample()

    data[DataColumns.theta_mean] = thetas_mean
    data[DataColumns.theta_std] = thetas_std

    return data


################
# PLOT


def plot_data(data: pd.DataFrame):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=data[DataColumns.time],
            y=data[DataColumns.theta_observation],
            name="Theta observation",
            mode="markers",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data[DataColumns.time],
            y=data[DataColumns.theta_mean],
            name="Theta estimates",
            mode="markers",
        )
    )

    fig.show()


def main():
    data = get_data()

    data = run_filter(data=data, num_particles=1000)

    plot_data(data)


if __name__ == "__main__":
    main()

