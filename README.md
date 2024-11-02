A simple particle filter.

theta_t+1 = theta_t + omega_t * dt + noise
omega_t+1 = [(-g * sin(theta_t)) / L] * dt + noise

Note: Not exactly the same implementations as in the project thesis.

Kalman filter would be
x = [theta, omega]
x_dot = [[0, 1], [-g / L, 0]] * x
