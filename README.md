# A basic particle filter implementation

## Implementation

$$
\theta_{t+1} = \theta_t + \omega_t \Delta t + \text{noise}
$$

$$
\omega_{t+1} = \left(\frac{-g \cdot \sin(\theta_t)}{L}\right) \Delta t + \text{noise}
$$

Note: Not exactly the same implementations as in the project thesis.

## Kalman filter would be

$$
x = \begin{bmatrix} \theta \\ \omega \end{bmatrix}
$$

$$
\dot{x} = \begin{bmatrix} 0 & 1 \\ -\frac{g}{L} & 0 \end{bmatrix} x
$$

Note: Matrices above are not displayed properly for some reason.
