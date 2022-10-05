Neuron Models
=============


Izhikevich
----------

Source: ... (2003)


Rate Reduced Rulkov Model
-------------------------

Source: A Rate-Reduced Neuron Model for Complex Spiking Behavior (2017)
Koen Dijkstra, Yuri A. Kuznetsov, Michel J.A.M. van Putten,
Stephan A. van Gils.


.. math::
    :label: eq:va

    v_{n+1} &= f(v_n, v_{n-1}, \kappa u_n - a_n - \theta), \\
    a_{n+1} &=  a_n - \epsilon (a_n + (1- \kappa) u_n - \gamma s_n).

where

.. math::
    :label: eq:fx

    f(x_1, x_2, x_3) =
    \begin{cases}
    \frac{2500 + 150 x_1}{50 - x_1} + 50 x_3 &\text{ if } x_1 < 0, \\
    50 + 50 x_3 & \text{ if } (0 \leq x_1 < 50 + 50 x_3) \land (x_{2} < 0)), \\
    -50 & \text{otherwise},
    \end{cases}


and

.. math::
    :label: eq::sn

    s_n = \begin{cases}
    1 & \text{ if the neuron spiked at iteration }n, \\
    0 & \text{ otherwise.}
    \end{cases}




A Rulkov neuron spikes at iteration n if its membrane potential is reset to :math:`v_{n+1} = 50` in the next iteration.
It follows from (2) that the spiking condition in (3) is satisfied if and only if

.. math::

    v_n \qquad \land \qquad \big((v_n \geq 50 + 50 (\kappa u_n - a_n - \theta) \qquad \lor \qquad (v_{n+1} \geq 0))\big).



