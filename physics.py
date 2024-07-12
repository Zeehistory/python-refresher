def calculate_buoyancy(V, density_fluid):
    """
    GOAL: Calculate the buoyant force exerted on an object submerged in a fluid.

    Parameters:
        V (float): Volume of the submerged object (in cubic meters).
        density_fluid (float): Density of the fluid (in kg/m^3).

    Returns:
        float: Buoyant force (in newtons).
    """
    g = 9.81  # acceleration due to gravity in m/s^2
    F_b = density_fluid * V * g  # buoyant force formula
    return F_b


def will_it_float(V, mass):
    """
    GOAL: Determine if an object will float in water based on its volume and mass.

    Parameters:
        V (float): Volume of the object (in cubic meters).
        mass (float): Mass of the object (in kilograms).

    Returns:
        bool: True if the buoyant force exceeds the weight, False otherwise.
    """
    density_water = 1000  # density of water in kg/m^3
    buoyant_force = calculate_buoyancy(V, density_water)  # calculate buoyant force
    weight = mass * 9.81  # calculate weight force
    return buoyant_force > weight


def calculate_pressure(depth):
    """
    GOAL: Calculate the hydrostatic pressure at a given depth underwater.

    Parameters:
        depth (float): Depth of the object underwater (in meters).

    Returns:
        float: Hydrostatic pressure (in pascals).
    """
    density_water = 1000  # density of water in kg/m^3
    g = 9.81  # acceleration due to gravity in m/s^2
    p = density_water * g * depth  # hydrostatic pressure formula
    return p


def calculate_acceleration(F, m):
    """
    GOAL: Calculate the acceleration of an object given the force and mass.

    Parameters:
        F (float): Force acting on the object (in newtons).
        m (float): Mass of the object (in kilograms).

    Returns:
        float: Acceleration (in m/s^2).

    Raises:
        ValueError: If mass is zero.
    """
    if m == 0:
        raise ValueError("Mass cannot be zero")
    a = F / m  # acceleration formula
    return a


def calculate_ang_acceleration(tau, I):
    """
    GOAL: Calculate the angular acceleration of an object given the torque and moment of inertia.

    Parameters:
        tau (float): Torque acting on the object (in newton-meters).
        I (float): Moment of inertia of the object (in kg*m^2).

    Returns:
        float: Angular acceleration (in rad/s^2).
    """
    alpha = tau / I  # angular acceleration formula
    return alpha


import math


def calculate_torque(F_magnitude, F_direction, r):
    """
    GOAL: Calculate the torque exerted on an object.

    Parameters:
        F_magnitude (float): Magnitude of the force (in newtons).
        F_direction (float): Direction of the force (in degrees).
        r (float): Lever arm distance (in meters).

    Returns:
        float: Torque (in newton-meters).
    """
    F_direction_radians = math.radians(F_direction)  # convert direction to radians
    tau = F_magnitude * r * math.sin(F_direction_radians)  # torque formula
    return tau


def calculate_MOI(m, r):
    """
    GOAL: Calculate the moment of inertia of a point mass.

    Parameters:
        m (float): Mass of the object (in kilograms).
        r (float): Distance from the axis of rotation (in meters).

    Returns:
        float: Moment of inertia (in kg*m^2).
    """
    I = m * r**2  # moment of inertia formula
    return I


def calculate_auv_acceleration(
    F_magnitude, F_angle, mass=100, volume=0.1, thruster_distance=0.5
):
    """
    GOAL: Calculate the acceleration of an Autonomous Underwater Vehicle (AUV) in water.

    Parameters:
        F_magnitude (float): Magnitude of the thruster force (in newtons).
        F_angle (float): Direction of the thruster force (in radians).
        mass (float, optional): Mass of the AUV (in kilograms). Default is 100.
        volume (float, optional): Volume of the AUV (in cubic meters). Default is 0.1.
        thruster_distance (float, optional): Distance of thrusters from the center of mass (in meters). Default is 0.5.

    Returns:
        float: Total acceleration of the AUV (in m/s^2).
    """
    g = 9.81  # acceleration due to gravity in m/s^2
    density_water = 1000  # density of water in kg/m^3

    F_x = F_magnitude * math.cos(F_angle)  # resolve force components
    F_y = F_magnitude * math.sin(F_angle)

    buoyant_force = density_water * volume * g  # calculate buoyant force

    weight = mass * g  # calculate weight force

    net_force_x = F_x  # net force in x-direction
    net_force_y = F_y + buoyant_force - weight  # net force in y-direction

    a_x = net_force_x / mass  # calculate acceleration components
    a_y = net_force_y / mass

    acceleration = math.sqrt(a_x**2 + a_y**2)  # total acceleration magnitude

    return acceleration


def calculate_angular_acceleration(
    F_magnitude, F_angle, inertia=1, thruster_distance=0.5
):
    """
    GOAL : Calculate the angular acceleration of an AUV.

    Parameters:
        F_magnitude (float): Magnitude of the thruster force (in newtons).
        F_angle (float): Direction of the thruster force (in radians).
        inertia (float, optional): Moment of inertia of the AUV (in kg*m^2). Default is 1.
        thruster_distance (float, optional): Distance of thrusters from the center of mass (in meters). Default is 0.5.

    Returns:
        float: Angular acceleration of the AUV (in rad/s^2).
    """
    torque = F_magnitude * thruster_distance * math.sin(F_angle)  # calculate torque
    angular_acceleration = torque / inertia  # angular acceleration formula
    return angular_acceleration


import numpy as np


def calculate_acc2(T, alpha, theta, mass=100):
    """
    Calculate the acceleration components of a system of forces.

    Parameters:
        T (numpy.ndarray): Array of force magnitudes (in newtons).
        alpha (numpy.ndarray): Array of force directions (in radians).
        theta (float): Angle of the object (in radians).
        mass (float, optional): Mass of the object (in kilograms). Default is 100.

    Returns:
        numpy.ndarray: Array of acceleration components [a_x, a_y] (in m/s^2).
    """
    F_x = np.sum(T * np.cos(alpha + theta))  # resolve force components in x-direction
    F_y = np.sum(T * np.sin(alpha + theta))  # resolve force components in y-direction
    a_x = F_x / mass  # calculate acceleration in x-direction
    a_y = F_y / mass  # calculate acceleration in y-direction
    return np.array([a_x, a_y])


def calculate_angular_acceleration2(T, alpha, L, l, inertia=100):
    """
    Calculate the angular acceleration of a system of torques.

    Parameters:
        T (numpy.ndarray): Array of torque magnitudes (in newton-meters).
        alpha (numpy.ndarray): Array of torque directions (in radians).
        L (float): Length parameter.
        l (float): Another length parameter.
        inertia (float, optional): Moment of inertia of the system (in kg*m^2). Default is 100.

    Returns:
        float: Angular acceleration (in rad/s^2).
    """
    torques = T * (
        L * np.cos(alpha) + l * np.sin(alpha)
    )  # calculate individual torques
    net_torque = np.sum(torques)  # sum of all torques
    angular_acceleration = net_torque / inertia  # angular acceleration formula
    return angular_acceleration


def simulate_auv2_motion(
    T, alpha, L, l, mass=100, inertia=100, dt=0.1, t_final=10, x0=0, y0=0, theta0=0
):
    """
    Simulate the motion of an Autonomous Underwater Vehicle (AUV) over time.

    Parameters:
        T (numpy.ndarray): Array of force magnitudes (in newtons).
        alpha (numpy.ndarray): Array of force directions (in radians).
        L (float): Length parameter.
        l (float): Another length parameter.
        mass (float, optional): Mass of the AUV (in kilograms). Default is 100.
        inertia (float, optional): Moment of inertia of the AUV (in kg*m^2). Default is 100.
        dt (float, optional): Time step for simulation (in seconds). Default is 0.1.
        t_final (float, optional): Final time for simulation (in seconds). Default is 10.
        x0 (float, optional): Initial x-position of the AUV. Default is 0.
        y0 (float, optional): Initial y-position of the AUV. Default is 0.
        theta0 (float, optional): Initial angle of the AUV (in radians). Default is 0.

    Returns:
        tuple: Arrays of time, x-position, y-position, angle, velocity, angular velocity, and acceleration over time.
    """
    t = np.arange(0, t_final + dt, dt)  # time array
    x = np.zeros_like(
        t
    )  # initialize arrays for position, velocity, angle, and acceleration
    y = np.zeros_like(t)
    theta = np.zeros_like(t)
    v = np.zeros((len(t), 2))
    omega = np.zeros_like(t)
    a = np.zeros((len(t), 2))
    x[0] = x0  # set initial conditions
    y[0] = y0
    theta[0] = theta0
    for i in range(1, len(t)):
        a[i] = calculate_acc2(T, alpha, theta[i - 1], mass)  # calculate acceleration
        angular_acc = calculate_angular_acceleration2(
            T, alpha, L, l, inertia
        )  # calculate angular acceleration
        v[i] = v[i - 1] + a[i] * dt  # update velocity
        omega[i] = omega[i - 1] + angular_acc * dt  # update angular velocity
        x[i] = x[i - 1] + v[i, 0] * dt  # update position
        y[i] = y[i - 1] + v[i, 1] * dt
        theta[i] = theta[i - 1] + omega[i] * dt  # update angle
    return t, x, y, theta, v, omega, a


# Example usage:
T = np.array([10, 10])
alpha = np.array([0, np.pi / 2])
L = 1
l = 1
t, x, y, theta, v, omega, a = simulate_auv2_motion(T, alpha, L, l)

print("Time:", t)
print("X-positions:", x)
print("Y-positions:", y)
print("Angles:", theta)
print("Velocities:", v)
print("Angular velocities:", omega)
print("Accelerations:", a)
