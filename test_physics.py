import unittest
import math
import numpy as np

# Assuming the functions are saved in a module named `physics`

import physics


class TestPhysics(unittest.TestCase):

    def test_calculate_buoyancy(self):
        self.assertAlmostEqual(physics.calculate_buoyancy(1, 1000), 9810)
        self.assertAlmostEqual(physics.calculate_buoyancy(0.5, 500), 2452.5)

    def test_will_it_float(self):
        self.assertTrue(physics.will_it_float(1, 500))
        self.assertFalse(physics.will_it_float(1, 1500))

    def test_calculate_pressure(self):
        self.assertAlmostEqual(physics.calculate_pressure(10), 98100)
        self.assertAlmostEqual(physics.calculate_pressure(5), 49050)

    def test_calculate_acceleration(self):
        self.assertAlmostEqual(physics.calculate_acceleration(10, 2), 5)
        self.assertAlmostEqual(physics.calculate_acceleration(9.81, 1), 9.81)

    def test_calculate_ang_acceleration(self):
        self.assertAlmostEqual(physics.calculate_ang_acceleration(10, 2), 5)
        self.assertAlmostEqual(physics.calculate_ang_acceleration(9.81, 1), 9.81)

    def test_calculate_torque(self):
        self.assertAlmostEqual(physics.calculate_torque(10, 90, 2), 20)
        self.assertAlmostEqual(
            physics.calculate_torque(10, 45, 2), 10 * math.sin(math.radians(45)) * 2
        )

    def test_calculate_MOI(self):
        self.assertAlmostEqual(physics.calculate_MOI(2, 3), 18)
        self.assertAlmostEqual(physics.calculate_MOI(5, 2), 20)

    def test_calculate_acceleration(self):
        self.assertAlmostEqual(physics.calculate_acceleration(10, 2), 5)
        self.assertAlmostEqual(physics.calculate_acceleration(9.81, 1), 9.81)

    def test_calculate_acceleration_zero_mass(self):
        with self.assertRaises(ValueError):
            physics.calculate_acceleration(10, 0)

    def test_calculate_angular_acceleration_advanced(self):
        self.assertAlmostEqual(physics.calculate_angular_acceleration(10, 0), 0.0)
        self.assertAlmostEqual(
            physics.calculate_angular_acceleration(10, math.pi / 2), 10 * 0.5
        )

    def test_calculate_acc2(self):
        T = np.array([10, 10])
        alpha = np.array([0, np.pi / 2])
        self.assertTrue(np.allclose(physics.calculate_acc2(T, alpha, 0), [0.1, 0.1]))
        self.assertTrue(
            np.allclose(
                physics.calculate_acc2(T, alpha, np.pi / 4), [0, 0.14142135623730953]
            )
        )

    def test_calculate_angular_acceleration2(self):
        T = np.array([10, 10])
        alpha = np.array([0, np.pi / 2])
        self.assertAlmostEqual(
            physics.calculate_angular_acceleration2(T, alpha, 1, 1), 0.1 * (1 + 1)
        )


if __name__ == "__main__":
    unittest.main()
