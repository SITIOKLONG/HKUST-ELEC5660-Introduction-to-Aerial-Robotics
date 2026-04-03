from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


def _poly_derivative(coeffs: np.ndarray) -> np.ndarray:
    """Derivative of a polynomial with coefficients in descending powers."""
    degree = len(coeffs) - 1
    if degree <= 0:
        return np.array([0.0])
    return np.array([coeffs[i] * (degree - i) for i in range(degree)], dtype=float)


def _solve_equality_qp(H: np.ndarray, Aeq: np.ndarray, beq: np.ndarray) -> np.ndarray:
    """Solve min 0.5 x^T H x s.t. Aeq x = beq using KKT system."""
    H = 0.5 * (H + H.T)
    n = H.shape[0]
    m = Aeq.shape[0]
    KKT = np.zeros((n + m, n + m))
    KKT[:n, :n] = H
    KKT[:n, n:] = Aeq.T
    KKT[n:, :n] = Aeq
    rhs = np.zeros(n + m)
    rhs[n:] = beq
    sol, _, _, _ = np.linalg.lstsq(KKT, rhs, rcond=None)
    return sol[:n]


def _basis_row(n_coef: int, deriv_order: int, t: float) -> np.ndarray:
    """Row mapping polynomial coefficients to derivative value at time t."""
    row = np.zeros(n_coef, dtype=float)
    for i in range(deriv_order, n_coef):
        row[i] = math.factorial(i) / math.factorial(i - deriv_order) * (t ** (i - deriv_order))
    return row


def _build_min_derivative_axis(
    p: np.ndarray,
    n_seg: int,
    T_scale: np.ndarray,
    n_coef: int,
    r: int,
) -> np.ndarray:
    """Solve one-axis minimum derivative trajectory (r=3 jerk, r=4 snap)."""
    n_var = n_seg * n_coef
    H = np.zeros((n_var, n_var), dtype=float)

    for k in range(n_seg):
        T = T_scale[k]
        Qk = np.zeros((n_coef, n_coef), dtype=float)
        for i in range(r, n_coef):
            ci = math.factorial(i) / math.factorial(i - r)
            for j in range(r, n_coef):
                cj = math.factorial(j) / math.factorial(j - r)
                power = i + j - 2 * r
                Qk[i, j] = ci * cj * (T ** (power + 1)) / (power + 1)
        s = k * n_coef
        e = s + n_coef
        H[s:e, s:e] = Qk

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for k in range(n_seg):
        s = k * n_coef
        e = s + n_coef

        row0 = np.zeros(n_var, dtype=float)
        row0[s:e] = _basis_row(n_coef, deriv_order=0, t=0.0)
        rows.append(row0)
        rhs.append(float(p[k]))

        rowT = np.zeros(n_var, dtype=float)
        rowT[s:e] = _basis_row(n_coef, deriv_order=0, t=T_scale[k])
        rows.append(rowT)
        rhs.append(float(p[k + 1]))

    for d in range(1, r):
        row_start = np.zeros(n_var, dtype=float)
        row_start[:n_coef] = _basis_row(n_coef, deriv_order=d, t=0.0)
        rows.append(row_start)
        rhs.append(0.0)

        row_end = np.zeros(n_var, dtype=float)
        s = (n_seg - 1) * n_coef
        e = s + n_coef
        row_end[s:e] = _basis_row(n_coef, deriv_order=d, t=T_scale[-1])
        rows.append(row_end)
        rhs.append(0.0)

    for k in range(n_seg - 1):
        s1 = k * n_coef
        e1 = s1 + n_coef
        s2 = (k + 1) * n_coef
        e2 = s2 + n_coef
        for d in range(1, r):
            row = np.zeros(n_var, dtype=float)
            row[s1:e1] = _basis_row(n_coef, deriv_order=d, t=T_scale[k])
            row[s2:e2] = -_basis_row(n_coef, deriv_order=d, t=0.0)
            rows.append(row)
            rhs.append(0.0)

    Aeq = np.vstack(rows)
    beq = np.array(rhs, dtype=float)
    x = _solve_equality_qp(H, Aeq, beq)

    c = np.zeros((n_coef, n_seg), dtype=float)
    for k in range(n_seg):
        s = k * n_coef
        e = s + n_coef
        c[:, k] = x[s:e]
    return c


def _generate_smooth_only(waypoints: np.ndarray, n_seg: int, total_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate smooth trajectory using quintic polynomials.
    Ensures position, velocity, and acceleration continuity at waypoints.
    Uses 5th order polynomials (6 coefficients per segment).
    """
    # ========================================================================
    # TODO: Implement your smooth only trajectory generation here
    # ========================================================================
    T_scale = np.full(n_seg, total_time / n_seg, dtype=float)
    
    def _fit_axis(p: np.ndarray) -> np.ndarray:
        c = np.zeros((6, n_seg), dtype=float)       # coefficient
        # init v, a
        v = np.zeros(n_seg + 1, dtype=float)
        a = np.zeros(n_seg + 1, dtype=float)

        # continuity constraints
        v[0] = 0.0
        v[-1] = 0.0
        a[0] = 0.0
        a[-1] = 0.0
        
        # estimate v,a
        for i in range(1, n_seg):
            dt_prev = T_scale[i-1]
            dt_next = T_scale[i]
            dt = 0.5 * (dt_prev + dt_next)
            v[i] = (p[i + 1] - p[i - 1]) / (2.0 * dt)
            a[i] = (p[i + 1] - 2.0 * p[i] + p[i - 1]) / (dt * dt)
        
        # solve coefficients
        for k in range(n_seg):
            T = T_scale[k]
            T2, T3, T4, T5 = T**2, T**3, T**4, T**5
            
            Aeq = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],          # p(0)
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],          # v(0)
                    [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],          # a(0)
                    [1.0, T,   T2,  T3,  T4,  T5],           # p(T)
                    [0.0, 1.0, 2*T, 3*T2, 4*T3, 5*T4],       # v(T)
                    [0.0, 0.0, 2.0, 6*T, 12*T2, 20*T3],      # a(T)
                ],
                dtype=float,
            )
            beq = np.array(
                [p[k], v[k], a[k], p[k + 1], v[k + 1], a[k + 1]],
                dtype=float,
            )
            H = np.eye(6, dtype=float)

            c[:, k] = _solve_equality_qp(H, Aeq, beq)
            
        return c
    
    c_x = _fit_axis(waypoints[:, 0])
    c_y = _fit_axis(waypoints[:, 1])
    c_z = _fit_axis(waypoints[:, 2])
    # ========================================================================
    # End of your implementation
    # ========================================================================
    return c_x, c_y, c_z, T_scale


def _generate_minimum_jerk(waypoints: np.ndarray, n_seg: int, total_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate minimum jerk trajectory using 6th order polynomials.
    Minimizes the integral of jerk (third derivative) squared.
    """
    # ========================================================================
    # TODO: Implement your minimum jerk trajectory generation here
    # ========================================================================
    T_scale = np.full(n_seg, total_time / n_seg, dtype=float)
    n_coef = 6
    c_x = _build_min_derivative_axis(waypoints[:, 0], n_seg, T_scale, n_coef, r=3)
    c_y = _build_min_derivative_axis(waypoints[:, 1], n_seg, T_scale, n_coef, r=3)
    c_z = _build_min_derivative_axis(waypoints[:, 2], n_seg, T_scale, n_coef, r=3)
    return c_x, c_y, c_z, T_scale


def _generate_minimum_snap(waypoints: np.ndarray, n_seg: int, total_time: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate minimum snap trajectory using 8th order polynomials.
    Minimizes the integral of snap (fourth derivative) squared.
    """
    # ========================================================================
    # TODO: Implement your minimum snap trajectory generation here
    # ========================================================================
    T_scale = np.full(n_seg, total_time / n_seg, dtype=float)
    n_coef = 8
    c_x = _build_min_derivative_axis(waypoints[:, 0], n_seg, T_scale, n_coef, r=4)
    c_y = _build_min_derivative_axis(waypoints[:, 1], n_seg, T_scale, n_coef, r=4)
    c_z = _build_min_derivative_axis(waypoints[:, 2], n_seg, T_scale, n_coef, r=4)
    return c_x, c_y, c_z, T_scale


@dataclass
class TrajectoryGenerator:
    """Polynomial trajectory generator with multiple methods."""

    waypoints: np.ndarray
    method: str = "snap"
    total_time: float = 25.0

    def __post_init__(self) -> None:
        self.waypoints = np.asarray(self.waypoints, dtype=float)
        if self.waypoints.ndim != 2 or self.waypoints.shape[1] != 3:
            raise ValueError("waypoints must be shaped (N, 3)")
        self.n_seg = self.waypoints.shape[0] - 1
        if self.n_seg < 1:
            raise ValueError("need at least two waypoints")

        if self.method not in {"smooth", "jerk", "snap"}:
            raise ValueError("method must be 'smooth', 'jerk', or 'snap'")

        # Set polynomial order based on method
        if self.method == "smooth":
            self.n = 6  # quintic (5th order, 6 coefficients)
        elif self.method == "jerk":
            self.n = 6  # 6th order
        else:  # snap
            self.n = 8  # 8th order

        self._prepare()

    def _prepare(self) -> None:
        """Prepare trajectory by calling appropriate generation method."""
        if self.method == "smooth":
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_smooth_only(
                self.waypoints, self.n_seg, self.total_time
            )
        elif self.method == "jerk":
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_minimum_jerk(
                self.waypoints, self.n_seg, self.total_time
            )
        else:  # snap
            self.c_x, self.c_y, self.c_z, self.T_scale = _generate_minimum_snap(
                self.waypoints, self.n_seg, self.total_time
            )

    def _segment_time(self, t: float) -> tuple[int, float]:
        t = float(t)
        for seg in range(self.n_seg):
            if t - self.T_scale[seg] <= 0.0:
                return seg, t
            t -= self.T_scale[seg]
        # Clamp to the end of the last segment
        return self.n_seg - 1, float(self.T_scale[self.n_seg - 1])

    def evaluate(self, t: float) -> np.ndarray:
        seg, local_t = self._segment_time(t)
        coeff_x = self.c_x[:, seg][::-1]
        coeff_y = self.c_y[:, seg][::-1]
        coeff_z = self.c_z[:, seg][::-1]

        s_des = np.zeros(11)
        s_des[0] = np.polyval(coeff_x, local_t)
        s_des[1] = np.polyval(coeff_y, local_t)
        s_des[2] = np.polyval(coeff_z, local_t)

        dcoeff_x = _poly_derivative(coeff_x)
        dcoeff_y = _poly_derivative(coeff_y)
        dcoeff_z = _poly_derivative(coeff_z)

        s_des[3] = np.polyval(dcoeff_x, local_t)
        s_des[4] = np.polyval(dcoeff_y, local_t)
        s_des[5] = np.polyval(dcoeff_z, local_t)

        return s_des
