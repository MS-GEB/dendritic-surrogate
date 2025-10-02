TITLE Hippocampal HH channels
:
: Fast Na+ and K+ currents responsible for action potentials
: Iterative equations
:
: Equations modified by Traub, for Hippocampal Pyramidal cells, in:
: Traub & Miles, Neuronal Networks of the Hippocampus, Cambridge, 1991
:
: range variable vtraub adjust threshold
:
: Written by Alain Destexhe, Salk Institute, Aug 1992
:
: Added gradient computation (Gan He et al., 2025)

NEURON {
    POINT_PROCESS HH2_modified
    RANGE ena, ina, ek, ik
    NONSPECIFIC_CURRENT i
    RANGE ina_dv, ik_dv, i_dv
    RANGE gnabar, gkbar, vtraub

    RANGE m_inf, h_inf, n_inf
    RANGE m_tau, h_tau, n_tau
    RANGE m_exp, h_exp, n_exp
    RANGE dv, seg_area, didv, pure_i
    RANGE didv_clip
    RANGE w
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    gnabar	= .003  (mho/cm2)
    gkbar	= .005 	(mho/cm2)
    ena     = 50	(mV)
    ek	    = -90	(mV)
    vtraub	= -55	(mV)
    w = 1.
    seg_area = 1.   : in cm2
    didv_clip = 0.
    dv = 1e-3
}

STATE {
    m h n
    m_dv h_dv n_dv
}

ASSIGNED {
    v   (mV)
    i   (nA)
    ina ik
    ina_dv ik_dv i_dv
    m_inf h_inf n_inf
    m_tau h_tau n_tau
    m_exp h_exp n_exp
    tadj
    didv pure_i
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ina_dv = gnabar * m_dv*m_dv*m_dv*h_dv * (v + dv - ena)
    ik_dv  = gkbar * n_dv*n_dv*n_dv*n_dv * (v + dv - ek)
    i_dv = fabs(w) * (ina_dv + ik_dv) * seg_area * 1e6

    ina = gnabar * m*m*m*h * (v - ena)
    ik  = gkbar * n*n*n*n * (v - ek)
    pure_i = -(ina + ik) * seg_area * 1e6
    i = fabs(w) * -pure_i
    didv = fabs(w) * (-gnabar * m*m*m*h - gkbar * n*n*n*n) * seg_area * 1e6
    if (v > didv_clip) {
        didv = 0.
    }
}

DERIVATIVE states {
    evaluate_fct(v + dv)
    m_dv' = (m_inf - m) / m_tau
    h_dv' = (h_inf - h) / h_tau
    n_dv' = (n_inf - n) / n_tau
    evaluate_fct(v)
    m' = (m_inf - m) / m_tau
    h' = (h_inf - h) / h_tau
    n' = (n_inf - n) / n_tau
}

UNITSOFF
INITIAL {
    evaluate_fct(v)
    m = m_inf
    m_dv = m_inf
    h = h_inf
    h_dv = h_inf
    n = n_inf
    n_dv = n_inf
:
:  Q10 was assumed to be 3 for both currents
:
: original measurements at roomtemperature?

    tadj = 3.0 ^ ((celsius-36) / 10)
}

PROCEDURE evaluate_fct(v (mV)) {
    LOCAL a, b, v2

    v2 = v - vtraub : convert to traub convention

    a = 0.32 * (13-v2) / (exp((13-v2)/4) - 1)
    b = 0.28 * (v2-40) / (exp((v2-40)/5) - 1)
    m_tau = 1 / (a + b) / tadj
    m_inf = a / (a + b)

    a = 0.128 * exp((17-v2)/18)
    b = 4 / (1 + exp((40-v2)/5))
    h_tau = 1 / (a + b) / tadj
    h_inf = a / (a + b)

    a = 0.032 * (15-v2) / (exp((15-v2)/5) - 1)
    b = 0.5 * exp((10-v2)/40)
    n_tau = 1 / (a + b) / tadj
    n_inf = a / (a + b)

    m_exp = 1 - exp(-dt/m_tau)
    h_exp = 1 - exp(-dt/h_tau)
    n_exp = 1 - exp(-dt/n_tau)
}

UNITSON