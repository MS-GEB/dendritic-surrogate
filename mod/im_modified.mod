TITLE Cortical M current
:
:   M-current, responsible for the adaptation of firing rate and the
:   afterhyperpolarization (AHP) of cortical pyramidal cells
:
:   First-order model described by hodgkin-Hyxley like equations.
:   K+ current, activated by depolarization, noninactivating.
:
:   Model taken from Yamada, W.M., Koch, C. and Adams, P.R.  Multiple
:   channels and calcium dynamics.  In: Methods in Neuronal Modeling,
:   edited by C. Koch and I. Segev, MIT press, 1989, p 97-134.
:
:   See also: McCormick, D.A., Wang, Z. and Huguenard, J. Neurotransmitter
:   control of neocortical neuronal activity and excitability.
:   Cerebral Cortex 3: 387-398, 1993.
:
:   Written by Alain Destexhe, Laval University, 1995
:
:   Changed initial from m=0 to m = m_inf, BAB 2018
:
: 	Added gradient computation (Gan He et al., 2025)

NEURON {
    POINT_PROCESS IM_modified
    RANGE ek, ik
    NONSPECIFIC_CURRENT i
    RANGE ik_dv, i_dv
    RANGE gkbar, m_inf, tau_m, taumax
    RANGE dv, seg_area, didv, pure_i
    RANGE didv_clip
    RANGE w
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    ek		= -80 (mV)
    gkbar	= 1e-6	(mho/cm2)
    taumax	= 1000	(ms)		: peak value of tau
    w = 1.
    seg_area = 1.   : in cm2
    didv_clip = 0.
    dv = 1e-3
}

STATE {
    m
    m_dv
}

ASSIGNED {
    v   (mV)
    i   (nA)
    ik
    ik_dv i_dv
    m_inf
    tau_m	    (ms)
    tau_peak	(ms)
    tadj
    didv pure_i
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    ik_dv = gkbar * m_dv * (v + dv - ek)
    i_dv = fabs(w) * ik_dv * seg_area * 1e6

    ik = gkbar * m * (v - ek)
    pure_i = -ik * seg_area * 1e6
    i = fabs(w) * -pure_i
    didv = fabs(w) * (-gkbar * m) * seg_area * 1e6
    if (v > didv_clip) {
        didv = 0.
    }
}

DERIVATIVE states {
    evaluate_fct(v + dv)
    m_dv' = (m_inf - m) / tau_m
    evaluate_fct(v)
    m' = (m_inf - m) / tau_m
}

UNITSOFF
INITIAL {
    evaluate_fct(v)
    m = m_inf
    m_dv = m_inf
:
:  The Q10 value is assumed to be 2.3
:
    tadj = 2.3 ^ ((celsius-36)/10)
    tau_peak = taumax / tadj
}

PROCEDURE evaluate_fct(v(mV)) {
    m_inf = 1 / ( 1 + exptable(-(v+35)/10) )
    tau_m = tau_peak / ( 3.3 * exptable((v+35)/20) + exptable(-(v+35)/20) )
}
UNITSON


FUNCTION exptable(x) {
    TABLE  FROM -25 TO 25 WITH 10000

    if ((x > -25) && (x < 25)) {
        exptable = exp(x)
    } else {
        exptable = 0.
    }
}