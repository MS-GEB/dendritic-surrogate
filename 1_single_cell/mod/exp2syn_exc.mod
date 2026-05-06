COMMENT
Two state kinetic scheme synapse described by rise time tau1,
and decay time constant tau2. The normalized peak condunductance is 1.
Decay time MUST be greater than rise time.

The solution of A->G->bath with rate constants 1/tau1 and 1/tau2 is
 A = a*exp(-t/tau1) and
 G = a*tau2/(tau2-tau1)*(-exp(-t/tau1) + exp(-t/tau2))
    where tau1 < tau2

If tau2-tau1 is very small compared to tau1, this is an alphasynapse with time constant tau2.
If tau1/tau2 is very small, this is single exponential decay with time constant tau2.

The factor is evaluated in the initial block
such that an event of weight 1 generates a
peak conductance of 1.

Because the solution is a sum of exponentials, the
coupled equations can be solved as a pair of independent equations
by the more efficient cnexp method.

## Modified NMDA voltage dependence with C=1/3.75, rho=0.062 (Bicknell & Hausser, 2019)
## Combined AMPA & NMDA through a single weight and added gradient computation (Gan He et al., 2025)

ENDCOMMENT

NEURON {
    POINT_PROCESS Exp2Syn_exc
    RANGE AMPA_tau1, AMPA_tau2, NMDA_tau1, NMDA_tau2, AMPA_e, NMDA_e
    NONSPECIFIC_CURRENT i
    RANGE NMDA_C, NMDA_rho
    RANGE AMPA_g, NMDA_g
    RANGE sigma, didv, pure_i
    RANGE w, r_na
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    AMPA_tau1 = 0.1 (ms) <1e-9,1e9>
    AMPA_tau2 = 10  (ms) <1e-9,1e9>
    NMDA_tau1 = 10  (ms) <1e-9,1e9>
    NMDA_tau2 = 50  (ms) <1e-9,1e9>
    AMPA_e = 0  (mV)
    NMDA_e = 0  (mV)
    NMDA_C = 0.267
    NMDA_rho = 0.062
    w = 1
    r_na = 2
}

ASSIGNED {
    v   (mV)
    i   (nA)
    AMPA_g  (uS)
    AMPA_factor
    NMDA_g  (uS)
    NMDA_factor
    sigma didv pure_i
}

STATE {
    AMPA_A  (uS)
    AMPA_B  (uS)
    NMDA_A  (uS)
    NMDA_B  (uS)
}

INITIAL {
    LOCAL AMPA_tp, NMDA_tp
    if (AMPA_tau1 / AMPA_tau2 > 0.9999) {
        AMPA_tau1 = 0.9999 * AMPA_tau2
    }
    if (AMPA_tau1 / AMPA_tau2 < 1e-9) {
        AMPA_tau1 = AMPA_tau2 * 1e-9
    }
    AMPA_A = 0
    AMPA_B = 0
    AMPA_tp = (AMPA_tau1 * AMPA_tau2) / (AMPA_tau2 - AMPA_tau1) * log(AMPA_tau2 / AMPA_tau1)
    AMPA_factor = -exp(-AMPA_tp / AMPA_tau1) + exp(-AMPA_tp / AMPA_tau2)
    AMPA_factor = 1 / AMPA_factor

    if (NMDA_tau1 / NMDA_tau2 > 0.9999) {
        NMDA_tau1 = 0.9999 * NMDA_tau2
    }
    if (NMDA_tau1 / NMDA_tau2 < 1e-9) {
        NMDA_tau1 = NMDA_tau2 * 1e-9
    }
    NMDA_A = 0
    NMDA_B = 0
    NMDA_tp = (NMDA_tau1 * NMDA_tau2) / (NMDA_tau2 - NMDA_tau1) * log(NMDA_tau2 / NMDA_tau1)
    NMDA_factor = -exp(-NMDA_tp / NMDA_tau1) + exp(-NMDA_tp / NMDA_tau2)
    NMDA_factor = 1 / NMDA_factor
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    AMPA_g = AMPA_B - AMPA_A
    NMDA_g = NMDA_B - NMDA_A
    sigma = 1 / (1 + NMDA_C * exp(-NMDA_rho * v))
    pure_i = AMPA_g * (AMPA_e - v) + NMDA_g * sigma * (NMDA_e - v)
    i = fabs(w) * -pure_i
    didv = fabs(w) * (-AMPA_g + NMDA_g * NMDA_rho * sigma * (1 - sigma) * (NMDA_e - v) - NMDA_g * sigma)
}

DERIVATIVE states {
    AMPA_A' = -AMPA_A/AMPA_tau1
    AMPA_B' = -AMPA_B/AMPA_tau2
    NMDA_A' = -NMDA_A/NMDA_tau1
    NMDA_B' = -NMDA_B/NMDA_tau2
}

NET_RECEIVE(weight (uS)) {
    AMPA_A = AMPA_A + 1 / (1 + r_na) * weight * AMPA_factor
    AMPA_B = AMPA_B + 1 / (1 + r_na) * weight * AMPA_factor
    NMDA_A = NMDA_A + r_na / (1 + r_na) * weight * NMDA_factor
    NMDA_B = NMDA_B + r_na / (1 + r_na) * weight * NMDA_factor
}
