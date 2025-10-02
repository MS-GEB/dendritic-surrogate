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

## Added gradient computation (Gan He et al., 2025)

ENDCOMMENT

NEURON {
    POINT_PROCESS Exp2Syn_inh
    RANGE GABA_tau1, GABA_tau2, GABA_e
    NONSPECIFIC_CURRENT i

    RANGE GABA_g
    RANGE didv, pure_i
    RANGE w
}

UNITS {
    (nA) = (nanoamp)
    (mV) = (millivolt)
    (uS) = (microsiemens)
}

PARAMETER {
    GABA_tau1 = 0.1 (ms) <1e-9,1e9>
    GABA_tau2 = 10  (ms) <1e-9,1e9>
    GABA_e = -80    (mV)
    w = 1
}

ASSIGNED {
    v   (mV)
    i   (nA)
    GABA_g  (uS)
    GABA_factor
    didv pure_i
}

STATE {
    GABA_A  (uS)
    GABA_B  (uS)
}

INITIAL {
    LOCAL GABA_tp
    if (GABA_tau1 / GABA_tau2 > 0.9999) {
        GABA_tau1 = 0.9999 * GABA_tau2
    }
    if (GABA_tau1 / GABA_tau2 < 1e-9) {
        GABA_tau1 = GABA_tau2 * 1e-9
    }
    GABA_A = 0
    GABA_B = 0
    GABA_tp = (GABA_tau1 * GABA_tau2) / (GABA_tau2 - GABA_tau1) * log(GABA_tau2 / GABA_tau1)
    GABA_factor = -exp(-GABA_tp / GABA_tau1) + exp(-GABA_tp / GABA_tau2)
    GABA_factor = 1 / GABA_factor
}

BREAKPOINT {
    SOLVE states METHOD cnexp
    GABA_g = GABA_B - GABA_A
    pure_i = GABA_g * (GABA_e - v)
    i = fabs(w) * -pure_i
    didv = fabs(w) * (-GABA_g)
}

DERIVATIVE states {
    GABA_A' = -GABA_A / GABA_tau1
    GABA_B' = -GABA_B / GABA_tau2
}

NET_RECEIVE(weight (uS)) {
    GABA_A = GABA_A + weight * GABA_factor
    GABA_B = GABA_B + weight * GABA_factor
}
