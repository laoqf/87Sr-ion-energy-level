# sr87_intermediate_B_app.py

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from sympy import Rational
from sympy.physics.wigner import clebsch_gordan
from fractions import Fraction


from matplotlib import colormaps as cmaps
import matplotlib.colors as mcolors

st.set_page_config(layout="wide")

# Fixed B range and probability cutoff
B_MIN = 0.0    # Gauss
B_MAX = 500.0  # Gauss

if "B_main" not in st.session_state:
    st.session_state.B_main = 10.0  # default B

def _sync_from_slider():
    # when slider moves, update shared B_main and input box
    st.session_state.B_main = st.session_state.B_slider
    st.session_state.B_input = st.session_state.B_slider

def _sync_from_input():
    # when number input changes, update shared B_main and slider
    st.session_state.B_main = float(st.session_state.B_input)
    st.session_state.B_slider = float(st.session_state.B_input)


PROB_EPS = 1e-12   # show all components with prob > 1e-6

# ==========================================================
# Constants (in frequency units: E/h, Hz)
# ==========================================================

h = 6.626_070_15e-34      # J/Hz from CODATA 2022
mu_B = 9.274_010_0657e-24 #J/T from CODATA 2022
mu_N = 5.050_783_7393e-27 #J/T from CODATA 2022
mu_B = mu_B/h/10000  # Hz/G  (E/h units) since the erengy here is frequency
mu_N = mu_N/h/10000
#mu_B = 1.399624e6 # value in Hz/G  (E/h units)

# 87Sr nuclear spin and nuclear g-factor (in units of μ_N)
I_Sr87 = 9/2
gI_Sr87 = -131.7712e-6               # gI = mu_I*(1-sigma)/mu_B from the experiement data : Phys. Rev. Lett. 135, 193001 – Published 3 November, 2025
gI_Sr87 = gI_Sr87*mu_B/mu_N


# Speed of light (for wavelength calculation)
c_light = 299_792_458.0  # m/s

# ==========================================================
# 87Sr+ level data (A, B in MHz; you can tweak these)
# ==========================================================

def lande_gJ(L, S, J, gL=1.0, gS=2.002_319_304_362_56):
    L = float(L); S = float(S); J = float(J)
    den = 2 * J * (J + 1)
    term1 = (J * (J + 1) - S * (S + 1) + L * (L + 1)) / den
    term2 = (J * (J + 1) + S * (S + 1) - L * (L + 1)) / den
    return gL * term1 + gS * term2

sr87_data = {
    "5S1/2": {
        "I": I_Sr87,
        "L": 0,
        "S": 1/2,
        "J": 1/2,
        "A_hfs": -1000.5e6,               # exp
        "B_hfs": 0.0,                     # J = 1/2 → no quadrupole term
        "gI": gI_Sr87,               # gI = mu_I*(1-sigma)/mu_B from the experiement data : Phys. Rev. Lett. 135, 193001 – Published 3 November, 2025
        "E0": 0.0,                        # define ground as zero
        "Emove": 0.0,
    },
    "5P1/2": {
        "I": I_Sr87,
        "L": 1,
        "S": 1/2,
        "J": 1/2,
        "A_hfs": -178.4e6,                # cal
        "B_hfs": 0.0,                     # J = 1/2 → no quadrupole term
        "gI": gI_Sr87,                    #exp
        "E0": -59e6 + 23715.19 * c_light * 100,   # cm^-1 → Hz
        "Emove": 710.95e12,
        "Emove": 0,
    },
    "5P3/2": {
        "I": I_Sr87,
        "L": 1,
        "S": 1/2,
        "J": 3/2,
        "A_hfs": -36.0e6,                 #exp
        "B_hfs": 88.5e6,                  #exp
        "gI": gI_Sr87,                    #exp
        "E0": -56e6 + 24516.65 * c_light * 100,
        "Emove": 734.974e12,
        "Emove": 0,
    },
    "4D3/2": {
        "I": I_Sr87,
        "L": 2,
        "S": 1/2,
        "J": 3/2,
        "A_hfs": -47.365e6,               #cal
        "B_hfs": 38.2e6,                  #cal
        "gI": gI_Sr87,                    #exp
        "E0": -206e6 + 14555.90 * c_light * 100,
        "Emove": 436.37e12,
        "Emove": 0,
    },
    "4D5/2": {
        "I": I_Sr87,
        "L": 2,
        "S": 1/2,
        "J": 5/2,
        "A_hfs": 2.1743e6,                #exp
        "B_hfs": 49.11e6,                 #exp
        "gI": gI_Sr87,                    #exp
        "E0": -207.7e6 + 14836.24 * c_light * 100,
        "Emove": 444.77e12,
        "Emove": 0,
    },
}


# create symmetric diverging colormap for mF
# modern Matplotlib API
div_map = cmaps.get_cmap("coolwarm_r")

def color_for_mF(mF, mF_max=7):
    """
    symmetric color map: mF=-mF_max → blue, mF=0 → white, mF=+mF_max → red
    Output formatted as hex for Plotly.
    """
    # normalize from [-mF_max, mF_max] → [0, 1]
    t = (mF + mF_max) / (2 * mF_max)
    r, g, b, _ = div_map(t)
    return mcolors.to_hex((r, g, b))

for name, d in sr87_data.items():
    d["gJ"] = lande_gJ(d["L"], d["S"], d["J"])

# ==========================================================
# Angular momentum matrices in |m=-j,...,+j> basis (ħ=1)
# ==========================================================

def J_matrices(j):
    """
    Return Jx, Jy, Jz and m values in basis |m=j, j-1, ..., -j>.
    """
    dim = int(2 * j + 1)
    m_vals = np.arange(j, -j-1, -1, dtype=float)  # descending

    Jp = np.zeros((dim, dim), dtype=complex)
    Jm = np.zeros((dim, dim), dtype=complex)

    for row, m in enumerate(m_vals):
        # J+|m> = sqrt(j(j+1)-m(m+1)) |m+1>
        if m < j:
            amp = np.sqrt(j * (j + 1) - m * (m + 1))
            Jp[row - 1, row] = amp
        # J-|m> = sqrt(j(j+1)-m(m-1)) |m-1>
        if m > -j:
            amp = np.sqrt(j * (j + 1) - m * (m - 1))
            Jm[row + 1, row] = amp

    Jx = 0.5 * (Jp + Jm)
    Jy = -0.5j * (Jp - Jm)
    Jz = np.diag(m_vals)
    return Jx, Jy, Jz, m_vals

# ==========================================================
# Build IJ basis and operators
# ==========================================================

def build_IJ_operators(I, J):
    """
    Returns:
      basis: list[(mI, mJ)]
      IJ_dot, Iz_full, Jz_full
    in |mI>⊗|mJ> basis.
    """
    Ix, Iy, Iz_single, mI_vals = J_matrices(I)
    Jx, Jy, Jz_single, mJ_vals = J_matrices(J)

    dimI = Ix.shape[0]
    dimJ = Jx.shape[0]
    IdI = np.eye(dimI)
    IdJ = np.eye(dimJ)

    Ix_full = np.kron(Ix, IdJ)
    Iy_full = np.kron(Iy, IdJ)
    Iz_full = np.kron(Iz_single, IdJ)

    Jx_full = np.kron(IdI, Jx)
    Jy_full = np.kron(IdI, Jy)
    Jz_full = np.kron(IdI, Jz_single)

    IJ_dot = Ix_full @ Jx_full + Iy_full @ Jy_full + Iz_full @ Jz_full

    basis = []
    for mI in mI_vals:
        for mJ in mJ_vals:
            basis.append((float(mI), float(mJ)))

    return basis, IJ_dot, Iz_full, Jz_full

# ==========================================================
# Hyperfine + Zeeman Hamiltonian (E/h, Hz)
# ==========================================================

def hamiltonian_intermediate_B(level_name, B_G):
    """
    Full hyperfine (A,B) + Zeeman (gJ,gI) Hamiltonian in the
    |I J mI mJ> basis. Returns (H, basis).
    H has units of Hz (E/h).
    """
    d = sr87_data[level_name]
    I = d["I"]
    J = d["J"]
    A = d["A_hfs"]   # Hz
    Bq = d["B_hfs"]  # Hz
    gJ = d["gJ"]
    gI = d["gI"]

    basis, IJ_dot, Iz_full, Jz_full = build_IJ_operators(I, J)
    dim = len(basis)

    H = np.zeros((dim, dim), dtype=complex)

    # Hyperfine A term: A I·J
    H += A * IJ_dot

    # Hyperfine B term (if applicable)
    if Bq != 0.0 and I >= 1.0 and J >= 1.0:
        Ival = float(I)
        Jval = float(J)
        denom = 2 * Ival * (2 * Ival - 1) * Jval * (2 * Jval - 1)
        IJ2 = IJ_dot @ IJ_dot
        termB = (3 * IJ2 + 1.5 * IJ_dot
                 - Ival * (Ival + 1) * Jval * (Jval + 1) * np.eye(dim)) / denom
        H += Bq * termB

    # Zeeman: μB gJ Jz B + μN gI Iz B
    H += mu_B * B_G * (gJ * Jz_full)
    H += mu_N * B_G * (gI * Iz_full)

    return H, basis

# ==========================================================
# Diagonalization by mF block
# ==========================================================

def diagonalize_by_mF(level_name, B_G):
    """
    Diagonalize H in blocks of fixed mF = mI+mJ.
    Returns:
      eig_data: dict[mF] -> (e_vals, e_vecs, idx_list)
      basis: list[(mI,mJ)]
    """
    H, basis = hamiltonian_intermediate_B(level_name, B_G)

    blocks = {}
    for idx, (mI, mJ) in enumerate(basis):
        mF = mI + mJ
        blocks.setdefault(mF, []).append(idx)

    eig_data = {}
    for mF, idx_list in blocks.items():
        idx_list = sorted(idx_list)
        H_block = H[np.ix_(idx_list, idx_list)]
        e_vals, e_vecs = np.linalg.eigh(H_block)  # ascending
        eig_data[mF] = (e_vals, e_vecs, idx_list)

    return eig_data, basis

# ==========================================================
# Build |F,mF> vectors in IJ basis using Clebsch–Gordan
# ==========================================================

def F_mF_vectors(I, J, basis):
    """
    Returns dict[(F,mF)] -> normalized vector (numpy array) in IJ basis.
    """
    index_of = {(mI, mJ): k for k, (mI, mJ) in enumerate(basis)}

    Isp = Rational(int(2 * I), 2)
    Jsp = Rational(int(2 * J), 2)

    vecs = {}
    F_min = abs(I - J)
    F_max = I + J

    for F in np.arange(F_min, F_max + 1):
        Fsp = Rational(int(2 * F), 2)
        for mF in np.arange(-F, F + 1):
            mFsp = Rational(int(2 * mF), 2)
            v = np.zeros(len(basis), dtype=complex)
            for (mI, mJ), k in index_of.items():
                if abs(mI + mJ - mF) > 1e-9:
                    continue
                mIsp = Rational(int(2 * mI), 2)
                mJsp = Rational(int(2 * mJ), 2)
                coeff = clebsch_gordan(Isp, Jsp, Fsp, mIsp, mJsp, mFsp)
                if coeff != 0:
                    v[k] = complex(coeff.evalf())
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v /= nrm
            vecs[(float(F), float(mF))] = v

    return vecs

def find_derivative_zero(B_vals, dnu_vals):
    """
    Given arrays B_vals and dnu(B) in GHz,
    return B* where derivative crosses zero.
    If no zero exists, return None.
    """
    # derivative d(Δν)/dB
    deriv = np.gradient(dnu_vals, B_vals)

    # find indices where derivative changes sign
    sign_changes = np.where(np.diff(np.sign(deriv)) != 0)[0]

    if len(sign_changes) == 0:
        return None  # no extremum

    i = sign_changes[0]

    # linear interpolation between B[i], B[i+1]
    B1, B2 = B_vals[i], B_vals[i+1]
    d1, d2 = deriv[i], deriv[i+1]

    # solve d1 + (d2–d1)*t = 0
    t = -d1 / (d2 - d1)
    B_star = B1 + t * (B2 - B1)

    return B_star

# ----------------------------
# Make an interpolation function
# ----------------------------
def derivative_at(B,B_vals, dnu_vals):
    """
    Return derivative d(Δν)/dB evaluated at arbitrary B,
    using linear interpolation of the derivative array.
    """

    # derivative d(Δν)/dB
    deriv = np.gradient(dnu_vals, B_vals)

    # print(deriv)

    return np.interp(B, B_vals, deriv)

# ==========================================================
# Cached calculation of energy branches vs B
# ==========================================================

@st.cache_data
def compute_branches(level_name, B_min, B_max, n_B):
    B_vals = np.linspace(B_min, B_max, n_B)
    eig0, basis = diagonalize_by_mF(level_name, B_vals[0])
    mF_list = sorted(eig0.keys())

    # For each mF and branch index, store E(B)
    branches = {}
    for mF in mF_list:
        n_states = len(eig0[mF][0])
        branches[mF] = np.zeros((n_states, n_B), dtype=float)

    for i, B in enumerate(B_vals):
        eig_data, _ = diagonalize_by_mF(level_name, B)
        for mF in mF_list:
            e_vals, _, _ = eig_data[mF]
            branches[mF][:, i] = e_vals / 1e6  # MHz

    return B_vals, mF_list, branches, basis

@st.cache_data
def cached_F_vectors(level_name):
    d = sr87_data[level_name]
    I = d["I"]; J = d["J"]
    basis, _, _, _ = build_IJ_operators(I, J)
    Fvecs = F_mF_vectors(I, J, basis)
    return basis, Fvecs

@st.cache_data
def transition_frequency_vs_B(level1, mF1, n1,
                              level2, mF2, n2,
                              B_min, B_max, n_B=200):
    """
    Compute transition frequency Δν(B) between two eigenstates
    (level1, mF1, n1) and (level2, mF2, n2).

    Returns:
        B_vals : array of B (Gauss)
        dnu_GHz : array of |E2(B)-E1(B)|/h in GHz
    """
    B_vals = np.linspace(B_min, B_max, n_B)
    dnu_vals = []

    for B in B_vals:
        eig1, _ = diagonalize_by_mF(level1, B)
        eig2, _ = diagonalize_by_mF(level2, B)

        E01 = sr87_data[level1]["E0"]  # Hz
        E02 = sr87_data[level2]["E0"]  # Hz

        e_vals1, _, _ = eig1[mF1]
        e_vals2, _, _ = eig2[mF2]

        E1 = E01 + e_vals1[n1]   # total energy in Hz
        E2 = E02 + e_vals2[n2]

        dnu_vals.append(abs(E2 - E1) / 1e9)  # GHz

    return B_vals, np.array(dnu_vals)


# ==========================================================
# Formatting helpers: Unicode fractions, Dirac notation
# ==========================================================

_superscript_map = {
    "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³",
    "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
}
_subscript_map = {
    "-": "₋", "0": "₀", "1": "₁", "2": "₂", "3": "₃",
    "4": "₄", "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉",
}

def to_superscript(s: str) -> str:
    return "".join(_superscript_map.get(ch, ch) for ch in s)

def to_subscript(s: str) -> str:
    return "".join(_subscript_map.get(ch, ch) for ch in s)

def unicode_fraction(x: float) -> str:
    """
    Convert e.g. 4.5 -> '⁹⁄₂', -3.5 -> '⁻⁷⁄₂'.
    """
    frac = Fraction(x).limit_denominator()
    n = frac.numerator
    d = frac.denominator
    sign = "-" if n < 0 else ""
    n_abs = abs(n)

    num_str = to_superscript(sign + str(n_abs))
    den_str = to_subscript(str(d))
    return f"{num_str}⁄{den_str}"

def frac_str_ascii(x: float) -> str:
    """
    ASCII version: 4.5 -> '9/2'
    """
    frac = Fraction(x).limit_denominator()
    return f"{frac.numerator}/{frac.denominator}"


# ==========================================================
# Streamlit UI
# ==========================================================

st.title("87Sr⁺ hyperfine levels – intermediate B (matrix diagonalization)")

st.markdown(
    r"""
This app diagonalizes the full hyperfine + Zeeman Hamiltonian
in the uncoupled basis $|I J m_I m_J\rangle$ and shows:

* SPD-style short-line energy diagram vs $m_F$
* Eigenstates ordered by energy in each $m_F$ block
* Decomposition of each eigenstate in:
  * $|I J m_I m_J\rangle$ basis (with Unicode fractions)
  * $|F,m_F\rangle$ basis (with integer $F,m_F$)

The Hamiltonian is expressed by
$$
\frac{H}{\mathrm{h}} = 
\underbrace{A_{\mathrm{hfs}} \, \hat{I} \cdot \hat{J} + B_{\mathrm{hfs}} \, \frac{3(\hat{I} \cdot \hat{J})^2 + \frac{3}{2} (\hat{I} \cdot \hat{J}) - I(I+1)J(J+1) }{2I(I-1)J(J-1)}}_{\text{Hyperfine}}
+
\underbrace{\frac{\mu_B}{\mathrm{h}}\, g_J \, \hat{J} \cdot \hat{B} + \frac{\mu_N}{\mathrm{h}}\, g_I \, \hat{I} \cdot \hat{B}}_{\text{Zeeman}}
$$
"""
)

# ==========================================================
# MAIN FIGURE: S1/2, P1/2, P3/2, D3/2, D5/2
# Short horizontal lines at a single B, color = mF.
# x-axis is just for visual separation.
# ==========================================================
st.subheader("Hyperfine levels at a single B (color = mF)")

all_manifolds = ["5S1/2", "5P1/2", "5P3/2", "4D3/2", "4D5/2"]

selected_manifolds = st.multiselect(
    "Select manifolds to display",
    options=all_manifolds,
    default=all_manifolds,
)

if not selected_manifolds:
    st.info("Select at least one manifold above.")
else:
    st.markdown("### Magnetic field B (Gauss)")

    col_slider, col_input = st.columns([4, 1])

    col_left, col_right = st.columns([1.2, 1])

    with col_slider:
        st.slider(
            "B slider input",
            B_MIN,
            B_MAX,
            st.session_state.B_main,
            step=1.0,
            key="B_slider",
            on_change=_sync_from_slider,
        )

    with col_input:
        st.number_input(
            "B value input",
            min_value=B_MIN,
            max_value=B_MAX,
            value=st.session_state.B_main,
            step=0.1,
            format="%.3f",
            key="B_input",
            on_change=_sync_from_input,
        )

    B_plot = st.session_state.B_main
    st.write(f"Current B: **{B_plot:.3f} G**")

    with col_left:
        # ---- Main Figure BLOCK ----
        fig = go.Figure()

        # geometry
        group_spacing = 1.2   # distance between manifolds on x-axis
        x_scale_mF   = 0.08   # offset inside each manifold by mF
        half_width   = 0.03   # half length of each short line

        # collect all mF values across selected manifolds
        all_mF_set = set()
        eig_by_manifold = {}

        for manifold in selected_manifolds:
            eig_data, basis = diagonalize_by_mF(manifold, B_plot)
            eig_by_manifold[manifold] = (eig_data, basis)
            all_mF_set.update(eig_data.keys())

        sorted_mF = sorted(all_mF_set)

        # list of all states for later transition selection
        available_states = []   # (label, (manifold, mF, n))

        for m_index, manifold in enumerate(selected_manifolds):
            E0 = sr87_data[manifold]["E0"]  # Hz
            Emove = sr87_data[manifold]["Emove"]  # Hz
            I_val = sr87_data[manifold]["I"]
            J_val = sr87_data[manifold]["J"]
            I_str = frac_str_ascii(I_val)
            J_str = frac_str_ascii(J_val)

            eig_data, basis = eig_by_manifold[manifold]
            x_group = m_index * group_spacing

            for mF in sorted(eig_data.keys()):
                e_vals, _, _ = eig_data[mF]
                color = color_for_mF(mF)

                for n, E_hfs in enumerate(e_vals):
                    E_tot_GHz = (E0 + E_hfs) / 1e9 
                    

                    x_center = x_group + mF * x_scale_mF
                    x0 = x_center - half_width
                    x1 = x_center + half_width

                    label = (
                        f"{manifold}: J={J_str}, mF={int(round(mF))}, n={n}"
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[E_tot_GHz - Emove/1e9, E_tot_GHz - Emove/1e9], # not to scale show
                            # y=[E_tot_GHz, E_tot_GHz], # real scale show
                            mode="lines",
                            line=dict(color=color, width=2),
                            name=label,
                            showlegend=True,
                            hovertemplate=(
                                f"{manifold}"
                                f"<br>I={I_str}, J={J_str}"
                                f"<br>mF = {mF:+.1f}, n = {n}"
                                f"<br>B = {B_plot:.3f} G"
                                # f"<br>E = {E_tot_GHz:.6f} GHz" # not to scale show
                                "<br>E = %{y:.6f} GHz" # real scale show
                                "<extra></extra>"
                            ),
                        )
                    )

                    # store this state for transition selection
                    available_states.append(
                        (label, (manifold, float(mF), int(n)))
                    )

        # mapping from label string -> (manifold, mF, n)
        label_to_state = {lbl: state for lbl, state in available_states}

        # ##############################################################
        # # TEST ARROW WITH YOUR REAL Y-SCALE
        # ##############################################################

        # # Use approximate coordinates based on your screenshot:
        # # bottom manifold visible around 0 GHz
        # # next manifold around 700k GHz

        # x_start = 0
        # y_start = 0       # bottom line

        # x_end = 0
        # y_end = 700000    # 700k GHz

        # fig.add_annotation(
        #     x=x_end,
        #     y=y_end,
        #     ax=x_start,
        #     ay=y_start,
        #     xref="x",  yref="y",   # head in data coords
        #     axref="x", ayref="y",  # tail in data coords  <-- important
        #     text="Here is an arrow",
        #     showarrow=True,
        #     arrowhead=3,
        #     arrowsize=2,
        #     arrowwidth=4,
        #     arrowcolor="red",
        #     opacity=1
        # )
        # ##############################################################

        

        # ----------------------------------------------------------------------
        # Draw arrow on MAIN FIGURE when two states are selected
        # ----------------------------------------------------------------------
        if "tr_state1" in st.session_state and "tr_state2" in st.session_state:
            sel1 = st.session_state["tr_state1"]
            sel2 = st.session_state["tr_state2"]

            if sel1 != "None" and sel2 != "None":

                man1, mF1, n1 = label_to_state[sel1]
                man2, mF2, n2 = label_to_state[sel2]

                # --- compute energies for arrow endpoints ---
                eig1, _ = eig_by_manifold[man1]
                eig2, _ = eig_by_manifold[man2]

                E_tot1 = (sr87_data[man1]["E0"] + eig1[mF1][0][n1]) / 1e9  # GHz
                E_tot2 = (sr87_data[man2]["E0"] + eig2[mF2][0][n2]) / 1e9

                # --- compute x-positions of the short lines in same way as main plot ---
                idx1 = selected_manifolds.index(man1)
                x_group1 = idx1 * group_spacing
                x_pos1 = x_group1 + mF1 * x_scale_mF

                idx2 = selected_manifolds.index(man2)
                x_group2 = idx2 * group_spacing
                x_pos2 = x_group2 + mF2 * x_scale_mF

                # --- Add arrow via annotations ---
                fig.add_annotation(
                    x=x_pos2,
                    y=E_tot2 - sr87_data[man2]["Emove"] / 1e9, # match not to scale
                    # y=E_tot2, 
                    ax=x_pos1,
                    ay=E_tot1 - sr87_data[man1]["Emove"] / 1e9, # match to scale
                    # ay=E_tot1,
                    xref="x",  yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=3,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor="red",
                    opacity=0.9,
                )

        
        fig.update_layout(
            title=f"87Sr⁺ hyperfine levels at B = {B_plot:.3f} G (color = mF)",
            xaxis=dict(visible=False),
            # yaxis=dict(visible=False), # not to scale
            yaxis_title="Energy (GHz, E/h)", # to scale
            legend_title_text="mF",
            height=480,
            margin=dict(l=40, r=20, t=60, b=40),
        )

        st.plotly_chart(fig, width="stretch", key="main_figure_plot")

        # ==========================================================
        # Eigenstate inspection (inside a pullbox / expander)
        # ==========================================================
        st.subheader("Inspect a single eigenstate (basis decomposition)")
    #with st.expander("Inspect a single eigenstate (basis decomposition)", expanded=False):

        # choose which manifold to inspect
        manifolds_for_inspect = ["5S1/2", "5P1/2", "5P3/2", "4D3/2", "4D5/2"]
        manifold_inspect = st.selectbox(
            "Manifold to inspect",
            options=manifolds_for_inspect,
            index=manifolds_for_inspect.index("5S1/2"),
        )

        # B slider for this inspection
        B_inspect = st.slider(
            "B for detailed eigenstate decomposition (G)",
            0.0,
            300.0,
            10.0,
            step=1.0,
            key="B_inspect_slider",
        )

        eig_data_inspect, basis = diagonalize_by_mF(manifold_inspect, B_inspect)
        basis_F, Fvecs = cached_F_vectors(manifold_inspect)

        assert basis == basis_F

        mF_choice = st.selectbox(
            "Choose mF block",
            options=[float(x) for x in sorted(eig_data_inspect.keys())],
        )

        e_vals, e_vecs, idx_list = eig_data_inspect[mF_choice]
        n_states_block = len(e_vals)

        st.write(
            f"mF = {mF_choice:+.1f} block, dimension = {n_states_block}, "
            f"B = {B_inspect:.1f} G"
        )

        df_e = pd.DataFrame(
            {
                "state_index": list(range(n_states_block)),
                "E_MHz": e_vals / 1e6,
            }
        )
        st.dataframe(df_e, hide_index=True)

        state_index = st.number_input(
            "Select state index in this mF block",
            min_value=0,
            max_value=n_states_block - 1,
            value=0,
            step=1,
        )

        # reconstruct full eigenvector in IJ basis
        psi_block = e_vecs[:, state_index]
        psi_full = np.zeros(len(basis), dtype=complex)
        for local_i, global_i in enumerate(idx_list):
            psi_full[global_i] = psi_block[local_i]

        # ===== Decomposition in |I J mI mJ> basis (same as before, but using manifold_inspect) =====
        st.markdown("### Decomposition in $|I J m_I m_J\\rangle$ basis")

        components = []
        ij_probs_for_latex = []

        for k, (mI, mJ) in enumerate(basis):
            amp = psi_full[k]
            prob = float(np.abs(amp) ** 2)
            if prob < PROB_EPS:
                continue
            amp_str = f"{amp.real:+.4f} {amp.imag:+.4f}i"
            mI_u = unicode_fraction(mI)
            mJ_u = unicode_fraction(mJ)
            label_ket = f"|{mI_u}, {mJ_u}⟩"
            components.append(
                {
                    "state": label_ket,
                    "probability": prob,
                    "amplitude": amp_str,
                }
            )
            ij_probs_for_latex.append((prob, mI, mJ))

        components.sort(key=lambda r: r["probability"], reverse=True)
        df_components = pd.DataFrame(components[:12])
        if len(df_components) == 0:
            st.write("All components are below the probability threshold.")
        else:
            st.table(df_components)

        ij_probs_for_latex.sort(reverse=True, key=lambda x: x[0])
        latex_terms_ij = []
        for prob, mI, mJ in ij_probs_for_latex[:4]:
            coeff = np.sqrt(prob)
            mI_frac = frac_str_ascii(mI)
            mJ_frac = frac_str_ascii(mJ)
            term = rf"{coeff:.3f}\,\left|{mI_frac},{mJ_frac}\right\rangle"
            latex_terms_ij.append(term)

        if latex_terms_ij:
            latex_ij = r"\left|\psi\right\rangle \approx " + " + ".join(latex_terms_ij)
            st.latex(latex_ij)

        # ===== Decomposition in |F,mF> basis =====
        st.markdown("### Decomposition in $|F,m_F\\rangle$ basis")

        I_val = sr87_data[manifold_inspect]["I"]
        J_val = sr87_data[manifold_inspect]["J"]
        F_min = abs(I_val - J_val)
        F_max = I_val + J_val

        FmF_rows = []
        FmF_probs_for_latex = []

        for F in np.arange(F_min, F_max + 1):
            key = (float(F), float(mF_choice))
            if key not in Fvecs:
                continue
            vF = Fvecs[key]
            amp = np.vdot(vF, psi_full)
            prob = float(np.abs(amp) ** 2)
            if prob < PROB_EPS:
                continue
            amp_str = f"{amp.real:+.4f} {amp.imag:+.4f}i"
            F_int = int(round(F))
            mF_int = int(round(mF_choice))
            FmF_rows.append(
                {
                    "F": F_int,
                    "mF": mF_int,
                    "probability": prob,
                    "amplitude": amp_str,
                }
            )
            FmF_probs_for_latex.append((prob, F_int))

        FmF_rows.sort(key=lambda r: r["probability"], reverse=True)
        df_FmF = pd.DataFrame(FmF_rows)
        if len(df_FmF) == 0:
            st.write("No significant $|F,m_F\\rangle$ components above threshold.")
        else:
            st.table(df_FmF)

        FmF_probs_for_latex.sort(reverse=True, key=lambda x: x[0])
        latex_terms_F = []
        mF_int = int(round(mF_choice))
        for prob, F_int in FmF_probs_for_latex[:4]:
            coeff = np.sqrt(prob)
            term = rf"{coeff:.3f}\,\left|F={F_int},\,m_F={mF_int}\right\rangle"
            latex_terms_F.append(term)

        if latex_terms_F:
            latex_F = r"\left|\psi\right\rangle \approx " + " + ".join(latex_terms_F)
            st.latex(latex_F)


    with col_right:
        # ------------------------------------------------------
        # Transition frequency & wavelength between two states
        # ------------------------------------------------------
        st.subheader("Transition frequency between two eigenstates")

        if available_states:

            # labels = ["None"] + [lbl for lbl, _ in available_states]

            labels = [lbl for lbl, _ in available_states]
            label_to_state = {lbl: state for lbl, state in available_states}

            #default_idx2 = 1 if len(labels) > 1 else 0

            sel1 = st.selectbox(
                "State 1", options=labels, index=0, key="tr_state1"
            )
            sel2 = st.selectbox(
                "State 2", options=labels, index=0, key="tr_state2"
            )


            if sel1 != "None" and sel2 != "None":
                if sel1 and sel2:
                    man1, mF1, n1 = label_to_state[sel1]
                    man2, mF2, n2 = label_to_state[sel2]

                    # store in session_state so the LEFT plot can draw the arrow
                    st.session_state["state1"] = (man1, mF1, n1)
                    st.session_state["state2"] = (man2, mF2, n2)

                    # energies at CURRENT B_plot using cached spectra
                    def total_energy_at_B(man, mF, n):
                        eig_data, _ = eig_by_manifold[man]
                        e_vals, _, _ = eig_data[mF]
                        return sr87_data[man]["E0"] + e_vals[n]  # Hz

                    # ---------- B slider for this transition section ----------
                    # B_tr = st.slider(
                    #     "B value (Gauss)",
                    #     B_MIN,
                    #     B_MAX,
                    #     300.0,   # or choose a default like 100.0
                    #     step=1.0,
                    #     key="B_tr_slider",
                    # )
                    B_tr = B_plot

                    # ---------- energies at THIS B_tr ----------
                    def total_energy_at(manifold, mF, n, B_G):
                        eig_data, _ = diagonalize_by_mF(manifold, B_G)
                        e_vals, _, _ = eig_data[mF]
                        return sr87_data[manifold]["E0"] + e_vals[n]  # Hz

                    E1 = total_energy_at(man1, mF1, n1, B_tr)
                    E2 = total_energy_at(man2, mF2, n2, B_tr)

                    dE = E2 - E1
                    dnu = abs(dE)           # Hz
                    dnu_GHz = dnu / 1e9
                    dnu_THz = dnu / 1e12

                    wavelength_m = c_light / dnu if dnu != 0 else float("inf")
                    wavelength_nm = wavelength_m * 1e9

                    st.markdown(
                        f"""
                    **{sel1} → {sel2} at B = {B_tr:.3f} G**

                    - Δν = {dnu:_.3f} Hz  
                    

                    - λ = {wavelength_nm:_.9f} nm
                    """
                    )
                    with st.expander("Transition frequency vs magnetic field", expanded=False):
                        # ---------- Δν(B) curve from 0 to B_tr ----------
                        # st.markdown("### Transition frequency vs magnetic field")

                        B_vals, dnu_GHz_curve = transition_frequency_vs_B(
                            man1, mF1, n1,
                            man2, mF2, n2,
                            B_MIN, B_MAX,
                        )

                        # ------------------------------------------------
                        # Find derivative-zero point
                        # ------------------------------------------------
                        B_star = find_derivative_zero(B_vals, dnu_GHz_curve)

                        fig_tr = go.Figure()
                        fig_tr.add_trace(
                            go.Scatter(
                                x=B_vals,
                                y=dnu_GHz_curve,
                                mode="lines",
                                name="Δν(B)",
                            )
                        )

                        # mark extremum
                        if B_star is not None:
                            # compute Δν(B_star)
                            # simply interpolate the curve
                            dnu_star = np.interp(B_star, B_vals, dnu_GHz_curve)

                            fig_tr.add_trace(
                                go.Scatter(
                                    x=[B_star],
                                    y=[dnu_star],
                                    mode="markers",
                                    marker=dict(size=10, color="red"),
                                    name=f"B* = {B_star:.3f} G",
                                )
                            )

                            st.markdown(
                                f"""
                                    ### **Extremum found**
                                    - B\\* = **{B_star:.3f} G**
                                    - Δν(B\\*) = **{dnu_star:.6f} GHz**
                                    - λ(B\\*) = **{(c_light/(dnu_star*1e9)*1e9):.3f} nm**
                                    """
                            )
                        else:
                            st.info("No extremum found (monotonic curve).")

                        devB = derivative_at(B_plot, B_vals, dnu_GHz_curve) * 1e9 # from GHz/G to Hz/G

                        st.markdown(
                            f"""Derivate at B  = **{B_plot:.3f} G**  is **{devB:.3f} Hz/G**
                            """
                        )

                        fig_tr.update_layout(
                            title=f"Δν(B) for {sel1} → {sel2}",
                            xaxis_title="B (Gauss)",
                            yaxis_title="Δν (GHz)",
                            height=450,
                            margin=dict(l=40, r=20, t=40, b=40),
                        )
                        st.plotly_chart(fig_tr, width="stretch", key="transition_plot")

                        

            else:
                st.info("No states available for the current selection.")
        else:
            st.info("Select two states to compute a transition.")


        st.subheader("Breit-Rabi Function")
        # --- controls ---

        level_name = st.selectbox(
            "Select manifold",
            options=list(sr87_data.keys()),
            index=4,  # default: 4D5/2
        )

        col1, col2 = st.columns(2)
        with col1:
            B_min, B_max = st.slider(
                "Magnetic field range B (Gauss)",
                0.0,
                300.0,
                (0.0, 20.0),
                step=1.0,
            )
        with col2:
            n_B = st.slider("Number of B points", 11, 201, 61, step=10)

        # --- compute and plot branches ---

        B_vals, mF_list, branches, basis_plot = compute_branches(
            level_name, B_min, B_max, n_B
        )

        fig = go.Figure()

        # simple color palette by mF index
        palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf",
        ]

        for mF_index, mF in enumerate(mF_list):
            E_mF = branches[mF]  # shape (n_states, n_B)
            n_states = E_mF.shape[0]
            # color = palette[mF_index % len(palette)]
            color = color_for_mF(mF)
            for n in range(n_states):
                label = f"mF={mF:+.1f}, n = {n}"
                fig.add_trace(
                    go.Scatter(
                        x=B_vals,
                        y=E_mF[n, :],
                        mode="lines",
                        line=dict(width=1.5, color=color),
                        name=label,
                        hovertemplate=f"{label}<br>B = %{{x:.1f}} G<br>E = %{{y:.6f}} MHz<extra></extra>",
                    )
                )

        fig.update_layout(
            title=f"{level_name} hyperfine-Zeeman levels of 87Sr⁺",
            xaxis_title="Magnetic field B (G)",
            yaxis_title="Energy (MHz, E/h)",
            width=900,
            height=600,
        )

        st.plotly_chart(fig, width="stretch")

       
    st.markdown(
        r"""
    Notes:

    * Energies are shown as $E/h$ in MHz.
    * Eigenstates are ordered by eigenvalue within each fixed $m_F$ block.
    * At $B\to 0$ you recover pure $|F,m_F\rangle$.
    * Unicode fractions (e.g. ⁹⁄₂) are used for $m_I,m_J$.
    * The LaTeX decompositions use $\sqrt{p}$ as real coefficients,
    where $p$ is the probability of each component.
    """
    )
