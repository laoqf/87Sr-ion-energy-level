# sr87_intermediate_B_app.py
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from sympy import Rational
from sympy.physics.wigner import clebsch_gordan

# ==========================================================
# Basic constants (frequencies in Hz, B in Gauss)
# ==========================================================

# ==========================================================
# Constants (in frequency units: E/h, Hz)
# ==========================================================

# Bohr and nuclear magnetons divided by h
# mu_B_over_h_T = 13.996_245_55e9   # Hz/T
# mu_N_over_h_T = 7.622_593_285e6   # Hz/T

# mu_B_over_h_G = mu_B_over_h_T / 1e4  # Hz/G
# mu_N_over_h_G = mu_N_over_h_T / 1e4  # Hz/G

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
    },
    "5P1/2": {
        "I": I_Sr87,
        "L": 1,
        "S": 1/2,
        "J": 1/2,
        "A_hfs": -178.4e6,                # cal
        "B_hfs": 0.0,                     # J = 1/2 → no quadrupole term
        "gI": gI_Sr87,                    #exp
        "E0": 23715.19 * c_light * 100,   # cm^-1 → Hz
    },
    "5P3/2": {
        "I": I_Sr87,
        "L": 1,
        "S": 1/2,
        "J": 3/2,
        "A_hfs": -36.0e6,                 #exp
        "B_hfs": 88.5e6,                  #exp
        "gI": gI_Sr87,                    #exp
        "E0": 24516.65 * c_light * 100,
    },
    "4D3/2": {
        "I": I_Sr87,
        "L": 2,
        "S": 1/2,
        "J": 3/2,
        "A_hfs": -47.365e6,               #cal
        "B_hfs": 38.2e6,                  #cal
        "gI": gI_Sr87,                    #exp
        "E0": 14555.90 * c_light * 100,
    },
    "4D5/2": {
        "I": I_Sr87,
        "L": 2,
        "S": 1/2,
        "J": 5/2,
        "A_hfs": 2.1743e6,                #exp
        "B_hfs": 49.11e6,                 #exp
        "gI": gI_Sr87,                    #exp
        "E0": 14836.24 * c_light * 100,
    },
}

# Precompute gJ
for name, d in sr87_data.items():
    d["gJ"] = lande_gJ(d["L"], d["S"], d["J"])

# ==========================================================
# Angular momentum matrices in |m=-j,...,+j> basis (ħ=1)
# ==========================================================

def J_matrices(j):
    """
    Return Jx, Jy, Jz and the list of m values in basis |m=j, j-1,...,-j>.
    """
    dim = int(2 * j + 1)
    m_vals = np.arange(j, -j-1, -1, dtype=float)  # descending

    Jp = np.zeros((dim, dim), dtype=complex)
    Jm = np.zeros((dim, dim), dtype=complex)

    for row, m in enumerate(m_vals):
        # J+ |m> = sqrt(j(j+1)-m(m+1)) |m+1>
        if m < j:
            amp = np.sqrt(j * (j + 1) - m * (m + 1))
            Jp[row - 1, row] = amp
        # J- |m> = sqrt(j(j+1)-m(m-1)) |m-1>
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
      basis: list of (mI, mJ)
      IJ_dot, Iz_full, Jz_full operators in that |mI>⊗|mJ> basis
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
# Hyperfine + Zeeman Hamiltonian (E/h – i.e. frequencies)
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
    Bq = d["B_hfs"] # Hz
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

    # Zeeman: μB gJ Jz B + μN gI Iz B (all divided by h already)
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
    where e_vecs columns are eigenvectors in the subspace defined by idx_list.
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
        e_vals, e_vecs = np.linalg.eigh(H_block)  # sorted ascending
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

# ==========================================================
# Streamlit UI
# ==========================================================

st.title("87Sr⁺ hyperfine levels – intermediate B (matrix diagonalization)")
# st.markdown(
#     """
# This app diagonalizes the full hyperfine + Zeeman Hamiltonian
# in the uncoupled basis $|I J m_I m_J\\rangle$ and shows the
# eigenstates ordered by eigenvalue for each fixed $m_F$.
# You can also inspect each eigenstate as a superposition of

# * $|I J m_I m_J\\rangle$ (uncoupled basis)
# * $|F m_F\\rangle$ (coupled basis via Clebsch–Gordan coefficients).
# """
# )

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
# --- controls ---

level_name = st.selectbox(
    "Select manifold",
    options=list(sr87_data.keys()),
    index=3,  # default: 4D5/2
)

col1, col2 = st.columns(2)
with col1:
    B_min, B_max = st.slider(
        "Magnetic field range B (Gauss)",
        0.0,
        300.0,
        (0.0, 200.0),
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
    color = palette[mF_index % len(palette)]
    for n in range(n_states):
        label = f"mF={mF:+.1f}, state #{n}"
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

# ==========================================================
# Eigenstate inspection at a chosen B
# ==========================================================

st.subheader("Inspect eigenstate at a specific B")

B_inspect = st.slider(
    "B for detailed state decomposition (G)",
    B_min,
    B_max,
    (B_min + B_max) / 2,
    step=1.0,
)

eig_data_inspect, basis = diagonalize_by_mF(level_name, B_inspect)
basis_F, Fvecs = cached_F_vectors(level_name)

# safety: basis used for F-vectors and for H must match
assert basis == basis_F

mF_choice = st.selectbox(
    "Choose mF block", options=[float(x) for x in sorted(eig_data_inspect.keys())]
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

# --- decomposition in |mI,mJ> ---

st.markdown("**Decomposition in $|I J m_I m_J\\rangle$ basis**")

components = []
for k, (mI, mJ) in enumerate(basis):
    amp = psi_full[k]
    prob = float(np.abs(amp) ** 2)
    if prob < 1e-4:
        continue
    amp_str = f"{amp.real:+.4f} {amp.imag:+.4f}i"
    components.append(
        {
            "|mI,mJ>": f"|{mI:+.1f}, {mJ:+.1f}>",
            "probability": prob,
            "amplitude": amp_str,  # <-- string, not complex
        }
    )

components.sort(key=lambda r: r["probability"], reverse=True)
df_components = pd.DataFrame(components[:10])
if len(df_components) == 0:
    st.write("All components are below threshold.")
else:
    st.table(df_components)

# --- decomposition in |F,mF> ---

st.markdown("**Decomposition in $|F, m_F\\rangle$ basis**")

I = sr87_data[level_name]["I"]
J = sr87_data[level_name]["J"]
F_min = abs(I - J)
F_max = I + J

FmF_rows = []
for F in np.arange(F_min, F_max + 1):
    key = (float(F), float(mF_choice))
    if key not in Fvecs:
        continue
    vF = Fvecs[key]
    amp = np.vdot(vF, psi_full)
    prob = float(np.abs(amp) ** 2)
    if prob < 1e-4:
        continue
    amp_str = f"{amp.real:+.4f} {amp.imag:+.4f}i"
    FmF_rows.append(
        {
            "F": F,
            "mF": mF_choice,
            "probability": prob,
            "amplitude": amp_str,  # <-- string, not complex
        }
    )

FmF_rows.sort(key=lambda r: r["probability"], reverse=True)
df_FmF = pd.DataFrame(FmF_rows)
if len(df_FmF) == 0:
    st.write("No significant |F,mF> components (all below threshold).")
else:
    st.table(df_FmF)


st.markdown(
    """
Notes:

* Energies are shown as $E/h$ in MHz.
* Eigenstates are ordered by eigenvalue *within each fixed* $m_F$ block.
* At $B \\to 0$ you should recover pure $|F,m_F\\rangle$ states.
* You can adjust the hyperfine constants in the `sr87_levels` dictionary
  to match your preferred values (experiment vs theory).
"""
)
