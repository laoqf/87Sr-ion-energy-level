# app.py
import sympy as sp
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

# ----------------------------------------------------------
# Symbolic magnetic field (Gauss)
# ----------------------------------------------------------
B_sym = sp.symbols("B", real=True)

# Speed of light (for wavelength calculation)
c_light = 299_792_458.0  # m/s

gI_Sr87 = -131.7712e-6               # gI = mu_I*(1-sigma)/mu_B from the experiement data : Phys. Rev. Lett. 135, 193001 – Published 3 November, 2025

# ----------------------------------------------------------
# 87Sr+ level data (your values)
#   A_hfs, B_hfs in Hz
#   L, S, J for LS-coupling g_J
#   gI is the atomic-style nuclear g-factor (μ_I(1-σ_d)/(μ_B I))
#   !!!E0 here is the frequency, E0 = E/h from NIST https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum=Sr+II&submit=Retrieve+Data&units=0&format=0&output=0&page_size=15&multiplet_ordered=0&conf_out=on&term_out=on&level_out=on&unc_out=1&j_out=on&lande_out=on&perc_out=on&biblio=on&temp=
# ----------------------------------------------------------
sr87_data = {
    "5S1/2": {
        "I": 9/2,
        "L": 0,
        "S": 1/2,
        "J": 1/2,
        "A_hfs": -1000.5e6,               # exp
        "B_hfs": 0.0,                     # J = 1/2 → no quadrupole term
        "gI": gI_Sr87,                    #exp
        "E0": 0.0,                        # define ground as zero
    },
    "5P1/2": {
        "I": 9/2,
        "L": 1,
        "S": 1/2,
        "J": 1/2,
        "A_hfs": -178.4e6,                # cal
        "B_hfs": 0.0,                     # J = 1/2 → no quadrupole term
        "gI": gI_Sr87,                    #exp
        "E0": 23715.19 * c_light * 100,   # cm^-1 → Hz
    },
    "5P3/2": {
        "I": 9/2,
        "L": 1,
        "S": 1/2,
        "J": 3/2,
        "A_hfs": -36.0e6,                 #exp
        "B_hfs": 88.5e6,                  #exp
        "gI": gI_Sr87,                    #exp
        "E0": 24516.65 * c_light * 100,
    },
    "4D3/2": {
        "I": 9/2,
        "L": 2,
        "S": 1/2,
        "J": 3/2,
        "A_hfs": -47.365e6,               #cal
        "B_hfs": 38.2e6,                  #cal
        "gI": gI_Sr87,                    #exp
        "E0": 14555.90 * c_light * 100,
    },
    "4D5/2": {
        "I": 9/2,
        "L": 2,
        "S": 1/2,
        "J": 5/2,
        "A_hfs": 2.1743e6,                #exp
        "B_hfs": 49.11e6,                 #exp
        "gI": gI_Sr87,                    #exp
        "E0": 14836.24 * c_light * 100,
    },
}

# ----------------------------------------------------------
# Magnetic moment (only μ_B needed in this convention)
# ----------------------------------------------------------
h = 6.626_070_15e-34   # J/Hz
mu_B = 9.274_010_0657e-24 #J/T
mu_B = mu_B/h/10000  # Hz/G  (E/h units) since the erengy here is frequency
#mu_B = 1.399624e6 # value in Hz/G  (E/h units)


# ----------------------------------------------------------
# LS-coupling Landé g_J (symbolic, fraction-friendly)
#   g_J = g_L [J(J+1) - S(S+1) + L(L+1)] / (2 J (J+1))
#       + g_S [J(J+1) + S(S+1) - L(L+1)] / (2 J (J+1))
# ----------------------------------------------------------
def lande_gJ(L, S, J, gL=1.0, gS=2.0023):
    # Convert to SymPy rationals / exact numbers
    J  = sp.nsimplify(J, rational=True)
    L  = sp.nsimplify(L, rational=True)
    S  = sp.nsimplify(S, rational=True)
    gL = sp.nsimplify(gL, rational=True)
    gS = sp.nsimplify(gS, rational=True)
    #gS = 2

    den = 2 * J * (J + 1)

    # Using exact SymPy arithmetic; this stays rational if gS is rational
    term1 = (J * (J + 1) - S * (S + 1) + L * (L + 1)) / den
    term2 = (J * (J + 1) + S * (S + 1) - L * (L + 1)) / den

    gJ = sp.simplify(gL * term1 + gS * term2)
    return gJ  # symbolic

# ----------------------------------------------------------
# Hyperfine energy E_F with Casimir A + B (numeric)
# (A_hfs, B_hfs in Hz)
# ----------------------------------------------------------
def E_F(I, J, F, A_hfs, B_hfs):
    I = float(I)
    J = float(J)
    F = float(F)

    K = F * (F + 1.0) - I * (I + 1.0) - J * (J + 1.0)

    term_A = 0.5 * A_hfs * K

    if B_hfs == 0.0 or I < 1.0 or J < 1.0:
        return term_A

    denom = 2.0 * I * (2.0 * I - 1.0) * 2.0 * J * (2.0 * J - 1.0)
    term_B = B_hfs * ((3/2 * K * (K + 1.0) - 2.0 * I * (I + 1.0) * J * (J + 1.0)) / denom)

    return term_A + term_B

# ----------------------------------------------------------
# Hyperfine Landé g_F (symbolic, using atomic gI)
# ----------------------------------------------------------
def g_F(I, J, F, gJ_sym, gI):
    I = sp.nsimplify(I, rational=True)
    J = sp.nsimplify(J, rational=True)
    F = sp.nsimplify(F, rational=True)
    gJ_sym = sp.nsimplify(gJ_sym, rational=True)
    gI_sym = sp.nsimplify(gI, rational=True)  # small, but keep symbolic
    #gI_sym=0

    num1 = F * (F + 1) + J * (J + 1) - I * (I + 1)
    num2 = F * (F + 1) + I * (I + 1) - J * (J + 1)
    den = 2 * F * (F + 1)

    gF_sym = sp.simplify(gJ_sym * num1 / den + gI_sym * num2 / den)
    return gF_sym

def F_values_for_manifold(manifold_name):
    d = sr87_data[manifold_name]
    I = d["I"]
    J = d["J"]
    F_min = abs(I - J)
    F_max = I + J
    return [F_min + k for k in range(int(F_max - F_min) + 1)]

# ----------------------------------------------------------
# Build a state table for all manifolds
#   key: (manifold, 2F, 2mF)
#   stores both symbolic and numeric gJ/gF
# ----------------------------------------------------------
def build_state_table():
    table = {}
    for manifold_name, d in sr87_data.items():
        I = d["I"]
        L = d["L"]
        S = d["S"]
        J = d["J"]
        A_hfs = d["A_hfs"]
        B_hfs = d["B_hfs"]
        gI = d["gI"]
        E0_elec = d["E0"]

        gJ_sym = lande_gJ(L, S, J)
        gJ_val = float(gJ_sym.evalf())

        F_list = F_values_for_manifold(manifold_name)
        for F in F_list:
            EF = E_F(I, J, F, A_hfs, B_hfs)
            gF_sym = g_F(I, J, F, gJ_sym, gI)
            gF_val = float(gF_sym.evalf())

            for mF_int in range(-int(2 * F), int(2 * F) + 1, 2):
                mF = mF_int / 2.0
                # Zeeman: numeric gF_val used for the energy
                E_sym = sp.simplify(EF + gF_val * mu_B * B_sym * mF + E0_elec)

                key = (manifold_name, int(2 * F), int(2 * mF))
                table[key] = {
                    "manifold": manifold_name,
                    "F": float(F),
                    "mF": float(mF),
                    "gJ_sym": gJ_sym,
                    "gJ": gJ_val,
                    "gF_sym": gF_sym,
                    "gF": gF_val,
                    "E0": EF + E0_elec,
                    "E_sym": E_sym,
                }
    return table



state_table = build_state_table()

# ----------------------------------------------------------
# Convenience accessors
# ----------------------------------------------------------
def get_gJ(manifold):
    d = sr87_data[manifold]
    gJ_sym = lande_gJ(d["L"], d["S"], d["J"])
    return float(gJ_sym.evalf()), gJ_sym

def get_gF_for_F(manifold, F):
    F2 = int(round(2 * F))
    key = (manifold, F2, 0)  # mF = 0
    info = state_table[key]
    return info["gF"], info["gF_sym"]

def energy_at_B(manifold, F, mF, B_val):
    F2 = int(round(2 * F))
    mF2 = int(round(2 * mF))
    key = (manifold, F2, mF2)
    info = state_table[key]
    return float(info["E_sym"].subs(B_sym, B_val))  # Hz

# ----------------------------------------------------------
# Streamlit UI
# ----------------------------------------------------------
st.title("87Sr⁺ Hyperfine–Zeeman Levels (5S, 5P, 4D)")

st.markdown(
    r"""
This app plots the **hyperfine–Zeeman energies** of several manifolds of
$^{87}\mathrm{Sr}^+$ in **small B field** using:

- $E_\text{hfs}$ from the **Casimir A + B formula**  
- $g_J$ from **LS-coupling** ($L,S,J$), kept as a symbolic fraction  
- first-order Zeeman: $g_F \mu_B B m_F$, with $g_F$ also stored symbolically  

Note:
<span style='color:red'>small B field is for 5s1/2, B < A/mu_B = 1000/1.399 = 715, but not for other states.</span>

The erengy **in frequency** is expressed by
$$
E = 
\underbrace{\frac{1}{2}A_{\mathrm{hfs}} K + B_{\mathrm{hfs}} \frac{\frac{3}{2}K(K+1)-2I(I+1)J(J+1)}{2I(I-1)2J(J-1)}}_{\text{Hyperfine}}
+
\underbrace{\mu_B\, g_F \, B \, m_F}_{\text{Zeeman}}
$$
where $K = F(F+1) -J(J+1) - I(I+1)$

In the tables we show both the **numeric** and **fractional** forms of $g_J$ and $g_F$.
"""
, unsafe_allow_html=True
)

# Select manifolds
all_manifolds = list(sr87_data.keys())
selected_manifolds = st.multiselect(
    "Select manifolds to plot:",
    options=all_manifolds,
    default=all_manifolds,
)

# Magnetic field (Gauss)
B_val = st.slider("Magnetic field B (Gauss)", 0.0, 200.0, 10.0, step=1.0)
st.write(f"Current B: **{B_val:.1f} G**")

# ----------------------------------------------------------
# F-dependent colors: F = 2 ... 7
# ----------------------------------------------------------
F_colors = {
    2: "blue",
    3: "green",
    4: "red",
    5: "black",
    6: "magenta",
    7: "orange",
}

# ----------------------------------------------------------
# Plot with Plotly
# ----------------------------------------------------------
fig = go.Figure()
x_scale = 0.08
half_width = 0.03

available_states = []  # (label, (manifold, F, mF))

for (manifold, F2, mF2), info in state_table.items():
    if manifold not in selected_manifolds:
        continue

    F = info["F"]
    mF = info["mF"]

    E_eval = float(info["E_sym"].subs(B_sym, B_val))  # Hz
    E_GHz = E_eval / 1e9

    x_offset = mF * x_scale
    x0 = x_offset - half_width
    x1 = x_offset + half_width

    F_int = int(F)
    color = F_colors.get(F_int, "black")

    label = f"{manifold}, F={F_int}, mF={mF:g}"
    available_states.append((label, (manifold, F, mF)))

    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[E_GHz, E_GHz],
            mode="lines",
            line=dict(color=color, width=2),
            name=label,
            hovertemplate=f"{label}<br>E = %{{y:.6f}} GHz<extra></extra>",
            showlegend=True,
        )
    )

fig.update_layout(
    title=f"87Sr⁺ Hyperfine Levels vs B (B = {B_val:.1f} G)",
    xaxis=dict(visible=False),
    yaxis_title="Energy (GHz)",
    legend_title="State, F, mF",
    width=900,
    height=700,
)

st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------
# Level selection: frequency difference and wavelength
# ----------------------------------------------------------
st.subheader("Transition frequency and wavelength between two levels")

if available_states:
    labels = [lbl for lbl, _ in available_states]
    label_to_state = {lbl: state for lbl, state in available_states}

    default_idx2 = 1 if len(labels) > 1 else 0

    sel1 = st.selectbox("Level 1", options=labels, index=0)
    sel2 = st.selectbox("Level 2", options=labels, index=default_idx2)

    if sel1 and sel2:
        man1, F1, mF1 = label_to_state[sel1]
        man2, F2, mF2 = label_to_state[sel2]

        E1 = energy_at_B(man1, F1, mF1, B_val)
        E2 = energy_at_B(man2, F2, mF2, B_val)

        dE = E2 - E1            # Hz
        dnu = abs(dE)           # Hz
        dnu_MHz = dnu / 1e6
        dnu_GHz = dnu / 1e9

        wavelength_m = c_light / dnu if dnu != 0 else float("inf")
        wavelength_nm = wavelength_m * 1e9

        st.markdown(
            f"""
**{sel1} → {sel2}**

- Δν = {dnu:.3f} Hz  
- Δν = {dnu_MHz:.6f} MHz  
- Δν = {dnu_GHz:.6f} GHz  

- λ = {wavelength_nm:.3f} nm
"""
        )

# ----------------------------------------------------------
# Tables: g_J and g_F (numeric + fraction)
# ----------------------------------------------------------
st.subheader("Landé g_J (from L,S,J)")

rows_gJ = []
for manifold in selected_manifolds:
    d = sr87_data[manifold]
    gJ_val, gJ_sym = get_gJ(manifold)
    rows_gJ.append(
        {
            "manifold": manifold,
            "L": d["L"],
            "S": d["S"],
            "J": d["J"],
            "g_J": gJ_val,
            "g_J_frac": str(sp.nsimplify(gJ_sym)),
        }
    )
if rows_gJ:
    df_gJ = pd.DataFrame(rows_gJ)
    st.dataframe(df_gJ, hide_index=True)

st.subheader("Hyperfine g_F values (first-order)")

rows_gF = []
for manifold in selected_manifolds:
    for F in F_values_for_manifold(manifold):
        gF_val, gF_sym = get_gF_for_F(manifold, F)
        rows_gF.append(
            {
                "manifold": manifold,
                "F": F,
                "g_F": gF_val,
                "g_F_frac": str(sp.nsimplify(gF_sym)),
            }
        )
if rows_gF:
    df_gF = pd.DataFrame(rows_gF)
    st.dataframe(df_gF, hide_index=True)

# ----------------------------------------------------------
# Sr87⁺ Parameter Table (from sr87_data) – formatted
# ----------------------------------------------------------
st.subheader("87Sr⁺ Parameter Table")

def make_sr87_table(data):
    rows = []
    for manifold, params in data.items():
        row = {"manifold": manifold}
        row.update(params)
        rows.append(row)
    return pd.DataFrame(rows)

df_params = make_sr87_table(sr87_data)

# Convert E0 → wavelength (nm)
def e0_to_lambda_nm(e0_hz):
    if e0_hz is None or e0_hz == 0:
        return ""      # Ground state → no wavelength
    wavelength_nm = c_light / float(e0_hz) * 1e9
    return f"{wavelength_nm:.2f}"   # <-- normal float formatting (no scientific notation)

# Apply conversion
df_params["E0_lambda_nm"] = df_params["E0"].apply(e0_to_lambda_nm)

# Scientific notation for A_hfs, B_hfs, E0
def sci(x):
    if x is None:
        return ""
    try:
        x = float(x)
    except:
        return x
    if x == 0:
        return "0"
    return f"{x:.3e}"

df_params["A_hfs"] = df_params["A_hfs"].apply(sci)
df_params["B_hfs"] = df_params["B_hfs"].apply(sci)
df_params["E0"]     = df_params["E0"].apply(sci)

# nice display order
preferred_order = [
    "manifold", "L", "S", "J", "I",
    "A_hfs", "B_hfs", "gI", "E0",
    "E0_lambda_nm",    # wavelength in nm (not scientific)
]

df_params = df_params.reindex(columns=preferred_order)

st.dataframe(df_params, hide_index=True)
