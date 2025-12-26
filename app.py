import math
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------
st.set_page_config(page_title="OPTION PRICER & PAYOFF VISUALIZER", layout="centered")
st.title("BLACK-SCHOLES PRICER & PAYOFF VISUALIZER")

# ---------------------------------------------------
# BLACK-SCHOLES FORMULAS (no external dependencies)
# ---------------------------------------------------
def norm_cdf(x):
    """Fonction de répartition de la loi normale standard."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, r, q, T, sigma):
    """Prix d'un call européen selon Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(0.0, S * math.exp(-q*T) - K * math.exp(-r*T))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-q * T) * S * norm_cdf(d1) - math.exp(-r * T) * K * norm_cdf(d2)

def bs_put(S, K, r, q, T, sigma):
    """Prix d'un put européen selon Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return max(0.0, K * math.exp(-r*T) - S * math.exp(-q*T))
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * K * norm_cdf(-d2) - math.exp(-q * T) * S * norm_cdf(-d1)

# ---------------------------------------------------
# SESSION STATE (mémoire persistante pendant l'app)
# ---------------------------------------------------
if "legs" not in st.session_state:
    st.session_state.legs = []
if "show_plot" not in st.session_state:
    st.session_state.show_plot = False

# ---------------------------------------------------
# SIDEBAR INPUT SECTION — all parameters in one place
# ---------------------------------------------------
st.sidebar.header("Parameters & Pricing")

with st.sidebar.form("add_leg_form"):
    option_type = st.selectbox("Option type", ["call", "put"])
    position = st.selectbox("Position", ["long", "short"])
    strike = st.number_input("Strike (K)", value=100.0, step=1.0)
    quantity = st.number_input("Quantity", value=1, step=1, min_value=1)

    st.markdown("---")
    spot = st.number_input("Spot (S)", value=100.0, step=1.0)
    rate = st.number_input("Risk-free rate (r)", value=0.02, step=0.01, format="%.4f")
    div = st.number_input("Dividend yield (q)", value=0.00, step=0.01, format="%.4f")
    maturity_days = st.number_input("Maturity (T, days)", value=1, step=1, min_value=1)
    maturity = maturity_days / 365 #conversion en année de la matu en jours
    vol = st.number_input("Volatility (σ)", value=0.20, step=0.05, format="%.4f")

    # Prix BS automatique
    if option_type == "call":
        price = bs_call(spot, strike, rate, div, maturity, vol)
    else:
        price = bs_put(spot, strike, rate, div, maturity, vol)

    st.markdown(f"### Black-Scholes premium: **{price:.4f}**")

    # Boutons
    col1, col2, col3 = st.columns(3)
    add_leg_clicked = col1.form_submit_button("Add leg")
    visualise_clicked = col2.form_submit_button("Visualise")
    reset_clicked = col3.form_submit_button("Reset")

    # Ajouter une jambe
    if add_leg_clicked:
        st.session_state.legs.append({
            "type": option_type,
            "position": position,
            "strike": strike,
            "quantity": quantity,
            "premium": price,
            "spot": spot,
            "r": rate,
            "q": div,
            "T": maturity,
            "vol": vol
        })
        st.success("✅ Leg added successfully.")

    # Visualiser
    if visualise_clicked:
        st.session_state.show_plot = True

    # Reset
    if reset_clicked:
        st.session_state.legs = []
        st.session_state.show_plot = False
        st.warning("Strategy cleared.")

# ---------------------------------------------------
# DISPLAY LEGS + TOTAL PREMIUM
# ---------------------------------------------------
st.subheader("Current Strategy")
if st.session_state.legs:
    total_premium = 0.0
    for i, leg in enumerate(st.session_state.legs, start=1):
        # Premium total = somme pondérée en fonction de la position
        signed_premium = leg["premium"] * leg["quantity"]
        total_premium += signed_premium if leg["position"] == "short" else -signed_premium

        st.markdown(
            f"**{i}. {leg['position'].capitalize()} {leg['type'].capitalize()}** — "
            f"K={leg['strike']}, Qty={leg['quantity']}, σ={leg['vol']*100:.1f}%, "
            f"T={leg['T']:.2f}y → 💰 {leg['premium']:.4f}"
        )

    st.markdown(f"### 💵 Total net premium of strategy: **{total_premium:.4f}**")
else:
    st.info("No legs yet. Fill parameters and click 'Add leg'.")

# ---------------------------------------------------
# PAYOFF VISUALIZATION
# ---------------------------------------------------
if st.session_state.show_plot and st.session_state.legs:
    # Grille de prix à maturité
    S_min = min(l["spot"] for l in st.session_state.legs) * 0.5
    S_max = max(l["spot"] for l in st.session_state.legs) * 1.5
    S = np.linspace(S_min, S_max, 600)

    payoff_total = np.zeros_like(S)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))

    for leg in st.session_state.legs:
        K = leg["strike"]
        p = leg["premium"]
        q = leg["quantity"]

        # Payoff brut
        if leg["type"] == "call":
            payoff = np.maximum(S - K, 0)
        else:
            payoff = np.maximum(K - S, 0)

        # Intègre position et prime
        if leg["position"] == "long":
            payoff = payoff - p
        else:
            payoff = -payoff + p

        payoff *= q
        payoff_total += payoff

        # Tracé jambe
        ax.plot(S, payoff, linestyle='--', alpha=0.6,
        label=f"{leg['position']} {leg['type']} K={K}")

    # Courbe totale
    ax.plot(S, payoff_total, color='#FF8C00', linewidth=2, label="Total Strategy")
    ax.axhline(0, color='white', linestyle='-')
    ax.axvline(st.session_state.legs[0]["spot"], color='red', linestyle='-', alpha=0.5, label="Spot")

    ax.set_xlabel("Underlying price at maturity (S)")
    ax.set_ylabel("Profit / Loss")
    ax.set_title("Payoff at Maturity")
    ax.legend( loc='center left',          # position verticale centrée
    bbox_to_anchor=(1, 0.5),    # décale la légende en dehors du graphe
    frameon=False               # pas de cadre autour
)
    plt.tight_layout(pad=2, rect=[0, 0, 0.85, 1])  # laisse de la place à droite
    ax.grid(
        True, which='major',
        linestyle='--',
        linewidth=0.6,
        color='white',
        alpha=0.18
    )

    st.pyplot(fig)
elif st.session_state.show_plot:
    st.warning("No legs to display — add at least one leg first.")

