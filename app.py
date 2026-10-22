# -*- coding: utf-8 -*-
from __future__ import annotations

import streamlit as st
import numpy as np
from scipy.optimize import minimize

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA WEB
# ==============================================================================
st.set_page_config(
    page_title="Q-MAP | Enterprise Suite v2.0",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilo visual general corporativo oscuro
st.markdown("""
    <style>
    .main { background-color: #050b14; color: #f8fafc; }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# 1. PARÁMETROS Y MATRICES DEL PROBLEMA[cite: 10]
# ==============================================================================
N_VARIABLES = 7

ACTION_NAMES = [
    "x1: Sensor piezoelectrico cavidad directo",
    "x2: Higrometro punto de rocio alta gama",
    "x3: Modificacion enfriamiento del molde",
    "x4: Re-balanceo termopares / PID canon",
    "x5: Perfil inyeccion 3 etapas",
    "x6: Valvula venteo asistida en molde",
    "x7: Sensor humedad tolva simplificado"
]

COSTS = np.array([18000, 12000, 22000, 1500, 2000, 8000, 6500])
BUDGET_DEFAULT = 35000

RISK_REDUCTION = np.array([
    [6, 0, 0, 1, 2, 5, 0],  # Falla A: Llenado
    [0, 6, 0, 1, 0, 0, 6],  # Falla B: Humedad
    [1, 0, 5, 0, 5, 1, 0],  # Falla C: Deformacion
    [2, 0, 1, 0, 2, 5, 0]   # Falla D: Soldadura
])

W_MATRIX = np.zeros((N_VARIABLES, N_VARIABLES))
W_MATRIX[4, 5] = -3.0
W_MATRIX[5, 4] = -3.0
W_MATRIX[3, 6] = -1.5
W_MATRIX[6, 3] = -1.5

LAMBDA_BUDGET = 50.0  
LAMBDA_RISK = 2.0
LAMBDA_COUPLING = 1.0

# ==============================================================================
# 2. MOTOR MATEMÁTICO Q-MAP + Q-CORE[cite: 10]
# ==============================================================================
def evaluate_bitstring_cost(bitstring, budget, initial_ap_list):
    x = np.array([int(b) for b in bitstring])
    total_cost = np.dot(COSTS, x)
    
    budget_penalty = 0.0
    if total_cost > budget:
        excess = total_cost - budget
        budget_penalty = (excess ** 2) + (10000.0 * excess)
        
    reduction_per_failure = np.dot(RISK_REDUCTION, x)
    residual_risk = np.sum(np.maximum(0, np.array(initial_ap_list) - reduction_per_failure))
    coupling_term = 0.5 * np.dot(x, np.dot(W_MATRIX, x))
    
    H_value = (LAMBDA_RISK * residual_risk) + (LAMBDA_BUDGET * budget_penalty) + (LAMBDA_COUPLING * coupling_term)
    return H_value, total_cost, residual_risk

def get_qaoa_state_vector(gamma, beta, budget, initial_ap_list):
    N = 2 ** N_VARIABLES
    state = np.ones(N, dtype=complex) / np.sqrt(N)
    
    for i in range(N):
        bitstring = format(i, f'0{N_VARIABLES}b')
        H_val, _, _ = evaluate_bitstring_cost(bitstring, budget, initial_ap_list)
        state[i] *= np.exp(-1j * gamma * H_val)
        
    for q in range(N_VARIABLES):
        cos_b = np.cos(beta)
        sin_b = -1j * np.sin(beta)
        new_state = np.zeros_like(state)
        for i in range(N):
            i_flipped = i ^ (1 << (N_VARIABLES - 1 - q))
            new_state[i] += cos_b * state[i] + sin_b * state[i_flipped]
        state = new_state
        
    return np.abs(state) ** 2

def run_qaoa_optimization(budget, initial_ap_list):
    def qaoa_obj(params):
        gamma, beta = params
        probs = get_qaoa_state_vector(gamma, beta, budget, initial_ap_list)
        val = 0.0
        for i in range(2 ** N_VARIABLES):
            bitstring = format(i, f'0{N_VARIABLES}b')
            H_val, _, _ = evaluate_bitstring_cost(bitstring, budget, initial_ap_list)
            val += H_val * probs[i]
        return val

    res = minimize(qaoa_obj, [0.5, 0.5], method='COBYLA', options={'maxiter': 50})
    best_gamma, best_beta = res.x
    final_probs = get_qaoa_state_vector(best_gamma, best_beta, budget, initial_ap_list)
    
    best_idx = None
    min_energy = float('inf')
    
    for i, prob in enumerate(final_probs):
        bitstring = format(i, f'0{N_VARIABLES}b')
        H_val, total_cost, _ = evaluate_bitstring_cost(bitstring, budget, initial_ap_list)
        if total_cost <= budget and H_val < min_energy:
            min_energy = H_val
            best_idx = i
            
    if best_idx is None:
        best_idx = np.argmax(final_probs)

    best_bitstring = format(best_idx, f'0{N_VARIABLES}b')
    H_val, total_cost, residual_risk = evaluate_bitstring_cost(best_bitstring, budget, initial_ap_list)
    
    return best_bitstring, total_cost, residual_risk, best_gamma, best_beta

def calculate_ap(s, o, d):
    if s >= 8 and (o >= 4 or d >= 4):
        return 8.0, "AP: ALTA (H)"
    elif s >= 6 or o >= 4 or d >= 4:
        return 5.0, "AP: MEDIA (M)"
    else:
        return 2.0, "AP: BAJA (L)"

# ==============================================================================
# 3. INTERFAZ GRÁFICA STREAMLIT (WEB)[cite: 10]
# ==============================================================================

# Encabezado principal
st.markdown("### Q-MAP: MULTIVARIABLE-ALLOCATION-PRIORITIZATION")
st.markdown("Industrial PFMEA Optimizer (AIAG-VDA / Q-MAP + Q-CORE) | ● KERNEL LISTO")
st.markdown("---")

# 1. Restricción Presupuestaria de CAPEX
st.markdown("#### 1. RESTRICCION PRESUPUESTARIA DE CAPEX")
budget = st.number_input(
    "Presupuesto Maximo ($ USD):", 
    min_value=1000.0, 
    max_value=200000.0, 
    value=float(BUDGET_DEFAULT), 
    step=1000.0
)

st.markdown("---")

# 2. Parametrización Dinámica de Fallas (S / O / D)
st.markdown("#### 2. PARAMETRIZACION DINAMICA DE FALLAS (S / O / D)")

failures_def = [
    {"id": "A", "name": "Falla A: Llenado Incompleto (Short Shot)", "s": 8, "o": 4, "d": 4},
    {"id": "B", "name": "Falla B: Degradacion por Humedad", "s": 8, "o": 3, "d": 5},
    {"id": "C", "name": "Falla C: Deformacion / Alabeo (Warpage)", "s": 6, "o": 5, "d": 4},
    {"id": "D", "name": "Falla D: Linea de Soldadura Debil", "s": 7, "o": 4, "d": 4}
]

col1, col2 = st.columns(2)
slider_values = {}

for idx, fail in enumerate(failures_def):
    target_col = col1 if idx % 2 == 0 else col2
    with target_col:
        st.markdown(f"**{fail['name']}**")
        s_val = st.slider(f"Severidad (S) - {fail['id']}", 1, 10, fail["s"], key=f"s_{fail['id']}")
        o_val = st.slider(f"Ocurrencia (O) - {fail['id']}", 1, 10, fail["o"], key=f"o_{fail['id']}")
        d_val = st.slider(f"Deteccion (D) - {fail['id']}", 1, 10, fail["d"], key=f"d_{fail['id']}")
        
        _, ap_text = calculate_ap(s_val, o_val, d_val)
        st.markdown(f"Nivel AP actual: `{ap_text}`")
        slider_values[fail["id"]] = {"s": s_val, "o": o_val, "d": d_val}
        st.markdown("---")

# Botón de ejecución principal con estilo destacado
if st.button("⚛ EJECUTAR OPTIMIZACION Q-MAP + Q-CORE ⚛", type="primary", use_container_width=True):
    with st.spinner("Ejecutando motor de optimización cuántica y cálculo de riesgos..."):
        initial_ap_list = []
        for fail in failures_def:
            f_id = fail["id"]
            s = slider_values[f_id]["s"]
            o = slider_values[f_id]["o"]
            d = slider_values[f_id]["d"]
            ap_val, _ = calculate_ap(s, o, d)
            initial_ap_list.append(ap_val)

        best_bitstring, total_cost, residual_risk, gamma, beta = run_qaoa_optimization(budget, initial_ap_list)

    st.success(f"Optimización completada con éxito. (Gamma: {gamma:.4f} | Beta: {beta:.4f})")

    # 3. Salida del Solver y Matriz de Decisión
    st.markdown("### 3. SALIDA DEL SOLVER Y MATRIZ DE DECISION")
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("ESTADO ÓPTIMO |q>", f"|{best_bitstring}>")
    with m2:
        st.metric("CAPEX REQUERIDO", f"${total_cost:,.2f}")
    with m3:
        st.metric("PRESUPUESTO LÍMITE", f"${budget:,.2f}")
    
    savings = budget - total_cost
    with m4:
        if savings >= 0:
            st.metric("AHORRO", f"${savings:,.2f}")
        else:
            st.metric("DÉFICIT", f"-${abs(savings):,.2f}")

    st.markdown(f"**Riesgo Residual Total:** {residual_risk:.1f} Puntos")

    # Tablas de resultados en dos columnas
    t_col1, t_col2 = st.columns(2)
    
    with t_col1:
        st.markdown("#### Acciones Técnicas Seleccionadas:")
        action_rows = []
        for idx, bit in enumerate(best_bitstring):
            state_str = "OK" if bit == '1' else "-"
            cost_str = f"${COSTS[idx]:,.2f}" if bit == '1' else "$0.00"
            short_name = ACTION_NAMES[idx].split(":")[1].strip()
            action_rows.append({"Acción": short_name, "Est": state_str, "Costo ($)": cost_str})
        st.table(action_rows)

    with t_col2:
        st.markdown("#### Estado Final de Riesgo Post:Q-MAP + Q-CORE:")
        risk_rows = []
        x_vector = np.array([int(b) for b in best_bitstring])
        reduction_per_failure = np.dot(RISK_REDUCTION, x_vector)

        for idx, fail in enumerate(failures_def):
            f_id = fail["id"]
            init_val = initial_ap_list[idx]
            rem_points = max(0.0, init_val - reduction_per_failure[idx])
            
            if rem_points >= 6.0:
                status_str = "ALTA (H)"
            elif rem_points >= 4.0:
                status_str = "MEDIA (M)"
            else:
                status_str = "BAJA (L)"
                
            risk_rows.append({"Falla FMEA": f"Falla {f_id}", "Nivel AP": status_str, "Restante": f"{rem_points:.1f} pts"})
        st.table(risk_rows)