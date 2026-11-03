# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime

# Librerías ReportLab para generación de PDF profesional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Configuración inicial de la página web en Streamlit
st.set_page_config(
    page_title="Q-PAC | AMEF de Proceso Avanzado (Reco Engine)",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. PARÁMETROS Y MATRICES DEL PROBLEMA
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
BUDGET_DEFAULT = 35000.0

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

FAILURES_INFO = [
    {"id": "A", "name": "Falla A: Llenado Incompleto (Short Shot)", "s": 8, "o": 4, "d": 4},
    {"id": "B", "name": "Falla B: Degradacion por Humedad", "s": 8, "o": 3, "d": 5},
    {"id": "C", "name": "Falla C: Deformacion / Alabeo (Warpage)", "s": 6, "o": 5, "d": 4},
    {"id": "D", "name": "Falla D: Linea de Soldadura Debil", "s": 7, "o": 4, "d": 4}
]

# ==============================================================================
# 2. MOTOR MATEMÁTICO QUBO + QAOA
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
        return 8.0, "ALTA"
    elif s >= 6 or o >= 4 or d >= 4:
        return 5.0, "MEDIA"
    else:
        return 2.0, "BAJA"

# ==============================================================================
# 3. GENERADOR DE PDF CON REPORTLAB
# ==============================================================================
def create_pdf_bytes(last_results, fig_ba, fig_pareto):
    pdf_filename = "temp_report.pdf"
    img_ba_path = "temp_chart_ba.png"
    img_pareto_path = "temp_chart_pareto.png"

    fig_ba.savefig(img_ba_path, dpi=200, bbox_inches='tight')
    fig_pareto.savefig(img_pareto_path, dpi=200, bbox_inches='tight')

    doc = SimpleDocTemplate(
        pdf_filename,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    style_title = ParagraphStyle('DocTitle', fontName='Helvetica-Bold', fontSize=16, textColor=colors.HexColor('#0f172a'), spaceAfter=4)
    style_subtitle = ParagraphStyle('DocSubTitle', fontName='Helvetica', fontSize=9, textColor=colors.HexColor('#64748b'), spaceAfter=12)
    style_h1 = ParagraphStyle('Heading1_Custom', fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#0284c7'), spaceBefore=8, spaceAfter=4)
    style_cell = ParagraphStyle('CellText', fontName='Helvetica', fontSize=7.5, textColor=colors.HexColor('#1e293b'), leading=9)
    style_cell_bold = ParagraphStyle('CellTextBold', fontName='Helvetica-Bold', fontSize=7.5, textColor=colors.HexColor('#0f172a'), leading=9)
    style_cell_header = ParagraphStyle('CellHeader', fontName='Helvetica-Bold', fontSize=7.5, textColor=colors.white, leading=9)

    elements = []

    elements.append(Paragraph("Q-PAC - ENTERPRISE REPORT (AIAG-VDA 2022 EDITION)", style_title))
    elements.append(Paragraph(f"Industrial FMEA Optimization Report (Reco Engine) | {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", style_subtitle))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#0284c7'), spaceAfter=10))

    elements.append(Paragraph("1. Resumen Ejecutivo de Optimización", style_h1))
    res = last_results
    
    exec_data = [
        [
            Paragraph("Presupuesto Límite (CAPEX):", style_cell_bold),
            Paragraph(f"${res['budget']:,.2f} USD", style_cell),
            Paragraph("Estado Óptimo |q>:", style_cell_bold),
            Paragraph(f"|{res['best_bitstring']}>", style_cell)
        ],
        [
            Paragraph("CAPEX Requerido:", style_cell_bold),
            Paragraph(f"${res['total_cost']:,.2f} USD", style_cell),
            Paragraph("Ahorro / Superávit:", style_cell_bold),
            Paragraph(f"${res['savings']:,.2f} USD", style_cell)
        ],
        [
            Paragraph("Riesgo Residual Total:", style_cell_bold),
            Paragraph(f"{res['residual_risk']:.1f} Puntos", style_cell),
            Paragraph("Parámetros del Solver:", style_cell_bold),
            Paragraph(f"γ: {res['gamma']:.3f} | β: {res['beta']:.3f}", style_cell)
        ]
    ]

    t_exec = Table(exec_data, colWidths=[120, 140, 120, 160])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#f8fafc')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_exec)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("2. Análisis de Reducción de Riesgo por Falla (Antes vs. Después - AIAG-VDA 2022)", style_h1))
    
    risk_data = [[
        Paragraph("Falla FMEA", style_cell_header),
        Paragraph("S / O / D Inicial", style_cell_header),
        Paragraph("AP Inicial", style_cell_header),
        Paragraph("AP Residual", style_cell_header),
        Paragraph("Nivel de Riesgo", style_cell_header)
    ]]

    for r_item in res['risk_table_data']:
        risk_data.append([
            Paragraph(r_item[0], style_cell),
            Paragraph(r_item[1], style_cell),
            Paragraph(r_item[2], style_cell_bold),
            Paragraph(r_item[3], style_cell_bold),
            Paragraph(r_item[4], style_cell_bold)
        ])

    t_risk = Table(risk_data, colWidths=[180, 90, 80, 80, 110])
    t_risk.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e293b')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_risk)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("3. Matriz de Acciones Técnicas Seleccionadas", style_h1))
    
    act_data = [[
        Paragraph("Acción Técnica", style_cell_header),
        Paragraph("Estado Solver", style_cell_header),
        Paragraph("Inversión ($ USD)", style_cell_header)
    ]]

    for a_item in res['selected_actions_data']:
        act_data.append([
            Paragraph(a_item[0], style_cell),
            Paragraph(a_item[1], style_cell_bold if a_item[1] == "OK" else style_cell),
            Paragraph(f"${a_item[2]:,.2f}", style_cell)
        ])

    t_act = Table(act_data, colWidths=[300, 100, 140])
    t_act.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0284c7')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#0284c7')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_act)
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("4. Evidencia Gráfica y Frontera Combinatoria", style_h1))
    
    img_table_data = [[
        RLImage(img_ba_path, width=250, height=140),
        RLImage(img_pareto_path, width=250, height=140)
    ]]
    
    t_imgs = Table(img_table_data, colWidths=[270, 270])
    t_imgs.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#cbd5e1')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_imgs)
    elements.append(Spacer(1, 10))

    # ANEXOS NORMATIVOS
    elements.append(PageBreak())
    elements.append(Paragraph("ANEXOS: TABLAS DE REFERENCIA NORMATIVA (AIAG-VDA EDICIÓN 2022)", style_title))
    elements.append(Paragraph("Tablas oficiales de Severidad, Ocurrencia y Detección para la evaluación de Action Priority (AP).", style_subtitle))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#0f172a'), spaceAfter=10))

    elements.append(Paragraph("Anexo A: Criterios de Severidad (S) - AIAG-VDA", style_h1))
    sev_data = [
        [Paragraph("<b>Efecto</b>", style_cell_header), Paragraph("<b>Criterio de Severidad en el Producto / Proceso</b>", style_cell_header), Paragraph("<b>Rango (S)</b>", style_cell_header)],
        [Paragraph("Muy Alto", style_cell), Paragraph("Afecta la seguridad del operador o incumplimiento de normativas legales sin advertencia.", style_cell), Paragraph("9 - 10", style_cell)],
        [Paragraph("Alto", style_cell), Paragraph("Interrupción severa de la línea de producción o fallo importante en el vehículo con advertencia.", style_cell), Paragraph("7 - 8", style_cell)],
        [Paragraph("Medio", style_cell), Paragraph("Daño moderado, reducción del rendimiento, retrabajo necesario en estación.", style_cell), Paragraph("4 - 6", style_cell)],
        [Paragraph("Bajo / Muy Bajo", style_cell), Paragraph("Molestia menor, desajuste estético menor perceptible por el cliente.", style_cell), Paragraph("1 - 3", style_cell)]
    ]
    t_sev = Table(sev_data, colWidths=[100, 360, 80])
    t_sev.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e293b')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_sev)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Anexo B: Criterios de Ocurrencia (O) - AIAG-VDA", style_h1))
    occ_data = [
        [Paragraph("<b>Probabilidad de Falla</b>", style_cell_header), Paragraph("<b>Tasas / Frecuencia de Incidencia Estimada en Proceso</b>", style_cell_header), Paragraph("<b>Rango (O)</b>", style_cell_header)],
        [Paragraph("Muy Alta", style_cell), Paragraph("Fallos casi inevitables, tasa estadística > 1 de cada 2 piezas.", style_cell), Paragraph("9 - 10", style_cell)],
        [Paragraph("Alta", style_cell), Paragraph("Fallos recurrentes en procesos similares, tasa de 1 en 20 a 1 en 80.", style_cell), Paragraph("6 - 8", style_cell)],
        [Paragraph("Media", style_cell), Paragraph("Proceso con controles estándar, fallos ocasionales (1 en 400 a 1 en 2,000).", style_cell), Paragraph("4 - 5", style_cell)],
        [Paragraph("Baja / Muy Baja", style_cell), Paragraph("Proceso altamente robusto y controlado, fallos raros o inexistentes.", style_cell), Paragraph("1 - 3", style_cell)]
    ]
    t_occ = Table(occ_data, colWidths=[100, 360, 80])
    t_occ.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e293b')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_occ)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Anexo C: Criterios de Detección (D) - AIAG-VDA", style_h1))
    det_data = [
        [Paragraph("<b>Oportunidad de Detección</b>", style_cell_header), Paragraph("<b>Capacidad del Control Actual para Detectar la Causa / Modo de Falla</b>", style_cell_header), Paragraph("<b>Rango (D)</b>", style_cell_header)],
        [Paragraph("Imposible", style_cell), Paragraph("El control actual no detecta el fallo ni permite identificar la anomalía.", style_cell), Paragraph("9 - 10", style_cell)],
        [Paragraph("Baja / Muy Baja", style_cell), Paragraph("Controles basados en inspección visual humana post-proceso o muestreos.", style_cell), Paragraph("6 - 8", style_cell)],
        [Paragraph("Moderada / Alta", style_cell), Paragraph("Controles automáticos en estación a prueba de errores humanos menores.", style_cell), Paragraph("3 - 5", style_cell)],
        [Paragraph("Muy Alta", style_cell), Paragraph("Pasa-no pasa automático integrado, prevención total de salida de defectos.", style_cell), Paragraph("1 - 2", style_cell)]
    ]
    t_det = Table(det_data, colWidths=[100, 360, 80])
    t_det.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0f172a')),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#1e293b')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    elements.append(t_det)

    doc.build(elements)

    with open(pdf_filename, "rb") as f:
        pdf_bytes = f.read()

    if os.path.exists(pdf_filename):
        os.remove(pdf_filename)
    if os.path.exists(img_ba_path):
        os.remove(img_ba_path)
    if os.path.exists(img_pareto_path):
        os.remove(img_pareto_path)

    return pdf_bytes

# ==============================================================================
# 4. INTERFAZ GRÁFICA WEB STREAMLIT
# ==============================================================================

# Encabezado
st.title("⚡ Q-PAC AutoAero Systems")
st.subheader("Industrial FMEA Optimizer (AIAG-VDA 2022 / Reco Engine)")
st.markdown("---")

# Barra lateral para entradas de control
st.sidebar.header("1. Restricción Presupuestaria CAPEX")
budget = st.sidebar.number_input("Presupuesto Máximo ($ USD)", value=BUDGET_DEFAULT, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.header("2. Parametrización de Fallas (S / O / D)")

sod_inputs = {}
for fail in FAILURES_INFO:
    f_id = fail["id"]
    st.sidebar.subheader(f"{fail['name']}")
    col_s, col_o, col_d = st.sidebar.columns(3)
    
    with col_s:
        s_val = st.slider(f"S ({f_id})", 1, 10, fail["s"], key=f"s_{f_id}")
    with col_o:
        o_val = st.slider(f"O ({f_id})", 1, 10, fail["o"], key=f"o_{f_id}")
    with col_d:
        d_val = st.slider(f"D ({f_id})", 1, 10, fail["d"], key=f"d_{f_id}")
        
    sod_inputs[f_id] = (s_val, o_val, d_val)

run_button = st.sidebar.button("🚀 EJECUTAR OPTIMIZACIÓN RECO ENGINE", type="primary", use_container_width=True)

# Cálculo de estado inicial
initial_ap_list = []
for fail in FAILURES_INFO:
    s, o, d = sod_inputs[fail["id"]]
    ap_val, _ = calculate_ap(s, o, d)
    initial_ap_list.append(ap_val)

if run_button or 'last_results' in st.session_state:
    if run_button:
        with st.spinner("Procesando espacio combinatorio con Reco Engine..."):
            best_bitstring, total_cost, residual_risk, gamma, beta = run_qaoa_optimization(budget, initial_ap_list)
            
            savings = budget - total_cost
            
            selected_actions_data = []
            for idx, bit in enumerate(best_bitstring):
                state_str = "OK" if bit == '1' else "-"
                cost_num = COSTS[idx] if bit == '1' else 0
                selected_actions_data.append((ACTION_NAMES[idx], state_str, cost_num))

            x_vector = np.array([int(b) for b in best_bitstring])
            reduction_per_failure = np.dot(RISK_REDUCTION, x_vector)

            residual_ap_list = []
            risk_table_data = []
            for idx, fail in enumerate(FAILURES_INFO):
                f_id = fail["id"]
                init_val = initial_ap_list[idx]
                s, o, d = sod_inputs[f_id]
                _, ap_txt_init = calculate_ap(s, o, d)
                
                rem_points = max(0.0, init_val - reduction_per_failure[idx])
                residual_ap_list.append(rem_points)
                
                status_str = "ALTA" if rem_points >= 6.0 else ("MEDIA" if rem_points >= 4.0 else "BAJA")
                risk_table_data.append((fail["name"], f"S:{s} O:{o} D:{d}", ap_txt_init, f"{rem_points:.1f} pts", status_str))

            st.session_state['last_results'] = {
                "budget": budget,
                "best_bitstring": best_bitstring,
                "total_cost": total_cost,
                "savings": savings,
                "residual_risk": residual_risk,
                "gamma": gamma,
                "beta": beta,
                "selected_actions_data": selected_actions_data,
                "risk_table_data": risk_table_data,
                "residual_ap_list": residual_ap_list,
                "initial_ap_list": initial_ap_list
            }

    res = st.session_state['last_results']

    # Métricas principales
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ESTADO ÓPTIMO |q>", f"|{res['best_bitstring']}>")
    m2.metric("CAPEX REQUERIDO", f"${res['total_cost']:,.2f} USD")
    m3.metric("PRESUPUESTO LÍMITE", f"${res['budget']:,.2f} USD")
    m4.metric("AHORRO / SUPERÁVIT", f"${res['savings']:,.2f} USD", delta_color="normal" if res['savings'] >= 0 else "inverse")

    st.markdown("---")

    # Layout de tablas y gráficos
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Acciones Técnicas Seleccionadas")
        df_actions = pd.DataFrame(res['selected_actions_data'], columns=["Acción Técnica", "Estado", "Costo ($ USD)"])
        st.dataframe(df_actions.style.format({"Costo ($ USD)": "${:,.2f}"}), use_container_width=True)

        st.subheader("Estado Final de Riesgo Post-Optimización")
        df_risks = pd.DataFrame(res['risk_table_data'], columns=["Falla FMEA", "S/O/D Inicial", "AP Inicial", "AP Residual", "Nivel de Riesgo"])
        st.dataframe(df_risks, use_container_width=True)

        # Gráfica Antes vs Después
        fig_ba, ax_ba = plt.subplots(figsize=(5, 3))
        failures_labels = ['Falla A', 'Falla B', 'Falla C', 'Falla D']
        x_indices = np.arange(len(failures_labels))
        bar_width = 0.35

        plot_res = [max(val, 0.2) if val == 0.0 else val for val in res['residual_ap_list']]

        bars_antes = ax_ba.bar(x_indices - bar_width/2, res['initial_ap_list'], bar_width, label='Antes', color='#ef4444', alpha=0.85)
        bars_despues = ax_ba.bar(x_indices + bar_width/2, plot_res, bar_width, label='Después', color='#22c55e', alpha=0.85)

        for bar in bars_antes:
            h = bar.get_height()
            ax_ba.annotate(f'{h:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8)

        for bar, real_val in zip(bars_despues, res['residual_ap_list']):
            h = bar.get_height()
            ax_ba.annotate(f'{real_val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax_ba.set_xticks(x_indices)
        ax_ba.set_xticklabels(failures_labels)
        ax_ba.set_ylabel("Puntos AP")
        ax_ba.set_title("Impacto FMEA: Antes vs. Después")
        ax_ba.legend()
        fig_ba.tight_layout()

        st.pyplot(fig_ba)

    with col_right:
        st.subheader("Frontera de Pareto (128 Estados Combinatorios)")

        costs_all, risks_all, valid_costs, risks_valid = [], [], [], []
        for i in range(2 ** N_VARIABLES):
            b_str = format(i, f'0{N_VARIABLES}b')
            _, c_val, r_val = evaluate_bitstring_cost(b_str, res['budget'], res['initial_ap_list'])
            costs_all.append(c_val)
            risks_all.append(r_val)
            if c_val <= res['budget']:
                valid_costs.append(c_val)
                risks_valid.append(r_val)

        fig_pareto, ax_p = plt.subplots(figsize=(5, 5.5))
        ax_p.scatter(costs_all, risks_all, c='#94a3b8', s=20, alpha=0.5, label='128 Estados')
        if valid_costs:
            ax_p.scatter(valid_costs, risks_valid, c='#0284c7', s=35, alpha=0.7, label='Factibles (CAPEX OK)')
            
        ax_p.scatter([res['total_cost']], [res['residual_risk']], c='#eab308', s=150, marker='*', zorder=5, label=f"Óptimo |{res['best_bitstring']}>")
        ax_p.axvline(x=res['budget'], color='#ef4444', linestyle='--', alpha=0.7, label=f"Presupuesto (${res['budget']:,.0f})")
        
        ax_p.set_xlabel("Costo Total ($ USD)")
        ax_p.set_ylabel("Riesgo Residual")
        ax_p.legend(loc='upper right')
        fig_pareto.tight_layout()

        st.pyplot(fig_pareto)

        # Generar descarga de PDF
        pdf_bytes = create_pdf_bytes(res, fig_ba, fig_pareto)
        st.download_button(
            label="📄 DESCARGAR REPORTE PDF OFICIAL (AIAG-VDA 2022)",
            data=pdf_bytes,
            file_name=f"Reporte_Q-PAC_AMEF_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )

else:
    st.info("👈 Modifica los parámetros en la barra lateral y presiona 'EJECUTAR OPTIMIZACIÓN RECO ENGINE' para iniciar la simulación.")

### Cambios clave implementados:
1. **Zero Tkinter:** Eliminado cualquier rastro de `tkinter`, `ttk`, `messagebox` y `filedialog`.
2. **Interfaz de Streamlit:** Entradas numéricas y sliders adaptados en la barra lateral (`st.sidebar`) y paneles principales con columnas (`st.columns`).
3. **Reportes PDF al instante:** Tu motor de PDF en ReportLab genera los bytes del documento directamente en la memoria y se descargan con un clic mediante `st.download_button`.
