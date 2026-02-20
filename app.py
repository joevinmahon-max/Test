import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(page_title="Battery Sizer", layout="wide")
st.title("Battery Sizer - Simulation automatique")

# ==========================================================
# SIDEBAR PARAMETERS (ALL HARD-CODE REMOVED)
# ==========================================================

st.sidebar.header("⚙️ Paramètres")

dt_hours = st.sidebar.number_input("Pas de temps (heures)", value=0.25)

values_are_kw = st.sidebar.checkbox("Valeurs en kW (sinon kWh)", value=True)

tariff_import = st.sidebar.number_input("Tarif import (CHF/kWh)", value=0.32)
tariff_export = st.sidebar.number_input("Tarif export (CHF/kWh)", value=0.08)

roundtrip_eff = st.sidebar.slider("Rendement aller-retour", 0.5, 1.0, 0.92)

cap_min = st.sidebar.number_input("Capacité min (kWh)", 1, 100, 5)
cap_max = st.sidebar.number_input("Capacité max (kWh)", 1, 200, 30)
cap_step = st.sidebar.number_input("Pas capacité (kWh)", 1, 20, 1)

p_min = st.sidebar.number_input("Puissance min (kW)", 1, 50, 3)
p_max = st.sidebar.number_input("Puissance max (kW)", 1, 100, 10)
p_step = st.sidebar.number_input("Pas puissance (kW)", 1, 20, 1)

gain_threshold = st.sidebar.slider("Seuil % du gain max", 0.5, 1.0, 0.95)
daily_percentile = st.sidebar.slider("Percentile export journalier (Pxx)", 0.5, 0.99, 0.8)

# ===== Upload Excel =====
uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx", "xls"])
if uploaded_file:
    # On lit toutes les lignes possibles pour trouver l'en-tête
    df_full = pd.read_excel(uploaded_file, header=None)  # pas de header initial

    # Tokens comme en VBA
    date_tokens = ["date", "datetime", "horodatage", "timestamp", "date/heure", "date heure"]
    import_tokens = ["soutirage", "import", "achat", "reseau", "consommation"]
    export_tokens = ["surplus", "surplus solaire", "export", "injection", "reinjection", "réinjection"]

    def find_header_row(df, max_rows=120):
        for r in range(max_rows):
            row = df.iloc[r].astype(str).str.lower().str.strip().tolist()
            row_text = " | ".join(row)
            if any(t in row_text for t in date_tokens) \
               and any(t in row_text for t in import_tokens) \
               and any(t in row_text for t in export_tokens):
                return r
        return None

    header_row = find_header_row(df_full)
    if header_row is None:
        st.error("Impossible de détecter la ligne d'en-tête dans les 120 premières lignes.")
    else:
        st.success(f"Ligne d'en-tête détectée : {header_row+1}")
        df = pd.read_excel(uploaded_file, header=header_row)

        # Fonction pour trouver les colonnes automatiquement
        def find_column(df, tokens):
            for col in df.columns:
                col_lower = str(col).lower()
                for t in tokens:
                    if t in col_lower:
                        return col
            return None

        date_col = find_column(df, date_tokens)
        imp_col = find_column(df, import_tokens)
        exp_col = find_column(df, export_tokens)

        if date_col is None or imp_col is None or exp_col is None:
            st.error("Impossible de détecter automatiquement les colonnes date/import/export après lecture de la ligne d'en-tête.")
        else:
            st.success(f"Colonnes détectées : date={date_col}, import={imp_col}, export={exp_col}")
            
            # Conversion en numérique, remplacer tout non-numérique par 0
            df[imp_col] = pd.to_numeric(df[imp_col], errors='coerce').fillna(0)
            df[exp_col] = pd.to_numeric(df[exp_col], errors='coerce').fillna(0)
            df[date_col] = pd.to_datetime(df[date_col])
            
            BAT_DT_HOURS = 0.25  # 15 min
            BAT_VALUES_ARE_KW = True

            if BAT_VALUES_ARE_KW:
                df["import_kWh"] = df[imp_col] * BAT_DT_HOURS
                df["export_kWh"] = df[exp_col] * BAT_DT_HOURS
            else:
                df["import_kWh"] = df[imp_col]
                df["export_kWh"] = df[exp_col]

            st.write("Aperçu des données converties :")
            st.dataframe(df.head())

    # ==========================================================
    # DYNAMIC CAP FROM DAILY EXPORT
    # ==========================================================

    daily_export = df.groupby(df[date_col].dt.date)["export_kWh"].sum()
    cap_max_dyn = np.ceil(np.percentile(daily_export, daily_percentile*100))

    cap_max_dyn = min(cap_max_dyn, cap_max)
    cap_max_dyn = max(cap_max_dyn, cap_min)

    st.sidebar.markdown(f"### Capacité max dynamique: **{cap_max_dyn} kWh**")

    # ==========================================================
    # QUICK SIM FUNCTION
    # ==========================================================

    def simulate(cap_kwh, p_kw):
        eta = np.sqrt(roundtrip_eff)
        soc = 0
        p_step = p_kw * dt_hours

        imp_after = 0
        exp_after = 0
        sum_charge = 0
        sum_dis = 0

        for i,row in df.iterrows():

            imp = row["import_kWh"]
            exp = row["export_kWh"]

            charge = min(exp, p_step, max(cap_kwh - soc,0))
            soc += charge * eta
            exp_after += exp - charge

            discharge = min(imp, p_step, soc)
            soc -= discharge / eta
            imp_after += imp - discharge

            sum_charge += charge
            sum_dis += discharge

        eq_cycles = (sum_charge + sum_dis)/(2*cap_kwh) if cap_kwh>0 else 0

        imp_before = df["import_kWh"].sum()
        exp_before = df["export_kWh"].sum()

        gain = (imp_before - imp_after)*tariff_import - (exp_before - exp_after)*tariff_export

        return gain, imp_after, exp_after, eq_cycles

    # ==========================================================
    # GRID SEARCH
    # ==========================================================

    results = []

    for cap in np.arange(cap_min, cap_max_dyn+1, cap_step):
        for p in np.arange(p_min, p_max+1, p_step):

            gain, imp_a, exp_a, cyc = simulate(cap,p)
            results.append([cap,p,gain,cyc])

    results = pd.DataFrame(results, columns=["Cap_kWh","Power_kW","Gain_CHF","Cycles"])

    gain_max = results["Gain_CHF"].max()
    threshold = gain_threshold * gain_max

    candidates = results[results["Gain_CHF"]>=threshold]
    best = candidates.sort_values(["Cap_kWh","Power_kW"]).iloc[0]

    st.success(f"🔋 Batterie optimale : {best.Cap_kWh} kWh / {best.Power_kW} kW")
    st.write(f"Gain annuel: {round(best.Gain_CHF,2)} CHF")

    # ==========================================================
    # GRAPH 1 – Gain surface
    # ==========================================================

    fig = go.Figure(data=[go.Scatter3d(
        x=results["Cap_kWh"],
        y=results["Power_kW"],
        z=results["Gain_CHF"],
        mode='markers',
        marker=dict(size=4,color=results["Gain_CHF"],colorscale="Viridis")
    )])

    fig.update_layout(title="Surface Gain CHF")
    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # DETAILED SIM BEST
    # ==========================================================

    eta = np.sqrt(roundtrip_eff)
    soc=0
    soc_list=[]

    for i,row in df.iterrows():
        imp=row["import_kWh"]
        exp=row["export_kWh"]

        charge=min(exp,best.Power_kW*dt_hours,max(best.Cap_kWh-soc,0))
        soc+=charge*eta

        discharge=min(imp,best.Power_kW*dt_hours,soc)
        soc-=discharge/eta

        soc_list.append(soc)

    df["SOC"] = soc_list

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df[date_col], y=df["SOC"], name="SoC"))
    fig2.update_layout(title="État de charge batterie")
    st.plotly_chart(fig2, use_container_width=True)

    # ==========================================================
    # GENERATION PDF FINAL
    # ==========================================================
    from fpdf import FPDF
    import plotly.io as pio
    
    # Sauvegarde le graphique SoC en PNG
    fig2.write_image("soc_plot.png")
    
    # Crée le PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Résumé Simulation Batterie Optimisée", ln=True, align="C")
    pdf.ln(10)
    
    # Indicateurs
    pdf.set_font("Arial", '', 12)
    indicateurs = {
        "Capacité retenue (kWh)": best.Cap_kWh,
        "Puissance retenue (kW)": best.Power_KW,
        "Rendement aller-retour": round(roundtrip_eff,2),
        "Import avant (kWh/an)": round(df["import_kWh"].sum(),2),
        "Export avant (kWh/an)": round(df["export_kWh"].sum(),2),
        "Import après (kWh/an)": round(simulate(best.Cap_kWh,best.Power_KW)[1],2),
        "Export après (kWh/an)": round(simulate(best.Cap_kWh,best.Power_KW)[2],2),
        "Import évité (kWh/an)": round(df["import_kWh"].sum() - simulate(best.Cap_kWh,best.Power_KW)[1],2),
        "Export évité (kWh/an)": round(df["export_kWh"].sum() - simulate(best.Cap_kWh,best.Power_KW)[2],2),
        "Cycles équivalents/an": round(simulate(best.Cap_kWh,best.Power_KW)[3],2),
        "Gain net (CHF/an)": round(best.Gain_CHF,2)
    }
    
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Indicateurs:", ln=True)
    pdf.set_font("Arial", '', 12)
    for k,v in indicateurs.items():
        pdf.cell(90, 8, str(k))
        pdf.cell(30, 8, str(v), ln=True)
    
    pdf.ln(5)
    
    # Meilleur choix
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Meilleur choix:", ln=True)
    pdf.set_font("Arial", '', 12)
    best_choice = {
        "Capacité retenue (kWh)": best.Cap_kWh,
        "Puissance retenue (kW)": best.Power_KW,
        "Gain choisi (CHF/an)": best.Gain_CHF,
        "Gain maximum (CHF/an)": results["Gain_CHF"].max(),
        "Seuil 95% (CHF/an)": gain_threshold * results["Gain_CHF"].max(),
        "Cap max dynamique (kWh)": cap_max_dyn,
        "Percentile export journalier": f"P{int(daily_percentile*100)}"
    }
    for k,v in best_choice.items():
            pdf.cell(90, 8, str(k))
            pdf.cell(30, 8, str(v), ln=True)
        
        pdf.ln(5)
        
        # Ajouter le graphique SoC
        pdf.image("soc_plot.png", x=15, w=180)
        
        # Export PDF
        pdf_file = "bilan_batterie.pdf"
        pdf.output(pdf_file)
        
        # Téléchargement depuis Streamlit
        with open(pdf_file, "rb") as f:
        st.download_button("Télécharger PDF final", f, file_name="bilan_batterie.pdf", mime="application/pdf")

    # ==========================================================
    # SUMMARY
    # ==========================================================

    st.header("📊 Résumé annuel")

    st.dataframe(results.sort_values("Gain_CHF",ascending=False).head(10))
