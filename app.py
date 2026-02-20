import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from fpdf import FPDF
from io import BytesIO

# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(page_title="Battery Sizer", layout="wide")
st.title("Battery Sizer - Simulation automatique")

# ==========================================================
# SIDEBAR PARAMETERS
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

# ==========================================================
# UPLOAD EXCEL
# ==========================================================
uploaded_file = st.file_uploader("Choisir un fichier Excel", type=["xlsx", "xls"])
if uploaded_file:
    with st.spinner("Lecture du fichier et détection des colonnes..."):
        df_full = pd.read_excel(uploaded_file, header=None)

        date_tokens = ["date", "datetime", "horodatage", "timestamp", "date/heure", "date heure"]
        import_tokens = ["soutirage", "import", "achat", "reseau", "consommation"]
        export_tokens = ["surplus", "surplus solaire", "export", "injection", "reinjection", "réinjection"]

        # Détection de l'en-tête
        def find_header_row(df, max_rows=120):
            for r in range(max_rows):
                row = df.iloc[r].astype(str).str.lower().str.strip().tolist()
                row_text = " | ".join(row)
                if any(t in row_text for t in date_tokens) and any(t in row_text for t in import_tokens) and any(t in row_text for t in export_tokens):
                    return r
            return None

        header_row = find_header_row(df_full)
        if header_row is None:
            st.error("Impossible de détecter la ligne d'en-tête.")
            st.stop()
        df = pd.read_excel(uploaded_file, header=header_row)

        # Détection colonnes
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
            st.error("Impossible de détecter automatiquement les colonnes date/import/export.")
            st.stop()

        # Conversion et calcul kWh
        df[imp_col] = pd.to_numeric(df[imp_col], errors='coerce').fillna(0)
        df[exp_col] = pd.to_numeric(df[exp_col], errors='coerce').fillna(0)
        df[date_col] = pd.to_datetime(df[date_col])

        if values_are_kw:
            df["import_kWh"] = df[imp_col] * dt_hours
            df["export_kWh"] = df[exp_col] * dt_hours
        else:
            df["import_kWh"] = df[imp_col]
            df["export_kWh"] = df[exp_col]

        st.success("Fichier et colonnes OK !")
        st.write("Aperçu des données converties :")
        st.dataframe(df.head())

    # ==========================================================
    # CALCUL CAPACITÉ MAX DYNAMIQUE
    # ==========================================================
    with st.spinner("Calcul de la capacité maximale dynamique..."):
        daily_export = df.groupby(df[date_col].dt.date)["export_kWh"].sum()
        cap_max_dyn = min(max(np.ceil(np.percentile(daily_export, daily_percentile*100)), cap_min), cap_max)
        st.sidebar.markdown(f"### Capacité max dynamique: **{cap_max_dyn} kWh**")

    # ==========================================================
    # SIMULATION VECTORISÉE
    # ==========================================================
    with st.spinner("Simulation et recherche du meilleur choix..."):
        exp_array = df["export_kWh"].values
        imp_array = df["import_kWh"].values
        eta = np.sqrt(roundtrip_eff)

        caps = np.arange(cap_min, cap_max_dyn+1, cap_step)
        powers = np.arange(p_min, p_max+1, p_step)
        results = []

        for cap in caps:
            for p in powers:
                p_step_val = p * dt_hours
                soc = np.zeros_like(exp_array)
                soc_val = 0
                exp_after = np.zeros_like(exp_array)
                imp_after = np.zeros_like(exp_array)

                charge = np.minimum(exp_array, p_step_val)
                discharge = np.minimum(imp_array, p_step_val)

                for i in range(len(exp_array)):
                    charge_i = min(exp_array[i], p_step_val, max(cap - soc_val,0))
                    soc_val += charge_i * eta
                    exp_after[i] = exp_array[i] - charge_i

                    discharge_i = min(imp_array[i], p_step_val, soc_val)
                    soc_val -= discharge_i / eta
                    imp_after[i] = imp_array[i] - discharge_i

                gain = (imp_array.sum() - imp_after.sum())*tariff_import - (exp_array.sum() - exp_after.sum())*tariff_export
                eq_cycles = (charge.sum() + discharge.sum())/(2*cap) if cap>0 else 0
                results.append([cap,p,gain,eq_cycles])

        results_df = pd.DataFrame(results, columns=["Cap_kWh","","Gain_CHF","Cycles"])
        gain_max = results_df["Gain_CHF"].max()
        threshold = gain_threshold * gain_max
        candidates = results_df[results_df["Gain_CHF"] >= threshold]
        best = candidates.sort_values(["Cap_kWh",""], ignore_index=True).iloc[0]

    st.success(f"🔋 Batterie optimale : {best.Cap_kWh} kWh / {best.} kW")
    st.write(f"Gain annuel: {round(best.Gain_CHF,2)} CHF")

    # ==========================================================
    # SOC VECTORISÉ
    # ==========================================================
    soc_val = 0
    soc_list = []
    p_step_val = best. * dt_hours
    for i in range(len(exp_array)):
        charge_i = min(exp_array[i], p_step_val, max(best.Cap_kWh - soc_val, 0))
        soc_val += charge_i * eta
        discharge_i = min(imp_array[i], p_step_val, soc_val)
        soc_val -= discharge_i / eta
        soc_list.append(soc_val)

    df["SOC"] = soc_list

    # ==========================================================
    # GRAPHIQUE SOC MATPLOTLIB
    # ==========================================================
    st.header("📈 État de charge batterie")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df[date_col], df["SOC"], label="SOC")
    ax.set_xlabel("Date")
    ax.set_ylabel("SOC (kWh)")
    ax.set_title("État de charge batterie optimisée")
    ax.legend()
    st.pyplot(fig)

    # ==========================================================
    # EXPORT PDF
    # ==========================================================
    with st.spinner("Génération PDF final..."):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Résumé Simulation Batterie Optimisée", ln=True, align="C")
        pdf.ln(10)

        # Indicateurs
        indicateurs = {
            "Capacité retenue (kWh)": best.Cap_kWh,
            "Puissance retenue (kW)": best.,
            "Rendement aller-retour": round(roundtrip_eff,2),
            "Import avant (kWh/an)": round(df["import_kWh"].sum(),2),
            "Export avant (kWh/an)": round(df["export_kWh"].sum(),2),
            "Import après (kWh/an)": round(imp_array.sum() - (imp_array - np.minimum(imp_array, p_step_val)).sum(),2),
            "Export après (kWh/an)": round(exp_array.sum() - (exp_array - np.minimum(exp_array, p_step_val)).sum(),2),
        }
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Indicateurs:", ln=True)
        pdf.set_font("Arial", '', 12)
        for k, v in indicateurs.items():
            pdf.cell(90, 8, str(k))
            pdf.cell(30, 8, str(v), ln=True)

        pdf.ln(5)

        # SOC Graphique matplotlib en mémoire
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png")
        img_buf.seek(0)
        pdf.image(img_buf, x=15, w=180)

        # Export PDF Streamlit
        pdf_buffer = BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)
        st.download_button(
            "Télécharger PDF final",
            pdf_buffer,
            file_name="bilan_batterie.pdf",
            mime="application/pdf"
        )

    st.success("PDF généré !")
