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

# Fonction de détection ligne d'en-tête
def find_header_row(df, date_tokens, import_tokens, export_tokens, max_rows=120):
    for r in range(min(max_rows, len(df))):
        row = df.iloc[r].astype(str).str.lower().str.strip().tolist()
        row_text = " | ".join(row)
        if any(t in row_text for t in date_tokens) \
           and any(t in row_text for t in import_tokens) \
           and any(t in row_text for t in export_tokens):
            return r
    return None

uploaded_file = st.file_uploader("Choisir un fichier Excel ou CSV", type=["xlsx", "xls", "csv"])
if uploaded_file:
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    date_tokens = ["date", "datetime", "horodatage", "timestamp", "date/heure", "date heure"]
    import_tokens = ["soutirage", "import", "achat", "reseau", "consommation"]
    export_tokens = ["surplus", "surplus solaire", "export", "excedent", "reinjection", "réinjection"]

    if file_type == "csv":
        # Lire sans header pour détecter la ligne d'en-tête
        df_full = pd.read_csv(uploaded_file, header=None, sep=';', engine='python')
        uploaded_file.seek(0)  # Remettre le curseur au début
    else:
        df_full = pd.read_excel(uploaded_file, header=None)
        uploaded_file.seek(0)  # Remettre le curseur au début

    with st.spinner("Détection de la ligne d'en-tête…"):
        header_row = find_header_row(df_full, date_tokens, import_tokens, export_tokens)

    if header_row is None:
        st.error("Impossible de détecter la ligne d'en-tête dans les 120 premières lignes.")
        st.stop()
    else:
        st.success(f"Ligne d'en-tête détectée : {header_row + 1}")

        if file_type == "csv":
            df = pd.read_csv(uploaded_file, header=header_row, sep=';', engine='python')
        else:
            df = pd.read_excel(uploaded_file, header=header_row)

    st.write("Aperçu des 5 premières lignes du fichier :")
    st.dataframe(df.head())


    # Détection automatique des colonnes
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
    else:
        st.success(f"Colonnes détectées : date={date_col}, import={imp_col}, export={exp_col}")

    # Conversion en datetime, gestion automatique du timezone

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', utc=True)  # convertit les dates, NaN si erreur
    df[date_col] = df[date_col].dt.tz_convert(None)  # supprime timezone

        
    with st.spinner("Conversion des données en kWh…"):
        df[imp_col] = pd.to_numeric(df[imp_col], errors='coerce').fillna(0)
        df[exp_col] = pd.to_numeric(df[exp_col], errors='coerce').fillna(0)
        df[date_col] = pd.to_datetime(df[date_col])
        if values_are_kw:
            df["import_kWh"] = df[imp_col] * dt_hours
            df["export_kWh"] = df[exp_col] * dt_hours
        else:
            df["import_kWh"] = df[imp_col]
            df["export_kWh"] = df[exp_col]

    st.write("✅ Aperçu des données converties :")
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

        results_df = pd.DataFrame(results, columns=["Cap_kWh","Power_kW","Gain_CHF","Cycles"])
        gain_max = results_df["Gain_CHF"].max()
        threshold = gain_threshold * gain_max
        candidates = results_df[results_df["Gain_CHF"] >= threshold]
        best = candidates.sort_values(["Cap_kWh","Power_kW"], ignore_index=True).iloc[0]

    st.success(f"🔋 Batterie optimale : {best.Cap_kWh} kWh / {best.Power_kW} kW")
    st.write(f"Gain annuel: {round(best.Gain_CHF,2)} CHF")

    # ==========================================================
    # SOC VECTORISÉ
    # ==========================================================
    soc_val = 0
    soc_list = []
    p_step_val = best.Power_kW * dt_hours
    for i in range(len(exp_array)):
        charge_i = min(exp_array[i], p_step_val, max(best.Cap_kWh - soc_val, 0))
        soc_val += charge_i * eta
        discharge_i = min(imp_array[i], p_step_val, soc_val)
        soc_val -= discharge_i / eta
        soc_list.append(soc_val)

    df["SOC_pct"] = [ (s / best.Cap_kWh)*100 for s in soc_list ]

    # ==========================================================
    # GRAPHIQUE SOC MATPLOTLIB
    # ==========================================================
    st.header("📈 État de charge batterie (%)")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df[date_col], df["SOC_pct"], label="SOC (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("SOC (%)")
    ax.set_title("État de charge batterie optimisée")
    ax.legend()
    st.pyplot(fig)

    # ==========================================================
    # PRÉPARATION DES DONNÉES
    # ==========================================================
    df = df.sort_values(by=date_col)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    
    # Nettoyage
    df["import_kWh"] = df["import_kWh"].clip(lower=0).fillna(0)
    df["export_kWh"] = df["export_kWh"].clip(lower=0).fillna(0)
    
    imp_after = pd.Series(imp_after, index=df.index).clip(lower=0).fillna(0)
    exp_after = pd.Series(exp_after, index=df.index).clip(lower=0).fillna(0)
    
    # ==========================================================
    # CHOIX AGRÉGATION
    # ==========================================================
    aggregation_choice = st.selectbox(
        "📅 Niveau d'agrégation",
        ["Journalier", "Hebdomadaire", "Mensuel"]
    )
    freq_map = {
        "Journalier": "D",
        "Hebdomadaire": "W",
        "Mensuel": "M"
    }
    freq = freq_map[aggregation_choice]
    
    # ==========================================================
    # AGRÉGATION
    # ==========================================================
    before_agg = df[["import_kWh", "export_kWh"]].resample(freq).sum()
    
    after_agg = pd.DataFrame({
        "import_after": imp_after,
        "export_after": exp_after
    }).resample(freq).sum()
    
    # ==========================================================
    # GRAPHIQUE AVANT
    # ==========================================================
    
    st.header("📊 Import / Export AVANT optimisation")
    fig_before, ax_before = plt.subplots(figsize=(12,5))
    ax_before.plot(before_agg.index, before_agg["import_kWh"], label="Import (kWh)")
    ax_before.plot(before_agg.index, before_agg["export_kWh"], label="Export (kWh)")
    ax_before.set_ylabel("Énergie (kWh)")
    ax_before.set_xlabel("Date")
    ax_before.set_title("Import / Export AVANT optimisation")
    ax_before.legend()
    ax_before.grid(alpha=0.3)
    st.pyplot(fig_before)
    
    # ==========================================================
    # GRAPHIQUE APRÈS
    # ==========================================================
    st.header("📊 Import / Export APRÈS optimisation")
    fig_after, ax_after = plt.subplots(figsize=(12,5))
    ax_after.plot(after_agg.index, after_agg["import_after"], label="Import après (kWh)")
    ax_after.plot(after_agg.index, after_agg["export_after"], label="Export après (kWh)")
    ax_after.set_ylabel("Énergie (kWh)")
    ax_after.set_xlabel("Date")
    ax_after.set_title("Import / Export APRÈS optimisation")
    ax_after.legend()
    ax_after.grid(alpha=0.3)
    st.pyplot(fig_after)

    # ==========================================================
    # CALCULS POUR LE RAPPORT
    # ==========================================================
    # Import/Export avant
    import_before = df["import_kWh"].sum()
    export_before = df["export_kWh"].sum()
    
    # Simulation SOC pour batterie optimale
    soc_val = 0
    soc_list = []
    p_step_val = best.Power_kW * dt_hours
    imp_after = np.zeros_like(imp_array)
    exp_after = np.zeros_like(exp_array)
    charge_total = 0
    discharge_total = 0
    
    for i in range(len(exp_array)):
        # Charge batterie
        charge_i = min(exp_array[i], p_step_val, max(best.Cap_kWh - soc_val, 0))
        soc_val += charge_i * eta
        exp_after[i] = exp_array[i] - charge_i
        charge_total += charge_i
    
        # Décharge batterie
        discharge_i = min(imp_array[i], p_step_val, soc_val)
        soc_val -= discharge_i / eta
        imp_after[i] = imp_array[i] - discharge_i
        discharge_total += discharge_i
    
        soc_list.append(soc_val)
    
    df["SOC"] = soc_list
    
    # Gains et cycles
    import_avoided = imp_array.sum() - imp_after.sum()
    export_avoided = exp_array.sum() - exp_after.sum()
    gain_net = import_avoided * tariff_import - export_avoided * tariff_export
    eq_cycles = (charge_total + discharge_total) / (2 * best.Cap_kWh)
    
    # ==========================================================
    # EXPORT PDF COMPLET
    # ==========================================================
    with st.spinner("Génération PDF final..."):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Résumé Simulation Batterie Optimisée", ln=True, align="C")
        pdf.ln(10)
    
        # Paramètres
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Paramètres de simulation :", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(90, 8, "Tarif achat réseau (CHF/kWh)"); pdf.cell(30, 8, f"{tariff_import}", ln=True)
        pdf.cell(90, 8, "Tarif reprise injection (CHF/kWh)"); pdf.cell(30, 8, f"{tariff_export}", ln=True)
        pdf.cell(90, 8, "Percentile export journalier"); pdf.cell(30, 8, f"{daily_percentile:.2f}", ln=True)
        pdf.ln(5)
    
        # Batterie recommandée
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Batterie recommandée :", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(90, 8, "Capacité (kWh)"); pdf.cell(30, 8, f"{best.Cap_kWh}", ln=True)
        pdf.cell(90, 8, "Puissance (kW)"); pdf.cell(30, 8, f"{best.Power_kW}", ln=True)
        pdf.cell(90, 8, "Gain annuel choisi (CHF/an)"); pdf.cell(30, 8, f"{gain_net:.2f}", ln=True)
        pdf.cell(90, 8, "Gain maximum (CHF/an)"); pdf.cell(30, 8, f"{gain_max:.2f}", ln=True)
        pdf.cell(90, 8, "Seuil 95% (CHF/an)"); pdf.cell(30, 8, f"{threshold:.2f}", ln=True)
        pdf.cell(90, 8, "Capacité max dynamique (kWh)"); pdf.cell(30, 8, f"{cap_max_dyn}", ln=True)
        pdf.ln(5)
    
        # Résultats annuels
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, "Résultats annuels estimés :", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(90, 8, "SOC moyen (%)"); pdf.cell(30, 8, f"{np.mean(df['SOC_pct']):.1f}", ln=True)
        pdf.cell(90, 8, "SOC max (%)"); pdf.cell(30, 8, f"{np.max(df['SOC_pct']):.1f}", ln=True)
        pdf.cell(90, 8, "Import avant (kWh/an)"); pdf.cell(30, 8, f"{import_before:.2f}", ln=True)
        pdf.cell(90, 8, "Export avant (kWh/an)"); pdf.cell(30, 8, f"{export_before:.2f}", ln=True)
        pdf.cell(90, 8, "Import évité (kWh/an)"); pdf.cell(30, 8, f"{import_avoided:.2f}", ln=True)
        pdf.cell(90, 8, "Export évité (kWh/an)"); pdf.cell(30, 8, f"{export_avoided:.2f}", ln=True)
        pdf.cell(90, 8, "Cycles équivalents/an"); pdf.cell(30, 8, f"{eq_cycles:.2f}", ln=True)
        pdf.cell(90, 8, "Gain net (CHF/an)"); pdf.cell(30, 8, f"{gain_net:.2f}", ln=True)
        pdf.ln(30)
    
        # SOC Graphique matplotlib en mémoire
        img_buf = BytesIO()
        fig.savefig(img_buf, format="png")
        img_buf.seek(0)
        pdf.image(img_buf, x=15, w=180)

        # Graphique AVANT
        img_buf_before = BytesIO()
        fig_before.savefig(img_buf_before, format="png")
        img_buf_before.seek(0)
        pdf.image(img_buf_before, x=15, w=180)
        pdf.ln(5)
        
        # Graphique APRÈS
        img_buf_after = BytesIO()
        fig_after.savefig(img_buf_after, format="png")
        img_buf_after.seek(0)
        pdf.image(img_buf_after, x=15, w=180)
        pdf.ln(5)
    
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
