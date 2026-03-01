import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

st.set_page_config(page_title="Awesome Dashboard", layout="wide")
st.title("Awesome Dashboard")

# ===================== SCENARIO SAVE/LOAD =====================

SCENARIOS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_scenarios")
os.makedirs(SCENARIOS_DIR, exist_ok=True)

# All widget keys and their default values
CONFIG_DEFAULTS = {
    # General
    "cfg_total_months": 60, "cfg_phase1_dur": 3, "cfg_phase2_dur": 3,
    # Scenarios
    "cfg_sens_conv": 0, "cfg_sens_churn": 0, "cfg_sens_cpi": 0, "cfg_sens_organic": 0,
    "cfg_scenario_bound": 20, "cfg_mc_enabled": False, "mc_iter": 200, "mc_var": 20.0,
    # Taxes
    "cfg_corp_tax": 1.0, "cfg_store_split": 50, "cfg_app_store_comm": 15.0,
    "cfg_web_comm_pct": 3.5, "cfg_web_comm_fixed": 0.50, "cfg_bank_fee": 1.0,
    # Phase 1
    "cfg_p1_inv": 100000.0, "cfg_p1_sal": 17475.0, "cfg_p1_misc": 8419.0,
    "cfg_p1_ad": 0.0, "cfg_p1_cpi": 7.50, "cfg_p1_ct": 0.0, "cfg_p1_cp": 0.0,
    # Phase 2
    "cfg_p2_inv": 0.0, "cfg_p2_sal": 3600.0, "cfg_p2_misc": 750.0,
    "cfg_p2_ad": 5000.0, "cfg_p2_cpi": 7.50, "cfg_p2_ct": 20.0, "cfg_p2_cp": 20.0,
    # Phase 3
    "cfg_p3_inv": 0.0, "cfg_p3_sal": 64800.0, "cfg_p3_misc": 13500.0,
    "cfg_p3_ad": 150000.0, "cfg_p3_cpi": 7.50, "cfg_p3_ct": 25.0, "cfg_p3_cp": 25.0,
    # Ad Growth
    "p2_adg_mode": "Percentage (%)", "p2_adg_pct": 5.0, "p2_adg_abs": 5000.0, "p2_cpid": 1.0,
    "p3_adg_mode": "Percentage (%)", "p3_adg_pct": 5.0, "p3_adg_abs": 5000.0, "p3_cpid": 1.0,
    # Organic
    "cfg_org_start": 0.0,
    "p1_ogm": "Percentage (%)", "p1_ogp": 0.0, "p1_oga": 0.0, "p1_oct": 0.0, "p1_ocp": 0.0, "p1_osp": 0.0,
    "p2_ogm": "Percentage (%)", "p2_ogp": 10.0, "p2_oga": 50.0, "p2_oct": 25.0, "p2_ocp": 25.0, "p2_osp": 500.0,
    "p3_ogm": "Percentage (%)", "p3_ogp": 15.0, "p3_oga": 500.0, "p3_oct": 35.0, "p3_ocp": 35.0, "p3_osp": 2000.0,
    # Pricing
    "pr_w": 4.99, "pr_m": 7.99, "pr_a": 49.99, "pp_pr": False,
    "p2_pw": 4.99, "p2_pm": 7.99, "p2_pa": 49.99,
    "p3_pw": 4.99, "p3_pm": 7.99, "p3_pa": 49.99,
    # Mix
    "mx_w": 0.0, "mx_m": 48.0, "mx_a": 52.0, "pp_mx": False,
    "p2_mw": 0.0, "p2_mm": 48.0, "p2_ma": 52.0,
    "p3_mw": 0.0, "p3_mm": 48.0, "p3_ma": 52.0,
    # Retention
    "cfg_wk_cancel": 15.0, "cfg_mo_churn": 10.0, "cfg_yr_nonrenew": 30.0,
    "cfg_p2_churn_mult": 1.5, "cfg_p3_churn_mult": 1.0,
    # Trial & COGS
    "cfg_trial_days": 7, "cfg_refund_rate": 2.0, "cfg_cogs_global": 0.10,
    "pp_cogs": False, "p1_cogs": 0.05, "p2_cogs": 0.15, "p3_cogs": 0.08,
    # Expansion
    "cfg_upgrade_rate": 2.0, "cfg_downgrade_rate": 5.0,
}


def list_saved_scenarios():
    files = [f[:-5] for f in os.listdir(SCENARIOS_DIR) if f.endswith(".json")]
    files.sort()
    return files


def save_scenario(name, notes=""):
    data = {"name": name, "notes": notes, "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "params": {}}
    for key in CONFIG_DEFAULTS:
        if key in st.session_state:
            val = st.session_state[key]
            if isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            data["params"][key] = val
    filepath = os.path.join(SCENARIOS_DIR, f"{name}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filepath


def load_scenario(name):
    filepath = os.path.join(SCENARIOS_DIR, f"{name}.json")
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key, val in data.get("params", {}).items():
        st.session_state[key] = val
    return data


def delete_scenario(name):
    filepath = os.path.join(SCENARIOS_DIR, f"{name}.json")
    if os.path.exists(filepath):
        os.remove(filepath)


# ===================== SIDEBAR =====================

# --- Scenario Management ---
with st.sidebar.expander("Scenario Management", expanded=False):
    saved_list = list_saved_scenarios()

    st.markdown("**Save Current**")
    save_name = st.text_input("Scenario Name", value="", key="save_name_input",
        help="Введите имя для сохранения текущей конфигурации.")
    save_notes = st.text_input("Notes (optional)", value="", key="save_notes_input")
    if st.button("Save", key="btn_save"):
        if save_name.strip():
            save_scenario(save_name.strip(), save_notes.strip())
            st.success(f"Saved: {save_name.strip()}")
            st.rerun()
        else:
            st.warning("Введите имя сценария.")

    st.markdown("---")
    st.markdown("**Load Saved**")
    if saved_list:
        load_choice = st.selectbox("Select Scenario", saved_list, key="load_choice_select")
        lc1, lc2 = st.columns(2)
        if lc1.button("Load", key="btn_load"):
            loaded = load_scenario(load_choice)
            st.success(f"Loaded: {load_choice}")
            if loaded.get("notes"):
                st.info(f"Notes: {loaded['notes']}")
            st.rerun()
        if lc2.button("Delete", key="btn_delete"):
            delete_scenario(load_choice)
            st.success(f"Deleted: {load_choice}")
            st.rerun()

        # Show info about selected scenario
        info_path = os.path.join(SCENARIOS_DIR, f"{load_choice}.json")
        if os.path.exists(info_path):
            with open(info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            st.caption(f"Saved: {info.get('saved_at', '?')} | Notes: {info.get('notes', '—')}")
    else:
        st.info("No saved scenarios yet.")

# --- General Settings ---
with st.sidebar.expander("General Settings", expanded=True):
    total_months = st.number_input("Projection Horizon (months)", min_value=12, max_value=120, value=60, key="cfg_total_months",
        help="Горизонт моделирования. По умолчанию 60 месяцев (5 лет).")
    phase1_dur = st.number_input("Phase 1 Duration (months)", min_value=1, max_value=48, value=3, key="cfg_phase1_dur",
        help="Длительность фазы Pre-MVP — разработка, нет рекламы и пользователей.")
    phase2_dur = st.number_input("Phase 2 Duration (months)", min_value=1, max_value=48, value=3, key="cfg_phase2_dur",
        help="Длительность фазы MVP — мягкий запуск, первые пользователи.")
    phase3_dur = total_months - phase1_dur - phase2_dur
    if phase3_dur < 1:
        st.error(f"Phase 1 + Phase 2 должны быть < {total_months} месяцев!")
        st.stop()
    st.caption(f"Phase 3 (Scaling): {phase3_dur} мес.")

# --- Scenario Sensitivity ---
with st.sidebar.expander("Scenario Sensitivity"):
    st.markdown("**Per-Variable Sensitivity**")
    sens_conv = st.slider("Conversion Sensitivity %", -50, 50, 0, key="cfg_sens_conv",
        help="Корректировка конверсий. + = лучше конверсии, - = хуже.")
    sens_churn = st.slider("Churn Sensitivity %", -50, 50, 0, key="cfg_sens_churn",
        help="Корректировка оттока. + = больше отток (хуже), - = меньше (лучше).")
    sens_cpi = st.slider("CPI Sensitivity %", -50, 50, 0, key="cfg_sens_cpi",
        help="Корректировка CPI. + = дороже установки, - = дешевле.")
    sens_organic = st.slider("Organic Sensitivity %", -50, 50, 0, key="cfg_sens_organic",
        help="Корректировка органического роста. + = больше органики.")
    scenario_bound = st.slider("Scenario Bound %", 5, 50, 20, key="cfg_scenario_bound",
        help="Границы пессимистичного/оптимистичного сценариев. Применяются ко всем переменным.")
    st.markdown("---")
    mc_enabled = st.checkbox("Monte Carlo Simulation", value=False, key="cfg_mc_enabled",
        help="Запустить N итераций с рандомизацией параметров для получения распределения исходов.")
    if mc_enabled:
        mc_iterations = st.number_input("MC Iterations", min_value=50, max_value=1000, value=200, key="mc_iter")
        mc_variance = st.number_input("MC Max Variance %", min_value=5.0, max_value=50.0, value=20.0, key="mc_var")
    else:
        mc_iterations = 200
        mc_variance = 20.0

# --- Taxes & Payment Fees ---
with st.sidebar.expander("Taxes & Payment Fees"):
    corporate_tax = st.number_input("Corporate Tax %", min_value=0.0, max_value=50.0, value=1.0, key="cfg_corp_tax",
        help="Налог на прибыль. 1% Грузия, 5% Армения, 3-10% Казахстан.")
    store_split = st.slider("Store vs Web % (Store)", 0, 100, 50, key="cfg_store_split",
        help="Процент пользователей, покупающих через App Store.")
    app_store_comm = st.number_input("App Store Commission %", min_value=0.0, max_value=50.0, value=15.0, key="cfg_app_store_comm",
        help="Комиссия App Store. 15% по Small Business Program до $1M/год.")
    web_comm_pct = st.number_input("Web Commission %", min_value=0.0, max_value=20.0, value=3.5, key="cfg_web_comm_pct",
        help="Процентная комиссия Lemon Squeezy / Paddle.")
    web_comm_fixed = st.number_input("Web Fixed Fee per Txn ($)", min_value=0.0, max_value=5.0, value=0.50, key="cfg_web_comm_fixed",
        help="Фиксированная комиссия за каждую Web-транзакцию.")
    bank_fee = st.number_input("Banking Fee %", min_value=0.0, max_value=10.0, value=1.0, key="cfg_bank_fee",
        help="Комиссия банка за переводы и конвертации.")

# --- Phase 1: Pre-MVP ---
with st.sidebar.expander("Phase 1: Pre-MVP"):
    p1_investment = st.number_input("Phase 1 Investment ($)", min_value=0.0, value=100000.0, key="cfg_p1_inv",
        help="Инвестиции в начале Phase 1.")
    p1_salaries_total = st.number_input("Phase 1 Total Salaries ($)", min_value=0.0, value=17475.0, key="cfg_p1_sal",
        help="Общая сумма зарплат за всю фазу.")
    p1_misc_total = st.number_input("Phase 1 Total Misc Costs ($)", min_value=0.0, value=8419.0, key="cfg_p1_misc",
        help="Прочие расходы за всю фазу.")
    p1_ad_budget = st.number_input("Phase 1 Monthly Ad Budget ($)", min_value=0.0, value=0.0, key="cfg_p1_ad",
        help="Рекламный бюджет в месяц. Обычно 0 на Pre-MVP.")
    p1_cpi = st.number_input("Phase 1 CPI ($)", min_value=0.01, value=7.50, key="cfg_p1_cpi",
        help="Стоимость одной установки.")
    p1_conv_trial = st.number_input("Phase 1 Conv. to Trial %", min_value=0.0, max_value=100.0, value=0.0, key="cfg_p1_ct",
        help="Конверсия установки в триал. На Pre-MVP обычно 0.")
    p1_conv_paid = st.number_input("Phase 1 Trial-to-Paid %", min_value=0.0, max_value=100.0, value=0.0, key="cfg_p1_cp",
        help="Конверсия триала в оплату. На Pre-MVP обычно 0.")

# --- Phase 2: MVP ---
with st.sidebar.expander("Phase 2: MVP"):
    p2_investment = st.number_input("Phase 2 Investment ($)", min_value=0.0, value=0.0, key="cfg_p2_inv",
        help="Дополнительные инвестиции в начале Phase 2.")
    p2_salaries_total = st.number_input("Phase 2 Total Salaries ($)", min_value=0.0, value=3600.0, key="cfg_p2_sal",
        help="Общая сумма зарплат за фазу MVP.")
    p2_misc_total = st.number_input("Phase 2 Total Misc Costs ($)", min_value=0.0, value=750.0, key="cfg_p2_misc",
        help="Прочие расходы за фазу MVP.")
    p2_ad_budget = st.number_input("Phase 2 Monthly Ad Budget ($)", min_value=0.0, value=5000.0, key="cfg_p2_ad",
        help="Стартовый рекламный бюджет на MVP.")
    p2_cpi = st.number_input("Phase 2 CPI ($)", min_value=0.01, value=7.50, key="cfg_p2_cpi",
        help="Стоимость установки на этапе MVP.")
    p2_conv_trial = st.number_input("Phase 2 Conv. to Trial %", min_value=0.0, max_value=100.0, value=20.0, key="cfg_p2_ct",
        help="Конверсия в триал на MVP.")
    p2_conv_paid = st.number_input("Phase 2 Trial-to-Paid %", min_value=0.0, max_value=100.0, value=20.0, key="cfg_p2_cp",
        help="Конверсия в оплату на MVP.")

# --- Phase 3: Scaling ---
with st.sidebar.expander("Phase 3: Scaling"):
    p3_investment = st.number_input("Phase 3 Investment ($)", min_value=0.0, value=0.0, key="cfg_p3_inv",
        help="Дополнительные инвестиции в начале Phase 3.")
    p3_salaries_total = st.number_input("Phase 3 Total Salaries ($)", min_value=0.0, value=64800.0, key="cfg_p3_sal",
        help="Общая сумма зарплат за фазу масштабирования.")
    p3_misc_total = st.number_input("Phase 3 Total Misc Costs ($)", min_value=0.0, value=13500.0, key="cfg_p3_misc",
        help="Прочие расходы за фазу масштабирования.")
    p3_ad_budget = st.number_input("Phase 3 Monthly Ad Budget ($)", min_value=0.0, value=150000.0, key="cfg_p3_ad",
        help="Стартовый рекламный бюджет на этапе масштабирования.")
    p3_cpi = st.number_input("Phase 3 CPI ($)", min_value=0.01, value=7.50, key="cfg_p3_cpi",
        help="Стоимость установки на зрелом продукте.")
    p3_conv_trial = st.number_input("Phase 3 Conv. to Trial %", min_value=0.0, max_value=100.0, value=25.0, key="cfg_p3_ct",
        help="Конверсия в триал на зрелом продукте.")
    p3_conv_paid = st.number_input("Phase 3 Trial-to-Paid %", min_value=0.0, max_value=100.0, value=25.0, key="cfg_p3_cp",
        help="Конверсия в оплату на зрелом продукте.")

# --- Ad Budget Growth (per phase) ---
with st.sidebar.expander("Ad Budget Growth"):
    st.markdown("**Phase 2: MVP**")
    p2_ad_growth_mode = st.radio("P2 Growth Mode", ["Percentage (%)", "Absolute ($)"], key="p2_adg_mode",
        help="Как растёт рекламный бюджет на MVP.")
    if p2_ad_growth_mode == "Percentage (%)":
        p2_ad_growth_pct = st.number_input("P2 MoM Growth %", min_value=0.0, max_value=100.0, value=5.0, key="p2_adg_pct")
        p2_ad_growth_abs = 0.0
    else:
        p2_ad_growth_abs = st.number_input("P2 MoM Growth ($)", min_value=0.0, value=5000.0, key="p2_adg_abs")
        p2_ad_growth_pct = 0.0
    p2_cpi_deg = st.number_input("P2 CPI Degradation %", min_value=0.0, max_value=10.0, value=1.0, key="p2_cpid",
        help="Рост CPI за каждые +$1000 к бюджету на MVP.")

    st.markdown("**Phase 3: Scaling**")
    p3_ad_growth_mode = st.radio("P3 Growth Mode", ["Percentage (%)", "Absolute ($)"], key="p3_adg_mode",
        help="Как растёт рекламный бюджет на Scaling.")
    if p3_ad_growth_mode == "Percentage (%)":
        p3_ad_growth_pct = st.number_input("P3 MoM Growth %", min_value=0.0, max_value=100.0, value=5.0, key="p3_adg_pct")
        p3_ad_growth_abs = 0.0
    else:
        p3_ad_growth_abs = st.number_input("P3 MoM Growth ($)", min_value=0.0, value=5000.0, key="p3_adg_abs")
        p3_ad_growth_pct = 0.0
    p3_cpi_deg = st.number_input("P3 CPI Degradation %", min_value=0.0, max_value=10.0, value=1.0, key="p3_cpid",
        help="Рост CPI за каждые +$1000 к бюджету на Scaling.")

# --- Organic Acquisition (per phase) ---
with st.sidebar.expander("Organic Acquisition"):
    starting_organic = st.number_input("Starting Organic Traffic", min_value=0.0, value=0.0, key="cfg_org_start",
        help="Начальное количество органических посетителей в месяц (месяц 1).")

    st.markdown("**Phase 1: Pre-MVP**")
    p1_organic_growth_mode = st.radio("P1 Organic Growth", ["Percentage (%)", "Absolute (users)"], key="p1_ogm",
        help="Рост органики на Pre-MVP. Обычно 0.")
    if p1_organic_growth_mode == "Percentage (%)":
        p1_organic_growth_pct = st.number_input("P1 Organic MoM %", min_value=0.0, max_value=200.0, value=0.0, key="p1_ogp")
        p1_organic_growth_abs = 0.0
    else:
        p1_organic_growth_abs = st.number_input("P1 Organic MoM (users)", min_value=0.0, value=0.0, key="p1_oga")
        p1_organic_growth_pct = 0.0
    p1_organic_conv_trial = st.number_input("P1 Organic Conv Trial %", min_value=0.0, max_value=100.0, value=0.0, key="p1_oct")
    p1_organic_conv_paid = st.number_input("P1 Organic Conv Paid %", min_value=0.0, max_value=100.0, value=0.0, key="p1_ocp")
    p1_organic_spend = st.number_input("P1 Monthly Organic Spend ($)", min_value=0.0, value=0.0, key="p1_osp")

    st.markdown("**Phase 2: MVP**")
    p2_organic_growth_mode = st.radio("P2 Organic Growth", ["Percentage (%)", "Absolute (users)"], key="p2_ogm",
        help="Рост органики на MVP.")
    if p2_organic_growth_mode == "Percentage (%)":
        p2_organic_growth_pct = st.number_input("P2 Organic MoM %", min_value=0.0, max_value=200.0, value=10.0, key="p2_ogp")
        p2_organic_growth_abs = 0.0
    else:
        p2_organic_growth_abs = st.number_input("P2 Organic MoM (users)", min_value=0.0, value=50.0, key="p2_oga")
        p2_organic_growth_pct = 0.0
    p2_organic_conv_trial = st.number_input("P2 Organic Conv Trial %", min_value=0.0, max_value=100.0, value=25.0, key="p2_oct")
    p2_organic_conv_paid = st.number_input("P2 Organic Conv Paid %", min_value=0.0, max_value=100.0, value=25.0, key="p2_ocp")
    p2_organic_spend = st.number_input("P2 Monthly Organic Spend ($)", min_value=0.0, value=500.0, key="p2_osp")

    st.markdown("**Phase 3: Scaling**")
    p3_organic_growth_mode = st.radio("P3 Organic Growth", ["Percentage (%)", "Absolute (users)"], key="p3_ogm",
        help="Рост органики на Scaling.")
    if p3_organic_growth_mode == "Percentage (%)":
        p3_organic_growth_pct = st.number_input("P3 Organic MoM %", min_value=0.0, max_value=200.0, value=15.0, key="p3_ogp")
        p3_organic_growth_abs = 0.0
    else:
        p3_organic_growth_abs = st.number_input("P3 Organic MoM (users)", min_value=0.0, value=500.0, key="p3_oga")
        p3_organic_growth_pct = 0.0
    p3_organic_conv_trial = st.number_input("P3 Organic Conv Trial %", min_value=0.0, max_value=100.0, value=35.0, key="p3_oct")
    p3_organic_conv_paid = st.number_input("P3 Organic Conv Paid %", min_value=0.0, max_value=100.0, value=35.0, key="p3_ocp")
    p3_organic_spend = st.number_input("P3 Monthly Organic Spend ($)", min_value=0.0, value=2000.0, key="p3_osp")

# --- Subscription Mix & Pricing ---
with st.sidebar.expander("Subscription Mix & Pricing"):
    st.markdown("**Default Pricing**")
    price_weekly = st.number_input("Price: Weekly ($)", min_value=0.0, value=4.99, key="pr_w")
    price_monthly = st.number_input("Price: Monthly ($)", min_value=0.0, value=7.99, key="pr_m")
    price_annual = st.number_input("Price: Annual ($)", min_value=0.0, value=49.99, key="pr_a")

    per_phase_pricing = st.checkbox("Customize pricing per phase", value=False, key="pp_pr")
    if per_phase_pricing:
        st.markdown("**Phase 2 Pricing**")
        p2_price_weekly = st.number_input("P2 Weekly ($)", min_value=0.0, value=price_weekly, key="p2_pw")
        p2_price_monthly = st.number_input("P2 Monthly ($)", min_value=0.0, value=price_monthly, key="p2_pm")
        p2_price_annual = st.number_input("P2 Annual ($)", min_value=0.0, value=price_annual, key="p2_pa")
        st.markdown("**Phase 3 Pricing**")
        p3_price_weekly = st.number_input("P3 Weekly ($)", min_value=0.0, value=price_weekly, key="p3_pw")
        p3_price_monthly = st.number_input("P3 Monthly ($)", min_value=0.0, value=price_monthly, key="p3_pm")
        p3_price_annual = st.number_input("P3 Annual ($)", min_value=0.0, value=price_annual, key="p3_pa")
    else:
        p2_price_weekly = p3_price_weekly = price_weekly
        p2_price_monthly = p3_price_monthly = price_monthly
        p2_price_annual = p3_price_annual = price_annual

    st.markdown("---")
    st.markdown("**Subscription Mix**")
    mix_weekly = st.number_input("Mix: Weekly %", min_value=0.0, max_value=100.0, value=0.0, key="mx_w")
    mix_monthly = st.number_input("Mix: Monthly %", min_value=0.0, max_value=100.0, value=48.0, key="mx_m")
    mix_annual = st.number_input("Mix: Annual %", min_value=0.0, max_value=100.0, value=52.0, key="mx_a")
    total_mix = mix_weekly + mix_monthly + mix_annual
    if total_mix > 0:
        mix_weekly /= total_mix
        mix_monthly /= total_mix
        mix_annual /= total_mix
    else:
        mix_monthly = 1.0

    per_phase_mix = st.checkbox("Customize mix per phase", value=False, key="pp_mx")
    if per_phase_mix:
        st.markdown("**Phase 2 Mix**")
        p2_mix_weekly = st.number_input("P2 Weekly %", min_value=0.0, max_value=100.0, value=mix_weekly * 100, key="p2_mw")
        p2_mix_monthly = st.number_input("P2 Monthly %", min_value=0.0, max_value=100.0, value=mix_monthly * 100, key="p2_mm")
        p2_mix_annual = st.number_input("P2 Annual %", min_value=0.0, max_value=100.0, value=mix_annual * 100, key="p2_ma")
        p2_total = p2_mix_weekly + p2_mix_monthly + p2_mix_annual
        if p2_total > 0:
            p2_mix_weekly /= p2_total; p2_mix_monthly /= p2_total; p2_mix_annual /= p2_total
        else:
            p2_mix_monthly = 1.0

        st.markdown("**Phase 3 Mix**")
        p3_mix_weekly = st.number_input("P3 Weekly %", min_value=0.0, max_value=100.0, value=mix_weekly * 100, key="p3_mw")
        p3_mix_monthly = st.number_input("P3 Monthly %", min_value=0.0, max_value=100.0, value=mix_monthly * 100, key="p3_mm")
        p3_mix_annual = st.number_input("P3 Annual %", min_value=0.0, max_value=100.0, value=mix_annual * 100, key="p3_ma")
        p3_total = p3_mix_weekly + p3_mix_monthly + p3_mix_annual
        if p3_total > 0:
            p3_mix_weekly /= p3_total; p3_mix_monthly /= p3_total; p3_mix_annual /= p3_total
        else:
            p3_mix_monthly = 1.0
    else:
        p2_mix_weekly = p3_mix_weekly = mix_weekly
        p2_mix_monthly = p3_mix_monthly = mix_monthly
        p2_mix_annual = p3_mix_annual = mix_annual

# --- Retention & Churn ---
with st.sidebar.expander("Retention & Churn"):
    weekly_cancel_rate = st.number_input("Weekly Cancellation Rate %", min_value=0.0, max_value=100.0, value=15.0, key="cfg_wk_cancel",
        help="Процент недельных подписчиков, отменяющих каждую неделю.")
    monthly_churn_rate = st.number_input("Monthly Churn Rate %", min_value=0.0, max_value=100.0, value=10.0, key="cfg_mo_churn",
        help="Процент месячных подписчиков, уходящих каждый месяц.")
    annual_non_renewal = st.number_input("Annual Non-Renewal Rate %", min_value=0.0, max_value=100.0, value=30.0, key="cfg_yr_nonrenew",
        help="Процент годовых подписчиков, НЕ продлевающих через 12 месяцев.")
    st.markdown("**Churn Multiplier by Phase**")
    p2_churn_mult = st.number_input("Phase 2 Churn Multiplier", min_value=0.1, max_value=5.0, value=1.5, key="cfg_p2_churn_mult",
        help="Множитель оттока на MVP. 1.5 = отток на 50% выше базового.")
    p3_churn_mult = st.number_input("Phase 3 Churn Multiplier", min_value=0.1, max_value=5.0, value=1.0, key="cfg_p3_churn_mult",
        help="Множитель оттока на Scaling. 1.0 = базовый.")

# --- Trial, Refunds & COGS ---
with st.sidebar.expander("Trial, Refunds & COGS"):
    trial_days = st.number_input("Trial Duration (days)", min_value=0, max_value=90, value=7, key="cfg_trial_days",
        help="Длительность бесплатного триала. 0 = оплата сразу.")
    refund_rate = st.number_input("Refund Rate %", min_value=0.0, max_value=30.0, value=2.0, key="cfg_refund_rate",
        help="Процент возвратов от валовой выручки.")
    cogs_global = st.number_input("COGS per Active User ($)", min_value=0.0, value=0.10, key="cfg_cogs_global",
        help="Затраты на серверы/хостинг на одного активного пользователя в месяц.")
    per_phase_cogs = st.checkbox("Customize COGS per phase", value=False, key="pp_cogs")
    if per_phase_cogs:
        p1_cogs = st.number_input("P1 COGS ($)", min_value=0.0, value=0.05, key="p1_cogs",
            help="COGS на Pre-MVP (обычно ниже — мало пользователей).")
        p2_cogs = st.number_input("P2 COGS ($)", min_value=0.0, value=0.15, key="p2_cogs",
            help="COGS на MVP (неоптимизированная инфра).")
        p3_cogs = st.number_input("P3 COGS ($)", min_value=0.0, value=0.08, key="p3_cogs",
            help="COGS на Scaling (экономия масштаба).")
    else:
        p1_cogs = p2_cogs = p3_cogs = cogs_global

# --- Expansion & Contraction ---
with st.sidebar.expander("Expansion & Contraction"):
    upgrade_rate = st.number_input("Monthly→Annual Upgrade %/mo", min_value=0.0, max_value=20.0, value=2.0, key="cfg_upgrade_rate",
        help="Процент месячных подписчиков, переходящих на годовой план каждый месяц.")
    downgrade_rate = st.number_input("Annual→Monthly Downgrade %/yr", min_value=0.0, max_value=50.0, value=5.0, key="cfg_downgrade_rate",
        help="Процент годовых подписчиков, переходящих на месячный план при продлении.")


# ===================== DATA ENGINE =====================

def add_phase_lines(fig, p1_end, p2_end):
    fig.add_vline(x=p1_end + 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Phase 2", annotation_position="top")
    fig.add_vline(x=p2_end + 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Phase 3", annotation_position="top")
    return fig


def add_milestone_markers(fig, ms, keys_labels, color="orange"):
    for key, label in keys_labels:
        val = ms.get(key)
        if val is not None:
            fig.add_vline(x=val, line_dash="dashdot", line_color=color, line_width=1,
                          annotation_text=label, annotation_position="bottom",
                          annotation_font_size=9, annotation_font_color=color)


def run_model(sens_params=None):
    if sens_params is None:
        sens_params = {"conv": 0, "churn": 0, "cpi": 0, "organic": 0}

    N = total_months
    months = np.arange(1, N + 1)
    df = pd.DataFrame({"Month": months})

    conv_factor = 1 + sens_params.get("conv", 0)
    churn_factor = 1 + sens_params.get("churn", 0)
    cpi_factor = 1 + sens_params.get("cpi", 0)
    organic_factor = 1 + sens_params.get("organic", 0)

    p1_end = phase1_dur
    p2_end = phase1_dur + phase2_dur

    def get_phase(m):
        if m <= p1_end:
            return 1
        elif m <= p2_end:
            return 2
        return 3

    df["Product Phase"] = df["Month"].apply(get_phase)

    # Per-phase config
    phase_cfg = {
        1: {
            "ad": p1_ad_budget, "cpi": p1_cpi,
            "ct": p1_conv_trial / 100.0 * conv_factor,
            "cp": p1_conv_paid / 100.0 * conv_factor,
            "sal": p1_salaries_total / phase1_dur,
            "misc": p1_misc_total / phase1_dur,
            "inv": p1_investment, "churn_m": 1.0,
            "ad_growth_mode": "Percentage (%)", "ad_growth_pct": 0.0, "ad_growth_abs": 0.0,
            "cpi_deg": 0.0,
            "org_growth_mode": p1_organic_growth_mode,
            "org_growth_pct": p1_organic_growth_pct, "org_growth_abs": p1_organic_growth_abs,
            "org_conv_trial": p1_organic_conv_trial / 100.0 * conv_factor,
            "org_conv_paid": p1_organic_conv_paid / 100.0 * conv_factor,
            "org_spend": p1_organic_spend,
            "mix_w": mix_weekly, "mix_m": mix_monthly, "mix_a": mix_annual,
            "pr_w": price_weekly, "pr_m": price_monthly, "pr_a": price_annual,
            "cogs": p1_cogs,
        },
        2: {
            "ad": p2_ad_budget, "cpi": p2_cpi,
            "ct": p2_conv_trial / 100.0 * conv_factor,
            "cp": p2_conv_paid / 100.0 * conv_factor,
            "sal": p2_salaries_total / phase2_dur,
            "misc": p2_misc_total / phase2_dur,
            "inv": p2_investment, "churn_m": p2_churn_mult,
            "ad_growth_mode": p2_ad_growth_mode, "ad_growth_pct": p2_ad_growth_pct, "ad_growth_abs": p2_ad_growth_abs,
            "cpi_deg": p2_cpi_deg,
            "org_growth_mode": p2_organic_growth_mode,
            "org_growth_pct": p2_organic_growth_pct, "org_growth_abs": p2_organic_growth_abs,
            "org_conv_trial": p2_organic_conv_trial / 100.0 * conv_factor,
            "org_conv_paid": p2_organic_conv_paid / 100.0 * conv_factor,
            "org_spend": p2_organic_spend,
            "mix_w": p2_mix_weekly, "mix_m": p2_mix_monthly, "mix_a": p2_mix_annual,
            "pr_w": p2_price_weekly, "pr_m": p2_price_monthly, "pr_a": p2_price_annual,
            "cogs": p2_cogs,
        },
        3: {
            "ad": p3_ad_budget, "cpi": p3_cpi,
            "ct": p3_conv_trial / 100.0 * conv_factor,
            "cp": p3_conv_paid / 100.0 * conv_factor,
            "sal": p3_salaries_total / phase3_dur,
            "misc": p3_misc_total / phase3_dur,
            "inv": p3_investment, "churn_m": p3_churn_mult,
            "ad_growth_mode": p3_ad_growth_mode, "ad_growth_pct": p3_ad_growth_pct, "ad_growth_abs": p3_ad_growth_abs,
            "cpi_deg": p3_cpi_deg,
            "org_growth_mode": p3_organic_growth_mode,
            "org_growth_pct": p3_organic_growth_pct, "org_growth_abs": p3_organic_growth_abs,
            "org_conv_trial": p3_organic_conv_trial / 100.0 * conv_factor,
            "org_conv_paid": p3_organic_conv_paid / 100.0 * conv_factor,
            "org_spend": p3_organic_spend,
            "mix_w": p3_mix_weekly, "mix_m": p3_mix_monthly, "mix_a": p3_mix_annual,
            "pr_w": p3_price_weekly, "pr_m": p3_price_monthly, "pr_a": p3_price_annual,
            "cogs": p3_cogs,
        },
    }

    # --- Ad Budget with per-phase growth ---
    ad_budgets = np.zeros(N)
    for i in range(N):
        phase = get_phase(i + 1)
        cfg = phase_cfg[phase]
        base = cfg["ad"]
        if phase == 1:
            m_in = i
        elif phase == 2:
            m_in = i - p1_end
        else:
            m_in = i - p2_end
        if cfg["ad_growth_mode"] == "Percentage (%)":
            ad_budgets[i] = base * ((1 + cfg["ad_growth_pct"] / 100.0) ** m_in)
        else:
            ad_budgets[i] = base + cfg["ad_growth_abs"] * m_in
    df["Ad Budget"] = ad_budgets

    # --- CPI with per-phase degradation and sensitivity ---
    cpi_arr = np.zeros(N)
    for i in range(N):
        phase = get_phase(i + 1)
        cfg = phase_cfg[phase]
        base_cpi = cfg["cpi"] * cpi_factor
        base_ad = cfg["ad"]
        deg = cfg["cpi_deg"]
        extra = max(0, ad_budgets[i] - base_ad) / 1000.0
        cpi_arr[i] = base_cpi * (1 + (deg / 100.0) * extra)
    df["CPI"] = cpi_arr
    df["Installs"] = np.where(df["Ad Budget"] > 0, df["Ad Budget"] / df["CPI"], 0)

    # --- Per-phase conversions ---
    conv_t = np.array([phase_cfg[get_phase(m)]["ct"] for m in months])
    conv_p = np.array([phase_cfg[get_phase(m)]["cp"] for m in months])
    df["New Trials"] = df["Installs"].values * conv_t

    # Trial delay: only full months count (3-day trial = 0 delay, 30-day = 1 month)
    trial_delay = trial_days // 30
    paid_new = df["New Trials"].values * conv_p
    if trial_delay > 0:
        paid_new = np.concatenate([np.zeros(trial_delay), paid_new[:N - trial_delay]])
    df["Paid New Paid Users"] = paid_new

    # --- Organic (per phase, traffic carries over) ---
    org_traffic = np.zeros(N)
    current_organic = starting_organic
    for i in range(N):
        phase = get_phase(i + 1)
        cfg = phase_cfg[phase]
        if i == 0:
            org_traffic[i] = current_organic
        else:
            prev_phase = get_phase(i)
            if prev_phase != phase:
                current_organic = org_traffic[i - 1]
            if cfg["org_growth_mode"] == "Percentage (%)":
                org_traffic[i] = org_traffic[i - 1] * (1 + cfg["org_growth_pct"] / 100.0 * organic_factor)
            else:
                org_traffic[i] = org_traffic[i - 1] + cfg["org_growth_abs"] * organic_factor
        org_traffic[i] = max(0, org_traffic[i])
    df["Organic Traffic"] = org_traffic

    # Organic conversions per phase
    org_ct = np.array([phase_cfg[get_phase(m)]["org_conv_trial"] for m in months])
    org_cp = np.array([phase_cfg[get_phase(m)]["org_conv_paid"] for m in months])
    org_new = org_traffic * org_ct * org_cp
    if trial_delay > 0:
        org_new = np.concatenate([np.zeros(trial_delay), org_new[:N - trial_delay]])
    df["Organic New Paid Users"] = org_new
    df["New Paid Users"] = df["Paid New Paid Users"] + df["Organic New Paid Users"]
    df["New Web Users"] = df["New Paid Users"] * (100 - store_split) / 100.0
    df["New Store Users"] = df["New Paid Users"] * store_split / 100.0

    # --- Churn rates ---
    base_churn_w = 1 - (1 - weekly_cancel_rate / 100.0) ** 4.33
    base_churn_m = monthly_churn_rate / 100.0
    base_non_renewal = annual_non_renewal / 100.0
    churn_mult_map = {1: 1.0, 2: p2_churn_mult, 3: p3_churn_mult}

    # --- Per-phase mix and pricing ---
    def get_mix(phase):
        cfg = phase_cfg[phase]
        return cfg["mix_w"], cfg["mix_m"], cfg["mix_a"]

    def get_prices(phase):
        cfg = phase_cfg[phase]
        return cfg["pr_w"], cfg["pr_m"], cfg["pr_a"]

    # --- Cohort matrices ---
    cohorts = {}
    for plat in ["web", "store"]:
        for plan in ["weekly", "monthly", "annual"]:
            cohorts[f"{plat}_{plan}"] = np.zeros((N, N))

    for i in range(N):
        nw = df.loc[i, "New Web Users"]
        ns = df.loc[i, "New Store Users"]
        phase_i = get_phase(i + 1)
        mw, mm, ma = get_mix(phase_i)

        cohorts["web_weekly"][i, i] = nw * mw
        cohorts["web_monthly"][i, i] = nw * mm
        cohorts["web_annual"][i, i] = nw * ma
        cohorts["store_weekly"][i, i] = ns * mw
        cohorts["store_monthly"][i, i] = ns * mm
        cohorts["store_annual"][i, i] = ns * ma

        for j in range(i + 1, N):
            phase_j = get_phase(j + 1)
            mult = churn_mult_map[phase_j] * churn_factor

            cw = min(1.0, base_churn_w * mult)
            cohorts["web_weekly"][i, j] = cohorts["web_weekly"][i, j - 1] * (1 - cw)
            cohorts["store_weekly"][i, j] = cohorts["store_weekly"][i, j - 1] * (1 - cw)

            cm = min(1.0, base_churn_m * mult)
            cohorts["web_monthly"][i, j] = cohorts["web_monthly"][i, j - 1] * (1 - cm)
            cohorts["store_monthly"][i, j] = cohorts["store_monthly"][i, j - 1] * (1 - cm)

            months_since = j - i
            if months_since > 0 and months_since % 12 == 0:
                ca = min(1.0, base_non_renewal * mult)
                cohorts["web_annual"][i, j] = cohorts["web_annual"][i, j - 1] * (1 - ca)
                cohorts["store_annual"][i, j] = cohorts["store_annual"][i, j - 1] * (1 - ca)
            else:
                cohorts["web_annual"][i, j] = cohorts["web_annual"][i, j - 1]
                cohorts["store_annual"][i, j] = cohorts["store_annual"][i, j - 1]

    # --- Active users by plan ---
    active_w = np.zeros(N)
    active_m_arr = np.zeros(N)
    active_a = np.zeros(N)
    for j in range(N):
        active_w[j] = np.sum(cohorts["web_weekly"][:, j] + cohorts["store_weekly"][:, j])
        active_m_arr[j] = np.sum(cohorts["web_monthly"][:, j] + cohorts["store_monthly"][:, j])
        active_a[j] = np.sum(cohorts["web_annual"][:, j] + cohorts["store_annual"][:, j])

    df["Active Web Users"] = [np.sum(cohorts["web_weekly"][:, j] + cohorts["web_monthly"][:, j] + cohorts["web_annual"][:, j]) for j in range(N)]
    df["Active Store Users"] = [np.sum(cohorts["store_weekly"][:, j] + cohorts["store_monthly"][:, j] + cohorts["store_annual"][:, j]) for j in range(N)]
    df["Total Active Users"] = df["Active Web Users"] + df["Active Store Users"]

    # --- Revenue (per-phase pricing) ---
    gross_rev_web = np.zeros(N)
    gross_rev_store = np.zeros(N)
    mrr_web = np.zeros(N)
    mrr_store = np.zeros(N)
    mrr_weekly_a = np.zeros(N)
    mrr_monthly_a = np.zeros(N)
    mrr_annual_a = np.zeros(N)
    tx_web = np.zeros(N)
    new_mrr_arr = np.zeros(N)

    for j in range(N):
        # For revenue, use the pricing of the phase when the user subscribed
        # Simplified: use current phase pricing for all active users
        pw, pm, pa = get_prices(get_phase(j + 1))

        mw_web = np.sum(cohorts["web_weekly"][:, j]) * pw * 4.33
        mm_web = np.sum(cohorts["web_monthly"][:, j]) * pm
        ma_web = np.sum(cohorts["web_annual"][:, j]) * pa / 12.0
        mw_st = np.sum(cohorts["store_weekly"][:, j]) * pw * 4.33
        mm_st = np.sum(cohorts["store_monthly"][:, j]) * pm
        ma_st = np.sum(cohorts["store_annual"][:, j]) * pa / 12.0

        mrr_web[j] = mw_web + mm_web + ma_web
        mrr_store[j] = mw_st + mm_st + ma_st
        mrr_weekly_a[j] = mw_web + mw_st
        mrr_monthly_a[j] = mm_web + mm_st
        mrr_annual_a[j] = ma_web + ma_st

        # New MRR from users acquired this month
        nw_j = df.loc[j, "New Web Users"]
        ns_j = df.loc[j, "New Store Users"]
        phase_j = get_phase(j + 1)
        mw_j, mm_j, ma_j = get_mix(phase_j)
        new_mrr_arr[j] = (nw_j + ns_j) * (mw_j * pw * 4.33 + mm_j * pm + ma_j * pa / 12.0)

        # Cash revenue (annual paid upfront)
        cash_a_web = 0
        cash_a_store = 0
        tx_a = 0
        for i in range(j + 1):
            if (j - i) % 12 == 0:
                cash_a_web += cohorts["web_annual"][i, j] * pa
                cash_a_store += cohorts["store_annual"][i, j] * pa
                tx_a += cohorts["web_annual"][i, j]
        gross_rev_web[j] = mw_web + mm_web + cash_a_web
        gross_rev_store[j] = mw_st + mm_st + cash_a_store
        tx_web[j] = np.sum(cohorts["web_weekly"][:, j]) * 4.33 + np.sum(cohorts["web_monthly"][:, j]) + tx_a

    # Refunds
    rf = 1 - refund_rate / 100.0
    gross_rev_web *= rf
    gross_rev_store *= rf

    df["Gross Revenue Web"] = gross_rev_web
    df["Gross Revenue Store"] = gross_rev_store
    df["Total Gross Revenue"] = gross_rev_web + gross_rev_store
    df["MRR Web"] = mrr_web * rf
    df["MRR Store"] = mrr_store * rf
    df["Total MRR"] = df["MRR Web"] + df["MRR Store"]
    df["MRR Weekly"] = mrr_weekly_a * rf
    df["MRR Monthly"] = mrr_monthly_a * rf
    df["MRR Annual"] = mrr_annual_a * rf
    df["Recognized Revenue"] = df["Total MRR"]
    df["New MRR"] = new_mrr_arr * rf

    # --- Expansion / Contraction MRR ---
    expansion_mrr = np.zeros(N)
    contraction_mrr = np.zeros(N)
    for j in range(N):
        pw, pm, pa = get_prices(get_phase(j + 1))
        upgraders = active_m_arr[j] * (upgrade_rate / 100.0)
        expansion_mrr[j] = upgraders * abs(pa / 12.0 - pm)
        downgraders = active_a[j] * (downgrade_rate / 100.0 / 12.0)
        contraction_mrr[j] = downgraders * abs(pm - pa / 12.0)
    df["Expansion MRR"] = expansion_mrr
    df["Contraction MRR"] = contraction_mrr

    # --- Churned MRR ---
    total_mrr_series = df["Total MRR"].values
    churned_mrr = np.zeros(N)
    for j in range(1, N):
        existing_mrr_now = total_mrr_series[j] - new_mrr_arr[j] * rf
        churned_mrr[j] = max(0, total_mrr_series[j - 1] - existing_mrr_now)
    df["Churned MRR"] = churned_mrr
    df["Net New MRR"] = df["New MRR"] + df["Expansion MRR"] - df["Contraction MRR"] - df["Churned MRR"]

    # --- Commissions ---
    df["Store Commission"] = df["Gross Revenue Store"] * (app_store_comm / 100.0)
    df["Web Commission"] = df["Gross Revenue Web"] * (web_comm_pct / 100.0) + tx_web * web_comm_fixed * rf
    df["Bank Fee"] = df["Total Gross Revenue"] * (bank_fee / 100.0)
    df["Total Commissions"] = df["Store Commission"] + df["Web Commission"] + df["Bank Fee"]
    df["Net Revenue"] = df["Total Gross Revenue"] - df["Total Commissions"]

    # --- Costs (per-phase COGS and organic spend) ---
    cogs_per_phase = {1: phase_cfg[1]["cogs"], 2: phase_cfg[2]["cogs"], 3: phase_cfg[3]["cogs"]}
    df["COGS"] = df.apply(lambda r: r["Total Active Users"] * cogs_per_phase[r["Product Phase"]], axis=1)
    org_spend_per_phase = {1: phase_cfg[1]["org_spend"], 2: phase_cfg[2]["org_spend"], 3: phase_cfg[3]["org_spend"]}
    df["Organic Spend"] = df["Product Phase"].map(org_spend_per_phase)
    df["Marketing"] = df["Ad Budget"] + df["Organic Spend"]
    df["Salaries"] = df["Product Phase"].map({1: phase_cfg[1]["sal"], 2: phase_cfg[2]["sal"], 3: phase_cfg[3]["sal"]})
    df["Misc Costs"] = df["Product Phase"].map({1: phase_cfg[1]["misc"], 2: phase_cfg[2]["misc"], 3: phase_cfg[3]["misc"]})
    df["Total Expenses"] = df["COGS"] + df["Marketing"] + df["Salaries"] + df["Misc Costs"]

    df["Gross Profit"] = df["Recognized Revenue"] - df["COGS"] - df["Total Commissions"]
    df["EBITDA"] = df["Gross Profit"] - df["Marketing"] - df["Salaries"] - df["Misc Costs"]
    df["Corporate Tax"] = df["EBITDA"].apply(lambda x: x * (corporate_tax / 100.0) if x > 0 else 0)
    df["Net Profit"] = df["EBITDA"] - df["Corporate Tax"]
    df["Net Cash Flow"] = df["Total Gross Revenue"] - df["Total Commissions"] - df["Total Expenses"] - df["Corporate Tax"]

    # Cash Balance with per-phase investments
    total_investment = phase_cfg[1]["inv"] + phase_cfg[2]["inv"] + phase_cfg[3]["inv"]
    cash_bal = np.zeros(N)
    for j in range(N):
        inv = 0
        if j == 0:
            inv = phase_cfg[1]["inv"]
        elif j == p1_end:
            inv = phase_cfg[2]["inv"]
        elif j == p2_end:
            inv = phase_cfg[3]["inv"]
        if j == 0:
            cash_bal[j] = inv + df.loc[j, "Net Cash Flow"]
        else:
            cash_bal[j] = cash_bal[j - 1] + inv + df.loc[j, "Net Cash Flow"]
    df["Cash Balance"] = cash_bal
    df["Deferred Revenue"] = (df["Total Gross Revenue"] - df["Recognized Revenue"]).cumsum()
    df["Cumulative Net Profit"] = df["Net Profit"].cumsum()
    df["Cumulative Revenue"] = df["Total Gross Revenue"].cumsum()
    df["Cumulative Marketing"] = df["Marketing"].cumsum()
    df["Cumulative Ad Spend"] = df["Ad Budget"].cumsum()

    # ===================== METRICS =====================

    df["Paid CAC"] = df["Ad Budget"] / df["Paid New Paid Users"].replace(0, np.nan)
    df["Organic CAC"] = df["Organic Spend"] / df["Organic New Paid Users"].replace(0, np.nan)
    df["Blended CAC"] = df["Marketing"] / df["New Paid Users"].replace(0, np.nan)
    df["ARPU"] = df["Total MRR"] / df["Total Active Users"].replace(0, np.nan)

    # Blended churn
    blended_churn = np.zeros(N)
    for j in range(N):
        phase_j = get_phase(j + 1)
        mult = churn_mult_map[phase_j] * churn_factor
        cw = min(1.0, base_churn_w * mult)
        cm_val = min(1.0, base_churn_m * mult)
        ca_monthly = min(1.0, base_non_renewal * mult) / 12.0
        total = active_w[j] + active_m_arr[j] + active_a[j]
        if total > 0:
            blended_churn[j] = (cw * active_w[j] + cm_val * active_m_arr[j] + ca_monthly * active_a[j]) / total
    df["Blended Churn"] = blended_churn

    # CRR (Customer Retention Rate)
    df["CRR %"] = (1 - df["Blended Churn"]) * 100

    df["Gross Margin %"] = df["Gross Profit"] / df["Recognized Revenue"].replace(0, np.nan)
    df["LTV"] = (df["ARPU"] * df["Gross Margin %"]) / pd.Series(blended_churn).replace(0, np.nan)
    df["LTV/CAC"] = df["LTV"] / df["Blended CAC"].replace(0, np.nan)
    df["MER"] = df["Total Gross Revenue"] / df["Marketing"].replace(0, np.nan)
    df["Payback Period (Months)"] = df["Blended CAC"] / (df["ARPU"] * df["Gross Margin %"]).replace(0, np.nan)

    # ROI (cumulative)
    total_costs_cum = (df["Total Expenses"] + df["Total Commissions"] + df["Corporate Tax"]).cumsum()
    df["ROI %"] = ((df["Cumulative Revenue"] - total_costs_cum - total_investment) / max(total_investment, 1)) * 100

    # ROAS (monthly and cumulative)
    df["ROAS"] = df["Total Gross Revenue"] / df["Ad Budget"].replace(0, np.nan)
    df["Cumulative ROAS"] = df["Cumulative Revenue"] / df["Cumulative Ad Spend"].replace(0, np.nan)

    # NRR (Net Revenue Retention) — revenue from existing users / previous MRR
    nrr = np.full(N, np.nan)
    for j in range(1, N):
        if total_mrr_series[j - 1] > 0:
            existing_mrr = total_mrr_series[j] - new_mrr_arr[j] * rf + expansion_mrr[j] - contraction_mrr[j]
            nrr[j] = (existing_mrr / total_mrr_series[j - 1]) * 100
    df["NRR %"] = nrr

    # Quick Ratio (SaaS)
    denominator = df["Churned MRR"] + df["Contraction MRR"]
    df["Quick Ratio"] = (df["New MRR"] + df["Expansion MRR"]) / denominator.replace(0, np.nan)

    # Burn Rate & Runway
    df["Burn Rate"] = df["Net Cash Flow"].apply(lambda x: abs(x) if x < 0 else 0)
    runway = np.full(N, np.nan)
    for j in range(N):
        if df.loc[j, "Net Cash Flow"] < 0 and cash_bal[j] > 0:
            runway[j] = cash_bal[j] / abs(df.loc[j, "Net Cash Flow"])
    df["Runway (Months)"] = runway

    # CAE (Customer Acquisition Efficiency)
    df["CAE"] = df["Net New MRR"] / df["Marketing"].replace(0, np.nan)

    # Revenue per Install
    df["Revenue per Install"] = df["Total Gross Revenue"] / df["Installs"].replace(0, np.nan)

    # Churn rates for display
    df["Weekly Churn %"] = [min(100.0, base_churn_w * churn_mult_map[get_phase(m)] * churn_factor * 100) for m in months]
    df["Monthly Churn %"] = [min(100.0, base_churn_m * churn_mult_map[get_phase(m)] * churn_factor * 100) for m in months]
    df["Annual Non-Renewal %"] = [min(100.0, base_non_renewal * churn_mult_map[get_phase(m)] * churn_factor * 100) for m in months]

    # ===================== MILESTONES =====================
    milestones = {}

    # Break-Even Month (Net Profit > 0)
    be_months = df[df["Net Profit"] > 0]["Month"]
    milestones["break_even_month"] = int(be_months.iloc[0]) if len(be_months) > 0 else None

    # Cumulative Break-Even (Cumulative Net Profit > 0)
    cum_be = df[df["Cumulative Net Profit"] > 0]["Month"]
    milestones["cumulative_break_even"] = int(cum_be.iloc[0]) if len(cum_be) > 0 else None

    # Cash Flow Positive Month
    cf_pos = df[df["Net Cash Flow"] > 0]["Month"]
    milestones["cf_positive_month"] = int(cf_pos.iloc[0]) if len(cf_pos) > 0 else None

    # Investment Payback (cumulative profit > total investment)
    payback = df[df["Cumulative Net Profit"] >= total_investment]["Month"]
    milestones["investment_payback_month"] = int(payback.iloc[0]) if len(payback) > 0 else None

    # Runway Out (Cash Balance < 0)
    cash_neg = df[df["Cash Balance"] < 0]["Month"]
    milestones["runway_out_month"] = int(cash_neg.iloc[0]) if len(cash_neg) > 0 else None

    # User milestones
    for threshold in [1000, 10000, 100000]:
        um = df[df["Total Active Users"] >= threshold]["Month"]
        milestones[f"users_{threshold}"] = int(um.iloc[0]) if len(um) > 0 else None

    # MRR milestones
    for threshold in [10000, 50000, 100000, 1000000]:
        mm = df[df["Total MRR"] >= threshold]["Month"]
        milestones[f"mrr_{threshold}"] = int(mm.iloc[0]) if len(mm) > 0 else None

    # Cohort retention matrix for heatmap
    cohort_sizes = np.zeros(N)
    retention_matrix = np.zeros((N, N))
    for i in range(N):
        total_cohort_i = sum(cohorts[k][i, i] for k in cohorts)
        cohort_sizes[i] = total_cohort_i
        for j in range(i, N):
            total_remaining = sum(cohorts[k][i, j] for k in cohorts)
            if total_cohort_i > 0:
                retention_matrix[i, j] = total_remaining / total_cohort_i * 100

    return df, milestones, retention_matrix


# ===================== RUN MODELS =====================

base_sens = {
    "conv": sens_conv / 100.0,
    "churn": sens_churn / 100.0,
    "cpi": sens_cpi / 100.0,
    "organic": sens_organic / 100.0,
}

pessimistic_sens = {
    "conv": base_sens["conv"] - scenario_bound / 100.0,
    "churn": base_sens["churn"] + scenario_bound / 100.0,
    "cpi": base_sens["cpi"] + scenario_bound / 100.0,
    "organic": base_sens["organic"] - scenario_bound / 100.0,
}

optimistic_sens = {
    "conv": base_sens["conv"] + scenario_bound / 100.0,
    "churn": base_sens["churn"] - scenario_bound / 100.0,
    "cpi": base_sens["cpi"] - scenario_bound / 100.0,
    "organic": base_sens["organic"] + scenario_bound / 100.0,
}

df_main, milestones_main, retention_main = run_model(base_sens)
df_pessimistic, milestones_pess, _ = run_model(pessimistic_sens)
df_optimistic, milestones_opt, _ = run_model(optimistic_sens)

p1_end = phase1_dur
p2_end = phase1_dur + phase2_dur

# ===================== GLOBAL FILTER =====================
st.header("Global Dashboard Filters")
month_range = st.slider("Select Month Range", 1, total_months, (1, total_months),
    help="Фильтр по месяцам — влияет на все графики и отчёты.")
start_m, end_m = month_range
f_df = df_main[(df_main["Month"] >= start_m) & (df_main["Month"] <= end_m)]
f_pess = df_pessimistic[(df_pessimistic["Month"] >= start_m) & (df_pessimistic["Month"] <= end_m)]
f_opt = df_optimistic[(df_optimistic["Month"] >= start_m) & (df_optimistic["Month"] <= end_m)]

# ===================== MILESTONES DISPLAY =====================
st.header("Key Milestones")
ms_cols = st.columns(5)

def fmt_milestone(val, suffix="мес."):
    return f"{val} {suffix}" if val is not None else "—"

ms_cols[0].metric("Break-Even (Monthly P&L)", fmt_milestone(milestones_main.get("break_even_month")),
    help="Месяц, когда Net Profit впервые > 0.")
ms_cols[1].metric("Cumulative Break-Even", fmt_milestone(milestones_main.get("cumulative_break_even")),
    help="Месяц, когда кумулятивная прибыль вышла в плюс.")
ms_cols[2].metric("Cash Flow Positive", fmt_milestone(milestones_main.get("cf_positive_month")),
    help="Месяц, когда денежный поток впервые > 0.")
ms_cols[3].metric("Investment Payback", fmt_milestone(milestones_main.get("investment_payback_month")),
    help="Месяц, когда кумулятивная прибыль покрыла все инвестиции.")
ms_cols[4].metric("Runway Out", fmt_milestone(milestones_main.get("runway_out_month")),
    help="Месяц, когда деньги на счету закончатся (Cash Balance < 0). '—' = денег хватает.")

ms_cols2 = st.columns(4)
ms_cols2[0].metric("1K Users", fmt_milestone(milestones_main.get("users_1000")),
    help="Месяц достижения 1,000 активных пользователей.")
ms_cols2[1].metric("10K Users", fmt_milestone(milestones_main.get("users_10000")),
    help="Месяц достижения 10,000 активных пользователей.")
ms_cols2[2].metric("MRR $10K", fmt_milestone(milestones_main.get("mrr_10000")),
    help="Месяц достижения MRR $10,000.")
ms_cols2[3].metric("MRR $100K", fmt_milestone(milestones_main.get("mrr_100000")),
    help="Месяц достижения MRR $100,000.")

# ===================== TOP METRICS =====================
st.markdown("---")
st.header("Key Metrics")

row1 = st.columns(4)
row1[0].metric("Total Revenue", f"${f_df['Total Gross Revenue'].sum():,.0f}",
    help="Сумма валовой выручки за период.")
row1[1].metric("Net Profit", f"${f_df['Net Profit'].sum():,.0f}",
    help="Чистая прибыль за период. Формула: SUM(EBITDA - Corporate Tax).")
row1[2].metric("End MRR", f"${f_df['Total MRR'].iloc[-1]:,.0f}",
    help="MRR на конец периода.")
row1[3].metric("Avg LTV/CAC", f"{f_df['LTV/CAC'].mean():.2f}x",
    help="Среднее LTV/CAC. >3x = отлично.")

row2 = st.columns(6)
total_inv = p1_investment + p2_investment + p3_investment
roi_val = f_df["ROI %"].iloc[-1] if not f_df["ROI %"].isna().all() else 0
roas_val = f_df["Cumulative ROAS"].iloc[-1] if not f_df["Cumulative ROAS"].isna().all() else 0
arpu_val = f_df["ARPU"].dropna().iloc[-1] if not f_df["ARPU"].dropna().empty else 0
gm_val = f_df["Gross Margin %"].dropna().iloc[-1] if not f_df["Gross Margin %"].dropna().empty else 0
burn_vals = f_df[f_df["Burn Rate"] > 0]["Burn Rate"]
burn_val = burn_vals.iloc[-1] if not burn_vals.empty else 0
runway_vals = f_df["Runway (Months)"].dropna()
runway_val = runway_vals.iloc[-1] if not runway_vals.empty else None

row2[0].metric("ROI", f"{roi_val:,.0f}%",
    help="Return on Investment (кумулятивный). Формула: (Revenue - Costs - Investment) / Investment.")
row2[1].metric("ROAS", f"{roas_val:,.1f}x",
    help="Return on Ad Spend (кумулятивный). Формула: Cumulative Revenue / Cumulative Ad Spend.")
row2[2].metric("ARPU", f"${arpu_val:,.2f}",
    help="Average Revenue Per User (последний месяц). Формула: MRR / Active Users.")
row2[3].metric("Gross Margin", f"{gm_val * 100:,.1f}%" if gm_val else "—",
    help="Валовая маржа. Формула: Gross Profit / Revenue.")
row2[4].metric("Burn Rate", f"${burn_val:,.0f}/mo",
    help="Текущая скорость сжигания денег (если CF < 0).")
row2[5].metric("Runway", f"{runway_val:,.0f} мес." if runway_val else "∞",
    help="Сколько месяцев до конца денег при текущем Burn Rate.")

# ===================== EXECUTIVE DASHBOARD =====================
st.markdown("---")
st.header("Executive Dashboard")

def health_indicator(value, good_threshold, bad_threshold, higher_is_better=True):
    """Return colored status emoji based on thresholds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "⚪ N/A"
    if higher_is_better:
        if value >= good_threshold:
            return f"🟢 {value:,.1f}" if isinstance(value, float) else f"🟢 {value:,}"
        elif value >= bad_threshold:
            return f"🟡 {value:,.1f}" if isinstance(value, float) else f"🟡 {value:,}"
        else:
            return f"🔴 {value:,.1f}" if isinstance(value, float) else f"🔴 {value:,}"
    else:
        if value <= good_threshold:
            return f"🟢 {value:,.1f}" if isinstance(value, float) else f"🟢 {value:,}"
        elif value <= bad_threshold:
            return f"🟡 {value:,.1f}" if isinstance(value, float) else f"🟡 {value:,}"
        else:
            return f"🔴 {value:,.1f}" if isinstance(value, float) else f"🔴 {value:,}"

# Compute latest values for health check
latest_ltv_cac = f_df["LTV/CAC"].dropna().iloc[-1] if not f_df["LTV/CAC"].dropna().empty else 0
latest_gm = (f_df["Gross Margin %"].dropna().iloc[-1] * 100) if not f_df["Gross Margin %"].dropna().empty else 0
latest_churn = (f_df["Blended Churn"].dropna().iloc[-1] * 100) if not f_df["Blended Churn"].dropna().empty else 0
latest_payback = f_df["Payback Period (Months)"].dropna().iloc[-1] if not f_df["Payback Period (Months)"].dropna().empty else None
latest_nrr = f_df["NRR %"].dropna().iloc[-1] if not f_df["NRR %"].dropna().empty else None
latest_qr = f_df["Quick Ratio"].dropna().iloc[-1] if not f_df["Quick Ratio"].dropna().empty else None
latest_mer = f_df["MER"].dropna().iloc[-1] if not f_df["MER"].dropna().empty else 0
total_net_profit = f_df["Net Profit"].sum()
end_cash = f_df["Cash Balance"].iloc[-1]

exec_data = {
    "Metric": [
        "LTV/CAC",
        "Gross Margin %",
        "Blended Churn %",
        "Payback (months)",
        "NRR %",
        "Quick Ratio",
        "MER",
        "Net Profit (total)",
        "End Cash Balance",
    ],
    "Value": [
        f"{latest_ltv_cac:.2f}x",
        f"{latest_gm:.1f}%",
        f"{latest_churn:.1f}%",
        f"{latest_payback:.1f}" if latest_payback else "—",
        f"{latest_nrr:.1f}%" if latest_nrr else "—",
        f"{latest_qr:.1f}" if latest_qr else "—",
        f"{latest_mer:.2f}x",
        f"${total_net_profit:,.0f}",
        f"${end_cash:,.0f}",
    ],
    "Status": [
        health_indicator(latest_ltv_cac, 3.0, 1.0),
        health_indicator(latest_gm, 70, 50),
        health_indicator(latest_churn, 5, 15, higher_is_better=False),
        health_indicator(latest_payback, 6, 18, higher_is_better=False) if latest_payback else "⚪ N/A",
        health_indicator(latest_nrr, 100, 80) if latest_nrr else "⚪ N/A",
        health_indicator(latest_qr, 4, 1) if latest_qr else "⚪ N/A",
        health_indicator(latest_mer, 3, 1),
        health_indicator(total_net_profit, 0, -50000),
        health_indicator(end_cash, 0, -10000),
    ],
    "Benchmark": [
        ">3x отлично, <1x плохо",
        ">70% отлично, <50% плохо",
        "<5% отлично, >15% плохо",
        "<6 мес отлично, >18 плохо",
        ">100% = рост, <80% потери",
        ">4 здоровый рост, <1 сжатие",
        ">3x эффективно, <1x убыточно",
        ">0 прибыль, <0 убыток",
        ">0 есть деньги, <0 дефицит",
    ],
}

ex_c1, ex_c2 = st.columns([2, 3])
with ex_c1:
    st.dataframe(pd.DataFrame(exec_data), use_container_width=True, hide_index=True)

with ex_c2:
    # Mini scenario summary
    st.markdown("**Scenario Summary (Period Total)**")
    sc_mini = pd.DataFrame({
        "": ["Revenue", "Net Profit", "End MRR", "Users", "ROI %", "Break-Even"],
        "🔴 Pessimistic": [
            f"${f_pess['Total Gross Revenue'].sum():,.0f}",
            f"${f_pess['Net Profit'].sum():,.0f}",
            f"${f_pess['Total MRR'].iloc[-1]:,.0f}",
            f"{f_pess['Total Active Users'].iloc[-1]:,.0f}",
            f"{f_pess['ROI %'].iloc[-1]:,.0f}%",
            fmt_milestone(milestones_pess.get("break_even_month")),
        ],
        "🔵 Base": [
            f"${f_df['Total Gross Revenue'].sum():,.0f}",
            f"${f_df['Net Profit'].sum():,.0f}",
            f"${f_df['Total MRR'].iloc[-1]:,.0f}",
            f"{f_df['Total Active Users'].iloc[-1]:,.0f}",
            f"{f_df['ROI %'].iloc[-1]:,.0f}%",
            fmt_milestone(milestones_main.get("break_even_month")),
        ],
        "🟢 Optimistic": [
            f"${f_opt['Total Gross Revenue'].sum():,.0f}",
            f"${f_opt['Net Profit'].sum():,.0f}",
            f"${f_opt['Total MRR'].iloc[-1]:,.0f}",
            f"{f_opt['Total Active Users'].iloc[-1]:,.0f}",
            f"{f_opt['ROI %'].iloc[-1]:,.0f}%",
            fmt_milestone(milestones_opt.get("break_even_month")),
        ],
    })
    st.dataframe(sc_mini, use_container_width=True, hide_index=True)

    # Key milestones inline
    ms_inline = []
    for key, label in [("break_even_month", "Break-Even"), ("cumulative_break_even", "Cum. Break-Even"),
                        ("cf_positive_month", "CF Positive"), ("investment_payback_month", "Inv. Payback"),
                        ("runway_out_month", "Runway Out")]:
        val = milestones_main.get(key)
        ms_inline.append(f"**{label}:** {val} мес." if val else f"**{label}:** —")
    st.markdown(" | ".join(ms_inline))

# ===================== CHARTS =====================
st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs(["Growth & Revenue", "Unit Economics & Efficiency", "P&L & Scenarios", "Cohorts & Deep Dive"])

# Milestone marker sets for different chart types
ms_financial = [("break_even_month", "BE"), ("cumulative_break_even", "Cum.BE"), ("runway_out_month", "Runway Out")]
ms_users = [("users_1000", "1K"), ("users_10000", "10K"), ("users_100000", "100K")]
ms_mrr = [("mrr_10000", "$10K"), ("mrr_50000", "$50K"), ("mrr_100000", "$100K")]

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MRR by Subscription Plan")
        fig1 = go.Figure()
        for col, name in [("MRR Weekly", "Weekly"), ("MRR Monthly", "Monthly"), ("MRR Annual", "Annual")]:
            fig1.add_trace(go.Scatter(x=f_df["Month"], y=f_df[col], mode="lines", stackgroup="one", name=f"{name} (Base)"))
            fig1.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess[col], mode="lines", line=dict(dash="dot"), name=f"{name} (Pess)", visible="legendonly"))
            fig1.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt[col], mode="lines", line=dict(dash="dash"), name=f"{name} (Opt)", visible="legendonly"))
        fig1.update_layout(title="MRR Breakdown")
        add_phase_lines(fig1, p1_end, p2_end)
        add_milestone_markers(fig1, milestones_main, ms_mrr, color="purple")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.subheader("Cash Balance")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=f_df["Month"], y=f_df["Cash Balance"], name="Base"))
        fig2.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["Cash Balance"], mode="lines", line=dict(dash="dot", color="red"), name="Pessimistic"))
        fig2.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["Cash Balance"], mode="lines", line=dict(dash="dash", color="green"), name="Optimistic"))
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        add_phase_lines(fig2, p1_end, p2_end)
        add_milestone_markers(fig2, milestones_main, ms_financial, color="orange")
        fig2.update_layout(title="Cash Balance (3 Scenarios)")
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Gross vs Net Revenue")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Gross Revenue Web"], name="Gross Web"))
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Gross Revenue Store"], name="Gross Store"))
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Net Revenue"], name="Net Revenue"))
        fig4.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["Net Revenue"], mode="lines", line=dict(dash="dot", color="red"), name="Net Rev (Pess)", visible="legendonly"))
        fig4.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["Net Revenue"], mode="lines", line=dict(dash="dash", color="green"), name="Net Rev (Opt)", visible="legendonly"))
        fig4.update_layout(barmode="group", title="Revenue Comparison (3 Scenarios)")
        add_phase_lines(fig4, p1_end, p2_end)
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        st.subheader("Churn Rates by Subscription Type")
        fig6 = go.Figure()
        for col in ["Weekly Churn %", "Monthly Churn %", "Annual Non-Renewal %"]:
            fig6.add_trace(go.Scatter(x=f_df["Month"], y=f_df[col], mode="lines", name=f"{col} (Base)"))
            fig6.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess[col], mode="lines", line=dict(dash="dot"), name=f"{col} (Pess)", visible="legendonly"))
            fig6.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt[col], mode="lines", line=dict(dash="dash"), name=f"{col} (Opt)", visible="legendonly"))
        fig6.update_layout(title="Churn Rates (3 Scenarios)")
        add_phase_lines(fig6, p1_end, p2_end)
        st.plotly_chart(fig6, use_container_width=True)

    # Deferred Revenue
    c3b, c4b = st.columns(2)
    with c3b:
        st.subheader("Deferred Revenue")
        fig_def = go.Figure()
        fig_def.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Deferred Revenue"], mode="lines+markers", name="Deferred Revenue (Base)", fill="tozeroy"))
        fig_def.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["Deferred Revenue"], mode="lines", line=dict(dash="dot"), name="Pess", visible="legendonly"))
        fig_def.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["Deferred Revenue"], mode="lines", line=dict(dash="dash"), name="Opt", visible="legendonly"))
        fig_def.update_layout(title="Deferred Revenue (Collected but Unrecognized)")
        add_phase_lines(fig_def, p1_end, p2_end)
        st.plotly_chart(fig_def, use_container_width=True)

    with c4b:
        st.subheader("Active Users")
        fig_users = go.Figure()
        fig_users.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Total Active Users"], mode="lines", name="Users (Base)"))
        fig_users.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["Total Active Users"], mode="lines", line=dict(dash="dot", color="red"), name="Users (Pess)"))
        fig_users.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["Total Active Users"], mode="lines", line=dict(dash="dash", color="green"), name="Users (Opt)"))
        fig_users.update_layout(title="Total Active Users (3 Scenarios)")
        add_phase_lines(fig_users, p1_end, p2_end)
        add_milestone_markers(fig_users, milestones_main, ms_users, color="blue")
        st.plotly_chart(fig_users, use_container_width=True)

with tab2:
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("LTV vs CAC")
        fig3 = go.Figure()
        for col in ["LTV", "Paid CAC", "Organic CAC", "Blended CAC"]:
            fig3.add_trace(go.Scatter(x=f_df["Month"], y=f_df[col], mode="lines", name=f"{col} (Base)"))
            fig3.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess[col], mode="lines", line=dict(dash="dot"), name=f"{col} (Pess)", visible="legendonly"))
            fig3.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt[col], mode="lines", line=dict(dash="dash"), name=f"{col} (Opt)", visible="legendonly"))
        fig3.update_layout(title="LTV vs CAC (3 Scenarios)")
        add_phase_lines(fig3, p1_end, p2_end)
        st.plotly_chart(fig3, use_container_width=True)

    with c6:
        st.subheader("Payback Period")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Payback Period (Months)"], mode="lines", name="Base"))
        fig5.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["Payback Period (Months)"], mode="lines", line=dict(dash="dot"), name="Pessimistic"))
        fig5.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["Payback Period (Months)"], mode="lines", line=dict(dash="dash"), name="Optimistic"))
        fig5.update_layout(title="Payback Period (3 Scenarios)")
        add_phase_lines(fig5, p1_end, p2_end)
        st.plotly_chart(fig5, use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("MER & ROAS")
        fig8 = go.Figure()
        fig8.add_trace(go.Scatter(x=f_df["Month"], y=f_df["MER"], mode="lines", name="MER (Base)"))
        fig8.add_trace(go.Scatter(x=f_df["Month"], y=f_df["ROAS"], mode="lines", name="ROAS (Base)"))
        fig8.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["MER"], mode="lines", line=dict(dash="dot"), name="MER (Pess)", visible="legendonly"))
        fig8.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["ROAS"], mode="lines", line=dict(dash="dot"), name="ROAS (Pess)", visible="legendonly"))
        fig8.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["MER"], mode="lines", line=dict(dash="dash"), name="MER (Opt)", visible="legendonly"))
        fig8.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["ROAS"], mode="lines", line=dict(dash="dash"), name="ROAS (Opt)", visible="legendonly"))
        fig8.update_layout(title="Marketing Efficiency (3 Scenarios)")
        add_phase_lines(fig8, p1_end, p2_end)
        st.plotly_chart(fig8, use_container_width=True)

    with c8:
        st.subheader("MRR Movement Waterfall")
        fig_mrr_w = go.Figure(go.Waterfall(
            name="MRR Movement", orientation="v",
            measure=["relative", "relative", "relative", "relative", "total"],
            x=["New MRR", "Expansion", "Contraction", "Churned", "Net New MRR"],
            y=[f_df["New MRR"].sum(), f_df["Expansion MRR"].sum(),
               -f_df["Contraction MRR"].sum(), -f_df["Churned MRR"].sum(),
               f_df["Net New MRR"].sum()],
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        fig_mrr_w.update_layout(title="MRR Movement (Period Total)")
        st.plotly_chart(fig_mrr_w, use_container_width=True)

    # Unit Economics Over Time (combined)
    st.subheader("Unit Economics Over Time")
    fig_ue = go.Figure()
    fig_ue.add_trace(go.Scatter(x=f_df["Month"], y=f_df["ARPU"], mode="lines", name="ARPU"))
    fig_ue.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Blended CAC"], mode="lines", name="Blended CAC"))
    fig_ue.add_trace(go.Scatter(x=f_df["Month"], y=f_df["LTV"], mode="lines", name="LTV"))
    fig_ue.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Payback Period (Months)"], mode="lines", name="Payback (months)", yaxis="y2"))
    fig_ue.update_layout(
        title="ARPU / CAC / LTV / Payback",
        yaxis=dict(title="$ Value"),
        yaxis2=dict(title="Payback (months)", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    add_phase_lines(fig_ue, p1_end, p2_end)
    add_milestone_markers(fig_ue, milestones_main, ms_financial, color="orange")
    st.plotly_chart(fig_ue, use_container_width=True)

with tab3:
    c9, c10 = st.columns(2)
    with c9:
        st.subheader("P&L Waterfall")
        fig7 = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["Revenue", "COGS", "Marketing", "Salaries & Misc", "Commissions & Tax", "Net Profit"],
            y=[f_df["Recognized Revenue"].sum(), -f_df["COGS"].sum(), -f_df["Marketing"].sum(),
               -(f_df["Salaries"].sum() + f_df["Misc Costs"].sum()),
               -(f_df["Total Commissions"].sum() + f_df["Corporate Tax"].sum()),
               f_df["Net Profit"].sum()],
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        fig7.update_layout(title="P&L Waterfall")
        st.plotly_chart(fig7, use_container_width=True)

    with c10:
        st.subheader("Scenario Comparison")
        metrics_names = ["Net Profit", "Total Revenue", "End MRR", "End Users"]
        scenario_data = []
        for label, sdf in [("Pessimistic", f_pess), ("Base", f_df), ("Optimistic", f_opt)]:
            scenario_data.append({
                "Scenario": label,
                "Net Profit": sdf["Net Profit"].sum(),
                "Total Revenue": sdf["Total Gross Revenue"].sum(),
                "End MRR": sdf["Total MRR"].iloc[-1],
                "End Users": sdf["Total Active Users"].iloc[-1],
            })
        sc_df = pd.DataFrame(scenario_data)

        fig9 = go.Figure()
        for metric in metrics_names:
            fig9.add_trace(go.Bar(
                x=sc_df["Scenario"], y=sc_df[metric], name=metric,
                visible=True if metric == "Net Profit" else "legendonly"
            ))
        fig9.update_layout(title="Scenario Comparison", barmode="group")
        st.plotly_chart(fig9, use_container_width=True)

    c11, c12 = st.columns(2)
    with c11:
        st.subheader("Cumulative Net Profit")
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Cumulative Net Profit"], mode="lines", name="Base"))
        fig_roi.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["Cumulative Net Profit"], mode="lines", line=dict(dash="dot", color="red"), name="Pessimistic"))
        fig_roi.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["Cumulative Net Profit"], mode="lines", line=dict(dash="dash", color="green"), name="Optimistic"))
        fig_roi.add_hline(y=0, line_dash="dash", line_color="red")
        add_phase_lines(fig_roi, p1_end, p2_end)
        add_milestone_markers(fig_roi, milestones_main, [("cumulative_break_even", "Cum.BE"), ("investment_payback_month", "Payback")], color="green")
        fig_roi.update_layout(title="Cumulative Net Profit (3 Scenarios)")
        st.plotly_chart(fig_roi, use_container_width=True)

    with c12:
        st.subheader("NRR & Quick Ratio")
        fig_nrr = go.Figure()
        fig_nrr.add_trace(go.Scatter(x=f_df["Month"], y=f_df["NRR %"], mode="lines", name="NRR % (Base)"))
        fig_nrr.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["NRR %"], mode="lines", line=dict(dash="dot"), name="NRR % (Pess)", visible="legendonly"))
        fig_nrr.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["NRR %"], mode="lines", line=dict(dash="dash"), name="NRR % (Opt)", visible="legendonly"))
        fig_nrr.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Quick Ratio"], mode="lines", name="Quick Ratio", yaxis="y2"))
        fig_nrr.add_hline(y=100, line_dash="dash", line_color="gray")
        fig_nrr.update_layout(
            title="NRR & Quick Ratio",
            yaxis=dict(title="NRR %"),
            yaxis2=dict(title="Quick Ratio", overlaying="y", side="right"),
        )
        add_phase_lines(fig_nrr, p1_end, p2_end)
        st.plotly_chart(fig_nrr, use_container_width=True)

with tab4:
    # Cohort Retention Heatmap
    st.subheader("Cohort Retention Heatmap")
    # Show every Nth cohort to keep heatmap readable
    max_cohorts = min(total_months, 30)
    step = max(1, total_months // max_cohorts)
    cohort_indices = list(range(0, total_months, step))
    retention_display = retention_main[np.ix_(cohort_indices, range(total_months))]
    # For each cohort row, show retention from cohort start month onwards
    heatmap_data = []
    heatmap_y = []
    for idx, ci in enumerate(cohort_indices):
        row = []
        for j in range(total_months):
            if j >= ci and retention_display[idx, j] > 0:
                row.append(round(retention_display[idx, j], 1))
            else:
                row.append(None)
        heatmap_data.append(row)
        heatmap_y.append(f"M{ci + 1}")

    fig_hm = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"M{m}" for m in range(1, total_months + 1)],
        y=heatmap_y,
        colorscale="RdYlGn",
        zmin=0, zmax=100,
        text=[[f"{v:.0f}%" if v is not None else "" for v in row] for row in heatmap_data],
        texttemplate="%{text}",
        colorbar=dict(title="Retention %"),
        hoverongaps=False,
    ))
    fig_hm.update_layout(
        title="Cohort Retention (% of initial cohort remaining)",
        xaxis_title="Month",
        yaxis_title="Cohort Start",
        height=500,
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Строки — когорты (месяц привлечения). Столбцы — текущий месяц. Значение — % оставшихся пользователей от начального размера когорты.")

    # Net New MRR over time (stacked area with components)
    c_dd1, c_dd2 = st.columns(2)
    with c_dd1:
        st.subheader("MRR Movement Over Time")
        fig_mrr_t = go.Figure()
        fig_mrr_t.add_trace(go.Bar(x=f_df["Month"], y=f_df["New MRR"], name="New MRR", marker_color="green"))
        fig_mrr_t.add_trace(go.Bar(x=f_df["Month"], y=f_df["Expansion MRR"], name="Expansion MRR", marker_color="lightgreen"))
        fig_mrr_t.add_trace(go.Bar(x=f_df["Month"], y=-f_df["Contraction MRR"], name="Contraction MRR", marker_color="orange"))
        fig_mrr_t.add_trace(go.Bar(x=f_df["Month"], y=-f_df["Churned MRR"], name="Churned MRR", marker_color="red"))
        fig_mrr_t.add_trace(go.Scatter(x=f_df["Month"], y=f_df["Net New MRR"], mode="lines+markers", name="Net New MRR", line=dict(color="black", width=2)))
        fig_mrr_t.update_layout(barmode="relative", title="MRR Movement by Month")
        add_phase_lines(fig_mrr_t, p1_end, p2_end)
        st.plotly_chart(fig_mrr_t, use_container_width=True)

    with c_dd2:
        st.subheader("ROI % Over Time")
        fig_roi_t = go.Figure()
        fig_roi_t.add_trace(go.Scatter(x=f_df["Month"], y=f_df["ROI %"], mode="lines", name="ROI % (Base)"))
        fig_roi_t.add_trace(go.Scatter(x=f_pess["Month"], y=f_pess["ROI %"], mode="lines", line=dict(dash="dot", color="red"), name="ROI % (Pess)"))
        fig_roi_t.add_trace(go.Scatter(x=f_opt["Month"], y=f_opt["ROI %"], mode="lines", line=dict(dash="dash", color="green"), name="ROI % (Opt)"))
        fig_roi_t.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_roi_t.update_layout(title="Cumulative ROI % (3 Scenarios)")
        add_phase_lines(fig_roi_t, p1_end, p2_end)
        st.plotly_chart(fig_roi_t, use_container_width=True)

# ===================== MONTE CARLO =====================
if mc_enabled:
    st.markdown("---")
    st.header("Monte Carlo Simulation")
    np.random.seed(42)
    mc_results = []
    mc_var = mc_variance / 100.0
    progress = st.progress(0, text="Running Monte Carlo...")
    for iteration in range(mc_iterations):
        rand_sens = {
            "conv": base_sens["conv"] + np.random.uniform(-mc_var, mc_var),
            "churn": base_sens["churn"] + np.random.uniform(-mc_var, mc_var),
            "cpi": base_sens["cpi"] + np.random.uniform(-mc_var, mc_var),
            "organic": base_sens["organic"] + np.random.uniform(-mc_var, mc_var),
        }
        mc_df, _, _ = run_model(rand_sens)
        mc_filtered = mc_df[(mc_df["Month"] >= start_m) & (mc_df["Month"] <= end_m)]
        mc_results.append({
            "Net Profit": mc_filtered["Net Profit"].sum(),
            "Total Revenue": mc_filtered["Total Gross Revenue"].sum(),
            "End MRR": mc_filtered["Total MRR"].iloc[-1],
        })
        if (iteration + 1) % max(1, mc_iterations // 20) == 0:
            progress.progress((iteration + 1) / mc_iterations, text=f"Monte Carlo: {iteration + 1}/{mc_iterations}")
    progress.empty()

    mc_df_results = pd.DataFrame(mc_results)
    mc_c1, mc_c2, mc_c3 = st.columns(3)
    with mc_c1:
        fig_mc1 = px.histogram(mc_df_results, x="Net Profit", nbins=30, title="Net Profit Distribution")
        fig_mc1.add_vline(x=mc_df_results["Net Profit"].median(), line_dash="dash", line_color="red",
                          annotation_text="Median")
        st.plotly_chart(fig_mc1, use_container_width=True)
    with mc_c2:
        fig_mc2 = px.histogram(mc_df_results, x="Total Revenue", nbins=30, title="Revenue Distribution")
        fig_mc2.add_vline(x=mc_df_results["Total Revenue"].median(), line_dash="dash", line_color="red",
                          annotation_text="Median")
        st.plotly_chart(fig_mc2, use_container_width=True)
    with mc_c3:
        fig_mc3 = px.histogram(mc_df_results, x="End MRR", nbins=30, title="End MRR Distribution")
        fig_mc3.add_vline(x=mc_df_results["End MRR"].median(), line_dash="dash", line_color="red",
                          annotation_text="Median")
        st.plotly_chart(fig_mc3, use_container_width=True)

    st.markdown("**Monte Carlo Summary**")
    mc_summary = mc_df_results.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    mc_summary.columns = ["Count", "Mean", "Std", "Min", "P5", "P25", "Median", "P75", "P95", "Max"]
    st.dataframe(mc_summary[["Mean", "Median", "Std", "P5", "P25", "P75", "P95", "Min", "Max"]], use_container_width=True)


# ===================== FINANCIAL REPORTS =====================
st.markdown("---")
st.header("Financial Reports")

rep_tab1, rep_tab2, rep_tab3, rep_tab4, rep_tab5 = st.tabs(
    ["P&L", "Cash Flow", "Balance Sheet", "Key Metrics", "Summary & Scenarios"])

pnl_cols = ["Month", "Product Phase", "Recognized Revenue", "COGS", "Gross Profit",
            "Marketing", "Salaries", "Misc Costs", "Total Commissions", "EBITDA",
            "Corporate Tax", "Net Profit"]
cf_cols = ["Month", "Product Phase", "Total Gross Revenue", "Total Commissions",
           "Total Expenses", "Corporate Tax", "Net Cash Flow", "Cash Balance"]

with rep_tab1:
    st.subheader("Profit & Loss Statement")
    pnl_sc1, pnl_sc2, pnl_sc3 = st.tabs(["Base", "Pessimistic", "Optimistic"])
    with pnl_sc1:
        st.dataframe(f_df[pnl_cols], use_container_width=True)
    with pnl_sc2:
        st.dataframe(f_pess[pnl_cols], use_container_width=True)
    with pnl_sc3:
        st.dataframe(f_opt[pnl_cols], use_container_width=True)

with rep_tab2:
    st.subheader("Cash Flow Statement")
    cf_sc1, cf_sc2, cf_sc3 = st.tabs(["Base", "Pessimistic", "Optimistic"])
    with cf_sc1:
        st.dataframe(f_df[cf_cols], use_container_width=True)
    with cf_sc2:
        st.dataframe(f_pess[cf_cols], use_container_width=True)
    with cf_sc3:
        st.dataframe(f_opt[cf_cols], use_container_width=True)

with rep_tab3:
    st.subheader("Balance Sheet (Simplified)")
    st.dataframe(f_df[["Month", "Product Phase", "Cash Balance", "Deferred Revenue",
                        "Cumulative Net Profit"]], use_container_width=True)

with rep_tab4:
    st.subheader("Key Metrics")
    metrics_cols = ["Month", "Product Phase", "Total Active Users", "ARPU", "Blended Churn", "CRR %",
                    "LTV", "Paid CAC", "Organic CAC", "Blended CAC", "LTV/CAC", "MER", "ROAS",
                    "Payback Period (Months)", "ROI %", "NRR %", "Quick Ratio",
                    "Burn Rate", "Runway (Months)", "CAE", "Revenue per Install"]
    km_sc1, km_sc2, km_sc3 = st.tabs(["Base", "Pessimistic", "Optimistic"])
    with km_sc1:
        st.dataframe(f_df[metrics_cols], use_container_width=True)
    with km_sc2:
        st.dataframe(f_pess[metrics_cols], use_container_width=True)
    with km_sc3:
        st.dataframe(f_opt[metrics_cols], use_container_width=True)

with rep_tab5:
    # Helper to build phase summary
    def build_phase_summary(sdf, label=""):
        ps = sdf.groupby("Product Phase").agg(
            Months=("Month", "count"),
            Total_Revenue=("Total Gross Revenue", "sum"),
            Total_Marketing=("Marketing", "sum"),
            Total_Salaries=("Salaries", "sum"),
            Total_Misc=("Misc Costs", "sum"),
            Total_COGS=("COGS", "sum"),
            Total_Commissions=("Total Commissions", "sum"),
            Net_Profit=("Net Profit", "sum"),
            End_Users=("Total Active Users", "last"),
            New_Users=("New Paid Users", "sum"),
            End_MRR=("Total MRR", "last"),
            Avg_ARPU=("ARPU", "mean"),
            Avg_LTV_CAC=("LTV/CAC", "mean"),
        ).reset_index()
        ps.columns = ["Phase", "Months", "Revenue", "Marketing", "Salaries",
                       "Misc", "COGS", "Commissions",
                       "Net Profit", "End Users", "New Users", "End MRR",
                       "Avg ARPU", "Avg LTV/CAC"]
        return ps

    st.subheader("Summary by Phase")
    ps_sc1, ps_sc2, ps_sc3 = st.tabs(["Base", "Pessimistic", "Optimistic"])
    with ps_sc1:
        st.dataframe(build_phase_summary(f_df), use_container_width=True)
    with ps_sc2:
        st.dataframe(build_phase_summary(f_pess), use_container_width=True)
    with ps_sc3:
        st.dataframe(build_phase_summary(f_opt), use_container_width=True)

    # Scenario comparison table
    st.subheader("Scenario Comparison Table")
    sc_table = pd.DataFrame({
        "Metric": ["Total Revenue", "Net Profit", "End MRR", "End Users", "Cumulative ROI %",
                    "Break-Even Month", "Cumulative BE", "Runway Out"],
        "Pessimistic": [
            f"${f_pess['Total Gross Revenue'].sum():,.0f}",
            f"${f_pess['Net Profit'].sum():,.0f}",
            f"${f_pess['Total MRR'].iloc[-1]:,.0f}",
            f"{f_pess['Total Active Users'].iloc[-1]:,.0f}",
            f"{f_pess['ROI %'].iloc[-1]:,.0f}%",
            fmt_milestone(milestones_pess.get("break_even_month")),
            fmt_milestone(milestones_pess.get("cumulative_break_even")),
            fmt_milestone(milestones_pess.get("runway_out_month")),
        ],
        "Base": [
            f"${f_df['Total Gross Revenue'].sum():,.0f}",
            f"${f_df['Net Profit'].sum():,.0f}",
            f"${f_df['Total MRR'].iloc[-1]:,.0f}",
            f"{f_df['Total Active Users'].iloc[-1]:,.0f}",
            f"{f_df['ROI %'].iloc[-1]:,.0f}%",
            fmt_milestone(milestones_main.get("break_even_month")),
            fmt_milestone(milestones_main.get("cumulative_break_even")),
            fmt_milestone(milestones_main.get("runway_out_month")),
        ],
        "Optimistic": [
            f"${f_opt['Total Gross Revenue'].sum():,.0f}",
            f"${f_opt['Net Profit'].sum():,.0f}",
            f"${f_opt['Total MRR'].iloc[-1]:,.0f}",
            f"{f_opt['Total Active Users'].iloc[-1]:,.0f}",
            f"{f_opt['ROI %'].iloc[-1]:,.0f}%",
            fmt_milestone(milestones_opt.get("break_even_month")),
            fmt_milestone(milestones_opt.get("cumulative_break_even")),
            fmt_milestone(milestones_opt.get("runway_out_month")),
        ],
    })
    st.dataframe(sc_table, use_container_width=True, hide_index=True)

    # Milestone comparison across scenarios
    st.subheader("Milestone Comparison")
    ms_compare = pd.DataFrame({
        "Milestone": ["Break-Even (P&L)", "Cumulative Break-Even", "Cash Flow Positive",
                       "Investment Payback", "1K Users", "10K Users", "MRR $10K", "MRR $100K", "Runway Out"],
        "Pessimistic": [
            fmt_milestone(milestones_pess.get(k)) for k in
            ["break_even_month", "cumulative_break_even", "cf_positive_month",
             "investment_payback_month", "users_1000", "users_10000", "mrr_10000", "mrr_100000", "runway_out_month"]
        ],
        "Base": [
            fmt_milestone(milestones_main.get(k)) for k in
            ["break_even_month", "cumulative_break_even", "cf_positive_month",
             "investment_payback_month", "users_1000", "users_10000", "mrr_10000", "mrr_100000", "runway_out_month"]
        ],
        "Optimistic": [
            fmt_milestone(milestones_opt.get(k)) for k in
            ["break_even_month", "cumulative_break_even", "cf_positive_month",
             "investment_payback_month", "users_1000", "users_10000", "mrr_10000", "mrr_100000", "runway_out_month"]
        ],
    })
    st.dataframe(ms_compare, use_container_width=True, hide_index=True)

# ===================== EXPORT =====================
st.markdown("---")
st.header("Export Data")

# Individual scenario CSVs
csv_base = df_main.to_csv(index=False).encode("utf-8")
csv_pess = df_pessimistic.to_csv(index=False).encode("utf-8")
csv_opt = df_optimistic.to_csv(index=False).encode("utf-8")

# Combined all-scenarios Excel-style CSV
df_base_tagged = df_main.copy()
df_base_tagged.insert(0, "Scenario", "Base")
df_pess_tagged = df_pessimistic.copy()
df_pess_tagged.insert(0, "Scenario", "Pessimistic")
df_opt_tagged = df_optimistic.copy()
df_opt_tagged.insert(0, "Scenario", "Optimistic")
df_all = pd.concat([df_base_tagged, df_pess_tagged, df_opt_tagged], ignore_index=True)

# Milestones summary
ms_rows = []
for scenario, ms in [("Base", milestones_main), ("Pessimistic", milestones_pess), ("Optimistic", milestones_opt)]:
    row = {"Scenario": scenario}
    for key, label in [
        ("break_even_month", "Break-Even Month"), ("cumulative_break_even", "Cumulative Break-Even"),
        ("cf_positive_month", "CF Positive Month"), ("investment_payback_month", "Investment Payback"),
        ("runway_out_month", "Runway Out"), ("users_1000", "1K Users Month"),
        ("users_10000", "10K Users Month"), ("users_100000", "100K Users Month"),
        ("mrr_10000", "MRR $10K Month"), ("mrr_50000", "MRR $50K Month"),
        ("mrr_100000", "MRR $100K Month"), ("mrr_1000000", "MRR $1M Month"),
    ]:
        row[label] = ms.get(key, None)
    ms_rows.append(row)
df_milestones = pd.DataFrame(ms_rows)

csv_all = df_all.to_csv(index=False).encode("utf-8")
csv_milestones = df_milestones.to_csv(index=False).encode("utf-8")

exp_c1, exp_c2, exp_c3, exp_c4, exp_c5 = st.columns(5)
exp_c1.download_button("Base (CSV)", csv_base, "base_scenario.csv", "text/csv")
exp_c2.download_button("Pessimistic (CSV)", csv_pess, "pessimistic_scenario.csv", "text/csv")
exp_c3.download_button("Optimistic (CSV)", csv_opt, "optimistic_scenario.csv", "text/csv")
exp_c4.download_button("All Scenarios (CSV)", csv_all, "all_scenarios.csv", "text/csv")
exp_c5.download_button("Milestones (CSV)", csv_milestones, "milestones.csv", "text/csv")
