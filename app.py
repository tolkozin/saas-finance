import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Awesome Dashboard", layout="wide")
st.title("Awesome Dashboard")

# ===================== SIDEBAR =====================

# --- Phase Configuration ---
with st.sidebar.expander("Phase Configuration", expanded=True):
    phase1_dur = st.number_input("Phase 1 Duration (months)", min_value=1, max_value=24, value=3,
        help="Длительность фазы Pre-MVP — разработка, нет рекламы и пользователей.")
    phase2_dur = st.number_input("Phase 2 Duration (months)", min_value=1, max_value=24, value=3,
        help="Длительность фазы MVP — мягкий запуск, первые пользователи.")
    phase3_dur = 60 - phase1_dur - phase2_dur
    if phase3_dur < 1:
        st.error("Phase 1 + Phase 2 должны быть < 60 месяцев!")
        st.stop()
    st.caption(f"Phase 3 (Scaling): {phase3_dur} мес.")

with st.sidebar.expander("Scenarios"):
    scenario_sensitivity = st.slider("Scenario Sensitivity %", -50, 50, 0,
        help="Сдвиг сценария — влияет на конверсии и удержание. 0 = базовый.")

# --- Taxes & Fees ---
with st.sidebar.expander("Taxes & Payment Fees"):
    corporate_tax = st.number_input("Corporate Tax %", min_value=0.0, max_value=50.0, value=1.0,
        help="Налог на прибыль. 1% Грузия, 5% Армения, 3-10% Казахстан.")
    store_split = st.slider("Store vs Web % (Store)", 0, 100, 50,
        help="Процент пользователей, покупающих через App Store.")
    app_store_comm = st.number_input("App Store Commission %", min_value=0.0, max_value=50.0, value=15.0,
        help="Комиссия App Store. 15% по Small Business Program до $1M/год.")
    web_comm_pct = st.number_input("Web Commission %", min_value=0.0, max_value=20.0, value=3.5,
        help="Процентная комиссия Lemon Squeezy / Paddle.")
    web_comm_fixed = st.number_input("Web Fixed Fee per Txn ($)", min_value=0.0, max_value=5.0, value=0.50,
        help="Фиксированная комиссия за каждую Web-транзакцию.")
    bank_fee = st.number_input("Banking Fee %", min_value=0.0, max_value=10.0, value=1.0,
        help="Комиссия банка за переводы и конвертации.")

# --- Phase 1: Pre-MVP ---
with st.sidebar.expander("Phase 1: Pre-MVP"):
    p1_investment = st.number_input("Phase 1 Investment ($)", min_value=0.0, value=100000.0,
        help="Инвестиции в начале Phase 1. Добавляются к балансу в первый месяц.")
    p1_salaries_total = st.number_input("Phase 1 Total Salaries ($)", min_value=0.0, value=17475.0,
        help="Общая сумма зарплат за всю фазу. Распределяется равномерно по месяцам.")
    p1_misc_total = st.number_input("Phase 1 Total Misc Costs ($)", min_value=0.0, value=8419.0,
        help="Прочие расходы за всю фазу (интеграции, контент, инфра).")
    p1_ad_budget = st.number_input("Phase 1 Monthly Ad Budget ($)", min_value=0.0, value=0.0,
        help="Рекламный бюджет в месяц. Обычно 0 — продукт не готов.")
    p1_cpi = st.number_input("Phase 1 CPI ($)", min_value=0.01, value=7.50,
        help="Стоимость одной установки приложения.")
    p1_conv_trial = st.number_input("Phase 1 Conv. to Trial %", min_value=0.0, max_value=100.0, value=0.0,
        help="Конверсия установки в триал. На Pre-MVP обычно 0.")
    p1_conv_paid = st.number_input("Phase 1 Trial-to-Paid %", min_value=0.0, max_value=100.0, value=0.0,
        help="Конверсия триала в оплату. На Pre-MVP обычно 0.")

# --- Phase 2: MVP ---
with st.sidebar.expander("Phase 2: MVP"):
    p2_investment = st.number_input("Phase 2 Investment ($)", min_value=0.0, value=0.0,
        help="Дополнительные инвестиции в начале Phase 2.")
    p2_salaries_total = st.number_input("Phase 2 Total Salaries ($)", min_value=0.0, value=3600.0,
        help="Общая сумма зарплат за фазу MVP.")
    p2_misc_total = st.number_input("Phase 2 Total Misc Costs ($)", min_value=0.0, value=750.0,
        help="Прочие расходы за фазу MVP.")
    p2_ad_budget = st.number_input("Phase 2 Monthly Ad Budget ($)", min_value=0.0, value=5000.0,
        help="Стартовый рекламный бюджет на MVP.")
    p2_cpi = st.number_input("Phase 2 CPI ($)", min_value=0.01, value=7.50,
        help="Стоимость установки на этапе MVP.")
    p2_conv_trial = st.number_input("Phase 2 Conv. to Trial %", min_value=0.0, max_value=100.0, value=20.0,
        help="Конверсия в триал на MVP — обычно ниже чем на зрелом продукте.")
    p2_conv_paid = st.number_input("Phase 2 Trial-to-Paid %", min_value=0.0, max_value=100.0, value=20.0,
        help="Конверсия в оплату на MVP.")

# --- Phase 3: Scaling ---
with st.sidebar.expander("Phase 3: Scaling"):
    p3_investment = st.number_input("Phase 3 Investment ($)", min_value=0.0, value=0.0,
        help="Дополнительные инвестиции в начале Phase 3.")
    p3_salaries_total = st.number_input("Phase 3 Total Salaries ($)", min_value=0.0, value=64800.0,
        help="Общая сумма зарплат за фазу масштабирования.")
    p3_misc_total = st.number_input("Phase 3 Total Misc Costs ($)", min_value=0.0, value=13500.0,
        help="Прочие расходы за фазу масштабирования.")
    p3_ad_budget = st.number_input("Phase 3 Monthly Ad Budget ($)", min_value=0.0, value=150000.0,
        help="Стартовый рекламный бюджет на этапе масштабирования.")
    p3_cpi = st.number_input("Phase 3 CPI ($)", min_value=0.01, value=7.50,
        help="Стоимость установки на зрелом продукте.")
    p3_conv_trial = st.number_input("Phase 3 Conv. to Trial %", min_value=0.0, max_value=100.0, value=25.0,
        help="Конверсия в триал на зрелом продукте.")
    p3_conv_paid = st.number_input("Phase 3 Trial-to-Paid %", min_value=0.0, max_value=100.0, value=25.0,
        help="Конверсия в оплату на зрелом продукте.")

# --- Ad Growth ---
with st.sidebar.expander("Ad Budget Growth"):
    ad_growth_mode = st.radio("Growth Mode", ["Percentage (%)", "Absolute ($)"],
        help="Как растёт рекламный бюджет: процент от текущего или фиксированная сумма в месяц.")
    if ad_growth_mode == "Percentage (%)":
        ad_growth_pct = st.number_input("MoM Growth %", min_value=0.0, max_value=100.0, value=5.0,
            help="Ежемесячный рост бюджета в процентах.")
        ad_growth_abs = 0.0
    else:
        ad_growth_abs = st.number_input("MoM Growth ($)", min_value=0.0, value=5000.0,
            help="Ежемесячный рост бюджета в долларах.")
        ad_growth_pct = 0.0
    cpi_deg = st.number_input("CPI Degradation %", min_value=0.0, max_value=10.0, value=1.0,
        help="Рост стоимости установки за каждые +$1000 к бюджету.")

# --- Organic ---
with st.sidebar.expander("Organic Acquisition"):
    starting_organic = st.number_input("Starting Organic Traffic", min_value=0.0, value=0.0,
        help="Начальное количество органических посетителей в месяц.")
    organic_growth_mode = st.radio("Organic Growth Mode", ["Percentage (%)", "Absolute (users)"],
        help="Как растёт органика: в процентах или фиксированное число новых посетителей.")
    if organic_growth_mode == "Percentage (%)":
        organic_growth_pct = st.number_input("Organic MoM Growth %", min_value=0.0, max_value=200.0, value=10.0,
            help="Ежемесячный процентный рост органического трафика.")
        organic_growth_abs = 0.0
    else:
        organic_growth_abs = st.number_input("Organic MoM Growth (users)", min_value=0.0, value=500.0,
            help="Сколько органических посетителей добавляется каждый месяц.")
        organic_growth_pct = 0.0
    organic_conv_trial = st.number_input("Organic Conv. to Trial %", min_value=0.0, max_value=100.0, value=30.0,
        help="Конверсия органики в триал. Обычно выше платного трафика.")
    organic_conv_paid = st.number_input("Organic Trial-to-Paid %", min_value=0.0, max_value=100.0, value=30.0,
        help="Конверсия органических триалов в оплату.")
    organic_spend = st.number_input("Monthly Organic Spend ($)", min_value=0.0, value=0.0,
        help="Расходы на контент, SEO, ASO в месяц.")

# --- Pricing ---
with st.sidebar.expander("Subscription Mix & Pricing"):
    mix_weekly = st.number_input("Mix: Weekly %", min_value=0.0, max_value=100.0, value=0.0,
        help="Доля новых подписчиков на недельном плане.")
    mix_monthly = st.number_input("Mix: Monthly %", min_value=0.0, max_value=100.0, value=48.0,
        help="Доля на месячном плане.")
    mix_annual = st.number_input("Mix: Annual %", min_value=0.0, max_value=100.0, value=52.0,
        help="Доля на годовом плане.")
    total_mix = mix_weekly + mix_monthly + mix_annual
    if total_mix > 0:
        mix_weekly /= total_mix
        mix_monthly /= total_mix
        mix_annual /= total_mix
    else:
        mix_monthly = 1.0
    price_weekly = st.number_input("Price: Weekly ($)", min_value=0.0, value=4.99,
        help="Цена недельной подписки.")
    price_monthly = st.number_input("Price: Monthly ($)", min_value=0.0, value=7.99,
        help="Цена месячной подписки.")
    price_annual = st.number_input("Price: Annual ($)", min_value=0.0, value=49.99,
        help="Цена годовой подписки (один платёж за год).")

# --- Retention ---
with st.sidebar.expander("Retention & Churn"):
    weekly_cancel_rate = st.number_input("Weekly Cancellation Rate %", min_value=0.0, max_value=100.0, value=15.0,
        help="Процент недельных подписчиков, отменяющих каждую неделю.")
    monthly_churn_rate = st.number_input("Monthly Churn Rate %", min_value=0.0, max_value=100.0, value=10.0,
        help="Процент месячных подписчиков, уходящих каждый месяц.")
    annual_non_renewal = st.number_input("Annual Non-Renewal Rate %", min_value=0.0, max_value=100.0, value=30.0,
        help="Процент годовых подписчиков, НЕ продлевающих через 12 месяцев.")
    st.markdown("**Churn Multiplier by Phase**")
    p2_churn_mult = st.number_input("Phase 2 Churn Multiplier", min_value=0.1, max_value=5.0, value=1.5,
        help="Множитель оттока на MVP. 1.5 = отток на 50% выше базового.")
    p3_churn_mult = st.number_input("Phase 3 Churn Multiplier", min_value=0.1, max_value=5.0, value=1.0,
        help="Множитель оттока на этапе масштабирования. 1.0 = базовый.")

# --- Trial & Refunds ---
with st.sidebar.expander("Trial & Refunds"):
    trial_days = st.number_input("Trial Duration (days)", min_value=0, max_value=90, value=7,
        help="Длительность бесплатного триала. 0 = оплата сразу.")
    refund_rate = st.number_input("Refund Rate %", min_value=0.0, max_value=30.0, value=2.0,
        help="Процент возвратов от валовой выручки.")
    cogs_per_user = st.number_input("COGS per Active User ($)", min_value=0.0, value=0.10,
        help="Затраты на серверы/хостинг на одного активного пользователя в месяц.")


# ===================== DATA ENGINE =====================

def add_phase_lines(fig, p1_end, p2_end):
    """Add vertical phase boundary lines to a chart."""
    fig.add_vline(x=p1_end + 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Phase 2", annotation_position="top")
    fig.add_vline(x=p2_end + 0.5, line_dash="dot", line_color="gray",
                  annotation_text="Phase 3", annotation_position="top")
    return fig


def run_model(sens_percent):
    months = np.arange(1, 61)
    df = pd.DataFrame({"Month": months})

    sens_factor = 1 + (sens_percent / 100.0)

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
        1: {"ad": p1_ad_budget, "cpi": p1_cpi,
            "ct": p1_conv_trial / 100.0 * sens_factor,
            "cp": p1_conv_paid / 100.0 * sens_factor,
            "sal": p1_salaries_total / phase1_dur,
            "misc": p1_misc_total / phase1_dur,
            "inv": p1_investment, "churn_m": 1.0},
        2: {"ad": p2_ad_budget, "cpi": p2_cpi,
            "ct": p2_conv_trial / 100.0 * sens_factor,
            "cp": p2_conv_paid / 100.0 * sens_factor,
            "sal": p2_salaries_total / phase2_dur,
            "misc": p2_misc_total / phase2_dur,
            "inv": p2_investment, "churn_m": p2_churn_mult},
        3: {"ad": p3_ad_budget, "cpi": p3_cpi,
            "ct": p3_conv_trial / 100.0 * sens_factor,
            "cp": p3_conv_paid / 100.0 * sens_factor,
            "sal": p3_salaries_total / phase3_dur,
            "misc": p3_misc_total / phase3_dur,
            "inv": p3_investment, "churn_m": p3_churn_mult},
    }

    # --- Ad Budget with growth ---
    ad_budgets = np.zeros(60)
    for i in range(60):
        phase = get_phase(i + 1)
        base = phase_cfg[phase]["ad"]
        if phase == 1:
            m_in = i
        elif phase == 2:
            m_in = i - p1_end
        else:
            m_in = i - p2_end
        if ad_growth_mode == "Percentage (%)":
            ad_budgets[i] = base * ((1 + ad_growth_pct / 100.0) ** m_in)
        else:
            ad_budgets[i] = base + ad_growth_abs * m_in

    df["Ad Budget"] = ad_budgets

    # --- CPI ---
    phase_cpi = df["Product Phase"].map({1: p1_cpi, 2: p2_cpi, 3: p3_cpi})
    phase_base_ad = df["Product Phase"].map({1: p1_ad_budget, 2: p2_ad_budget, 3: p3_ad_budget})
    df["CPI"] = phase_cpi * (1 + (cpi_deg / 100.0) * ((df["Ad Budget"] - phase_base_ad).clip(lower=0) / 1000.0))
    df["Installs"] = np.where(df["Ad Budget"] > 0, df["Ad Budget"] / df["CPI"], 0)

    # --- Per-phase conversions ---
    conv_t = np.array([phase_cfg[get_phase(m)]["ct"] for m in months])
    conv_p = np.array([phase_cfg[get_phase(m)]["cp"] for m in months])

    df["New Trials"] = df["Installs"].values * conv_t

    # Trial delay
    trial_delay = trial_days // 30
    paid_new = df["New Trials"].values * conv_p
    if trial_delay > 0:
        paid_new = np.concatenate([np.zeros(trial_delay), paid_new[:60 - trial_delay]])
    df["Paid New Paid Users"] = paid_new

    # --- Organic ---
    org_traffic = np.zeros(60)
    if starting_organic > 0:
        for i in range(60):
            if organic_growth_mode == "Percentage (%)":
                org_traffic[i] = starting_organic * ((1 + organic_growth_pct / 100.0) ** i)
            else:
                org_traffic[i] = starting_organic + organic_growth_abs * i
    df["Organic Traffic"] = org_traffic

    org_ct = (organic_conv_trial / 100.0) * sens_factor
    org_cp = (organic_conv_paid / 100.0) * sens_factor
    org_new = org_traffic * org_ct * org_cp
    if trial_delay > 0:
        org_new = np.concatenate([np.zeros(trial_delay), org_new[:60 - trial_delay]])
    df["Organic New Paid Users"] = org_new
    df["New Paid Users"] = df["Paid New Paid Users"] + df["Organic New Paid Users"]
    df["New Web Users"] = df["New Paid Users"] * (100 - store_split) / 100.0
    df["New Store Users"] = df["New Paid Users"] * store_split / 100.0

    # --- Churn rates ---
    base_churn_w = 1 - (1 - weekly_cancel_rate / 100.0) ** 4.33
    base_churn_m = monthly_churn_rate / 100.0
    base_non_renewal = annual_non_renewal / 100.0
    churn_mult_map = {1: 1.0, 2: p2_churn_mult, 3: p3_churn_mult}

    # --- Cohort matrices ---
    cohorts = {}
    for plat in ["web", "store"]:
        for plan in ["weekly", "monthly", "annual"]:
            cohorts[f"{plat}_{plan}"] = np.zeros((60, 60))

    for i in range(60):
        nw = df.loc[i, "New Web Users"]
        ns = df.loc[i, "New Store Users"]
        cohorts["web_weekly"][i, i] = nw * mix_weekly
        cohorts["web_monthly"][i, i] = nw * mix_monthly
        cohorts["web_annual"][i, i] = nw * mix_annual
        cohorts["store_weekly"][i, i] = ns * mix_weekly
        cohorts["store_monthly"][i, i] = ns * mix_monthly
        cohorts["store_annual"][i, i] = ns * mix_annual

        for j in range(i + 1, 60):
            phase_j = get_phase(j + 1)
            mult = churn_mult_map[phase_j] / sens_factor

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

    # --- Revenue ---
    gross_rev_web = np.zeros(60)
    gross_rev_store = np.zeros(60)
    mrr_web = np.zeros(60)
    mrr_store = np.zeros(60)
    mrr_weekly_a = np.zeros(60)
    mrr_monthly_a = np.zeros(60)
    mrr_annual_a = np.zeros(60)
    tx_web = np.zeros(60)

    for j in range(60):
        mw_web = np.sum(cohorts["web_weekly"][:, j]) * price_weekly * 4.33
        mm_web = np.sum(cohorts["web_monthly"][:, j]) * price_monthly
        ma_web = np.sum(cohorts["web_annual"][:, j]) * price_annual / 12.0
        mw_st = np.sum(cohorts["store_weekly"][:, j]) * price_weekly * 4.33
        mm_st = np.sum(cohorts["store_monthly"][:, j]) * price_monthly
        ma_st = np.sum(cohorts["store_annual"][:, j]) * price_annual / 12.0

        mrr_web[j] = mw_web + mm_web + ma_web
        mrr_store[j] = mw_st + mm_st + ma_st
        mrr_weekly_a[j] = mw_web + mw_st
        mrr_monthly_a[j] = mm_web + mm_st
        mrr_annual_a[j] = ma_web + ma_st

        cash_a_web = 0
        cash_a_store = 0
        tx_a = 0
        for i in range(j + 1):
            if (j - i) % 12 == 0:
                cash_a_web += cohorts["web_annual"][i, j] * price_annual
                cash_a_store += cohorts["store_annual"][i, j] * price_annual
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

    # Commissions
    df["Store Commission"] = df["Gross Revenue Store"] * (app_store_comm / 100.0)
    df["Web Commission"] = df["Gross Revenue Web"] * (web_comm_pct / 100.0) + tx_web * web_comm_fixed * rf
    df["Bank Fee"] = df["Total Gross Revenue"] * (bank_fee / 100.0)
    df["Total Commissions"] = df["Store Commission"] + df["Web Commission"] + df["Bank Fee"]
    df["Net Revenue"] = df["Total Gross Revenue"] - df["Total Commissions"]

    # Active users
    active_w = np.array([np.sum(cohorts["web_weekly"][:, j] + cohorts["store_weekly"][:, j]) for j in range(60)])
    active_m = np.array([np.sum(cohorts["web_monthly"][:, j] + cohorts["store_monthly"][:, j]) for j in range(60)])
    active_a = np.array([np.sum(cohorts["web_annual"][:, j] + cohorts["store_annual"][:, j]) for j in range(60)])
    df["Active Web Users"] = [np.sum(cohorts["web_weekly"][:, j] + cohorts["web_monthly"][:, j] + cohorts["web_annual"][:, j]) for j in range(60)]
    df["Active Store Users"] = [np.sum(cohorts["store_weekly"][:, j] + cohorts["store_monthly"][:, j] + cohorts["store_annual"][:, j]) for j in range(60)]
    df["Total Active Users"] = df["Active Web Users"] + df["Active Store Users"]

    # Costs
    df["COGS"] = df["Total Active Users"] * cogs_per_user
    df["Organic Spend"] = organic_spend
    df["Marketing"] = df["Ad Budget"] + organic_spend
    df["Salaries"] = df["Product Phase"].map({1: phase_cfg[1]["sal"], 2: phase_cfg[2]["sal"], 3: phase_cfg[3]["sal"]})
    df["Misc Costs"] = df["Product Phase"].map({1: phase_cfg[1]["misc"], 2: phase_cfg[2]["misc"], 3: phase_cfg[3]["misc"]})
    df["Total Expenses"] = df["COGS"] + df["Marketing"] + df["Salaries"] + df["Misc Costs"]

    df["Gross Profit"] = df["Recognized Revenue"] - df["COGS"] - df["Total Commissions"]
    df["EBITDA"] = df["Gross Profit"] - df["Marketing"] - df["Salaries"] - df["Misc Costs"]
    df["Corporate Tax"] = df["EBITDA"].apply(lambda x: x * (corporate_tax / 100.0) if x > 0 else 0)
    df["Net Profit"] = df["EBITDA"] - df["Corporate Tax"]
    df["Net Cash Flow"] = df["Total Gross Revenue"] - df["Total Commissions"] - df["Total Expenses"] - df["Corporate Tax"]

    # Cash Balance with per-phase investments
    cash_bal = np.zeros(60)
    for j in range(60):
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

    # Metrics
    df["Paid CAC"] = df["Ad Budget"] / df["Paid New Paid Users"].replace(0, np.nan)
    df["Organic CAC"] = organic_spend / df["Organic New Paid Users"].replace(0, np.nan)
    df["Blended CAC"] = df["Marketing"] / df["New Paid Users"].replace(0, np.nan)
    df["ARPU"] = df["Total MRR"] / df["Total Active Users"].replace(0, np.nan)

    # Blended churn weighted by sub type
    blended_churn = np.zeros(60)
    for j in range(60):
        phase_j = get_phase(j + 1)
        mult = churn_mult_map[phase_j] / sens_factor
        cw = min(1.0, base_churn_w * mult)
        cm_val = min(1.0, base_churn_m * mult)
        ca_monthly = min(1.0, base_non_renewal * mult) / 12.0
        total = active_w[j] + active_m[j] + active_a[j]
        if total > 0:
            blended_churn[j] = (cw * active_w[j] + cm_val * active_m[j] + ca_monthly * active_a[j]) / total
    df["Blended Churn"] = blended_churn
    df["Gross Margin %"] = df["Gross Profit"] / df["Recognized Revenue"].replace(0, np.nan)
    df["LTV"] = (df["ARPU"] * df["Gross Margin %"]) / pd.Series(blended_churn).replace(0, np.nan)
    df["LTV/CAC"] = df["LTV"] / df["Blended CAC"].replace(0, np.nan)
    df["MER"] = df["Total Gross Revenue"] / df["Marketing"].replace(0, np.nan)
    df["Payback Period (Months)"] = df["Blended CAC"] / (df["ARPU"] * df["Gross Margin %"]).replace(0, np.nan)

    # Churn rates for display
    df["Weekly Churn %"] = [min(100.0, base_churn_w * churn_mult_map[get_phase(m)] / sens_factor * 100) for m in months]
    df["Monthly Churn %"] = [min(100.0, base_churn_m * churn_mult_map[get_phase(m)] / sens_factor * 100) for m in months]
    df["Annual Non-Renewal %"] = [min(100.0, base_non_renewal * churn_mult_map[get_phase(m)] / sens_factor * 100) for m in months]

    return df


df_main = run_model(scenario_sensitivity)

p1_end = phase1_dur
p2_end = phase1_dur + phase2_dur

# ===================== GLOBAL FILTER =====================
st.header("Global Dashboard Filters")
month_range = st.slider("Select Month Range", 1, 60, (1, 60),
    help="Фильтр по месяцам — влияет на все графики и отчёты.")
start_m, end_m = month_range
f_df = df_main[(df_main["Month"] >= start_m) & (df_main["Month"] <= end_m)]

# ===================== TOP METRICS =====================
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${f_df['Total Gross Revenue'].sum():,.0f}",
    help="Сумма валовой выручки за период (за вычетом возвратов). Формула: SUM(Gross Revenue Web + Store).")
col2.metric("Net Profit", f"${f_df['Net Profit'].sum():,.0f}",
    help="Чистая прибыль за период. Формула: SUM(EBITDA - Corporate Tax).")
col3.metric("End MRR", f"${f_df['Total MRR'].iloc[-1]:,.0f}",
    help="MRR на конец периода — регулярная месячная выручка. Формула: MRR Web + MRR Store.")
col4.metric("Avg LTV/CAC", f"{f_df['LTV/CAC'].mean():.2f}x",
    help="Среднее отношение ценности клиента к стоимости привлечения. >3x = отлично. Формула: AVG(LTV / Blended CAC).")

# ===================== CHARTS =====================
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["Growth & Revenue", "Unit Economics & Efficiency", "P&L & Scenarios"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MRR by Subscription Plan")
        fig1 = px.area(f_df, x="Month", y=["MRR Weekly", "MRR Monthly", "MRR Annual"],
                       title="MRR Breakdown", labels={"value": "MRR ($)", "variable": "Plan"})
        add_phase_lines(fig1, p1_end, p2_end)
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("*Выручка по тарифам. Weekly = пользователи * цена * 4.33; Monthly = пользователи * цена; Annual = пользователи * цена / 12.*")

    with c2:
        st.subheader("Cash Balance")
        fig2 = px.bar(f_df, x="Month", y="Cash Balance", title="Monthly Cash Balance")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        add_phase_lines(fig2, p1_end, p2_end)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("*Баланс денежных средств. Красная линия — ноль. Если баланс уходит ниже — деньги закончились.*")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Gross vs Net Revenue")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Gross Revenue Web"], name="Gross Web"))
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Net Revenue"], name="Net Revenue"))
        fig4.update_layout(barmode="group", title="Revenue Comparison")
        add_phase_lines(fig4, p1_end, p2_end)
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("*Разница между валовой и чистой выручкой — показывает сколько забирают комиссии.*")

    with c4:
        st.subheader("Churn Rates by Subscription Type")
        fig6 = px.line(f_df, x="Month", y=["Weekly Churn %", "Monthly Churn %", "Annual Non-Renewal %"],
                       title="Churn Rates")
        add_phase_lines(fig6, p1_end, p2_end)
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("*Отток по типам подписок. Weekly — еженедельная отмена (пересчитана в месячную), Monthly — месячный чурн, Annual — % непродлений (применяется раз в 12 мес).*")

with tab2:
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("LTV vs CAC")
        fig3 = px.line(f_df, x="Month", y=["LTV", "Paid CAC", "Organic CAC", "Blended CAC"], title="LTV vs CAC")
        add_phase_lines(fig3, p1_end, p2_end)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("*LTV — сколько приносит клиент за всё время. CAC — сколько стоит его привлечь. LTV должен быть выше CAC минимум в 3 раза.*")

    with c6:
        st.subheader("Payback Period")
        fig5 = px.line(f_df, x="Month", y="Payback Period (Months)", title="Payback Period (Months)")
        add_phase_lines(fig5, p1_end, p2_end)
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("*За сколько месяцев окупается привлечение одного пользователя. Чем меньше — тем лучше. Формула: Blended CAC / (ARPU * Gross Margin).*")

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Marketing Efficiency Ratio (MER)")
        fig8 = px.line(f_df, x="Month", y="MER", title="MER")
        add_phase_lines(fig8, p1_end, p2_end)
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("*Сколько долларов выручки приносит каждый доллар маркетинга. >1 = выручка больше затрат. Формула: Total Revenue / Marketing.*")

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
        st.markdown("*Как выручка превращается в прибыль — что именно съедает деньги.*")

    with c10:
        st.subheader("Scenario Comparison")
        df_base = run_model(0)
        df_worst = run_model(-20)
        df_best = run_model(20)
        scenarios = ["Pessimistic (-20%)", "Base (0%)", "Optimistic (+20%)"]
        profits = [
            df_worst[(df_worst["Month"] >= start_m) & (df_worst["Month"] <= end_m)]["Net Profit"].sum(),
            df_base[(df_base["Month"] >= start_m) & (df_base["Month"] <= end_m)]["Net Profit"].sum(),
            df_best[(df_best["Month"] >= start_m) & (df_best["Month"] <= end_m)]["Net Profit"].sum()
        ]
        fig9 = px.bar(x=scenarios, y=profits, title="Net Profit by Scenario",
                      labels={"x": "Scenario", "y": "Net Profit ($)"})
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("*Сценарии: Sensitivity влияет на конверсии (+) и чурн (-). -20% = пессимистичный, +20% = оптимистичный.*")

# ===================== FINANCIAL REPORTS =====================
st.markdown("---")
st.header("Financial Reports")

rep_tab1, rep_tab2, rep_tab3, rep_tab4, rep_tab5 = st.tabs(
    ["P&L", "Cash Flow", "Balance Sheet", "Key Metrics", "Summary by Phase"])


def display_df(data):
    st.dataframe(
        data,
        column_config={
            "Month": st.column_config.NumberColumn("Month", help="Порядковый номер месяца (1-60)."),
            "Product Phase": st.column_config.NumberColumn("Phase", help="Фаза: 1 = Pre-MVP, 2 = MVP, 3 = Scaling."),
            "Recognized Revenue": st.column_config.NumberColumn("Recognized Revenue",
                help="Признанная выручка = MRR. Формула: MRR Web + MRR Store."),
            "Net Profit": st.column_config.NumberColumn("Net Profit",
                help="Чистая прибыль после всех расходов и налогов. Формула: EBITDA - Tax."),
            "Cash Balance": st.column_config.NumberColumn("Cash Balance",
                help="Сколько денег на счету. Формула: предыдущий баланс + инвестиции + чистый денежный поток."),
            "Deferred Revenue": st.column_config.NumberColumn("Deferred Revenue",
                help="Собранные но ещё не признанные деньги (годовые подписки). Формула: CUMSUM(Gross - Recognized)."),
            "LTV": st.column_config.NumberColumn("LTV",
                help="Сколько денег принесёт клиент за всю жизнь. Формула: ARPU * Gross Margin / Churn."),
            "Blended CAC": st.column_config.NumberColumn("Blended CAC",
                help="Средняя стоимость привлечения клиента. Формула: (Ad Budget + Organic Spend) / New Paid Users."),
            "Paid CAC": st.column_config.NumberColumn("Paid CAC",
                help="Стоимость привлечения через рекламу. Формула: Ad Budget / Paid New Users."),
            "Organic CAC": st.column_config.NumberColumn("Organic CAC",
                help="Стоимость привлечения через органику. Формула: Organic Spend / Organic New Users."),
        },
        use_container_width=True
    )


with rep_tab1:
    st.subheader("Profit & Loss Statement")
    display_df(f_df[["Month", "Product Phase", "Recognized Revenue", "COGS", "Gross Profit",
                      "Marketing", "Salaries", "Misc Costs", "Total Commissions", "EBITDA",
                      "Corporate Tax", "Net Profit"]])

with rep_tab2:
    st.subheader("Cash Flow Statement")
    display_df(f_df[["Month", "Product Phase", "Total Gross Revenue", "Total Commissions",
                      "Total Expenses", "Corporate Tax", "Net Cash Flow", "Cash Balance"]])

with rep_tab3:
    st.subheader("Balance Sheet (Simplified)")
    display_df(f_df[["Month", "Product Phase", "Cash Balance", "Deferred Revenue"]])

with rep_tab4:
    st.subheader("Key Metrics")
    display_df(f_df[["Month", "Product Phase", "Total Active Users", "ARPU", "Blended Churn",
                      "LTV", "Paid CAC", "Organic CAC", "Blended CAC", "LTV/CAC", "MER",
                      "Payback Period (Months)"]])

with rep_tab5:
    st.subheader("Summary by Phase")
    phase_summary = f_df.groupby("Product Phase").agg(
        Months=("Month", "count"),
        Total_Revenue=("Total Gross Revenue", "sum"),
        Total_Marketing=("Marketing", "sum"),
        Total_Salaries=("Salaries", "sum"),
        Total_Misc=("Misc Costs", "sum"),
        Total_COGS=("COGS", "sum"),
        Total_Commissions=("Total Commissions", "sum"),
        Total_Expenses=("Total Expenses", "sum"),
        Net_Profit=("Net Profit", "sum"),
        End_Users=("Total Active Users", "last"),
        New_Users=("New Paid Users", "sum"),
    ).reset_index()
    phase_summary.columns = ["Phase", "Months", "Revenue", "Marketing", "Salaries",
                              "Misc Costs", "COGS", "Commissions", "Total Spend",
                              "Net Profit", "End Users", "New Users"]
    st.dataframe(phase_summary, use_container_width=True)

# ===================== EXPORT =====================
csv = df_main.to_csv(index=False).encode("utf-8")
st.download_button("Download Full Report (CSV)", csv, "financial_model_60_months.csv", "text/csv")
