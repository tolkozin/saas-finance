import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Awesome dashboard", layout="wide")

st.title("Awesome dashboard")

# --- SIDEBAR ---
with st.sidebar.expander("Scenarios, Taxes & Settings", expanded=True):
    scenario_sensitivity = st.slider("Scenario Sensitivity (+/- %)", -50, 50, 0, help="Настройка чувствительности сценария (влияет на конверсии и удержание). Формула: sens_factor = 1 + (sensitivity / 100).")
    corporate_tax = st.number_input("Corporate Tax %", value=1.0, help="Налог на прибыль. По умолчанию 1% (Грузия), возможны варианты Армения 5%, Казахстан 3-10%. Формула: [EBITDA] * [Corporate Tax %] / 100 (только если EBITDA > 0).")

with st.sidebar.expander("Payment Gateways & Transaction Fees", expanded=True):
    store_split = st.slider("Store vs. Web Split % (Store)", 0, 100, 50, help="Доля пользователей, покупающих через Store (остальные через Web). Формула: Store Users = [New Paid Users] * [Store Split] / 100; Web Users = [New Paid Users] * (100 - [Store Split]) / 100.")
    web_split = 100 - store_split
    app_store_comm = st.number_input("App Store Commission %", value=15.0, help="15% по Apple Small Business Program до $1M, далее 30%. Формула: [Gross Revenue Store] * [App Store Commission %] / 100.")
    web_comm_pct = st.number_input("Web Payment Commission %", value=3.5, help="Стандартные условия Lemon Squeezy / Paddle. Налоги Sales Tax они берут на себя. Формула: [Gross Revenue Web] * [Web Commission %] / 100.")
    web_comm_fixed = st.number_input("Web Payment Fixed Fee $", value=0.50, help="Фиксированная комиссия за транзакцию в Web. Формула: [Количество Web транзакций] * [Fixed Fee].")
    bank_fee = st.number_input("Banking Transaction Fee %", value=1.0, help="Закладываем 1% на последующие банковские операции и конвертации. Формула: [Total Gross Revenue] * [Bank Fee %] / 100.")

with st.sidebar.expander("Acquisition & Scaling", expanded=True):
    paid_budget = st.number_input("Initial Monthly Ad Budget ($)", value=150000.0, help="Начальный ежемесячный бюджет на рекламу. Формула: Ad Budget[m] = [Initial Budget] * (1 + [MoM Growth %] / 100) ^ (m - 1), затем масштабируется по фазе.")
    ad_growth = st.number_input("Ad Budget MoM Growth %", value=5.0, help="Ежемесячный рост рекламного бюджета в процентах. Формула: Ad Budget[m] = [Initial Budget] * (1 + [Ad Growth %] / 100) ^ (m - 1).")
    base_cpi = st.number_input("CPI (Cost Per Install) $", value=7.5, help="Базовая стоимость установки. Формула: CPI[m] = [Base CPI] * (1 + [CPI Degradation %] / 100 * ([Ad Budget] - [Initial Budget]) / 1000).")
    cpi_deg = st.number_input("CPI Degradation %", value=1.0, help="На сколько процентов увеличивается CPI за каждую добавленную $1,000 к бюджету. Формула: CPI[m] = [Base CPI] * (1 + [Degradation %] / 100 * max(0, ([Ad Budget] - [Initial Budget]) / 1000)).")
    conv_trial = st.number_input("Conv. to Trial %", value=25.0, help="Конверсия из установки в триал. Формула: New Trials = [Installs] * [Conv. to Trial %] / 100 * [Sensitivity Factor].")
    conv_paid = st.number_input("Trial-to-Paid %", value=25.0, help="Конверсия из триала в платную подписку. Формула: New Paid Users = [New Trials] * [Trial-to-Paid %] / 100 * [Sensitivity Factor].")

with st.sidebar.expander("Organic Acquisition", expanded=True):
    starting_organic_traffic = st.number_input("Starting Organic Traffic", value=0.0, help="Начальный органический трафик (посетители/мес). Формула: Organic Traffic[m] = [Starting Organic Traffic] * (1 + [Organic MoM Growth %] / 100) ^ (m - 1).")
    organic_growth = st.number_input("Organic MoM Growth %", value=10.0, help="Ежемесячный рост органического трафика в процентах. Формула: Organic Traffic[m] = [Starting] * (1 + [Growth %] / 100) ^ (m - 1).")
    organic_spend = st.number_input("Monthly Organic Spend ($)", value=0.0, help="Ежемесячные расходы на органическое привлечение (контент, SEO). Формула: добавляется к Marketing = [Ad Budget] + [Organic Spend]. Расходуется во всех фазах.")

with st.sidebar.expander("Monetization & Retention", expanded=True):
    mix_weekly = st.number_input("Mix: Weekly %", value=0.0, help="Доля еженедельных подписок. Формула: нормализуется так, чтобы Weekly + Monthly + Annual = 100%.")
    mix_monthly = st.number_input("Mix: Monthly %", value=48.0, help="Доля ежемесячных подписок. Формула: нормализуется так, чтобы Weekly + Monthly + Annual = 100%.")
    mix_annual = st.number_input("Mix: Annual %", value=52.0, help="Доля ежегодных подписок. Формула: нормализуется так, чтобы Weekly + Monthly + Annual = 100%.")

    total_mix = mix_weekly + mix_monthly + mix_annual
    if total_mix > 0:
        mix_weekly /= total_mix
        mix_monthly /= total_mix
        mix_annual /= total_mix
    else:
        mix_monthly = 1.0

    price_weekly = st.number_input("Price: Weekly $", value=4.99, help="Цена еженедельной подписки. Формула: MRR Weekly = [Active Weekly Users] * [Price Weekly] * 4.33.")
    price_monthly = st.number_input("Price: Monthly $", value=7.99, help="Цена ежемесячной подписки. Формула: MRR Monthly = [Active Monthly Users] * [Price Monthly].")
    price_annual = st.number_input("Price: Annual $", value=49.99, help="Цена ежегодной подписки (оплата сразу). Формула: MRR Annual = [Active Annual Users] * [Price Annual] / 12.")

    churn_web = st.number_input("Monthly Churn: Web %", value=10.0, help="Ежемесячный отток пользователей Web. Формула: Users[m+1] = Users[m] * (1 - [Churn Web %] / 100 / [Sensitivity Factor]).")
    churn_store = st.number_input("Monthly Churn: Store %", value=8.0, help="Ежемесячный отток пользователей Store. Формула: Users[m+1] = Users[m] * (1 - [Churn Store %] / 100 / [Sensitivity Factor]).")

with st.sidebar.expander("Cost Structure & Phases", expanded=True):
    seed_investment = st.number_input("Seed Investment ($)", value=100000.0, help="Начальные инвестиции. Формула: Cash Balance[0] = [Seed Investment] + Net Cash Flow[0].")
    cogs_per_user = st.number_input("COGS per Active User ($)", value=0.10, help="Затраты на сервера/хостинг на одного активного пользователя в месяц. Формула: COGS = [Total Active Users] * [COGS per User].")

    st.markdown("**Phase 1 (Months 1-3): Pre-MVP**")
    phase1_salaries = st.number_input("Phase 1 Monthly Salaries ($)", value=5825.0, help="Зарплаты в фазе Pre-MVP (мес. 1-3). Формула: Salaries = [Phase N Salaries] в зависимости от текущей фазы. По умолчанию: 17475/3 ~ $5825.")
    phase1_misc = st.number_input("Phase 1 Monthly Misc Costs ($)", value=2806.0, help="Прочие расходы в фазе Pre-MVP (мес. 1-3). Формула: Platform Subs = [Phase N Misc Costs]. По умолчанию: 8419/3 ~ $2806.")
    phase1_ad_pct = st.number_input("Phase 1 Ad Budget % of Full", value=0.0, help="Доля рекламного бюджета в фазе 1. Формула: Actual Ad Budget = [Full Ad Budget] * [Phase Ad %] / 100. 0% = нет рекламы (Pre-MVP).")

    st.markdown("**Phase 2 (Months 4-6): MVP**")
    phase2_salaries = st.number_input("Phase 2 Monthly Salaries ($)", value=1200.0, help="Зарплаты в фазе MVP (мес. 4-6). Формула: Salaries = [Phase N Salaries].")
    phase2_misc = st.number_input("Phase 2 Monthly Misc Costs ($)", value=250.0, help="Прочие расходы в фазе MVP (мес. 4-6). Формула: Platform Subs = [Phase N Misc Costs].")
    phase2_ad_pct = st.number_input("Phase 2 Ad Budget % of Full", value=25.0, help="Доля рекламного бюджета в фазе 2. Формула: Actual Ad Budget = [Full Ad Budget] * 25 / 100.")

    st.markdown("**Phase 3 (Months 7-60): Full Scaling**")
    phase3_salaries = st.number_input("Phase 3 Monthly Salaries ($)", value=1200.0, help="Зарплаты в фазе полного масштабирования (мес. 7-60). Формула: Salaries = [Phase N Salaries].")
    phase3_misc = st.number_input("Phase 3 Monthly Misc Costs ($)", value=250.0, help="Прочие расходы в фазе полного масштабирования (мес. 7-60). Формула: Platform Subs = [Phase N Misc Costs].")
    phase3_ad_pct = st.number_input("Phase 3 Ad Budget % of Full", value=100.0, help="Доля рекламного бюджета в фазе 3. Формула: Actual Ad Budget = [Full Ad Budget] * 100 / 100.")

# --- DATA ENGINE ---
def run_model(sens_percent):
    months = np.arange(1, 61)
    df = pd.DataFrame({"Month": months})

    sens_factor = 1 + (sens_percent / 100.0)

    # Phase assignment
    def get_phase(m):
        if m <= 3:
            return 1
        elif m <= 6:
            return 2
        else:
            return 3

    df["Product Phase"] = df["Month"].apply(get_phase)

    phase_salaries = {1: phase1_salaries, 2: phase2_salaries, 3: phase3_salaries}
    phase_misc = {1: phase1_misc, 2: phase2_misc, 3: phase3_misc}
    phase_ad_pct = {1: phase1_ad_pct / 100.0, 2: phase2_ad_pct / 100.0, 3: phase3_ad_pct / 100.0}

    # Full ad budget (before phase scaling)
    full_ad_budget = paid_budget * ((1 + ad_growth / 100.0) ** (df["Month"] - 1))
    # Scale by phase
    df["Ad Budget"] = full_ad_budget * df["Product Phase"].map(phase_ad_pct)

    df["CPI"] = base_cpi * (1 + (cpi_deg / 100.0) * ((df["Ad Budget"] - paid_budget).clip(lower=0) / 1000.0))
    df["Installs"] = np.where(df["Ad Budget"] > 0, df["Ad Budget"] / df["CPI"], 0)

    actual_conv_trial = (conv_trial / 100.0) * sens_factor
    actual_conv_paid = (conv_paid / 100.0) * sens_factor

    df["New Trials"] = df["Installs"] * actual_conv_trial
    df["Paid New Paid Users"] = df["New Trials"] * actual_conv_paid

    # Organic traffic
    if starting_organic_traffic > 0:
        df["Organic Traffic"] = starting_organic_traffic * ((1 + organic_growth / 100.0) ** (df["Month"] - 1))
    else:
        df["Organic Traffic"] = 0.0
    df["Organic New Paid Users"] = df["Organic Traffic"] * actual_conv_trial * actual_conv_paid

    df["New Paid Users"] = df["Paid New Paid Users"] + df["Organic New Paid Users"]

    df["New Web Users"] = df["New Paid Users"] * (100 - store_split) / 100.0
    df["New Store Users"] = df["New Paid Users"] * store_split / 100.0

    actual_churn_web = min(1.0, max(0.0, (churn_web / 100.0) / sens_factor))
    actual_churn_store = min(1.0, max(0.0, (churn_store / 100.0) / sens_factor))

    cohorts_web_w = np.zeros((60, 60))
    cohorts_web_m = np.zeros((60, 60))
    cohorts_web_a = np.zeros((60, 60))

    cohorts_store_w = np.zeros((60, 60))
    cohorts_store_m = np.zeros((60, 60))
    cohorts_store_a = np.zeros((60, 60))

    for i in range(60):
        new_web = df.loc[i, "New Web Users"]
        new_store = df.loc[i, "New Store Users"]

        cohorts_web_w[i, i] = new_web * mix_weekly
        cohorts_web_m[i, i] = new_web * mix_monthly
        cohorts_web_a[i, i] = new_web * mix_annual

        cohorts_store_w[i, i] = new_store * mix_weekly
        cohorts_store_m[i, i] = new_store * mix_monthly
        cohorts_store_a[i, i] = new_store * mix_annual

        for j in range(i + 1, 60):
            cohorts_web_w[i, j] = cohorts_web_w[i, j-1] * (1 - actual_churn_web)
            cohorts_web_m[i, j] = cohorts_web_m[i, j-1] * (1 - actual_churn_web)
            cohorts_web_a[i, j] = cohorts_web_a[i, j-1] * (1 - actual_churn_web)

            cohorts_store_w[i, j] = cohorts_store_w[i, j-1] * (1 - actual_churn_store)
            cohorts_store_m[i, j] = cohorts_store_m[i, j-1] * (1 - actual_churn_store)
            cohorts_store_a[i, j] = cohorts_store_a[i, j-1] * (1 - actual_churn_store)

    gross_revenue_web = np.zeros(60)
    gross_revenue_store = np.zeros(60)
    mrr_web = np.zeros(60)
    mrr_store = np.zeros(60)
    mrr_weekly = np.zeros(60)
    mrr_monthly = np.zeros(60)
    mrr_annual = np.zeros(60)
    transactions_web = np.zeros(60)

    for j in range(60):
        mrr_w_web = np.sum(cohorts_web_w[:, j]) * price_weekly * 4.33
        mrr_m_web = np.sum(cohorts_web_m[:, j]) * price_monthly
        mrr_a_web = np.sum(cohorts_web_a[:, j]) * price_annual / 12.0

        mrr_w_store = np.sum(cohorts_store_w[:, j]) * price_weekly * 4.33
        mrr_m_store = np.sum(cohorts_store_m[:, j]) * price_monthly
        mrr_a_store = np.sum(cohorts_store_a[:, j]) * price_annual / 12.0

        mrr_web[j] = mrr_w_web + mrr_m_web + mrr_a_web
        mrr_store[j] = mrr_w_store + mrr_m_store + mrr_a_store

        mrr_weekly[j] = mrr_w_web + mrr_w_store
        mrr_monthly[j] = mrr_m_web + mrr_m_store
        mrr_annual[j] = mrr_a_web + mrr_a_store

        cash_w_web = np.sum(cohorts_web_w[:, j]) * price_weekly * 4.33
        cash_m_web = np.sum(cohorts_web_m[:, j]) * price_monthly
        cash_w_store = np.sum(cohorts_store_w[:, j]) * price_weekly * 4.33
        cash_m_store = np.sum(cohorts_store_m[:, j]) * price_monthly

        cash_a_web = 0
        cash_a_store = 0
        tx_a = 0
        for i in range(j + 1):
            if (j - i) % 12 == 0:
                cash_a_web += cohorts_web_a[i, j] * price_annual
                cash_a_store += cohorts_store_a[i, j] * price_annual
                tx_a += cohorts_web_a[i, j]

        gross_revenue_web[j] = cash_w_web + cash_m_web + cash_a_web
        gross_revenue_store[j] = cash_w_store + cash_m_store + cash_a_store

        tx_w = np.sum(cohorts_web_w[:, j]) * 4.33
        tx_m = np.sum(cohorts_web_m[:, j])
        transactions_web[j] = tx_w + tx_m + tx_a

    df["Gross Revenue Web"] = gross_revenue_web
    df["Gross Revenue Store"] = gross_revenue_store
    df["Total Gross Revenue"] = gross_revenue_web + gross_revenue_store

    df["MRR Web"] = mrr_web
    df["MRR Store"] = mrr_store
    df["Total MRR"] = mrr_web + mrr_store
    df["MRR Weekly"] = mrr_weekly
    df["MRR Monthly"] = mrr_monthly
    df["MRR Annual"] = mrr_annual

    df["Recognized Revenue"] = df["Total MRR"]

    df["Store Commission"] = df["Gross Revenue Store"] * (app_store_comm / 100.0)
    df["Web Commission"] = df["Gross Revenue Web"] * (web_comm_pct / 100.0) + transactions_web * web_comm_fixed
    df["Bank Fee"] = df["Total Gross Revenue"] * (bank_fee / 100.0)

    df["Total Commissions"] = df["Store Commission"] + df["Web Commission"] + df["Bank Fee"]
    df["Net Revenue"] = df["Total Gross Revenue"] - df["Total Commissions"]

    df["Active Web Users"] = [np.sum(cohorts_web_w[:, j] + cohorts_web_m[:, j] + cohorts_web_a[:, j]) for j in range(60)]
    df["Active Store Users"] = [np.sum(cohorts_store_w[:, j] + cohorts_store_m[:, j] + cohorts_store_a[:, j]) for j in range(60)]
    df["Total Active Users"] = df["Active Web Users"] + df["Active Store Users"]

    df["COGS"] = df["Total Active Users"] * cogs_per_user
    df["Organic Spend"] = organic_spend
    df["Marketing"] = df["Ad Budget"] + df["Organic Spend"]
    df["Salaries"] = df["Product Phase"].map(phase_salaries)
    df["Platform Subs"] = df["Product Phase"].map(phase_misc)
    df["Total Expenses"] = df["COGS"] + df["Marketing"] + df["Salaries"] + df["Platform Subs"]

    df["Gross Profit"] = df["Recognized Revenue"] - df["COGS"] - df["Total Commissions"]
    df["EBITDA"] = df["Gross Profit"] - df["Marketing"] - df["Salaries"] - df["Platform Subs"]

    df["Corporate Tax"] = df["EBITDA"].apply(lambda x: x * (corporate_tax / 100.0) if x > 0 else 0)
    df["Net Profit"] = df["EBITDA"] - df["Corporate Tax"]

    df["Net Cash Flow"] = df["Total Gross Revenue"] - df["Total Commissions"] - df["Total Expenses"] - df["Corporate Tax"]

    initial_cash = seed_investment
    cash_balance = np.zeros(60)
    cash_balance[0] = initial_cash + df.loc[0, "Net Cash Flow"]
    for j in range(1, 60):
        cash_balance[j] = cash_balance[j-1] + df.loc[j, "Net Cash Flow"]

    df["Cash Balance"] = cash_balance
    df["Deferred Revenue"] = (df["Total Gross Revenue"] - df["Recognized Revenue"]).cumsum()

    # CAC metrics
    df["Paid CAC"] = df["Ad Budget"] / df["Paid New Paid Users"].replace(0, np.nan)
    df["Organic CAC"] = df["Organic Spend"] / df["Organic New Paid Users"].replace(0, np.nan)
    df["Blended CAC"] = df["Marketing"] / df["New Paid Users"].replace(0, np.nan)

    df["ARPU"] = df["Total MRR"] / df["Total Active Users"].replace(0, np.nan)
    df["Blended Churn"] = (actual_churn_web * df["Active Web Users"] + actual_churn_store * df["Active Store Users"]) / df["Total Active Users"].replace(0, np.nan)
    df["Gross Margin %"] = df["Gross Profit"] / df["Recognized Revenue"].replace(0, np.nan)

    df["LTV"] = (df["ARPU"] * df["Gross Margin %"]) / df["Blended Churn"].replace(0, np.nan)
    df["LTV/CAC"] = df["LTV"] / df["Blended CAC"].replace(0, np.nan)
    df["MER"] = df["Total Gross Revenue"] / df["Marketing"].replace(0, np.nan)

    df["Payback Period (Months)"] = df["Blended CAC"] / (df["ARPU"] * df["Gross Margin %"]).replace(0, np.nan)

    df["Web Churn Rate"] = actual_churn_web * 100
    df["Store Churn Rate"] = actual_churn_store * 100

    return df

df_main = run_model(scenario_sensitivity)

# --- GLOBAL FILTER ---
st.header("Global Dashboard Filters")
month_range = st.slider("Select Month Range", 1, 60, (1, 60), help="Глобальный фильтр по месяцам. Влияет на все графики и метрики. Формула: фильтрует DataFrame по [Start Month] <= Month <= [End Month].")
start_m, end_m = month_range

f_df = df_main[(df_main["Month"] >= start_m) & (df_main["Month"] <= end_m)]

# --- TOP METRICS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${f_df['Total Gross Revenue'].sum():,.0f}", help="Суммарная валовая выручка за выбранный период. Формула: SUM([Gross Revenue Web] + [Gross Revenue Store]).")
col2.metric("Net Profit", f"${f_df['Net Profit'].sum():,.0f}", help="Суммарная чистая прибыль за выбранный период. Формула: SUM([EBITDA] - [Corporate Tax]).")
col3.metric("End MRR", f"${f_df['Total MRR'].iloc[-1]:,.0f}", help="Регулярная месячная выручка на конец выбранного периода. Формула: [MRR Web] + [MRR Store] последнего месяца.")
col4.metric("Avg LTV/CAC", f"{f_df['LTV/CAC'].mean():.2f}x", help="Среднее отношение пожизненной ценности клиента к стоимости привлечения за период. Формула: AVG([LTV] / [Blended CAC]).")

# --- CHARTS ---
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Growth & Revenue", "Unit Economics & Efficiency", "P&L & Scenarios"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("MRR & Total Revenue over Time")
        fig1 = px.area(f_df, x="Month", y=["MRR Weekly", "MRR Monthly", "MRR Annual"],
                       title="MRR by Cohorts", labels={"value": "MRR ($)", "variable": "Plan"})
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("*Разделение выручки по когортам тарифов. Формула: MRR Weekly = [Users Weekly] * [Price Weekly] * 4.33; MRR Monthly = [Users Monthly] * [Price Monthly]; MRR Annual = [Users Annual] * [Price Annual] / 12.*")

    with c2:
        st.subheader("Cash Flow & Valley of Death")
        fig2 = px.bar(f_df, x="Month", y="Cash Balance", title="Monthly Cash Balance")
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("*Помесячный баланс наличных средств. Формула: Cash Balance[m] = Cash Balance[m-1] + Net Cash Flow[m]; Cash Balance[0] = [Seed Investment] + Net Cash Flow[0]. Ключевой график для оценки Burn Rate.*")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Gross vs Net Revenue: Web vs Store")
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Gross Revenue Web"], name="Gross Web"))
        fig4.add_trace(go.Bar(x=f_df["Month"], y=f_df["Net Revenue"], name="Net Revenue (Total)"))
        fig4.update_layout(barmode='group', title="Revenue Comparison")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("*Сколько денег съедают комиссии Apple (15-30%) по сравнению с Lemon Squeezy (3.5% + $0.50). Формула: Net Revenue = [Total Gross Revenue] - [Store Commission] - [Web Commission] - [Bank Fee].*")

    with c4:
        st.subheader("Churn Rate Comparison: Web vs Store")
        fig6 = px.line(f_df, x="Month", y=["Web Churn Rate", "Store Churn Rate"], title="Churn Rates (%)")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("*Визуальное сравнение оттока пользователей с разных платформ. Формула: Churn Rate = [Base Churn %] / [Sensitivity Factor].*")

with tab2:
    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Payback Funnel: LTV vs Paid CAC vs Blended CAC")
        fig3 = px.line(f_df, x="Month", y=["LTV", "Paid CAC", "Organic CAC", "Blended CAC"], title="LTV vs CAC")
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("*Показывает соотношение стоимости привлечения и LTV. Формула: LTV = [ARPU] * [Gross Margin %] / [Blended Churn]; Paid CAC = [Ad Budget] / [Paid New Paid Users]; Organic CAC = [Organic Spend] / [Organic New Paid Users]; Blended CAC = [Marketing] / [New Paid Users].*")

    with c6:
        st.subheader("ROAS & Payback Period")
        fig5 = px.line(f_df, x="Month", y="Payback Period (Months)", title="Payback Period (Months)")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("*На какой месяц после привлечения когорта окупает затраты на рекламу. Формула: Payback Period = [Blended CAC] / ([ARPU] * [Gross Margin %]).*")

    c7, c8 = st.columns(2)
    with c7:
        st.subheader("Marketing Efficiency Ratio (MER)")
        fig8 = px.line(f_df, x="Month", y="MER", title="MER (Total Revenue / Total Spend)")
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("*Отношение общей выручки к затратам на маркетинг. Формула: MER = [Total Gross Revenue] / [Marketing].*")

with tab3:
    c9, c10 = st.columns(2)
    with c9:
        st.subheader("P&L Waterfall Chart")
        fig7 = go.Figure(go.Waterfall(
            name="P&L", orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", "total"],
            x=["Gross Revenue", "COGS", "Marketing", "Salaries & Subs", "Commissions & Tax", "Net Profit"],
            y=[f_df["Recognized Revenue"].sum(), -f_df["COGS"].sum(), -f_df["Marketing"].sum(),
               -(f_df["Salaries"].sum() + f_df["Platform Subs"].sum()),
               -(f_df["Total Commissions"].sum() + f_df["Corporate Tax"].sum()),
               f_df["Net Profit"].sum()],
            connector={"line":{"color":"rgb(63, 63, 63)"}}
        ))
        fig7.update_layout(title="P&L Waterfall for Selected Period")
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("*Как Валовая выручка превращается в Чистую прибыль. Формула: Net Profit = [Recognized Revenue] - [COGS] - [Marketing] - [Salaries] - [Platform Subs] - [Commissions] - [Tax].*")

    with c10:
        st.subheader("Scenario Net Profit Comparison")
        df_base = run_model(0)
        df_worst = run_model(-20)
        df_best = run_model(20)

        scenarios = ["Worst Case (-20%)", "Base Case (0%)", "Best Case (+20%)"]
        profits = [
            df_worst[(df_worst["Month"] >= start_m) & (df_worst["Month"] <= end_m)]["Net Profit"].sum(),
            df_base[(df_base["Month"] >= start_m) & (df_base["Month"] <= end_m)]["Net Profit"].sum(),
            df_best[(df_best["Month"] >= start_m) & (df_best["Month"] <= end_m)]["Net Profit"].sum()
        ]

        fig9 = px.bar(x=scenarios, y=profits, title="Cumulative Net Profit by Scenario", labels={"x": "Scenario", "y": "Net Profit ($)"})
        st.plotly_chart(fig9, use_container_width=True)
        st.markdown("*Сравнение итоговой прибыли для сценариев. Формула: Sensitivity Factor = 1 + [Scenario %] / 100 (влияет на конверсии и удержание).*")

# --- FINANCIAL REPORTS ---
st.markdown("---")
st.header("Financial Reports")

rep_tab1, rep_tab2, rep_tab3, rep_tab4, rep_tab5 = st.tabs(["P&L", "Cash Flow", "Balance Sheet", "Key Metrics", "Summary by Phase"])

def display_df(df_to_show):
    st.dataframe(
        df_to_show,
        column_config={
            "Month": st.column_config.NumberColumn("Month", help="Месяц моделирования (1-60). Формула: порядковый номер месяца."),
            "Product Phase": st.column_config.NumberColumn("Product Phase", help="Фаза продукта. Формула: 1 (мес. 1-3), 2 (мес. 4-6), 3 (мес. 7-60)."),
            "Recognized Revenue": st.column_config.NumberColumn("Recognized Revenue", help="Признанная выручка (MRR). Формула: [MRR Web] + [MRR Store]."),
            "Net Profit": st.column_config.NumberColumn("Net Profit", help="Чистая прибыль. Формула: [EBITDA] - [Corporate Tax]."),
            "Cash Balance": st.column_config.NumberColumn("Cash Balance", help="Остаток денежных средств на конец месяца. Формула: Cash Balance[m-1] + [Net Cash Flow]."),
            "Deferred Revenue": st.column_config.NumberColumn("Deferred Revenue", help="Отложенная выручка. Формула: CUMSUM([Total Gross Revenue] - [Recognized Revenue])."),
            "LTV": st.column_config.NumberColumn("LTV", help="Пожизненная ценность клиента. Формула: [ARPU] * [Gross Margin %] / [Blended Churn]."),
            "Blended CAC": st.column_config.NumberColumn("Blended CAC", help="Смешанная стоимость привлечения. Формула: [Marketing] / [New Paid Users]."),
            "Paid CAC": st.column_config.NumberColumn("Paid CAC", help="Стоимость платного привлечения. Формула: [Ad Budget] / [Paid New Paid Users]."),
            "Organic CAC": st.column_config.NumberColumn("Organic CAC", help="Стоимость органического привлечения. Формула: [Organic Spend] / [Organic New Paid Users].")
        },
        use_container_width=True
    )

with rep_tab1:
    st.subheader("Profit & Loss Statement")
    display_df(f_df[["Month", "Product Phase", "Recognized Revenue", "COGS", "Gross Profit", "Marketing", "Salaries", "Platform Subs", "Total Commissions", "EBITDA", "Corporate Tax", "Net Profit"]])

with rep_tab2:
    st.subheader("Cash Flow Statement")
    display_df(f_df[["Month", "Product Phase", "Total Gross Revenue", "Total Commissions", "Total Expenses", "Corporate Tax", "Net Cash Flow", "Cash Balance"]])

with rep_tab3:
    st.subheader("Balance Sheet (Simplified)")
    display_df(f_df[["Month", "Product Phase", "Cash Balance", "Deferred Revenue"]])

with rep_tab4:
    st.subheader("Key Metrics")
    display_df(f_df[["Month", "Product Phase", "Total Active Users", "ARPU", "Blended Churn", "LTV", "Paid CAC", "Organic CAC", "Blended CAC", "LTV/CAC", "MER", "Payback Period (Months)"]])

with rep_tab5:
    st.subheader("Summary by Phase")
    phase_summary = f_df.groupby("Product Phase").agg(
        Months=("Month", "count"),
        Total_Revenue=("Total Gross Revenue", "sum"),
        Total_Marketing=("Marketing", "sum"),
        Total_Salaries=("Salaries", "sum"),
        Total_Misc=("Platform Subs", "sum"),
        Total_COGS=("COGS", "sum"),
        Total_Commissions=("Total Commissions", "sum"),
        Total_Expenses=("Total Expenses", "sum"),
        Net_Profit=("Net Profit", "sum"),
        End_Active_Users=("Total Active Users", "last"),
        New_Paid_Users=("New Paid Users", "sum"),
    ).reset_index()
    phase_summary.columns = ["Phase", "Months", "Total Revenue", "Total Marketing", "Total Salaries",
                              "Total Misc Costs", "Total COGS", "Total Commissions", "Total Spend",
                              "Net Profit", "End Active Users", "New Paid Users"]
    st.dataframe(phase_summary, use_container_width=True)

# Export
csv = df_main.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full Report (CSV)",
    data=csv,
    file_name='financial_model_60_months.csv',
    mime='text/csv',
)
