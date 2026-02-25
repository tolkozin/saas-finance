import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="SaaS Financial Model", layout="wide")

# Sidebar Inputs
st.sidebar.header("Model Assumptions")

with st.sidebar.expander("Global Settings", expanded=True):
    currency = st.selectbox("Currency Symbol", ["$", "€", "£"], help="Символ валюты для отображения в дашборде.")
    sensitivity = st.slider("Scenario Sensitivity (+/- %)", min_value=0, max_value=50, value=20, step=5, help="Процент отклонения для оптимистичного и консервативного сценариев. Влияет на конверсию и отток.")

with st.sidebar.expander("Acquisition Funnels"):
    st.markdown("### Organic")
    org_traffic = st.number_input("Starting Organic Traffic", value=10000, help="Начальный органический трафик в первый месяц.")
    org_growth = st.number_input("Organic MoM Growth (%)", value=5.0, help="Ежемесячный рост органического трафика в процентах.")
    org_conv_trial = st.number_input("Organic Conv. to Trial (%)", value=5.0, help="Конверсия из органического трафика в триальные регистрации.")
    
    st.markdown("### Paid")
    paid_budget = st.number_input("Monthly Ad Budget", value=5000, help="Ежемесячный бюджет на платную рекламу.")
    cpi = st.number_input("CPI (Cost Per Install/Lead)", value=2.0, help="Стоимость привлечения одного лида/установки (Cost Per Install).")
    paid_conv_trial = st.number_input("Paid Conv. to Trial (%)", value=10.0, help="Конверсия из платного трафика в триальные регистрации.")
    
    st.markdown("### Trial Logic")
    trial_days = st.number_input("Trial Duration (days)", value=14, help="Длительность пробного периода в днях.")
    trial_to_paid = st.number_input("Trial-to-Paid Conv. (%)", value=15.0, help="Конверсия из триала в платную подписку.")

with st.sidebar.expander("Monetization (Product Mix)"):
    st.markdown("### Allocation")
    mix_weekly = st.slider("% Users on Weekly", 0, 100, 20, help="Доля пользователей, выбирающих еженедельную подписку.")
    mix_monthly = st.slider("% Users on Monthly", 0, 100, 50, help="Доля пользователей, выбирающих ежемесячную подписку.")
    mix_annual = st.slider("% Users on Annual", 0, 100, 30, help="Доля пользователей, выбирающих ежегодную подписку.")
    
    if mix_weekly + mix_monthly + mix_annual != 100:
        st.sidebar.error("Allocation must sum to 100%")
        
    st.markdown("### Weekly Plan")
    price_weekly = st.number_input("Weekly Price", value=4.99, help="Стоимость еженедельной подписки.")
    churn_weekly = st.number_input("Weekly Plan Monthly Churn (%)", value=20.0, help="Ежемесячный отток (Churn) для еженедельной подписки.")
    
    st.markdown("### Monthly Plan")
    price_monthly = st.number_input("Monthly Price", value=14.99, help="Стоимость ежемесячной подписки.")
    churn_monthly = st.number_input("Monthly Plan Monthly Churn (%)", value=10.0, help="Ежемесячный отток (Churn) для ежемесячной подписки.")
    
    st.markdown("### Annual Plan")
    price_annual = st.number_input("Annual Price", value=99.99, help="Стоимость ежегодной подписки (оплачивается авансом).")
    churn_annual = st.number_input("Annual Plan Annualized Churn (%)", value=30.0, help="Годовой отток для ежегодной подписки. В модели конвертируется в ежемесячный.")
    
    st.markdown("### Fees")
    setup_fee = st.number_input("Setup Fee (One-time)", value=0.0, help="Единоразовая комиссия за подключение.")
    store_comm = st.number_input("Store Commission (%)", value=15.0, help="Комиссия платформы (Apple App Store, Google Play, Stripe).")
    vat_tax = st.number_input("VAT/Tax (%)", value=20.0, help="Налог на добавленную стоимость (НДС).")

with st.sidebar.expander("Costs & Investments"):
    cogs_per_user = st.number_input("Hosting Cost per Active User", value=0.5, help="Прямые расходы на одного активного пользователя (сервера, API).")
    salaries = st.number_input("Salaries (Fixed Monthly)", value=15000, help="Фиксированные ежемесячные расходы на зарплаты.")
    misc_fixed = st.number_input("Misc Fixed Costs", value=2000, help="Прочие фиксированные ежемесячные расходы (офис, софт).")
    initial_cash = st.number_input("Initial Cash Injection", value=100000, help="Стартовый капитал (инвестиции).")

# Core Logic Engine
def calculate_model(scenario="Baseline"):
    # Adjust inputs based on scenario
    sens_factor = sensitivity / 100.0
    
    if scenario == "Optimistic":
        adj_conv = 1 + sens_factor
        adj_churn = 1 - sens_factor
    elif scenario == "Conservative":
        adj_conv = 1 - sens_factor
        adj_churn = 1 + sens_factor
    else:
        adj_conv = 1.0
        adj_churn = 1.0

    months = range(1, 61)
    data = []
    
    cash_balance = initial_cash
    active_weekly = 0
    active_monthly = 0
    active_annual = 0
    
    deferred_revenue = 0
    
    for m in months:
        # Acquisition
        current_org_traffic = org_traffic * ((1 + org_growth/100.0) ** (m - 1))
        org_trials = current_org_traffic * (org_conv_trial/100.0) * adj_conv
        
        paid_leads = paid_budget / cpi if cpi > 0 else 0
        paid_trials = paid_leads * (paid_conv_trial/100.0) * adj_conv
        
        total_trials = org_trials + paid_trials
        new_paid = total_trials * (trial_to_paid/100.0) * adj_conv
        
        new_weekly = new_paid * (mix_weekly/100.0)
        new_monthly = new_paid * (mix_monthly/100.0)
        new_annual = new_paid * (mix_annual/100.0)
        
        # Churn
        churn_w_rate = min((churn_weekly/100.0) * adj_churn, 1.0)
        churn_m_rate = min((churn_monthly/100.0) * adj_churn, 1.0)
        # Convert annual churn to monthly
        churn_a_rate = min((1 - (1 - churn_annual/100.0)**(1/12)) * adj_churn, 1.0)
        
        churned_weekly = active_weekly * churn_w_rate
        churned_monthly = active_monthly * churn_m_rate
        churned_annual = active_annual * churn_a_rate
        
        active_weekly = max(0, active_weekly - churned_weekly + new_weekly)
        active_monthly = max(0, active_monthly - churned_monthly + new_monthly)
        active_annual = max(0, active_annual - churned_annual + new_annual)
        
        total_active = active_weekly + active_monthly + active_annual
        
        # Revenue
        # Weekly is billed roughly 4.33 times a month
        rev_weekly_cash = active_weekly * price_weekly * 4.33
        rev_monthly_cash = active_monthly * price_monthly
        rev_annual_cash = new_annual * price_annual # Cash received upfront
        setup_fees_cash = new_paid * setup_fee
        
        total_cash_in = rev_weekly_cash + rev_monthly_cash + rev_annual_cash + setup_fees_cash
        
        # P&L Revenue (Recognized)
        rev_weekly_pnl = rev_weekly_cash
        rev_monthly_pnl = rev_monthly_cash
        rev_annual_pnl = active_annual * (price_annual / 12)
        setup_fees_pnl = setup_fees_cash
        
        total_revenue_pnl = rev_weekly_pnl + rev_monthly_pnl + rev_annual_pnl + setup_fees_pnl
        
        # Deferred Revenue Tracking
        deferred_revenue = deferred_revenue + rev_annual_cash - rev_annual_pnl
        
        # Costs
        net_revenue_pnl = total_revenue_pnl * (1 - vat_tax/100.0)
        store_fees = net_revenue_pnl * (store_comm/100.0)
        cogs = total_active * cogs_per_user
        gross_profit = net_revenue_pnl - store_fees - cogs
        
        marketing_spend = paid_budget
        total_opex = salaries + marketing_spend + misc_fixed
        
        net_profit = gross_profit - total_opex
        
        # Cash Flow
        net_cash_revenue = total_cash_in * (1 - vat_tax/100.0)
        cash_store_fees = net_cash_revenue * (store_comm/100.0)
        net_cash_flow = net_cash_revenue - cash_store_fees - cogs - total_opex
        cash_balance += net_cash_flow
        
        # Metrics
        blended_cac = marketing_spend / new_paid if new_paid > 0 else 0
        paid_cac = marketing_spend / (paid_trials * (trial_to_paid/100.0) * adj_conv) if (paid_trials * (trial_to_paid/100.0) * adj_conv) > 0 else 0
        
        # Blended ARPU (Monthly)
        blended_arpu = total_revenue_pnl / total_active if total_active > 0 else 0
        gross_margin_pct = gross_profit / net_revenue_pnl if net_revenue_pnl > 0 else 0
        blended_churn = (churned_weekly + churned_monthly + churned_annual) / (total_active - new_paid) if (total_active - new_paid) > 0 else 0
        
        ltv = (blended_arpu * gross_margin_pct) / blended_churn if blended_churn > 0 else 0
        
        # LTV 30/60/90 (Simplified cumulative ARPU * margin)
        ltv_30 = blended_arpu * gross_margin_pct
        ltv_60 = ltv_30 + (blended_arpu * gross_margin_pct) * (1 - blended_churn)
        ltv_90 = ltv_60 + (blended_arpu * gross_margin_pct) * ((1 - blended_churn)**2)
        
        data.append({
            "Month": m,
            "Organic Traffic": current_org_traffic,
            "Paid Leads": paid_leads,
            "Total Trials": total_trials,
            "New Paid Customers": new_paid,
            "Active Weekly": active_weekly,
            "Active Monthly": active_monthly,
            "Active Annual": active_annual,
            "Total Active": total_active,
            "MRR": rev_weekly_pnl + rev_monthly_pnl + rev_annual_pnl,
            "Cash Inflow": total_cash_in,
            "Recognized Revenue": total_revenue_pnl,
            "Net Revenue (ex VAT)": net_revenue_pnl,
            "COGS": cogs,
            "Store Fees": store_fees,
            "Gross Profit": gross_profit,
            "OpEx": total_opex,
            "Net Profit": net_profit,
            "Net Cash Flow": net_cash_flow,
            "Cash Balance": cash_balance,
            "Deferred Revenue": deferred_revenue,
            "Blended CAC": blended_cac,
            "Paid CAC": paid_cac,
            "LTV": ltv,
            "LTV/CAC": ltv / blended_cac if blended_cac > 0 else 0,
            "LTV30": ltv_30,
            "LTV60": ltv_60,
            "LTV90": ltv_90,
            "Gross Margin %": gross_margin_pct * 100,
            "Blended Churn %": blended_churn * 100
        })
        
    return pd.DataFrame(data)

df_base = calculate_model("Baseline")
df_opt = calculate_model("Optimistic")
df_cons = calculate_model("Conservative")

# Global Timeframe Filtering
st.title("SaaS Financial Model Dashboard")

timeframe = st.slider("Select Timeframe (Months)", 1, 60, (1, 12))
start_m, end_m = timeframe

df_filtered = df_base[(df_base["Month"] >= start_m) & (df_base["Month"] <= end_m)]

# Aggregations for Scorecard
sum_revenue = df_filtered["Recognized Revenue"].sum()
sum_profit = df_filtered["Net Profit"].sum()
end_mrr = df_filtered.iloc[-1]["MRR"] if not df_filtered.empty else 0
end_cash = df_filtered.iloc[-1]["Cash Balance"] if not df_filtered.empty else 0
avg_ltv_cac = df_filtered["LTV/CAC"].mean()

# Runway Calculation
current_cash = df_filtered.iloc[-1]["Cash Balance"]
avg_burn = df_filtered[df_filtered["Net Cash Flow"] < 0]["Net Cash Flow"].mean()
runway = current_cash / abs(avg_burn) if pd.notnull(avg_burn) and avg_burn < 0 else float('inf')

# KPI Scorecard
col1, col2, col3, col4 = st.columns(4)

ltv_cac_color = "normal"
if avg_ltv_cac < 1:
    ltv_cac_color = "inverse" # Red-ish in Streamlit if we use delta, but we can just use text
    
col1.metric("LTV/CAC Ratio", f"{avg_ltv_cac:.2f}")
col2.metric("Total Revenue", f"{currency}{sum_revenue:,.0f}")
col3.metric("Net Profit", f"{currency}{sum_profit:,.0f}")
col4.metric("Runway (Months)", f"{runway:.1f}" if runway != float('inf') else "Infinite")

# Interactive Charts
st.markdown("### Interactive Charts")
c1, c2 = st.columns(2)

with c1:
    # MRR Growth
    fig_mrr = go.Figure()
    fig_mrr.add_trace(go.Scatter(x=df_base["Month"], y=df_base["MRR"], name="Baseline", line=dict(color='blue')))
    fig_mrr.add_trace(go.Scatter(x=df_opt["Month"], y=df_opt["MRR"], name="Optimistic", line=dict(color='green', dash='dash')))
    fig_mrr.add_trace(go.Scatter(x=df_cons["Month"], y=df_cons["MRR"], name="Conservative", line=dict(color='red', dash='dash')))
    fig_mrr.update_layout(title="MRR Growth Scenarios", xaxis_title="Month", yaxis_title=f"MRR ({currency})")
    fig_mrr.update_xaxes(range=[start_m, end_m])
    st.plotly_chart(fig_mrr, use_container_width=True)

with c2:
    # Unit Economics
    fig_ue = go.Figure()
    fig_ue.add_trace(go.Scatter(x=df_filtered["Month"], y=df_filtered["LTV"], name="LTV", fill='tozeroy'))
    fig_ue.add_trace(go.Scatter(x=df_filtered["Month"], y=df_filtered["Blended CAC"], name="CAC", fill='tozeroy'))
    fig_ue.update_layout(title="Unit Economics (LTV vs CAC)", xaxis_title="Month", yaxis_title=f"Amount ({currency})")
    st.plotly_chart(fig_ue, use_container_width=True)

c3, c4 = st.columns(2)

with c3:
    # Cash Flow Analysis
    fig_cf = px.bar(df_filtered, x="Month", y="Net Cash Flow", title="Monthly Net Cash Flow", color="Net Cash Flow", color_continuous_scale=px.colors.diverging.RdYlGn)
    st.plotly_chart(fig_cf, use_container_width=True)

with c4:
    # Revenue Composition
    rev_comp = pd.DataFrame({
        "Month": df_filtered["Month"],
        "Weekly": df_filtered["Active Weekly"] * price_weekly * 4.33,
        "Monthly": df_filtered["Active Monthly"] * price_monthly,
        "Annual": df_filtered["Active Annual"] * (price_annual / 12)
    })
    fig_rev = px.area(rev_comp, x="Month", y=["Weekly", "Monthly", "Annual"], title="Revenue Composition (Recognized)")
    st.plotly_chart(fig_rev, use_container_width=True)

# Detailed Financial Reports
st.markdown("### Detailed Financial Reports")
tab1, tab2, tab3, tab4 = st.tabs(["P&L", "Cash Flow", "Balance Sheet", "Cohort Analysis"])

def render_dataframe(df, column_config):
    st.dataframe(df, column_config=column_config, use_container_width=True, hide_index=True)

with tab1:
    pnl_cols = ["Month", "Recognized Revenue", "Net Revenue (ex VAT)", "COGS", "Store Fees", "Gross Profit", "Gross Margin %", "OpEx", "Net Profit"]
    pnl_config = {
        "Recognized Revenue": st.column_config.NumberColumn("Recognized Revenue", help="Признанная выручка. Включает ежемесячные платежи и 1/12 от годовых подписок."),
        "Net Revenue (ex VAT)": st.column_config.NumberColumn("Net Revenue (ex VAT)", help="Чистая выручка за вычетом НДС."),
        "COGS": st.column_config.NumberColumn("COGS", help="Прямые расходы на сервера и инфраструктуру."),
        "Store Fees": st.column_config.NumberColumn("Store Fees", help="Комиссии магазинов приложений и платежных шлюзов."),
        "Gross Profit": st.column_config.NumberColumn("Gross Profit", help="Валовая прибыль. Выручка минус COGS и комиссии."),
        "Gross Margin %": st.column_config.NumberColumn("Gross Margin %", help="Валовая маржа в процентах. Показывает эффективность продукта."),
        "OpEx": st.column_config.NumberColumn("OpEx", help="Операционные расходы (зарплаты, маркетинг, прочее)."),
        "Net Profit": st.column_config.NumberColumn("Net Profit", help="Чистая прибыль. Валовая прибыль минус операционные расходы.")
    }
    render_dataframe(df_filtered[pnl_cols], pnl_config)

with tab2:
    cf_cols = ["Month", "Cash Inflow", "OpEx", "COGS", "Store Fees", "Net Cash Flow", "Cash Balance"]
    cf_config = {
        "Cash Inflow": st.column_config.NumberColumn("Cash Inflow", help="Денежные поступления. Включает полные авансовые платежи за годовые подписки."),
        "Net Cash Flow": st.column_config.NumberColumn("Net Cash Flow", help="Чистый денежный поток за месяц."),
        "Cash Balance": st.column_config.NumberColumn("Cash Balance", help="Остаток денежных средств на конец месяца.")
    }
    render_dataframe(df_filtered[cf_cols], cf_config)

with tab3:
    bs_cols = ["Month", "Cash Balance", "Deferred Revenue"]
    bs_config = {
        "Cash Balance": st.column_config.NumberColumn("Cash Balance (Assets)", help="Активы: Остаток денежных средств на счетах."),
        "Deferred Revenue": st.column_config.NumberColumn("Deferred Revenue (Liabilities)", help="Обязательства: Отложенная выручка (полученные авансы за годовые подписки, еще не признанные в P&L).")
    }
    render_dataframe(df_filtered[bs_cols], bs_config)

with tab4:
    cohort_cols = ["Month", "New Paid Customers", "Blended CAC", "LTV", "LTV30", "LTV60", "LTV90"]
    cohort_config = {
        "New Paid Customers": st.column_config.NumberColumn("New Paid Customers", help="Количество новых платных клиентов в этом месяце."),
        "Blended CAC": st.column_config.NumberColumn("Blended CAC", help="Смешанная стоимость привлечения клиента (маркетинг / все новые клиенты)."),
        "LTV": st.column_config.NumberColumn("LTV", help="Пожизненная ценность клиента (Lifetime Value)."),
        "LTV30": st.column_config.NumberColumn("LTV30", help="Ценность клиента за первые 30 дней."),
        "LTV60": st.column_config.NumberColumn("LTV60", help="Ценность клиента за первые 60 дней."),
        "LTV90": st.column_config.NumberColumn("LTV90", help="Ценность клиента за первые 90 дней.")
    }
    render_dataframe(df_filtered[cohort_cols], cohort_config)

# Export
st.markdown("### Export Data")
csv = df_base.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download Full 60-Month Data (CSV)",
    data=csv,
    file_name='saas_financial_model_60m.csv',
    mime='text/csv',
)
