
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FreshGuard | Smart Inventory",
    page_icon="🌿",
    layout="wide"
)

ROOT = Path(__file__).parent
DATA = ROOT / "data/raw/synthetic"
REPORTS = ROOT / "reports"

TEST_START = "2025-07-01"
TEST_END = "2025-12-31"

GREEN = "#176B45"
LIGHT_GREEN = "#DCEFE3"
DARK = "#183528"
GREY = "#64756A"
RED = "#C74646"
ORANGE = "#C17A25"


# =====================================================
# DESIGN
# =====================================================

st.markdown("""
<style>
.stApp {
    background: #F4F7F4;
    color: #26372D;
}

.main .block-container {
    max-width: 1450px;
    padding: 2rem 2rem 3rem;
}

h1, h2, h3 {
    color: #183D2B !important;
    font-weight: 750 !important;
}

p, li, label, .stMarkdown {
    color: #34483B;
}

[data-testid="stSidebar"] {
    background: #153D2B;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE7DE;
    border-radius: 13px;
    padding: 17px;
    box-shadow: 0 3px 12px rgba(20,50,30,.04);
}

[data-testid="stMetricLabel"] {
    color: #53685A !important;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #183D2B !important;
    font-weight: 750;
}

[data-testid="stMetricDelta"] {
    color: #176B45 !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF;
    border: 1px solid #DCE7DE;
    border-radius: 13px;
}

.stButton button, .stDownloadButton button {
    background: #176B45;
    color: white;
    border: 0;
    border-radius: 8px;
    font-weight: 600;
}

.stButton button:hover, .stDownloadButton button:hover {
    background: #105235;
    color: white;
}

hr {
    border-color: #DCE5DE;
}

.fg-hero {
    background: linear-gradient(115deg, #153D2B, #28734D);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
}

.fg-hero h2 {
    color: #FFFFFF !important;
    margin: 5px 0;
}

.fg-hero p {
    color: #E2F0E6 !important;
    margin-bottom: 0;
}

.fg-kicker {
    color: #B9E1C6;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.fg-note {
    background: #E7F1E9;
    border-left: 4px solid #28734D;
    padding: 12px 15px;
    border-radius: 7px;
    color: #254B34;
    margin: 12px 0 20px;
}

.fg-muted {
    color: #617368;
    font-size: 13px;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    sales = pd.read_csv(DATA / "sales_daily.csv")
    truth = pd.read_csv(DATA / "ground_truth_demand.csv")
    products = pd.read_csv(DATA / "products.csv")
    stores = pd.read_csv(DATA / "stores.csv")

    predictions = pd.read_csv(REPORTS / "final_demand_predictions.csv")
    model = pd.read_csv(REPORTS / "final_model_results.csv")
    importance = pd.read_csv(REPORTS / "final_feature_importance.csv")
    inventory = pd.read_csv(REPORTS / "inventory_simulation_daily.csv")

    for frame in [sales, truth, predictions, inventory]:
        frame["date"] = pd.to_datetime(frame["date"])

    data = sales.merge(
        truth,
        on=["date", "store_id", "sku_id"],
        validate="one_to_one"
    ).merge(
        products,
        on="sku_id",
        how="left"
    ).merge(
        stores,
        on="store_id",
        how="left"
    )

    predictions = predictions.merge(
        products[["sku_id", "category"]],
        on="sku_id",
        how="left"
    )

    inventory = inventory.merge(
        products[["sku_id", "category"]],
        on="sku_id",
        how="left"
    )

    return data, predictions, model, importance, inventory


try:
    data, predictions, model, importance, inventory = load_data()
except Exception as e:
    st.error(f"Could not load project data: {e}")
    st.info("Check your data/raw/synthetic and reports folders.")
    st.stop()


# =====================================================
# CHART STYLE
# =====================================================

def style_chart(fig, height=370):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial",
            color="#26372D",
            size=13
        ),
        title_font=dict(
            color="#183D2B",
            size=17
        ),
        margin=dict(l=15, r=15, t=55, b=20),
        legend=dict(
            font=dict(color="#34483B"),
            bgcolor="#FFFFFF"
        )
    )

    fig.update_xaxes(
        title_font=dict(color="#34483B"),
        tickfont=dict(color="#34483B"),
        gridcolor="#E7EDE8",
        showgrid=False
    )

    fig.update_yaxes(
        title_font=dict(color="#34483B"),
        tickfont=dict(color="#34483B"),
        gridcolor="#E3EAE4",
        zeroline=False
    )

    return fig


def heading(title, description):
    st.title(title)
    st.markdown(
        f'<p class="fg-muted">{description}</p>',
        unsafe_allow_html=True
    )


def number(value):
    return f"{value:,.0f}"


def filtered(frame, date_filter=True):
    result = frame[
        frame["store_id"].isin(stores_selected)
        & frame["category"].isin(categories_selected)
    ].copy()

    if date_filter:
        result = result[
            result["date"].between(start_date, end_date)
        ]

    return result


def policy_metrics(base, fresh):
    old_waste = base["units_wasted"].sum()
    new_waste = fresh["expired_units"].sum()

    old_lost = base["lost_sales"].sum()
    new_lost = fresh["lost_sales"].sum()

    old_purchase = (
        base["units_received"] * base["unit_cost"]
    ).sum()

    new_purchase = fresh["purchase_cost"].sum()

    old_waste_cost = (
        base["units_wasted"]
        * (base["unit_cost"] - base["salvage_value"])
    ).sum()

    new_waste_cost = fresh["waste_cost"].sum()

    old_cost = old_purchase + old_waste_cost
    new_cost = new_purchase + new_waste_cost

    def reduction(old, new):
        return (old - new) / old * 100 if old else 0

    return {
        "old_waste": old_waste,
        "new_waste": new_waste,
        "old_lost": old_lost,
        "new_lost": new_lost,
        "old_cost": old_cost,
        "new_cost": new_cost,
        "waste_change": reduction(old_waste, new_waste),
        "lost_change": reduction(old_lost, new_lost),
        "cost_change": reduction(old_cost, new_cost)
    }


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("""
# 🌿 FreshGuard
**SMART INVENTORY INTELLIGENCE**

Helping retailers reduce waste and understand hidden demand.
""")

page = st.sidebar.radio(
    "MENU",
    [
        "Overview",
        "Demand Insights",
        "Inventory Planner",
        "Food Waste",
        "Model Results",
        "About FreshGuard"
    ]
)

st.sidebar.divider()
st.sidebar.markdown("### FILTER DATA")

stores_selected = st.sidebar.multiselect(
    "Choose stores",
    sorted(data["store_id"].unique()),
    default=sorted(data["store_id"].unique())
)

categories_selected = st.sidebar.multiselect(
    "Choose product categories",
    sorted(data["category"].dropna().unique()),
    default=sorted(data["category"].dropna().unique())
)

date_range = st.sidebar.date_input(
    "Choose date range",
    value=(data["date"].min().date(), data["date"].max().date()),
    min_value=data["date"].min().date(),
    max_value=data["date"].max().date()
)

if len(date_range) == 2:
    start_date, end_date = map(pd.Timestamp, date_range)
else:
    start_date = end_date = pd.Timestamp(date_range[0])

st.sidebar.divider()
st.sidebar.caption("Dataset: Synthetic retail benchmark")
st.sidebar.caption("Dataset period: Jan 2024 – Dec 2025")
st.sidebar.caption("Model test period: Jul – Dec 2025")


# Apply filters
d = filtered(data)
p = filtered(predictions)
i = filtered(inventory)

# Both strategies use the same test period.
base_test = filtered(data)
base_test = base_test[
    base_test["date"].between(TEST_START, TEST_END)
]

fresh_test = filtered(inventory)
fresh_test = fresh_test[
    fresh_test["date"].between(TEST_START, TEST_END)
]


# =====================================================
# 1. OVERVIEW
# =====================================================

if page == "Overview":

    st.markdown("""
    <div class="fg-hero">
        <div class="fg-kicker">SMART RETAIL • MACHINE LEARNING • SUSTAINABILITY</div>
        <h2>Make better inventory decisions.</h2>
        <p>Understand hidden demand, identify stock problems and reduce
        unnecessary food waste with FreshGuard.</p>
    </div>
    """, unsafe_allow_html=True)

    heading(
        "Business Overview",
        "A simple view of what is happening across your selected stores and products."
    )

    if d.empty:
        st.warning("No data available for these filters.")
        st.stop()

    demand = d["true_demand"].sum()
    sales = d["units_sold"].sum()
    lost = d["lost_sales"].sum()
    waste = d["units_wasted"].sum()

    a, b, c, e = st.columns(4)
    a.metric("Customer Demand", number(demand))
    b.metric("Sales Recorded", number(sales))
    c.metric("Missed Sales", number(lost))
    e.metric("Food Wasted", number(waste))

    st.markdown(
        '<div class="fg-note"><b>What does this mean?</b> '
        'Customer demand is higher than recorded sales when products '
        'are unavailable. The difference represents estimated missed '
        'sales in this synthetic dataset.</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Are we missing customer demand?")

        monthly = d.groupby(
            d["date"].dt.to_period("M")
        )[["units_sold", "true_demand"]].sum().reset_index()

        monthly["Month"] = monthly["date"].astype(str)

        chart = monthly.melt(
            id_vars="Month",
            value_vars=["units_sold", "true_demand"],
            var_name="Measure",
            value_name="Units"
        )

        chart["Measure"] = chart["Measure"].map({
            "units_sold": "Sales recorded",
            "true_demand": "Actual demand"
        })

        fig = px.line(
            chart,
            x="Month",
            y="Units",
            color="Measure",
            markers=True,
            color_discrete_sequence=[GREEN, "#E49A42"]
        )
        st.plotly_chart(style_chart(fig), use_container_width=True)

    with right:
        st.subheader("Which products have more demand?")

        category = d.groupby(
            "category", as_index=False
        )[["units_sold", "true_demand"]].sum()

        chart = category.melt(
            id_vars="category",
            var_name="Measure",
            value_name="Units"
        )

        chart["Measure"] = chart["Measure"].map({
            "units_sold": "Sales recorded",
            "true_demand": "Actual demand"
        })

        fig = px.bar(
            chart,
            x="category",
            y="Units",
            color="Measure",
            barmode="group",
            color_discrete_sequence=[GREEN, "#E49A42"]
        )
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(style_chart(fig), use_container_width=True)

    st.divider()
    st.subheader("What did the FreshGuard inventory test achieve?")

    if not base_test.empty and not fresh_test.empty:
        m = policy_metrics(base_test, fresh_test)

        x, y, z = st.columns(3)
        x.metric("Food waste reduction", f"{m['waste_change']:.1f}%")
        y.metric("Inventory cost reduction", f"{m['cost_change']:.1f}%")
        z.metric(
            "Missed sales reduction",
            f"{m['lost_change']:.1f}%",
            help="A negative number means missed sales increased."
        )

        st.caption(
            "Comparison uses the same July–December 2025 test period. "
            "These are simulated results, not real supermarket savings."
        )


# =====================================================
# 2. DEMAND INSIGHTS
# =====================================================

elif page == "Demand Insights":

    heading(
        "Demand Insights",
        "See how FreshGuard estimates sales that may have been lost during stockouts."
    )

    st.markdown(
        '<div class="fg-note"><b>In simple terms:</b> When a product runs '
        'out, recorded sales stop. FreshGuard estimates how much more '
        'customers may have wanted to buy.</div>',
        unsafe_allow_html=True
    )

    if p.empty:
        st.warning("No predictions available for the selected filters.")
        st.stop()

    a, b, c = st.columns(3)
    a.metric("Sales recorded", number(p["units_sold"].sum()))
    b.metric("FreshGuard estimate", number(p["estimated_demand"].sum()))
    c.metric("Actual demand (benchmark)", number(p["true_demand"].sum()))

    st.subheader("How close are the estimates?")

    fig = px.scatter(
        p,
        x="true_demand",
        y="estimated_demand",
        color="category",
        opacity=0.6,
        labels={
            "true_demand": "Actual demand",
            "estimated_demand": "FreshGuard estimate"
        }
    )

    maximum = max(
        p["true_demand"].max(),
        p["estimated_demand"].max()
    )

    fig.add_shape(
        type="line",
        x0=0, y0=0,
        x1=maximum, y1=maximum,
        line=dict(color=RED, dash="dash", width=2)
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.caption(
        "Each dot represents a product on a stockout day. "
        "Dots closer to the diagonal line indicate more accurate estimates."
    )

    st.subheader("Estimated missed sales by category")

    category = p.groupby(
        "category", as_index=False
    )[["actual_lost_sales", "predicted_lost_sales"]].sum()

    category = category.melt(
        id_vars="category",
        var_name="Measure",
        value_name="Units"
    )

    category["Measure"] = category["Measure"].map({
        "actual_lost_sales": "Actual missed sales",
        "predicted_lost_sales": "Estimated missed sales"
    })

    fig = px.bar(
        category,
        x="category",
        y="Units",
        color="Measure",
        barmode="group",
        color_discrete_sequence=[ORANGE, GREEN]
    )

    fig.update_xaxes(tickangle=-25)
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("Prediction details")

    columns = [
        "date", "store_id", "sku_id", "category",
        "units_sold", "actual_lost_sales",
        "predicted_lost_sales", "estimated_demand"
    ]

    st.dataframe(
        p[columns].sort_values("date", ascending=False).head(500),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download demand report",
        p[columns].to_csv(index=False),
        "freshguard_demand_report.csv",
        "text/csv"
    )


# =====================================================
# 3. INVENTORY PLANNER
# =====================================================

elif page == "Inventory Planner":

    heading(
        "Inventory Planner",
        "Find products that may need replenishment and review suggested order quantities."
    )

    st.markdown(
        '<div class="fg-note"><b>What should you do?</b> Review products '
        'marked as low stock first. Suggested orders are based on the '
        'simulated inventory policy.</div>',
        unsafe_allow_html=True
    )

    if i.empty:
        st.warning("No inventory records available.")
        st.stop()

    latest_date = i["date"].max()
    latest = i[i["date"] == latest_date].copy()

    latest["status"] = np.where(
        latest["closing_stock"] <= latest["forecast_demand"],
        "Low stock",
        "Stock available"
    )

    a, b, c = st.columns(3)
    a.metric("Latest inventory date", latest_date.strftime("%d %b %Y"))
    b.metric("Products / stores", number(len(latest)))
    c.metric("Suggested units to order", number(latest["order_quantity"].sum()))

    st.subheader("Products needing attention")

    attention = latest[
        latest["status"] == "Low stock"
    ].sort_values("order_quantity", ascending=False)

    if attention.empty:
        st.success("No low-stock products in the latest simulation records.")
    else:
        st.dataframe(
            attention[[
                "store_id", "sku_id", "category",
                "forecast_demand", "closing_stock",
                "order_quantity", "status"
            ]],
            use_container_width=True,
            hide_index=True
        )

    st.download_button(
        "Download order suggestions",
        latest.to_csv(index=False),
        "freshguard_order_suggestions.csv",
        "text/csv"
    )

    st.subheader("Do we have enough stock?")

    fig = px.scatter(
        latest,
        x="forecast_demand",
        y="closing_stock",
        color="status",
        hover_name="sku_id",
        color_discrete_map={
            "Low stock": RED,
            "Stock available": GREEN
        },
        labels={
            "forecast_demand": "Expected daily demand",
            "closing_stock": "Stock remaining"
        }
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.caption(
        "This is a historical simulation, not a live inventory feed. "
        "The simulation assumes same-day delivery."
    )


# =====================================================
# 4. FOOD WASTE
# =====================================================

elif page == "Food Waste":

    heading(
        "Food Waste",
        "Compare the existing inventory strategy with FreshGuard's shelf-life-aware strategy."
    )

    st.markdown(
        '<div class="fg-note"><b>Goal:</b> Reduce expired products '
        'without creating too many missed sales.</div>',
        unsafe_allow_html=True
    )

    if base_test.empty or fresh_test.empty:
        st.warning(
            "Select dates that include July–December 2025 to view "
            "the inventory simulation comparison."
        )
        st.stop()

    m = policy_metrics(base_test, fresh_test)

    a, b, c = st.columns(3)
    a.metric("Waste reduction", f"{m['waste_change']:.2f}%")
    b.metric("Cost reduction", f"{m['cost_change']:.2f}%")
    c.metric(
        "Missed sales reduction",
        f"{m['lost_change']:.2f}%",
        help="Negative means missed sales increased."
    )

    st.subheader("Waste by product category")

    old = base_test.groupby(
        "category", as_index=False
    )["units_wasted"].sum().rename(
        columns={"units_wasted": "Existing strategy"}
    )

    new = fresh_test.groupby(
        "category", as_index=False
    )["expired_units"].sum().rename(
        columns={"expired_units": "FreshGuard strategy"}
    )

    comparison = old.merge(new, on="category")
    chart = comparison.melt(
        id_vars="category",
        var_name="Strategy",
        value_name="Waste units"
    )

    fig = px.bar(
        chart,
        x="category",
        y="Waste units",
        color="Strategy",
        barmode="group",
        color_discrete_sequence=["#A8B7AD", GREEN]
    )

    fig.update_xaxes(tickangle=-20)
    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("What changed?")

    x, y = st.columns(2)

    with x:
        st.metric(
            "Existing strategy waste",
            number(m["old_waste"])
        )
        st.metric(
            "FreshGuard waste",
            number(m["new_waste"])
        )

    with y:
        st.metric(
            "Existing strategy missed sales",
            number(m["old_lost"])
        )
        st.metric(
            "FreshGuard missed sales",
            number(m["new_lost"])
        )

    st.warning(
        "FreshGuard's current simulation reduces waste and cost but "
        "slightly increases missed sales. Retailers must decide how "
        "to balance availability against expiry risk."
    )

    st.caption(
        "All figures are simulated. Cost values are dataset cost units, "
        "not necessarily Indian rupees."
    )


# =====================================================
# 5. MODEL RESULTS
# =====================================================

elif page == "Model Results":

    heading(
        "Model Results",
        "See how accurately LightGBM estimates demand that was missed during stockouts."
    )

    baseline = model[model["Model"] == "Observed Sales Baseline"]
    final = model[model["Model"] == "Final LightGBM"]

    if baseline.empty or final.empty:
        st.error("Model evaluation results are missing.")
        st.stop()

    baseline_mae = baseline["MAE"].iloc[0]
    final_mae = final["MAE"].iloc[0]
    improvement = (baseline_mae - final_mae) / baseline_mae * 100

    a, b, c = st.columns(3)
    a.metric("Baseline prediction error", f"{baseline_mae:.3f}")
    b.metric("FreshGuard prediction error", f"{final_mae:.3f}")
    c.metric("Improvement", f"{improvement:.2f}%")

    st.markdown(
        '<div class="fg-note"><b>What is MAE?</b> It measures the '
        'average size of prediction errors. A smaller number means '
        'the predictions are closer to actual demand.</div>',
        unsafe_allow_html=True
    )

    st.subheader("Prediction error comparison")

    chart = model.melt(
        id_vars="Model",
        value_vars=["MAE", "RMSE"],
        var_name="Error measure",
        value_name="Error"
    )

    fig = px.bar(
        chart,
        x="Error measure",
        y="Error",
        color="Model",
        barmode="group",
        color_discrete_sequence=[GREEN, "#E49A42"]
    )

    st.plotly_chart(style_chart(fig), use_container_width=True)

    st.subheader("What influences the model?")

    fig = px.bar(
        importance.sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#B9DCC5", GREEN]
    )

    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(
        style_chart(fig, 480),
        use_container_width=True
    )

    st.caption(
        "Feature importance shows which inputs the model used most "
        "when making predictions. It does not prove cause and effect."
    )

    st.info(
        "Evaluation uses 16,996 held-out stockout records from the "
        "synthetic July–December 2025 test period. The results are "
        "not a guarantee of real-world performance."
    )


# =====================================================
# 6. ABOUT
# =====================================================

elif page == "About FreshGuard":

    heading(
        "About FreshGuard",
        "The idea, technology and workflow behind the project."
    )

    st.markdown("""
    **FreshGuard – Perishable Food Waste and Hidden-Demand Intelligence**

    FreshGuard is an academic machine learning project that explores
    how retailers can estimate demand hidden by stockouts and improve
    inventory decisions for perishable products.

    ### The problem

    When a product becomes unavailable, recorded sales no longer show
    how much customers wanted to purchase. This can lead to
    underestimated demand and poor replenishment decisions.

    ### Our solution

    1. Analyse historical sales and stock availability.
    2. Estimate lost sales using LightGBM.
    3. Reconstruct total demand.
    4. Use demand estimates and product shelf life in an inventory simulation.
    5. Visualise the results through this dashboard.

    ### Technologies used

    - Python
    - Pandas and NumPy
    - LightGBM
    - Scikit-learn
    - Plotly
    - Streamlit

    ### Dataset

    The project uses a synthetic retail benchmark with 12 stores,
    40 products and two years of simulated daily data.

    The project also explores FreshRetailNet-50K for real retail
    stockout-pattern analysis. Its sales values are normalised and
    it does not provide verified ground-truth lost-sales labels.

    ### Important limitation

    Synthetic data allows controlled evaluation because actual demand
    is known. Real-world demand recovery and actual waste reduction
    require further validation using operational retail data.
    """)

    st.divider()

    st.subheader("Project summary")

    a, b, c = st.columns(3)
    a.metric("Stores", data["store_id"].nunique())
    b.metric("Products", data["sku_id"].nunique())
    c.metric("Daily records", f"{len(data):,}")


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; padding:10px;">
        <h3 style="color:#176B45;">🌿 FreshGuard</h3>
        <p style="color:#64756A; font-size:12px;">
            Perishable Food Waste & Hidden-Demand Intelligence<br>
            Academic Machine Learning Project · Synthetic Benchmark
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
