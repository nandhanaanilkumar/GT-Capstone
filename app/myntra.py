import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import StringIO
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Myntra Sales Analyzer", layout="wide")
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# --------------------------------------------------
# 🎨 CUSTOM PROFESSIONAL THEME (Myntra Colors)
# --------------------------------------------------
st.markdown("""
    <style>
        .main {
            background: #ffffff;
        }
        /* Title row */
        .title-container {
            display: flex;
            align-items: center;
            gap: 20px;
        }
        .title-text {
            font-size: 38px;
            font-weight: 700;
            color: #FF0066;
        }
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ff5f6d, #ffc371);
            color: white;
        }
        .sidebar-content {
            font-size: 18px !important;
            color: white !important;
        }
        /* Tabs styling */
        .stTabs [role="tab"] {
            font-size: 18px;
            padding: 12px;
            font-weight: bold;
            color: #FF0066;
        }
        .stTabs [aria-selected="true"] {
            border-bottom: 3px solid #FF0066;
        }
        /* Card style */
        .stCard {
            padding: 20px;
            border-radius: 12px;
            background: #ffffff;
            box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        }
        /* Buttons */
        .stButton>button {
            background-color: #FF0066;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            border: none;
            
        }
               div[data-testid="stFileUploader"] button {
        background-color: #000 !important;
        color: white !important;
        border-radius: 6px !important;
        border: 1px solid #000 !important;
        padding: 8px 16px !important;
    }

    /* Hover effect */
    div[data-testid="stFileUploader"] button:hover {
        background-color: #333 !important;
        color: white !important;
    }

    /* ---- Make upload text (Drag and drop / Limit text) black ---- */
    div[data-testid="stFileUploader"] div {
        color: black !important;
    }

    </style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 📌 SIDEBAR WITH UPLOAD + NAVIGATION
# --------------------------------------------------
st.sidebar.markdown("<h2 class='sidebar-content'>📁 Upload Dataset</h2>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel",
    type=["csv", "xlsx", "xls"],
    label_visibility="collapsed"
)

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, encoding="latin1")
        else:
            df = pd.read_excel(uploaded_file)

        df = df.sample(20000, random_state=42).reset_index(drop=True)
        st.session_state["df"] = df
        st.sidebar.success("Dataset Uploaded ✔️")

    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "📌 Navigate",
    ["📄 Dataset Overview","🧹 Data Cleaning", "📊 EDA", "🤖 Model Training", "🔮 Prediction"],
)
# --------------------------------------------------
# 📌 LINKEDIN LOGO AT BOTTOM
# --------------------------------------------------
st.sidebar.markdown("---")

linkedin_url = "https://www.linkedin.com/in/ann-maria-14ba92256"
image_path = "app/assets/linkedin-removebg-preview.png"

try:
    img_base64 = get_base64_image(image_path)

    st.sidebar.markdown(
        f"""
        <div style="display: flex; justify-content: center; padding-top: 10px;">
        <a href="{linkedin_url}" target="_blank">
            <img src="data:image/png;base64,{img_base64}" width="40" style="margin-left:10px;">
        </a>
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    st.sidebar.error("❌ Logo image not found. Check your file path.")

# --------------------------------------------------
# 🛍️ HEADER WITH LOGO + TITLE (INLINE)
# --------------------------------------------------
col1, col2 = st.columns([1, 8])

with col1:
    st.image("app/assets/myntra_logo-removebg-preview.png", width=80)

with col2:
    st.markdown(
        
        "<h1 style='color:#FF0066; padding-top: 20px;'>Myntra Sales Analyzer</h1>",
        unsafe_allow_html=True
    )
# ====================================================
# 📄 DATASET OVERVIEW
# ====================================================
if page == "📄 Dataset Overview":
    st.header("📄 Dataset Overview & Summary Statistics")

    if "df" not in st.session_state:
        st.warning("⬆️ Please upload a dataset first from the sidebar.")
    else:
        df = st.session_state["df"]

        st.subheader("🔍 First 5 Rows of Dataset")
        st.dataframe(df.head())

        st.subheader("📏 Shape of the Dataset")
        st.write(f"**Rows:** {df.shape[0]}  |  **Columns:** {df.shape[1]}")

        st.subheader("🔠 Column Types")
        st.write(df.dtypes)

        st.subheader("❗ Missing Value Summary")
        st.write(df.isnull().sum())

        st.subheader("📊 Summary Statistics (Numerical Columns)")
        st.write(df.describe())


        st.subheader("ℹ Dataset Info")
        buffer = StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.text(info_str)


# ====================================================
# 2. DATA CLEANING
# ====================================================
elif page == "🧹 Data Cleaning":
    if "df" not in st.session_state:
        st.warning("⬆️ Please upload a dataset first.")
    else:
        df = st.session_state["df"]
        st.header("🧹 Data Cleaning")

        st.subheader("Missing Values Before Cleaning")
        st.write(df.isnull().sum())

        # Remove duplicates
        df.drop_duplicates(inplace=True)

        # Numeric fill
        df.fillna(df.median(numeric_only=True), inplace=True)

        # Categorical fill
        df.fillna(df.mode().iloc[0], inplace=True)

        st.success("Data Cleaned Successfully!")

        st.subheader("Missing Values After Cleaning")
        st.write(df.isnull().sum())

        st.session_state["df"] = df
        st.write(df.head())

# ====================================================
# 3. EXPLORATORY DATA ANALYSIS
# ====================================================
elif page == "📊 EDA":
    if "df" not in st.session_state:
        st.warning("⬆️ Upload a dataset first.")
    else:
        df = st.session_state["df"]
        st.header("📊 Exploratory Data Analysis")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        # ====================================================
        # 1️⃣ CORRELATION HEATMAP
        # ====================================================
        st.subheader("Correlation Heatmap")

        if len(numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="viridis", ax=ax)
            st.pyplot(fig)

            # 🔍 AUTOMATIC INFERENCE
            st.markdown("### 📘 Inference:")
            corr_df = df[numeric_cols].corr()
            strong_corr = [
                f"**{c1}** ↔ **{c2}** (corr = {corr_df.loc[c1, c2]:.2f})"
                for c1 in corr_df.columns
                for c2 in corr_df.columns
                if c1 != c2 and abs(corr_df.loc[c1, c2]) > 0.7
            ]

            if strong_corr:
                for line in strong_corr:
                    st.write("• " + line)
            else:
                st.write("No strong correlations found.")
        else:
            st.info("Not enough numeric columns for heatmap.")

        # ====================================================
        # 2️⃣ HISTOGRAM (Distribution Plot)
        # ====================================================
        st.subheader("Distribution Plot")

        if len(numeric_cols) > 0:
            col = st.selectbox("Select numeric column for Histogram", numeric_cols)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

            # 🔍 AUTOMATIC INFERENCE
            mean_val = df[col].mean()
            median_val = df[col].median()
            std_val = df[col].std()
            skew = df[col].skew()

            if skew > 1:
                skew_text = "Highly right-skewed distribution"
            elif skew > 0.5:
                skew_text = "Moderately right-skewed distribution"
            elif skew < -1:
                skew_text = "Highly left-skewed distribution"
            elif skew < -0.5:
                skew_text = "Moderately left-skewed distribution"
            else:
                skew_text = "Approximately symmetric distribution"

            st.markdown(f"""
            ### 📘 Inference:
            - **Mean:** {mean_val:.2f}  
            - **Median:** {median_val:.2f}  
            - **Std Dev:** {std_val:.2f}  
            - **Skew:** {skew:.2f} → {skew_text}
            """)
        else:
            st.info("No numeric columns found")

        # ====================================================
        # 3️⃣ BAR CHART (Top Categories)
        # ====================================================
        st.subheader("Top Categories")

        # Remove URL columns from categorical list
        clean_cat_cols = [c for c in cat_cols if "url" not in c.lower()]

        if len(clean_cat_cols) > 0:
          cat = st.selectbox("Select categorical column", clean_cat_cols)
          fig, ax = plt.subplots(figsize=(5, 4))
          df[cat].value_counts().head(10).plot(kind="bar", ax=ax,color="purple" )
          st.pyplot(fig)

     # 🔍 AUTOMATIC INFERENCE
          top_cat = df[cat].value_counts().idxmax()
          top_count = df[cat].value_counts().max()

          st.markdown(f"""
          ### 📘 Inference:
          - The most frequent category in **{cat}** is **{top_cat}**
          - It appears **{top_count} times**
          """)
        else:
           st.info("No categorical columns found (URL removed).")


        # ====================================================
        # 4️⃣ DYNAMIC SCATTER PLOT
        # ====================================================
        st.subheader("Scatter Plot (Dynamic)")

        if len(numeric_cols) >= 2:
            col_x = st.selectbox("Select X-axis numeric column", numeric_cols)
            col_y = st.selectbox("Select Y-axis numeric column", numeric_cols)

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(df[col_x], df[col_y],color = "#073c62")
            ax.set_xlabel(col_x)
            ax.set_ylabel(col_y)
            ax.set_title(f"Scatter Plot: {col_x} vs {col_y}")
            st.pyplot(fig)

            # 🔍 AUTOMATIC INFERENCE USING CORRELATION
            corr_val = df[col_x].corr(df[col_y])

            if corr_val > 0.7:
                rel = "Strong Positive Relationship"
            elif corr_val > 0.3:
                rel = "Moderate Positive Relationship"
            elif corr_val > 0:
                rel = "Weak Positive Relationship"
            elif corr_val < -0.7:
                rel = "Strong Negative Relationship"
            elif corr_val < -0.3:
                rel = "Moderate Negative Relationship"
            elif corr_val < 0:
                rel = "Weak Negative Relationship"
            else:
                rel = "No Relationship"

            st.markdown(f"""
            ### 📘 Inference:
            - Correlation between **{col_x}** and **{col_y}**: **{corr_val:.2f}**
            - Relationship type: **{rel}**
            """)
        else:
            st.info("At least two numeric columns are required.")

        # ====================================================
        # 5️⃣ BOX PLOT (NEW)
        # ====================================================
        st.subheader("Box Plot (Outlier Detection)")

        if len(numeric_cols) > 0:
            col_box = st.selectbox("Select numeric column for Boxplot", numeric_cols)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.boxplot(x=df[col_box], ax=ax)
            st.pyplot(fig)

            # AUTOMATIC BOX PLOT INFERENCE
            q1 = df[col_box].quantile(0.25)
            q3 = df[col_box].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outliers = df[(df[col_box] < lower) | (df[col_box] > upper)].shape[0]

            st.markdown(f"""
            ### 📘 Inference:
            - **Q1:** {q1:.2f}, **Q3:** {q3:.2f}, **IQR:** {iqr:.2f}  
            - Outlier lower bound: {lower:.2f}  
            - Outlier upper bound: {upper:.2f}  
            - **Total outliers detected:** {outliers}
            """)
        else:
            st.info("No numeric columns available for box plot.")

        # ====================================================
        # 6️⃣ PRICE vs CATEGORY BOXPLOT (Your Custom Graph)
        # ====================================================
        st.subheader("Price vs Category (Boxplot)")
        try:
            # Clean the price column (same logic you provided)
            def clean_price(s):
                return pd.to_numeric(
                    s.astype(str).str.replace(r"[^\d\.]", "", regex=True),
                    errors="coerce"
                    )
            df["price_clean"] = clean_price(df["DiscountPrice (in Rs)"])
            # Top 10 categories
            categories = df["Category"].value_counts().head(10).index
            # Prepare price lists for each category
            box_data = [df[df["Category"] == c]["price_clean"].dropna() for c in categories]

            # Plot inside Streamlit
            fig, ax = plt.subplots(figsize=(14, 7))
            ax.boxplot(box_data, labels=categories, patch_artist=True)
            ax.set_xlabel("Category")
            ax.set_ylabel("Price (INR)")
            ax.set_title("Price vs Category (Boxplot)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            st.info("Shows how product prices vary across top categories.")
        except Exception as e:
            st.error("Could not generate Price vs Category plot.")
            st.error(str(e))
            st.markdown("""
### 📘 Inference

- **Different categories show clearly distinct price ranges.**  
  Premium categories like **jackets, dresses, and ethnic wear** tend to have much higher median prices compared to casual items.

- **Affordable categories like T-shirts, tops, and bags remain at the lower price range**, indicating consistent pricing strategies for fast-moving goods.

- **Large spreads (big whiskers)** in some categories reveal a mix of **budget, mid-range, and premium products**, showing higher variability in product pricing.

- **Median price differences across categories highlight the role of product type in pricing**, as fashion categories differ in materials, brand value, and manufacturing cost.

- Such patterns help identify which product groups are **price-sensitive** and which are **premium-focused** for business decisions.
""")

# ====================================================
# 4. MODEL TRAINING (FINAL FIXED VERSION)
# ====================================================
elif page == "🤖 Model Training":

    if "df" not in st.session_state:
        st.warning("⬆️ Please upload your dataset first.")
        st.stop()

    st.header("🤖 Train ML Model – Predict Product Price")

    df = st.session_state["df"].copy()

    # --------------------------
    # ✅ USE ONLY VALID FEATURES
    # --------------------------
    target = "DiscountPrice (in Rs)"   # y variable

    feature_cols = [
        "BrandName",
        "Category",
        "category_by_Gender",
        "OriginalPrice (in Rs)",
        "DiscountOffer",
        "Ratings",
        "Reviews"
    ]

    st.write("### Selected Features Used For Training:")
    st.write(feature_cols)

    # Keep only the required columns
    df = df[feature_cols + [target]].copy()

    # Drop rows where price is missing
    df = df.dropna(subset=[target])

    # --------------------------------
    # Split X and y
    # --------------------------------
    X = df[feature_cols].copy()
    y = df[target].copy()

    # --------------------------------
    # FIX missing values in X
    # --------------------------------
    from sklearn.impute import SimpleImputer

    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()

    # --------------------------
    # FIXED NUMERIC IMPUTATION
    # --------------------------
    if len(num_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        X[num_cols] = pd.DataFrame(
            num_imputer.fit_transform(X[num_cols]),
            columns=num_cols,
            index=X.index
        )

    # --------------------------
    # FIXED CATEGORICAL IMPUTATION
    # --------------------------
    if len(cat_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[cat_cols] = pd.DataFrame(
            cat_imputer.fit_transform(X[cat_cols]),
            columns=cat_cols,
            index=X.index
        )

    # --------------------------
    # Encode categorical columns
    # --------------------------
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le

    # --------------------------
    # Scale numeric features
    # --------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------
    # Train-test split
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model_choice = st.radio("Choose model", ["Linear Regression", "Random Forest"])

    if st.button("Train Model"):

        if model_choice == "Linear Regression":
            model = LinearRegression()
        else:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # --------------------------
        # Performance Metrics
        # --------------------------
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = model.score(X_test, y_test)

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", round(mae, 2))
        col2.metric("RMSE", round(rmse, 2))
        col3.metric("R²", round(r2, 3))

        # Save trained model
        pickle.dump((model, scaler, encoders, feature_cols), open("model.pkl", "wb"))

        st.success("🎉 Model trained & saved successfully!")
# ============================================================
        # 📈 1. ACTUAL VS PREDICTED GRAPH
        # ============================================================
        st.subheader("📈 Actual vs Predicted")

        fig1, ax1 = plt.subplots(figsize=(6,4))
        ax1.scatter(y_test, y_pred, alpha=0.6, label="Predicted")
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 'r--', linewidth=2, label="Perfect Fit")

        ax1.set_xlabel("Actual Price")
        ax1.set_ylabel("Predicted Price")
        ax1.set_title(f"Actual vs Predicted — {model_choice}")
        ax1.legend()

        st.pyplot(fig1)

        # ============================================================
        # 📉 2. RESIDUAL PLOT
        # ============================================================
        st.subheader("📉 Residual Plot")

        residuals = y_test - y_pred

        fig2, ax2 = plt.subplots(figsize=(6,4))
        ax2.scatter(y_pred, residuals, alpha=0.6)
        ax2.axhline(0, color="red", linestyle="--", linewidth=2)

        ax2.set_xlabel("Predicted Price")
        ax2.set_ylabel("Residual (Actual - Predicted)")
        ax2.set_title(f"Residual Plot — {model_choice}")

        st.pyplot(fig2)
# ============================================================
        # 🌳 Feature Importance (Random Forest Only)
        # ============================================================
        if model_choice == "Random Forest":
            st.subheader("🌳 Feature Importance")

            importances = model.feature_importances_
            fi = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances
            }).sort_values(by="Importance", ascending=True)

            fig3, ax3 = plt.subplots(figsize=(6,4))
            ax3.barh(fi["Feature"], fi["Importance"], color="teal")
            ax3.set_title("Feature Importance — Random Forest")
            st.pyplot(fig3)
# ====================================================
# 5. PREDICTION (FINAL CLEAN VERSION) — NO ORIGINAL PRICE
# ====================================================
elif page == "🔮 Prediction":

    st.header("🔮 Predict Price for a NEW Product")

    try:
        # LOAD MODEL
        model, scaler, encoders, feature_cols = pickle.load(open("model.pkl", "rb"))
        st.success("Model Loaded Successfully ✔")

        st.subheader("Enter product details:")

        user_input = {}

        # ASK ONLY FOR NECESSARY FEATURES
        for col in feature_cols:
            if col in encoders:   
                user_input[col] = st.text_input(f"{col}", value="")
            else:
                user_input[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Predict Price"):

            df_input = pd.DataFrame([user_input])

            # SAFE ENCODING
            for col in encoders:
                try:
                    df_input[col] = encoders[col].transform([df_input[col].iloc[0]])
                except:
                    df_input[col] = encoders[col].transform([encoders[col].classes_[0]])

            # NUMERIC FIX
            for col in df_input.columns:
                if col not in encoders:
                    df_input[col] = pd.to_numeric(df_input[col], errors="coerce").fillna(0.0)

            # SCALE
            scaled_input = scaler.transform(df_input[feature_cols])

            # PREDICT
            price = model.predict(scaled_input)[0]

            st.success(f"💰 Predicted Product Price: **₹{price:,.2f}**")

    except Exception as e:
        st.error("⚠ Model not trained yet. Please train the model first.")
        st.error(str(e))
