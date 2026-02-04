import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title='Wealth Migration Dashboard', layout='wide')
st.title('🌍 Wealth Migration — Dashboard (Grandmaster+)')

# Load artifacts
@st.cache_data
def load_data():
    try:
        res = pd.read_csv('model_comparison_full.csv')
    except Exception:
        res = pd.DataFrame()
    try:
        preds = pd.read_csv('country_predictions.csv')
    except Exception:
        preds = pd.DataFrame()
    return res, preds

res, preds = load_data()

st.sidebar.header('Options')
show_map = st.sidebar.checkbox('Show world map', value=True)

st.subheader('Model Rankings')
if res.empty:
    st.info('Model ranking file not found (run the notebook to generate `model_comparison_full.csv`).')
else:
    st.table(res)

st.markdown('---')

st.subheader('Country Predictions')
if preds.empty:
    st.info('Country predictions file not found. Run the notebook to save `country_predictions.csv`.')
else:
    country = st.selectbox('Select country', preds['country'].tolist())
    row = preds[preds['country'] == country].iloc[0]
    st.metric('Predicted Inflow Probability', f"{row['pred_prob']:.2%}")

    feat_cols = [c for c in preds.columns if c not in ['country','pred_prob']]
    st.write('Feature snapshot:')
    st.write(row[feat_cols].to_frame().T)

    if show_map:
        st.subheader('Predicted Inflow Probability by Country')
        fig = px.choropleth(preds, locations='country', locationmode='country names', color='pred_prob', color_continuous_scale='Viridis', title='Predicted Inflow Probability by Country')
        st.plotly_chart(fig, use_container_width=True)

st.markdown('---')
st.subheader('Download artifacts')
col1, col2 = st.columns(2)
with col1:
    if not res.empty:
        st.download_button('Download model rankings (CSV)', res.to_csv(index=False), file_name='model_comparison_full.csv')
with col2:
    if not preds.empty:
        st.download_button('Download country predictions (CSV)', preds.to_csv(index=False), file_name='country_predictions.csv')

st.markdown('---')
st.caption('Run `conda activate ml_env && streamlit run wealth_migration_app.py` to start the app locally.')