"""
Twitter Sentiment Analyzer - Full App - Clean Minimal UI
(Light Grey Sidebar, Soft Blue Accent)
- Batch analysis with manual feedback per tweet
- Feedback percentages on statistics page
- One-click clear history
- FIXED: Feedback logic for single and batch analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# Import your predictor module
from predict import SentimentPredictor

# ----------------------- Page config ----------------------- #
st.set_page_config(
    page_title="Twitter Sentiment Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- Custom CSS ----------------------- #
st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .sidebar .sidebar-content {
        background-color: #e5e7eb;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- Session state ----------------------- #
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
    st.session_state.model_loaded = False
    st.session_state.predictions_history = []

if 'page' not in st.session_state:
    st.session_state.page = "Home"

# ----------------------- Model loader ----------------------- #
@st.cache_resource
def load_model():
    try:
        predictor = SentimentPredictor()
        predictor.load_model(
            model_path='models/logistic_regression_model.pkl',
            vectorizer_path='models/tfidf_vectorizer.pkl'
        )
        return predictor, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

# ----------------------- Visual helpers ----------------------- #
def create_confidence_gauge(confidence, sentiment):
    color = "#2ecc71" if sentiment == "POSITIVE" else "#e74c3c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        title={'text': "Confidence (%)"},
        gauge={'axis': {'range':[0,100]},
               'bar':{'color':color},
               'steps':[{'range':[0,60],'color':"#f1f5f9"},{'range':[60,75],'color':"#e2e8f0"},
                       {'range':[75,90],'color':"#cbd5e1"},{'range':[90,100],'color':"#94a3b8"}]}
    ))
    fig.update_layout(height=240, margin=dict(l=10,r=10,t=40,b=10))
    return fig

def create_probability_bar(proba_positive, proba_negative):
    fig = go.Figure()
    fig.add_trace(go.Bar(y=['Sentiment'], x=[proba_negative*100], name='Negative',
                         orientation='h', marker=dict(color='#e74c3c'),
                         text=[f"{proba_negative*100:.1f}%"], textposition='inside'))
    fig.add_trace(go.Bar(y=['Sentiment'], x=[proba_positive*100], name='Positive',
                         orientation='h', marker=dict(color='#2ecc71'),
                         text=[f"{proba_positive*100:.1f}%"], textposition='inside'))
    fig.update_layout(barmode='stack', height=140, showlegend=True,
                     xaxis_title="Probability (%)", margin=dict(l=10,r=10,t=20,b=10),
                     xaxis=dict(range=[0,100]))
    return fig

def display_result(result):
    sentiment = result['sentiment']
    confidence = result['confidence']
    
    st.markdown("### Prediction Result")
    col1,col2 = st.columns([2,1])
    
    with col1:
        st.markdown(f"<h1 style='color:#2ecc71'>✓ {sentiment}</h1>" if sentiment=="POSITIVE" 
                   else f"<h1 style='color:#e74c3c'>✗ {sentiment}</h1>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("**Original Tweet:**")
        st.info(result['original_tweet'])
        with st.expander("Processed text"):
            st.code(result.get('cleaned_tweet',''))
        st.markdown("**Probability Distribution:**")
        st.plotly_chart(create_probability_bar(result['probabilities']['positive'],
                                              result['probabilities']['negative']), 
                       use_container_width=True)
    
    with col2:
        st.plotly_chart(create_confidence_gauge(confidence,sentiment), use_container_width=True)
        st.metric("Confidence", f"{confidence:.1f}%")
        st.metric("Positive Probability", f"{result['probabilities']['positive']*100:.1f}%")
        st.metric("Negative Probability", f"{result['probabilities']['negative']*100:.1f}%")
    
    st.markdown("---")
    st.markdown("**Interpretation:**")
    if confidence>=90:
        st.success("The model is very confident about this prediction.")
    elif confidence>=75:
        st.info("The model is confident about this prediction.")
    elif confidence>=60:
        st.warning("The model is moderately confident; sentiment leans in a direction.")
    else:
        st.warning("The model is uncertain; the sentiment may be ambiguous or neutral.")

# ----------------------- Sidebar ----------------------- #
with st.sidebar:
    st.markdown("<h1 style='text-align:center'>🐦 Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("**Model Status**")
    if not st.session_state.model_loaded:
        with st.spinner("Loading model..."):
            predictor, loaded = load_model()
            if loaded:
                st.session_state.predictor = predictor
                st.session_state.model_loaded = True
                st.success("✓ Model loaded")
            else:
                st.error("✗ Failed to load model")
    else:
        st.success("✓ Model ready")
    
    st.markdown("---")
    pages = ["Home","Batch Analysis","Statistics","About"]
    for p in pages:
        key = f"nav_{p}"
        is_active = (st.session_state.get('page','Home')==p)
        clicked = st.button(p,key=key, use_container_width=True)
        if clicked:
            st.session_state.page=p
        if is_active:
            st.markdown("<div style='height:2px;background:#3b82f6;margin-top:-10px'></div>", 
                       unsafe_allow_html=True)
    
    st.markdown("---")
    with st.expander("ℹ️ Model Information", expanded=False):
        st.write("**Primary Model:** Logistic Regression")
        st.write("**Training Samples:** 1,600,000")
        st.write("**Vocabulary Size:** 5,000 features")
    
    st.markdown("---")
    st.markdown("<h4>💡 Tips</h4>", unsafe_allow_html=True)
    st.markdown("<small>• Use clear language<br>• Avoid heavy sarcasm<br>• Keep tweets concise</small>", 
               unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("<center><small>© 2024 — Sentiment Analyzer</small></center>", unsafe_allow_html=True)

# ----------------------- MAIN PAGES ----------------------- #
page = st.session_state.get('page','Home')

# ----------------------- HOME PAGE ----------------------- #
if page=="Home":
    st.title("🐦 Twitter Sentiment Analyzer")
    st.write("Sentiment analysis for Twitter text.")
    st.markdown("---")
    
    c1,c2,c3 = st.columns(3)
    with c1:
        st.metric("Accuracy","78.46%")
    with c2:
        st.metric("Average Confidence","< 100%")
    with c3:
        st.metric("Training Data","1.6M+")
    
    st.markdown("---")
    st.subheader("Quick Analysis")
    
    # Initialize session state for single tweet analysis
    if 'quick_tweet' not in st.session_state:
        st.session_state.quick_tweet = ""
    if 'quick_result' not in st.session_state:
        st.session_state.quick_result = None
    if 'show_feedback' not in st.session_state:
        st.session_state.show_feedback = False
    
    st.session_state.quick_tweet = st.text_input(
        "Enter a tweet to analyze", 
        value=st.session_state.quick_tweet,
        placeholder="I love this product!", 
        max_chars=280
    )
    
    if st.button("🔍 Analyze Now", use_container_width=True):
        if not st.session_state.quick_tweet:
            st.warning("Please enter a tweet first.")
        elif not st.session_state.model_loaded:
            st.error("Model not loaded.")
        else:
            with st.spinner("Analyzing..."):
                result = st.session_state.predictor.predict_single(st.session_state.quick_tweet)
                st.session_state.quick_result = result
                st.session_state.show_feedback = True
                # Add to history without feedback initially
                st.session_state.predictions_history.append({
                    'tweet': st.session_state.quick_tweet,
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'timestamp': datetime.now(),
                    'feedback': None
                })
    
    # Display result if available
    if st.session_state.quick_result:
        display_result(st.session_state.quick_result)
        
        # ----------------------- Single Tweet Feedback ----------------------- #
        if st.session_state.show_feedback:
            st.markdown("---")
            st.subheader("📝 Submit Feedback for this Prediction")
            
            # Create a unique key for this feedback session
            feedback_choice = st.radio(
                "Was this prediction correct?",
                ["Yes", "No"],
                key="single_feedback_radio",
                index=0
            )
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("✓ Submit Feedback", use_container_width=True):
                    # Update the last prediction with feedback
                    if st.session_state.predictions_history:
                        st.session_state.predictions_history[-1]['feedback'] = feedback_choice
                    st.success(f"Feedback submitted: {feedback_choice}")
                    
                    # Clear for next analysis
                    st.session_state.quick_tweet = ""
                    st.session_state.quick_result = None
                    st.session_state.show_feedback = False
                    time.sleep(0.5)
                    st.rerun()

# ----------------------- BATCH ANALYSIS ----------------------- #
elif page=="Batch Analysis":
    st.title("📊 Batch Sentiment Analysis with Feedback")
    st.write("Analyze multiple tweets at once and submit feedback per tweet.")
    st.markdown("---")
    
    input_method = st.radio("Input method:", ["Text Input","File Upload"], horizontal=True)
    
    tweets=[]
    if input_method=="Text Input":
        tweets_text = st.text_area("Enter tweets (one per line):", height=300)
        if tweets_text:
            tweets=[line.strip() for line in tweets_text.split('\n') if line.strip()]
        if tweets:
            st.info(f"📝 {len(tweets)} tweets entered")
    else:
        uploaded_file = st.file_uploader("Upload CSV or TXT file", type=['csv','txt'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                tweets = df.iloc[:,0].astype(str).tolist()
            else:
                tweets = [t.strip() for t in uploaded_file.read().decode('utf-8').split('\n') if t.strip()]
            st.success(f"✓ {len(tweets)} tweets loaded from file")
    
    st.markdown("---")
    
    if st.button("🚀 Analyze All Tweets", use_container_width=True):
        if not tweets:
            st.warning("Please provide tweets to analyze.")
        elif not st.session_state.model_loaded:
            st.error("Model not loaded.")
        elif len(tweets) > 1000:
            st.error("Maximum 1000 tweets allowed per batch.")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize batch results in session state
            st.session_state.batch_results = []
            
            for i, tweet in enumerate(tweets):
                result = st.session_state.predictor.predict_single(tweet)
                st.session_state.batch_results.append(result)
                progress = (i+1)/len(tweets)
                progress_bar.progress(progress)
                status_text.text(f"Processing: {i+1}/{len(tweets)}")
            
            status_text.text("✓ Analysis complete")
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()
            
            # Initialize feedback state for all tweets
            if 'batch_feedback_submitted' not in st.session_state:
                st.session_state.batch_feedback_submitted = False
    
    # ----------------------- Display Batch Results with Feedback ----------------------- #
    if 'batch_results' in st.session_state and st.session_state.batch_results:
        st.markdown("---")
        st.markdown("## 📋 Batch Results & Feedback")
        
        # Store feedback choices in a dictionary
        if 'batch_feedback_choices' not in st.session_state:
            st.session_state.batch_feedback_choices = {}
        
        for idx, r in enumerate(st.session_state.batch_results):
            with st.expander(f"Tweet {idx+1}: {r['original_tweet'][:50]}...", expanded=(idx==0)):
                display_result(r)
                
                st.markdown("#### Was this prediction correct?")
                
                # Use a unique key for each radio button
                feedback_key = f"batch_fb_{idx}"
                
                # Get current value or default to "Yes"
                current_value = st.session_state.batch_feedback_choices.get(idx, "Yes")
                
                feedback = st.radio(
                    "Select feedback:",
                    ["Yes", "No"],
                    key=feedback_key,
                    index=0 if current_value == "Yes" else 1,
                    horizontal=True
                )
                
                # Store the feedback choice
                st.session_state.batch_feedback_choices[idx] = feedback
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("✅ Submit All Feedback", use_container_width=True, type="primary"):
                # Add all results with their feedback to history
                for idx, r in enumerate(st.session_state.batch_results):
                    feedback = st.session_state.batch_feedback_choices.get(idx, "Yes")
                    st.session_state.predictions_history.append({
                        'tweet': r['original_tweet'],
                        'sentiment': r['sentiment'],
                        'confidence': r['confidence'],
                        'timestamp': datetime.now(),
                        'feedback': feedback
                    })
                
                st.success(f"✓ Batch analysis complete! {len(st.session_state.batch_results)} predictions with feedback recorded!")
                
                # Clear batch results and feedback
                time.sleep(1)
                del st.session_state.batch_results
                del st.session_state.batch_feedback_choices
                st.rerun()

# ----------------------- STATISTICS ----------------------- #
elif page=="Statistics":
    st.title("📊 Prediction Statistics & Feedback")
    
    if not st.session_state.predictions_history:
        st.info("No predictions yet. Start analyzing tweets to see statistics!")
    else:
        history_df = pd.DataFrame(st.session_state.predictions_history)
        
        total = len(history_df)
        positive_count = (history_df['sentiment']=='POSITIVE').sum()
        negative_count = (history_df['sentiment']=='NEGATIVE').sum()
        avg_confidence = history_df['confidence'].mean()
        
        # Feedback statistics
        feedback_yes = (history_df['feedback']=='Yes').sum()
        feedback_no = (history_df['feedback']=='No').sum()
        feedback_none = history_df['feedback'].isna().sum()
        feedback_total = feedback_yes + feedback_no
        
        feedback_pct_yes = (feedback_yes/feedback_total*100) if feedback_total>0 else 0
        feedback_pct_no = (feedback_no/feedback_total*100) if feedback_total>0 else 0
        
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.metric("Positive", f"{positive_count}", delta=f"{(positive_count/total*100):.1f}%")
        with c2:
            st.metric("Negative", f"{negative_count}", delta=f"{(negative_count/total*100):.1f}%")
        with c3:
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        with c4:
            st.metric("Feedback Yes %", f"{feedback_pct_yes:.0f}%")
        
        if feedback_none > 0:
            st.info(f"ℹ️ {feedback_none} predictions have no feedback yet")
        
        st.markdown("---")
        st.subheader("📈 Visualizations")
        
        p1,p2=st.columns(2)
        with p1:
            st.plotly_chart(px.pie(values=[positive_count, negative_count], 
                                  names=['Positive','Negative'],
                                  title="Sentiment Distribution",
                                  color_discrete_sequence=['#2ecc71', '#e74c3c']),
                          use_container_width=True)
        with p2:
            if feedback_total > 0:
                st.plotly_chart(px.pie(values=[feedback_yes, feedback_no], 
                                      names=['Yes','No'],
                                      title="Feedback Distribution",
                                      color_discrete_sequence=['#3b82f6', '#ef4444']),
                              use_container_width=True)
            else:
                st.info("No feedback submitted yet")
        
        st.markdown("---")
        st.subheader("🕐 Recent Predictions")
        display_df = history_df[['tweet','sentiment','confidence','feedback','timestamp']].tail(20)
        display_df = display_df.sort_values('timestamp', ascending=False)
        st.dataframe(display_df, use_container_width=True, height=400)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.predictions_history=[]
                st.rerun()

# ----------------------- ABOUT ----------------------- #
elif page=="About":
    st.title("ℹ️ About this project")
    st.markdown("---")
    
    st.markdown("## Twitter Sentiment Analyzer")
    st.markdown("Minimal professional UI for sentiment analysis of tweets using a trained ML model.")
    
    st.markdown("---")
    st.subheader("📊 Model Performance (Test Dataset)")
    
    metrics_df = pd.DataFrame([
    {"Model": "Naive Bayes", "Accuracy": "76.52%", "Precision": "76.52%", "Recall": "76.52%", "F1-Score": "76.52%", "ROC-AUC": "0.848"},
    {"Model": "Logistic Regression", "Accuracy": "78.46%", "Precision": "77.39%", "Recall": "80.41%", "F1-Score": "78.87%", "ROC-AUC": "0.866"},
    {"Model": "Voting Ensemble", "Accuracy": "78.38%", "Precision": "77.24%", "Recall": "80.45%", "F1-Score": "78.82%", "ROC-AUC": "0.784"},
    {"Model": "Stacking Ensemble", "Accuracy": "78.41%", "Precision": "77.51%", "Recall": "80.04%", "F1-Score": "78.75%", "ROC-AUC": "0.865"}
    ])
    st.dataframe(metrics_df.set_index('Model'), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🛠️ Technical Stack")
    st.write("- **Frontend:** Streamlit")
    st.write("- **ML:** Scikit-learn")
    st.write("- **Visualization:** Plotly")
    st.write("- **NLP:** NLTK / custom preprocessing")
    
    st.markdown("---")
    st.subheader("⚠️ Limitations")
    st.write("- May not detect sarcasm reliably")
    st.write("- Works best with English tweets and clear emotional language")
    st.write("- Trained on Twitter data from 2009-2017")

# ----------------------- Footer ----------------------- #
st.markdown("---")
f1,f2,f3=st.columns(3)
with f1:
    st.markdown("**Accuracy:** 78.46%")
with f2:
    st.markdown("**Model:** Logistic Regression")
with f3:
    st.markdown("**Training Data:** 1.6M+")