import streamlit as st
import asyncio
from datetime import datetime
import nest_asyncio  # Corrected import statement
import pandas as pd
import time
import json
import plotly.graph_objects as go
import plotly.express as px

# Import your trading agent system
from trading_agent_system import analyze_trading_opportunity

# Apply nest_asyncio to make asyncio work properly in Streamlit
nest_asyncio.apply()

# Set page configuration
st.set_page_config(
    page_title="TradingGPT - AI Trading Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(to right, #3b82f6, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .agent-card {
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .market-agent {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3b82f6;
    }
    .fundamental-agent {
        background-color: rgba(16, 185, 129, 0.1);
        border-left: 4px solid #10b981;
    }
    .macro-agent {
        background-color: rgba(124, 58, 237, 0.1);
        border-left: 4px solid #7c3aed;
    }
    .news-agent {
        background-color: rgba(245, 158, 11, 0.1);
        border-left: 4px solid #f59e0b;
    }
    .response-area {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid #e5e7eb;
    }
    .stButton button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .stButton button:hover {
        background-color: #4338ca;
    }
    .example-button {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #3b82f6;
        border-radius: 15px;
        padding: 0.3rem 0.8rem;
        font-size: 0.9rem;
        margin-right: 0.5rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .example-button:hover {
        background-color: rgba(59, 130, 246, 0.2);
    }
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None

# Header
st.markdown('<h1 class="main-header">TradingGPT</h1>', unsafe_allow_html=True)
st.markdown("Your AI-powered trading research assistant")

# Main layout with columns
left_col, right_col = st.columns([2, 1])

# Left column - Main interaction area
with left_col:
    # Query input
    st.markdown("### Ask Your Trading Question")
    
    # Example queries
    example_queries = [
        "What's the current market regime?",
        "Analyze AAPL technical indicators",
        "Best sectors for this economic cycle"
    ]
    
    # Create a row of buttons for example queries
    cols = st.columns(len(example_queries))
    for i, query in enumerate(example_queries):
        if cols[i].button(query, key=f"example_{i}"):
            # If the user clicks an example, use it as the query
            st.session_state.example_query = query
    
    # Query form
    with st.form(key='query_form'):
        # Use the example query if one was selected
        default_query = st.session_state.get('example_query', '')
        
        user_query = st.text_area("Enter your trading question", height=100, 
                                   key="user_query", 
                                   value=default_query,
                                   placeholder="Example: What stocks are positioned to benefit from current market conditions?")
        submit_button = st.form_submit_button(label='Analyze')
    
    # Process the query
    if submit_button and user_query:
        with st.spinner("Analyzing your query..."):
            try:
                # Create an event loop to run the async function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Start a timer
                start_time = time.time()
                
                # Call your trading agent
                result = loop.run_until_complete(analyze_trading_opportunity(user_query))
                loop.close()
                
                # Calculate processing time
                processing_time = time.time() - start_time
                
                # Store response with timestamp
                response_data = {
                    "query": user_query,
                    "response": result,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "processing_time": f"{processing_time:.2f} seconds"
                }
                
                # Update session state
                st.session_state.current_response = response_data
                st.session_state.history.insert(0, response_data)
                
                # Clear the example query after submission
                if 'example_query' in st.session_state:
                    del st.session_state.example_query
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    # Display current response
    if st.session_state.current_response:
        st.markdown("### Analysis Results")
        
        # Response metadata
        meta_col1, meta_col2 = st.columns([3, 1])
        with meta_col1:
            st.markdown(f"**Query:** {st.session_state.current_response['query']}")
        with meta_col2:
            st.markdown(f"**Time:** {st.session_state.current_response['timestamp']}")
        
        # Display the response in a nicely formatted container
        st.markdown("---")
        with st.container():
            st.markdown(st.session_state.current_response['response'])
        
        # Add an example visualization if keywords are detected in the response
        if any(keyword in st.session_state.current_response['response'].lower() for keyword in ['technical', 'indicator', 'price', 'trend']):
            st.markdown("### Visualization")
            
            # This is a placeholder chart - in a real implementation, you would generate
            # charts based on actual data from your trading agent
            chart_tab1, chart_tab2 = st.tabs(["Price Chart", "Technical Indicators"])
            
            with chart_tab1:
                # Generate fake price data
                dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
                prices = [100]
                for i in range(1, 100):
                    prices.append(prices[-1] * (1 + (0.001 * (i % 5 - 2))))
                
                price_df = pd.DataFrame({
                    'Date': dates,
                    'Price': prices
                })
                
                fig = px.line(price_df, x='Date', y='Price', title='Sample Price Chart')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tab2:
                # Generate fake RSI data
                rsi_values = [50 + (i % 10 - 5) * 5 for i in range(100)]
                rsi_df = pd.DataFrame({
                    'Date': dates,
                    'RSI': rsi_values
                })
                
                fig = px.line(rsi_df, x='Date', y='RSI', title='Sample RSI Indicator')
                fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

# Right column - Agent capabilities and history
with right_col:
    # Agent capabilities section
    st.markdown("### Agent Capabilities")
    
    with st.container():
        st.markdown("#### 📊 Market Data Analyst")
        st.markdown("Technical analysis, support/resistance, volume profile, and market structure")
    
    with st.container():
        st.markdown("#### 📈 Fundamental Analyst")
        st.markdown("Financial statements, ratios, valuations, and company health assessment")
    
    with st.container():
        st.markdown("#### 🌍 Macro Environment Analyst")
        st.markdown("Economic indicators, central bank policies, and market regimes")
    
    with st.container():
        st.markdown("#### 📰 News Sentiment Analyst")
        st.markdown("News impact, sentiment trends, and media analysis")
    
    # History section
    st.markdown("### Query History")
    
    if len(st.session_state.history) > 0:
        for i, item in enumerate(st.session_state.history[:5]):  # Show only the 5 most recent
            with st.expander(f"{item['query'][:50]}..." if len(item['query']) > 50 else item['query']):
                st.markdown(f"**Time:** {item['timestamp']}")
                st.markdown(f"**Processing Time:** {item['processing_time']}")
                st.markdown("**Response:**")
                st.markdown(item['response'][:200] + "..." if len(item['response']) > 200 else item['response'])
                if st.button("Show Full Response", key=f"full_{i}"):
                    st.session_state.current_response = item
                    st.experimental_rerun()
    else:
        st.info("No query history yet. Try asking a question!")

# Footer
st.markdown("---")
st.markdown("© 2025 TradingGPT | Powered by AI Trading Agent System")