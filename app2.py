import streamlit as st
import asyncio
import time
import pandas as pd
import os
import json
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import your trading agent system
from trading_agent_system import analyze_trading_opportunity

# Set page configuration
st.set_page_config(
    page_title="Trading Portfolio Assistant",
    page_icon="📊",
    layout="wide"
)

# Portfolio file path
PORTFOLIO_FILE = "trading_portfolio.csv"

# Function to load portfolio
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    else:
        # Create empty portfolio with columns
        return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Date Added', 'Notes'])

# Function to save portfolio
def save_portfolio(portfolio_df):
    portfolio_df.to_csv(PORTFOLIO_FILE, index=False)

# Function to update portfolio based on agent's recommendation
def update_portfolio_from_recommendation(portfolio_df, recommendation_text):
    """
    Parses the recommendation text for portfolio update instructions and applies them
    Returns: (updated_portfolio, changes_summary)
    """
    # Track changes made
    changes = []
    modified = False
    
    # Look for buy instructions
    buy_pattern = r"BUY\s+(\d+(?:\.\d+)?)\s+shares\s+of\s+([A-Z]+)(?:\s+at\s+\$?(\d+(?:\.\d+)?))?"
    for match in re.finditer(buy_pattern, recommendation_text, re.IGNORECASE):
        shares = float(match.group(1))
        symbol = match.group(2).upper()
        price = float(match.group(3)) if match.group(3) else None
        
        # If no price provided, use placeholder
        if price is None:
            price = 0
            price_note = " (price not specified)"
        else:
            price_note = ""
            
        # Check if symbol already exists in portfolio
        symbol_exists = False
        for idx, row in portfolio_df.iterrows():
            if row['Symbol'] == symbol:
                # Update existing position
                portfolio_df.at[idx, 'Shares'] += shares
                # Only update price if provided and it's a new buy (not averaging)
                if price is not None and price > 0:
                    # Calculate new average price
                    old_shares = portfolio_df.at[idx, 'Shares'] - shares
                    old_price = portfolio_df.at[idx, 'Entry Price']
                    portfolio_df.at[idx, 'Entry Price'] = (old_shares * old_price + shares * price) / portfolio_df.at[idx, 'Shares']
                portfolio_df.at[idx, 'Notes'] += f" | Added {shares} shares on {datetime.now().strftime('%Y-%m-%d')}"
                changes.append(f"Added {shares} shares to existing {symbol} position{price_note}")
                symbol_exists = True
                modified = True
                break
        
        # If symbol doesn't exist, add new position
        if not symbol_exists:
            new_position = pd.DataFrame({
                'Symbol': [symbol],
                'Shares': [shares],
                'Entry Price': [price],
                'Date Added': [datetime.now().strftime("%Y-%m-%d")],
                'Notes': [f"Added by agent recommendation"]
            })
            portfolio_df = pd.concat([portfolio_df, new_position], ignore_index=True)
            changes.append(f"Added new position: {shares} shares of {symbol}{price_note}")
            modified = True
    
    # Look for sell instructions
    sell_pattern = r"SELL\s+(\d+(?:\.\d+)?)\s+shares\s+of\s+([A-Z]+)(?:\s+at\s+\$?(\d+(?:\.\d+)?))?"
    for match in re.finditer(sell_pattern, recommendation_text, re.IGNORECASE):
        shares = float(match.group(1))
        symbol = match.group(2).upper()
        price = match.group(3)  # We don't need the selling price for portfolio tracking
        
        # Check if symbol exists in portfolio
        for idx, row in portfolio_df.iterrows():
            if row['Symbol'] == symbol:
                if row['Shares'] >= shares:
                    # Partial sell
                    portfolio_df.at[idx, 'Shares'] -= shares
                    portfolio_df.at[idx, 'Notes'] += f" | Sold {shares} shares on {datetime.now().strftime('%Y-%m-%d')}"
                    changes.append(f"Sold {shares} shares of {symbol} (keeping {portfolio_df.at[idx, 'Shares']} shares)")
                    modified = True
                    # If shares reduced to 0, remove the row
                    if portfolio_df.at[idx, 'Shares'] == 0:
                        portfolio_df = portfolio_df.drop(idx).reset_index(drop=True)
                        changes.append(f"Removed {symbol} from portfolio (all shares sold)")
                else:
                    # Trying to sell more than owned
                    changes.append(f"⚠️ Warning: Attempted to sell {shares} shares of {symbol}, but only {row['Shares']} owned")
                break
        else:
            # Symbol not found in portfolio
            changes.append(f"⚠️ Warning: Cannot sell {symbol} - not found in portfolio")
    
    # Look for remove/delete instructions
    remove_pattern = r"(REMOVE|DELETE)\s+([A-Z]+)\s+from portfolio"
    for match in re.finditer(remove_pattern, recommendation_text, re.IGNORECASE):
        symbol = match.group(2).upper()
        
        # Check if symbol exists in portfolio
        for idx, row in portfolio_df.iterrows():
            if row['Symbol'] == symbol:
                portfolio_df = portfolio_df.drop(idx).reset_index(drop=True)
                changes.append(f"Removed {symbol} from portfolio completely")
                modified = True
                break
        else:
            # Symbol not found in portfolio
            changes.append(f"⚠️ Warning: Cannot remove {symbol} - not found in portfolio")
    
    # If no changes were detected but there are potential update instructions
    if not modified and any(kw in recommendation_text.lower() for kw in ['buy', 'sell', 'add', 'remove', 'portfolio']):
        changes.append("⚠️ Agent made recommendations but no portfolio changes were automatically applied.")
        changes.append("You can manually update your portfolio or rephrase your request.")
    
    return portfolio_df, changes

# Function to run async trading agent
def run_trading_analysis(query, include_portfolio=False):
    try:
        # Augment query with portfolio information if requested
        if include_portfolio and not st.session_state.portfolio.empty:
            portfolio_summary = "My current portfolio:\n"
            for _, row in st.session_state.portfolio.iterrows():
                portfolio_summary += f"- {row['Symbol']}: {row['Shares']} shares at ${row['Entry Price']:.2f}\n"
            
            enhanced_query = f"{query}\n\n{portfolio_summary}\n\nPlease include specific portfolio update instructions in your response using formats like 'BUY 10 shares of AAPL at $200' or 'SELL 5 shares of MSFT' if you recommend any changes."
        else:
            enhanced_query = query
        
        # Create a new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Run the analysis
        result = loop.run_until_complete(analyze_trading_opportunity(enhanced_query))
        loop.close()
        return result
    except Exception as e:
        st.error(f"Error running trading analysis: {str(e)}")
        return f"Error: {str(e)}"

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = load_portfolio()
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_response' not in st.session_state:
    st.session_state.current_response = None
if 'changes_made' not in st.session_state:
    st.session_state.changes_made = []

# Header
st.title("🚀 Trading Portfolio Assistant")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Portfolio", "Analysis", "History"])

# Portfolio Tab
with tab1:
    st.header("Your Investment Portfolio")
    
    # Display current portfolio
    if not st.session_state.portfolio.empty:
        # Calculate current values and totals
        portfolio = st.session_state.portfolio.copy()
        portfolio['Total Cost'] = portfolio['Shares'] * portfolio['Entry Price']
        
        # Display portfolio summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Holdings", f"{len(portfolio)} stocks")
        with col2:
            total_investment = portfolio['Total Cost'].sum()
            st.metric("Total Investment", f"${total_investment:,.2f}")
        with col3:
            st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
        
        # Portfolio table with edit functionality
        st.subheader("Holdings")
        
        # Edit mode toggle
        edit_mode = st.toggle("Enable Edit Mode", False)
        
        if edit_mode:
            # Editable dataframe
            edited_df = st.data_editor(
                portfolio,
                num_rows="dynamic",
                key="portfolio_editor",
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", help="Stock ticker symbol"),
                    "Shares": st.column_config.NumberColumn("Shares", help="Number of shares", min_value=0, step=1),
                    "Entry Price": st.column_config.NumberColumn("Entry Price", help="Price per share at purchase", min_value=0, format="$%.2f"),
                    "Date Added": st.column_config.DateColumn("Date Added", help="When you acquired the shares"),
                    "Notes": st.column_config.TextColumn("Notes", help="Any additional information"),
                    "Total Cost": None  # Hide calculated column during editing
                },
                hide_index=True
            )
            
            if st.button("Save Changes"):
                # Update the portfolio
                st.session_state.portfolio = edited_df[['Symbol', 'Shares', 'Entry Price', 'Date Added', 'Notes']]
                save_portfolio(st.session_state.portfolio)
                st.success("Portfolio saved successfully!")
                time.sleep(1)
                st.experimental_rerun()
        else:
            # Display as regular table with calculated values
            st.dataframe(
                portfolio,
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol"),
                    "Shares": st.column_config.NumberColumn("Shares", format="%.2f"),
                    "Entry Price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                    "Total Cost": st.column_config.NumberColumn("Total Cost", format="$%.2f"),
                    "Date Added": st.column_config.DateColumn("Date Added"),
                    "Notes": st.column_config.TextColumn("Notes")
                },
                hide_index=True
            )
        
        # Portfolio visualization
        if not portfolio.empty and len(portfolio) > 1:
            st.subheader("Portfolio Allocation")
            
            # Create allocation chart
            fig = px.pie(
                portfolio, 
                values='Total Cost', 
                names='Symbol',
                title='Portfolio Allocation by Investment',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
            
        # Show recent changes if any
        if st.session_state.changes_made:
            st.subheader("Recent Portfolio Updates")
            with st.expander("View recent changes made by the agent", expanded=True):
                for change in st.session_state.changes_made:
                    st.markdown(f"- {change}")
                if st.button("Clear Change History"):
                    st.session_state.changes_made = []
                    st.experimental_rerun()
    else:
        st.info("Your portfolio is empty. Add some holdings to get started!")
    
    # Portfolio upload/download section
    st.subheader("Import/Export Portfolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Upload portfolio
        uploaded_file = st.file_uploader("Import portfolio from CSV", type="csv")
        if uploaded_file is not None:
            try:
                imported_portfolio = pd.read_csv(uploaded_file)
                required_columns = ['Symbol', 'Shares', 'Entry Price']
                
                # Check if required columns exist
                if all(col in imported_portfolio.columns for col in required_columns):
                    # Add any missing columns with default values
                    if 'Date Added' not in imported_portfolio.columns:
                        imported_portfolio['Date Added'] = datetime.now().strftime("%Y-%m-%d")
                    if 'Notes' not in imported_portfolio.columns:
                        imported_portfolio['Notes'] = ""
                        
                    # Update and save
                    st.session_state.portfolio = imported_portfolio
                    save_portfolio(imported_portfolio)
                    st.success("Portfolio imported successfully!")
                else:
                    st.error(f"CSV must contain the columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"Error importing portfolio: {str(e)}")
    
    with col2:
        # Download portfolio
        if not st.session_state.portfolio.empty:
            csv = st.session_state.portfolio.to_csv(index=False)
            st.download_button(
                label="Download Portfolio as CSV",
                data=csv,
                file_name="my_trading_portfolio.csv",
                mime="text/csv"
            )
            
    # Quick add form
    st.subheader("Add New Position")
    with st.form("add_position_form"):
        cols = st.columns([2, 2, 2, 3, 3])
        new_symbol = cols[0].text_input("Symbol").upper()
        new_shares = cols[1].number_input("Shares", min_value=0.01, step=1.0)
        new_price = cols[2].number_input("Entry Price ($)", min_value=0.01, step=0.01)
        new_date = cols[3].date_input("Date Added")
        new_notes = cols[4].text_input("Notes")
        
        submitted = st.form_submit_button("Add Position")
        
        if submitted and new_symbol and new_shares > 0 and new_price > 0:
            # Create new row
            new_position = pd.DataFrame({
                'Symbol': [new_symbol],
                'Shares': [new_shares],
                'Entry Price': [new_price],
                'Date Added': [new_date.strftime("%Y-%m-%d")],
                'Notes': [new_notes]
            })
            
            # Append to portfolio
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
            save_portfolio(st.session_state.portfolio)
            st.success(f"Added {new_shares} shares of {new_symbol} to your portfolio!")
            time.sleep(1)
            st.experimental_rerun()

# Analysis Tab
with tab2:
    st.header("Trading Analysis")
    
    # Example queries for portfolio-specific analysis
    st.subheader("Ask the Trading Agent")
    examples = [
        "What changes should I make to my portfolio given current market conditions?",
        "Analyze my current holdings and suggest any adjustments",
        "Which stocks in my portfolio should I consider selling?",
        "What new positions would complement my existing portfolio?",
        "Rebalance my portfolio for optimal risk/reward"
    ]
    
    # Display example buttons in 2 rows
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i % 3].button(f"📝 {example}", key=f"example_{i}"):
            st.session_state.query = example
    
    # Display portfolio-based advice option
    include_portfolio = st.checkbox("Include my portfolio details in the query", value=True, 
                                    help="When checked, your portfolio holdings will be included in the query to the agent")
    
    allow_updates = st.checkbox("Allow the agent to update my portfolio", value=True,
                                help="When checked, the agent can automatically add/remove positions based on its recommendations")
    
    # Input form
    with st.form("analysis_form"):
        query = st.text_area(
            "Enter your trading question",
            value=st.session_state.get('query', ''),
            height=100,
            placeholder="Ask about your portfolio, market conditions, or specific stocks..."
        )
        
        analyze_button = st.form_submit_button("Get Analysis")
    
    # Process query
    if analyze_button and query:
        with st.spinner("Analyzing your query..."):
            # Run trading analysis
            result = run_trading_analysis(query, include_portfolio)
            
            # Save to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_item = {
                "query": query,
                "response": result,
                "timestamp": timestamp,
                "included_portfolio": include_portfolio
            }
            st.session_state.history.insert(0, history_item)
            st.session_state.current_response = history_item
            
            # Process portfolio updates if allowed
            if allow_updates and include_portfolio:
                updated_portfolio, changes = update_portfolio_from_recommendation(
                    st.session_state.portfolio, result
                )
                
                if changes:
                    # Save changes to history
                    st.session_state.changes_made = changes + st.session_state.changes_made
                    
                    # Update portfolio
                    st.session_state.portfolio = updated_portfolio
                    save_portfolio(updated_portfolio)
            
            # Clear the query after submission
            st.session_state.query = ""
    
    # Display response
    if st.session_state.current_response:
        st.subheader("Analysis Results")
        
        # Show metadata
        st.caption(f"Query time: {st.session_state.current_response['timestamp']}")
        
        # Show query and response in an expander
        with st.expander("Show Query", expanded=False):
            st.write(st.session_state.current_response['query'])
        
        # Show the response
        st.markdown(st.session_state.current_response['response'])
        
        # Show portfolio update summary if changes were made
        if st.session_state.changes_made:
            with st.expander("Portfolio Updates", expanded=True):
                st.subheader("Changes Made to Your Portfolio")
                for change in st.session_state.changes_made[:5]:  # Show only the most recent changes
                    st.markdown(f"- {change}")
                
                if st.button("View All Changes in Portfolio Tab"):
                    tab1.selectbox = True
                    st.experimental_rerun()

# History Tab
with tab3:
    st.header("Analysis History")
    
    if st.session_state.history:
        # Group history by date
        dates = {}
        for item in st.session_state.history:
            date = item['timestamp'].split(' ')[0]
            if date not in dates:
                dates[date] = []
            dates[date].append(item)
        
        # Display history by date
        for date, items in dates.items():
            st.subheader(date)
            
            for i, item in enumerate(items):
                with st.expander(f"{item['query'][:50]}..." if len(item['query']) > 50 else item['query'], expanded=False):
                    st.caption(f"Time: {item['timestamp'].split(' ')[1]}")
                    st.markdown(item['response'])
                    
                    col1, col2 = st.columns(2)
                    if col1.button("Load in Analysis Tab", key=f"load_{date}_{i}"):
                        st.session_state.current_response = item
                        st.experimental_rerun()
    else:
        st.info("No query history yet. Try asking a question in the Analysis tab!")

# Footer
st.markdown("---")
st.caption("Trading Portfolio Assistant | Data is stored locally on your computer")