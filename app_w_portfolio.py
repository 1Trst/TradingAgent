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

# Function to load portfolio and ensure correct data types
def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            # Load the CSV
            df = pd.read_csv(PORTFOLIO_FILE)
            
            # Ensure proper data types for each column
            if 'Symbol' in df.columns:
                df['Symbol'] = df['Symbol'].astype(str)
            if 'Shares' in df.columns:
                df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
            if 'Entry Price' in df.columns:
                df['Entry Price'] = pd.to_numeric(df['Entry Price'], errors='coerce')
            if 'Date Added' in df.columns:
                # Keep Date Added as string for consistency
                df['Date Added'] = df['Date Added'].astype(str)
            if 'Notes' in df.columns:
                # Convert Notes to string (important if it was loaded as float)
                df['Notes'] = df['Notes'].astype(str)
                # Replace 'nan' strings with empty strings
                df['Notes'] = df['Notes'].replace('nan', '')
            
            return df
        except Exception as e:
            st.error(f"Error loading portfolio: {str(e)}")
            return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Date Added', 'Notes'])
    else:
        # Create empty portfolio with columns
        return pd.DataFrame(columns=['Symbol', 'Shares', 'Entry Price', 'Date Added', 'Notes'])

# Function to save portfolio
def save_portfolio(portfolio_df):
    try:
        # Ensure Notes column is string type before saving
        if 'Notes' in portfolio_df.columns:
            portfolio_df['Notes'] = portfolio_df['Notes'].astype(str)
            portfolio_df['Notes'] = portfolio_df['Notes'].replace('nan', '')
        
        portfolio_df.to_csv(PORTFOLIO_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving portfolio: {str(e)}")
        return False

# Function to parse portfolio recommendations and return suggested changes
def parse_portfolio_recommendations(portfolio_df, recommendation_text):
    """
    Parses the recommendation text for portfolio update instructions and returns suggestions
    Returns: List of change suggestions
    """
    # Track suggested changes
    suggestions = []
    
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
                # Create an update suggestion
                suggestions.append({
                    'action': 'update',
                    'symbol': symbol,
                    'shares': shares,
                    'price': price,
                    'row_idx': idx,
                    'description': f"Add {shares} shares to existing {symbol} position{price_note}"
                })
                symbol_exists = True
                break
        
        # If symbol doesn't exist, suggest adding new position
        if not symbol_exists:
            suggestions.append({
                'action': 'add',
                'symbol': symbol,
                'shares': shares,
                'price': price,
                'description': f"Add new position: {shares} shares of {symbol}{price_note}"
            })
    
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
                    # Suggest sell
                    suggestions.append({
                        'action': 'sell',
                        'symbol': symbol,
                        'shares': shares,
                        'row_idx': idx,
                        'description': f"Sell {shares} shares of {symbol} (keeping {row['Shares'] - shares} shares)"
                    })
                else:
                    # Trying to sell more than owned
                    suggestions.append({
                        'action': 'warning',
                        'symbol': symbol,
                        'description': f"⚠️ Warning: Attempted to sell {shares} shares of {symbol}, but only {row['Shares']} owned"
                    })
                break
        else:
            # Symbol not found in portfolio
            suggestions.append({
                'action': 'warning',
                'symbol': symbol,
                'description': f"⚠️ Warning: Cannot sell {symbol} - not found in portfolio"
            })
    
    # Look for remove/delete instructions
    remove_pattern = r"(REMOVE|DELETE)\s+([A-Z]+)\s+from portfolio"
    for match in re.finditer(remove_pattern, recommendation_text, re.IGNORECASE):
        symbol = match.group(2).upper()
        
        # Check if symbol exists in portfolio
        for idx, row in portfolio_df.iterrows():
            if row['Symbol'] == symbol:
                suggestions.append({
                    'action': 'remove',
                    'symbol': symbol,
                    'row_idx': idx,
                    'description': f"Remove {symbol} from portfolio completely"
                })
                break
        else:
            # Symbol not found in portfolio
            suggestions.append({
                'action': 'warning',
                'symbol': symbol,
                'description': f"⚠️ Warning: Cannot remove {symbol} - not found in portfolio"
            })
    
    # Check if we found any potential change instructions
    if not suggestions and any(kw in recommendation_text.lower() for kw in ['buy', 'sell', 'add', 'remove', 'portfolio']):
        suggestions.append({
            'action': 'info',
            'description': "ℹ️ Agent made recommendations but no portfolio changes were automatically detected."
        })
    
    return suggestions

# Function to apply validated changes to portfolio
def apply_portfolio_changes(portfolio_df, approved_changes):
    """
    Applies the approved changes to the portfolio
    Returns: updated portfolio dataframe and list of changes made
    """
    changes_made = []
    modified = False
    
    # Create a copy of the portfolio to modify
    updated_portfolio = portfolio_df.copy()
    
    for change in approved_changes:
        action = change.get('action', '')
        
        if action == 'update':
            # Add shares to existing position
            idx = change['row_idx']
            shares = change['shares']
            price = change['price']
            symbol = change['symbol']
            
            # Update shares
            updated_portfolio.loc[idx, 'Shares'] += shares
            
            # Update average price if provided
            if price is not None and price > 0:
                old_shares = updated_portfolio.loc[idx, 'Shares'] - shares
                old_price = updated_portfolio.loc[idx, 'Entry Price']
                updated_portfolio.loc[idx, 'Entry Price'] = (old_shares * old_price + shares * price) / updated_portfolio.loc[idx, 'Shares']
            
            # Update notes
            updated_portfolio.loc[idx, 'Notes'] += f" | Added {shares} shares on {datetime.now().strftime('%Y-%m-%d')}"
            changes_made.append(change['description'])
            modified = True
            
        elif action == 'add':
            # Add new position
            new_position = pd.DataFrame({
                'Symbol': [change['symbol']],
                'Shares': [change['shares']],
                'Entry Price': [change['price']],
                'Date Added': [datetime.now().strftime("%Y-%m-%d")],
                'Notes': [f"Added by agent recommendation on {datetime.now().strftime('%Y-%m-%d')}"]
            })
            updated_portfolio = pd.concat([updated_portfolio, new_position], ignore_index=True)
            changes_made.append(change['description'])
            modified = True
            
        elif action == 'sell':
            # Sell shares from existing position
            idx = change['row_idx']
            shares = change['shares']
            symbol = change['symbol']
            
            # Update shares
            updated_portfolio.loc[idx, 'Shares'] -= shares
            updated_portfolio.loc[idx, 'Notes'] += f" | Sold {shares} shares on {datetime.now().strftime('%Y-%m-%d')}"
            changes_made.append(change['description'])
            modified = True
            
            # If shares reduced to 0, remove the row
            if updated_portfolio.loc[idx, 'Shares'] == 0:
                updated_portfolio = updated_portfolio.drop(idx).reset_index(drop=True)
                changes_made.append(f"Removed {symbol} from portfolio (all shares sold)")
                
        elif action == 'remove':
            # Remove position completely
            idx = change['row_idx']
            updated_portfolio = updated_portfolio.drop(idx).reset_index(drop=True)
            changes_made.append(change['description'])
            modified = True
            
        elif action in ['warning', 'info']:
            # Just add to the changes list for information
            changes_made.append(change['description'])
    
    return updated_portfolio, changes_made, modified

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
if 'pending_suggestions' not in st.session_state:
    st.session_state.pending_suggestions = []
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

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
            # Show instructions for editing
            st.info("For entering decimal shares: You can type the decimal directly, e.g. '10.5'")
            
            # Prepare data for editing
            edit_df = portfolio.copy()
            
            # Add a delete column
            edit_df['Delete'] = False
            
            # Don't include calculated Total Cost during editing
            if 'Total Cost' in edit_df.columns:
                edit_df = edit_df.drop('Total Cost', axis=1)
                
            # Convert shares to strings to allow more flexible editing
            edit_df['Shares'] = edit_df['Shares'].astype(str)
                
            # Create the editor with text fields and delete checkbox
            edited_df = st.data_editor(
                edit_df,
                num_rows="dynamic",
                key="portfolio_editor",
                column_config={
                    "Symbol": st.column_config.TextColumn("Symbol", help="Stock ticker symbol"),
                    "Shares": st.column_config.TextColumn("Shares", help="Number of shares (decimals allowed, e.g. 10.5)"),
                    "Entry Price": st.column_config.NumberColumn("Entry Price", help="Price per share at purchase", min_value=0, format="$%.2f"),
                    "Date Added": st.column_config.TextColumn("Date Added", help="When you acquired the shares (YYYY-MM-DD)"),
                    "Notes": st.column_config.TextColumn("Notes", help="Any additional information"),
                    "Delete": st.column_config.CheckboxColumn("Delete", help="Check to delete this position")
                },
                hide_index=True
            )
            
            if st.button("Save Changes"):
                try:
                    # Process the edited dataframe
                    processed_df = edited_df.copy()
                    
                    # Handle deletes
                    rows_to_delete = processed_df[processed_df['Delete'] == True]
                    if not rows_to_delete.empty:
                        for _, row in rows_to_delete.iterrows():
                            st.info(f"Deleted {row['Symbol']} from portfolio")
                        processed_df = processed_df[processed_df['Delete'] != True]
                    
                    # Remove the Delete column
                    processed_df = processed_df.drop('Delete', axis=1)
                    
                    # Convert Shares from text back to numeric, preserving decimals
                    processed_df['Shares'] = processed_df['Shares'].apply(
                        lambda x: float(str(x).replace(',', '')) if pd.notna(x) else 0
                    )
                    
                    # Ensure other columns have the right types
                    processed_df['Entry Price'] = pd.to_numeric(processed_df['Entry Price'], errors='coerce')
                    processed_df['Symbol'] = processed_df['Symbol'].astype(str)
                    processed_df['Notes'] = processed_df['Notes'].astype(str).replace('nan', '')
                    
                    # Update the portfolio
                    st.session_state.portfolio = processed_df
                    save_portfolio(processed_df)
                    st.success("Portfolio saved successfully!")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving portfolio: {str(e)}")
                    st.info("Please ensure all data is in the correct format before saving.")
        else:
            # Display as regular table with calculated values
            try:
                # For display, convert dates if needed
                display_portfolio = portfolio.copy()
                
                st.dataframe(
                    display_portfolio,
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol"),
                        "Shares": st.column_config.NumberColumn("Shares", format="%.4f"),  # Support 4 decimal places
                        "Entry Price": st.column_config.NumberColumn("Entry Price", format="$%.2f"),
                        "Total Cost": st.column_config.NumberColumn("Total Cost", format="$%.2f"),
                        "Date Added": st.column_config.TextColumn("Date Added"),
                        "Notes": st.column_config.TextColumn("Notes")
                    },
                    hide_index=True
                )
            except Exception as e:
                st.error(f"Error displaying portfolio: {str(e)}")
                # Simple fallback display
                st.dataframe(portfolio, hide_index=True)
        
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
                    st.rerun()
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
        new_shares = cols[1].text_input("Shares", "1", help="Number of shares (you can use commas and decimals, e.g. 1,000.5)")
        new_price = cols[2].number_input("Entry Price ($)", min_value=0.01, step=0.01)
        new_date = cols[3].date_input("Date Added")
        new_notes = cols[4].text_input("Notes")
        
        # Convert shares to numeric, handling commas
        try:
            shares_value = float(new_shares.replace(',', ''))
        except ValueError:
            shares_value = 0
            
        submitted = st.form_submit_button("Add Position")
        
        if submitted and new_symbol and shares_value > 0 and new_price > 0:
            # Create new row with proper types
            new_position = pd.DataFrame({
                'Symbol': [new_symbol],
                'Shares': [shares_value],
                'Entry Price': [float(new_price)],
                'Date Added': [new_date.strftime("%Y-%m-%d")],
                'Notes': [str(new_notes)]
            })
            
            # Append to portfolio
            st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_position], ignore_index=True)
            if save_portfolio(st.session_state.portfolio):
                st.success(f"Added {new_shares.replace(',', '')} shares of {new_symbol} to your portfolio!")
                time.sleep(1)
                st.rerun()

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
    
    allow_updates = st.checkbox("Allow the agent to suggest portfolio updates", value=True,
                                help="When checked, the agent can suggest changes to your portfolio based on its recommendations")
    
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
            
            # Parse portfolio suggestions if allowed
            if allow_updates and include_portfolio:
                suggestions = parse_portfolio_recommendations(
                    st.session_state.portfolio, result
                )
                
                if suggestions:
                    st.session_state.pending_suggestions = suggestions
            
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
        
        # Show portfolio update validation if there are pending suggestions
        if st.session_state.pending_suggestions:
            with st.expander("Portfolio Suggestions", expanded=True):
                st.subheader("Agent Suggested Portfolio Changes")
                st.write("The trading agent has suggested the following changes to your portfolio. Please review and approve the changes you want to apply.")
                
                # Show all suggestions with checkboxes for approval
                approved_changes = []
                
                for i, suggestion in enumerate(st.session_state.pending_suggestions):
                    action = suggestion.get('action', '')
                    if action not in ['warning', 'info']:
                        # Add checkbox for actionable suggestions
                        if st.checkbox(f"{suggestion['description']}", key=f"suggestion_{i}", value=True):
                            approved_changes.append(suggestion)
                    else:
                        # Just display warnings and info
                        st.info(suggestion['description'])
                
                # Add buttons to apply changes or reject all
                col1, col2 = st.columns(2)
                
                if col1.button("Apply Selected Changes"):
                    if approved_changes:
                        # Apply the approved changes
                        updated_portfolio, changes_made, modified = apply_portfolio_changes(
                            st.session_state.portfolio, approved_changes
                        )
                        
                        if modified:
                            # Save changes to history
                            st.session_state.changes_made = changes_made + st.session_state.changes_made
                            
                            # Update portfolio
                            st.session_state.portfolio = updated_portfolio
                            save_portfolio(updated_portfolio)
                            
                            # Clear pending suggestions
                            st.session_state.pending_suggestions = []
                            st.success("Portfolio updated with selected changes!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.info("No changes were made to the portfolio.")
                    else:
                        st.info("No changes were selected for approval.")
                
                if col2.button("Reject All Suggestions"):
                    # Clear pending suggestions
                    st.session_state.pending_suggestions = []
                    st.info("All suggestions rejected.")
                    st.rerun()
        
        # Show portfolio update summary if changes were made
        elif st.session_state.changes_made:
            with st.expander("Portfolio Updates", expanded=True):
                st.subheader("Changes Made to Your Portfolio")
                for change in st.session_state.changes_made[:5]:  # Show only the most recent changes
                    st.markdown(f"- {change}")
                
                if st.button("View All Changes in Portfolio Tab"):
                    st.rerun()

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
                        st.rerun()
    else:
        st.info("No query history yet. Try asking a question in the Analysis tab!")

# Footer
st.markdown("---")
st.caption("Trading Portfolio Assistant | Data is stored locally on your computer")