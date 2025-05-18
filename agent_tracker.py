import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
if not os.path.exists("logs"):
    os.makedirs("logs")

# Add file handler for logging
file_handler = logging.FileHandler("logs/agent_usage_tracker.log")
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

class AgentTracker:
    """
    Tracks agent and tool usage in a trading agent system
    Based on examination of the OpenAI Agent SDK documentation
    """
    
    def __init__(self):
        """Initialize the agent tracker"""
        self.reset()
    
    def reset(self):
        """Reset tracking state for a new session"""
        # Store tracked data
        self.current_agent = None
        self.agent_usage = {}
        self.tool_usage = {}
        self.execution_log = []
        self.start_time = datetime.now()
        self.end_time = None
        
        # Expected agents
        self.expected_agents = [
            "Macro Environment Analyst",
            "News Sentiment Analyst",
            "Fundamental Analyst", 
            "Market Data Analyst",
            "Risk Management Analyst"
        ]
        
        # Expected tools by agent (minimum requirements)
        self.expected_tools = {
            "Macro Environment Analyst": [
                "get_economic_indicators_us",
                "get_market_regime_analysis_current",
                "get_central_bank_analysis_fed",
                "get_sector_rotation_analysis_us",
                "analyze_geopolitical_risks_detailed"
            ],
            "News Sentiment Analyst": [
                "search_financial_news",
                "monitor_economic_news",
                "analyze_company_news_impact",
                "track_social_sentiment"
            ],
            "Fundamental Analyst": [
                "get_financial_statements",
                "calculate_financial_ratios",
                "analyze_earnings_reports",
                "run_valuation_model"
            ],
            "Market Data Analyst": [
                "get_historical_prices",
                "calculate_technical_indicators",
                "analyze_market_structure",
                "analyze_volume_profile",
                "assess_liquidity_conditions"
            ],
            "Risk Management Analyst": [
                "calculate_position_size",
                "analyze_portfolio_risk",
                "calculate_stop_loss",
                "analyze_correlation_risk",
                "optimize_portfolio_allocation"
            ]
        }
        
        logger.info("Agent tracker reset")
    
    def track_agent(self, agent_info):
        """
        Track the usage of an agent
        
        Args:
            agent_info: Any information about the agent (name, object, etc.)
        """
        # Try to extract agent name
        agent_name = self._extract_agent_name(agent_info)
        
        # Record usage
        if agent_name not in self.agent_usage:
            self.agent_usage[agent_name] = 0
        
        self.agent_usage[agent_name] += 1
        self.current_agent = agent_name
        
        # Record in execution log
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_handoff",
            "agent": agent_name,
            "count": self.agent_usage[agent_name]
        })
        
        logger.info(f"Agent used: {agent_name} (total: {self.agent_usage[agent_name]})")
    
    def track_tool(self, tool_name, agent_name=None):
        """
        Track the usage of a tool
        
        Args:
            tool_name: Name of the tool being used
            agent_name: Name of the agent using the tool (optional)
        """
        # Use current agent if no agent name provided
        if agent_name is None:
            agent_name = self.current_agent or "Unknown Agent"
        
        # Initialize agent in tool usage if needed
        if agent_name not in self.tool_usage:
            self.tool_usage[agent_name] = {}
        
        # Initialize tool count if needed
        if tool_name not in self.tool_usage[agent_name]:
            self.tool_usage[agent_name][tool_name] = 0
        
        # Increment tool usage count
        self.tool_usage[agent_name][tool_name] += 1
        
        # Record in execution log
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "type": "tool_usage",
            "agent": agent_name,
            "tool": tool_name,
            "count": self.tool_usage[agent_name][tool_name]
        })
        
        logger.info(f"Tool used: {tool_name} by {agent_name} (total: {self.tool_usage[agent_name][tool_name]})")
    
    def end_session(self):
        """End the tracking session and return a summary"""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Tracking session ended after {duration:.2f} seconds")
        
        return self.get_summary()
    
    def get_summary(self):
        """Generate a summary of agent and tool usage with compliance assessment"""
        # Check which expected agents were used
        missing_agents = []
        for agent in self.expected_agents:
            if agent not in self.agent_usage:
                missing_agents.append(agent)
        
        # Check which agents didn't use enough tools
        agents_missing_tools = []
        for agent in self.expected_agents:
            if agent in self.tool_usage:
                if len(self.tool_usage[agent]) < 2:  # Each agent should use at least 2 tools
                    agents_missing_tools.append(agent)
            elif agent in self.agent_usage:  # Agent was used but didn't use any tools
                agents_missing_tools.append(agent)
        
        # Check if agents were used in the correct order
        actual_order = []
        for log_entry in self.execution_log:
            if log_entry["type"] == "agent_handoff" and log_entry["agent"] in self.expected_agents:
                if log_entry["agent"] not in actual_order:
                    actual_order.append(log_entry["agent"])
        
        # Compare with expected order
        correct_order = True
        if len(actual_order) != len(self.expected_agents):
            correct_order = False
        else:
            for i, agent in enumerate(actual_order):
                if i >= len(self.expected_agents) or agent != self.expected_agents[i]:
                    correct_order = False
                    break
        
        # Calculate compliance percentages
        agent_compliance = 0
        if self.expected_agents:
            agent_compliance = (len(self.expected_agents) - len(missing_agents)) / len(self.expected_agents) * 100
        
        tool_compliance = 0
        if self.expected_agents:
            tools_compliant_count = 0
            for agent in self.expected_agents:
                if agent in self.tool_usage and len(self.tool_usage[agent]) >= 2:
                    tools_compliant_count += 1
            tool_compliance = tools_compliant_count / len(self.expected_agents) * 100
        
        # Generate most used tools by agent
        top_tools_by_agent = {}
        for agent, tools in self.tool_usage.items():
            if tools:
                sorted_tools = sorted(tools.items(), key=lambda x: x[1], reverse=True)
                top_tools_by_agent[agent] = [
                    {"tool": tool, "count": count} 
                    for tool, count in sorted_tools[:3]  # Top 3 tools
                ]
        
        # Define compliance status
        fully_compliant = len(missing_agents) == 0 and len(agents_missing_tools) == 0 and correct_order
        compliance_status = "COMPLIANT" if fully_compliant else "NON-COMPLIANT"
        
        # Generate summary
        return {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None,
            "agent_usage": self.agent_usage,
            "tool_usage": self.tool_usage,
            "agent_compliance_pct": round(agent_compliance, 1),
            "tool_compliance_pct": round(tool_compliance, 1),
            "all_agents_used": len(missing_agents) == 0,
            "missing_agents": missing_agents,
            "agents_missing_tools": agents_missing_tools,
            "correct_agent_order": correct_order,
            "actual_agent_sequence": actual_order,
            "top_tools_by_agent": top_tools_by_agent,
            "compliance_status": compliance_status
        }
    
    def save_report(self, query=None, filename=None):
        """
        Save a detailed report of the tracking session
        
        Args:
            query: The original query that was processed (optional)
            filename: Custom filename for the report (optional)
            
        Returns:
            str: Path to the saved report file
        """
        try:
            # Generate default filename if none provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs/agent_usage_report_{timestamp}.json"
            
            # Generate report data
            summary = self.get_summary()
            
            report = {
                "timestamp": datetime.now().isoformat(),
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "query": query,
                "agent_usage": self.agent_usage,
                "tool_usage": self.tool_usage,
                "execution_log": self.execution_log,
                "summary": summary
            }
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved agent usage report to {filename}")
            return filename
        
        except Exception as e:
            logger.error(f"Error saving report: {str(e)}")
            return None
    
    def _extract_agent_name(self, agent_info):
        """
        Extract agent name from various types of input
        
        Args:
            agent_info: Agent information in various formats
            
        Returns:
            str: Extracted agent name
        """
        # Handle different types of agent information
        # if hasattr(agent_info, 'name'):
        #     return agent_info.name
        if isinstance(agent_info, str):
            return agent_info
        elif isinstance(agent_info, dict) and 'name' in agent_info:
            return agent_info['name']
        else:
            return str(agent_info)

# Create global tracker instance
tracker = AgentTracker()

# Function to wrap Runner.run to track agent usage
def create_tracking_wrapper(original_run):
    """
    Create a wrapper around Runner.run that tracks agent usage
    
    Args:
        original_run: The original Runner.run function
        
    Returns:
        function: A wrapped version of Runner.run
    """
    async def wrapped_run(starting_agent, input, **kwargs):
        """Wrapped version of Runner.run that tracks agent usage"""
        # Track the starting agent
        tracker.track_agent(starting_agent)
        
        # Call the original function
        result = await original_run(starting_agent, input, **kwargs)
        
        # Save report at the end of a query
        if hasattr(starting_agent, 'name') and starting_agent.name == "Trading Orchestrator":
            query_text = input if isinstance(input, str) else "Complex input"
            tracker.save_report(query=query_text)
        
        return result
    
    return wrapped_run

# Function to wrap function_tool to track tool usage
def create_tool_tracking_wrapper(original_function_tool):
    """
    Create a wrapper around function_tool that tracks tool usage
    
    Args:
        original_function_tool: The original function_tool decorator
        
    Returns:
        function: A wrapped version of function_tool
    """
    def wrapped_function_tool(func=None, **kwargs):
        """Wrapped version of function_tool"""
        # Get the actual decorator
        decorator = original_function_tool(func, **kwargs) if func is not None else original_function_tool(**kwargs)
        
        # If called without arguments
        if func is not None:
            tool_name = kwargs.get('name_override', func.__name__)
            
            async def wrapper(*args, **kwds):
                # Track the tool usage
                tracker.track_tool(tool_name)
                
                # Call the original function
                return await decorator(*args, **kwds)
            
            return wrapper
        
        # If called with arguments
        def inner_decorator(func):
            tool_name = kwargs.get('name_override', func.__name__)
            decorated = decorator(func)
            
            async def wrapper(*args, **kwds):
                # Track the tool usage
                tracker.track_tool(tool_name)
                
                # Call the decorated function
                return await decorated(*args, **kwds)
            
            return wrapper
        
        return inner_decorator
    
    return wrapped_function_tool

def get_compliance_report(with_recommendations=True):
    """
    Get a detailed compliance report for the tracking session
    
    Args:
        with_recommendations: Whether to include recommendations for improvement
        
    Returns:
        dict: Compliance report with recommendations
    """
    # Get basic summary
    summary = tracker.get_summary()
    
    # Generate recommendations if requested
    recommendations = []
    if with_recommendations:
        if summary["missing_agents"]:
            recommendations.append(f"Make sure all required agents are used: {', '.join(summary['missing_agents'])}")
        
        if summary["agents_missing_tools"]:
            recommendations.append(f"Ensure each agent uses at least 2 tools: {', '.join(summary['agents_missing_tools'])}")
        
        if not summary["correct_agent_order"]:
            recommendations.append(f"Use agents in the correct order: {' -> '.join(tracker.expected_agents)}")
    
    # Add recommendations to the report
    enhanced_report = {**summary, "recommendations": recommendations}
    
    return enhanced_report

def save_compliance_report(filename=None):
    """
    Save a compliance report to a file
    
    Args:
        filename: Custom filename for the report (optional)
        
    Returns:
        str: Path to the saved report file
    """
    try:
        # Generate default filename if none provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"logs/compliance_report_{timestamp}.json"
        
        # Generate report
        report = get_compliance_report(with_recommendations=True)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved compliance report to {filename}")
        return filename
    
    except Exception as e:
        logger.error(f"Error saving compliance report: {str(e)}")
        return None

def reset_tracker():
    """Reset the tracker for a new session"""
    tracker.reset()
    logger.info("Tracker reset")
    return True