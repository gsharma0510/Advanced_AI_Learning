# A Smart Investor Agent that uses tools to fetch stock prices and news, with clean terminal output.
# This example demonstrates how to build an agent that can use tools, and how to parse its actions using regex. 
# It also shows how to maintain a clean terminal output by only printing the new thought
# Import necessary libraries
import os
import re
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()

# Gemini Client Setup
client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 1. DEFINE TOOLS ---
def get_stock_price(ticker):
    """Simulates fetching a stock price."""
    # In a real app, you would call Yahoo Finance or AlphaVantage API here
    return f"{ticker} is currently trading at $215.40"

def get_news(query):
    """Simulates a web search for news."""
    return f"Recent news for {query}: Analysts predict strong growth due to AI demand."

# Tool Registry
tools = {
    "get_stock_price": get_stock_price,
    "get_news": get_news
}

# --- 2. SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are a Financial Research Agent.
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop, output your Final Answer.

Use Action: tool_name[input] to use a tool.
Available tools: get_stock_price, get_news.

Example:
Thought: I need to check the price.
Action: get_stock_price[AAPL]
PAUSE
"""

# --- 3. THE AGENT LOOP ---
def run_agent(user_query):
    print(f"🚀 Starting Research for: {user_query}\n")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    for turn in range(5):
        # We only want to join history for the LLM, not for our terminal display
        full_history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        response = client_gemini.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=full_history,
            config={'temperature': 0, 'stop_sequences': ["PAUSE"]}
        )
        agent_text = response.text
        
        # --- CLEAN PRINTING START ---
        # Only print the NEW thought/action from the agent
        print(f"--- Turn {turn + 1} ---")
        print(f"🤖 Agent:\n{agent_text.strip()}\n")
        # --- CLEAN PRINTING END ---

        messages.append({"role": "assistant", "content": agent_text})
        
        if "Final Answer:" in agent_text:
            print("✅ Task Complete!")
            return agent_text
            
        action_match = re.search(r"Action: (\w+)\[(.*)\]", agent_text)
        
        if action_match:
            tool_name, tool_input = action_match.groups()
            if tool_name in tools:
                print(f"⚙️ Executing {tool_name}...")
                observation = tools[tool_name](tool_input)
                print(f"👀 Observation: {observation}\n")
                
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                messages.append({"role": "user", "content": f"Error: Tool {tool_name} not found."})
        else:
            # If the model is just 'thinking' but didn't call a tool, nudge it
            messages.append({"role": "user", "content": "Continue with the next Thought or Action."})

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY in your .env file
    run_agent("Is Nvidia a good buy right now? Check price and news.")
    