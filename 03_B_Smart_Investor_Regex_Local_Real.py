
import os
import re
from google import genai
import yfinance as yf
from dotenv import load_dotenv

# Load API Key from .env file
load_dotenv()
# Gemini Client Setup (if you want to use Gemini instead of OpenAI)
client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 1. DEFINE TOOLS (REAL DATA) ---
def get_stock_price(ticker: str):
    """Fetches the real-time stock price using Yahoo Finance."""
    print(f"📡 Accessing Yahoo Finance for {ticker} price...")
    try:
        stock = yf.Ticker(ticker)
        # We use fast_info for minimal latency
        info = stock.fast_info
        price = info['last_price']
        currency = info['currency']
        
        if price is None:
            return f"Could not find a price for {ticker}. Is the ticker symbol correct?"
            
        return f"{ticker} is currently trading at {price:.2f} {currency}"
    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"


def get_news(ticker):
    """Uses yf.Search for reliable headline and publisher retrieval."""
    print(f"🔍 Deep searching news for {ticker}...")
    try:
        # news_count=3 keeps the context window clean for Gemini
        search = yf.Search(ticker, news_count=3)
        news_list = search.news
        
        if not news_list:
            return f"No search results found for {ticker}."
            
        formatted_news = []
        for n in news_list:
            # yf.Search provides 'title', 'publisher', and 'link' reliably
            title = n.get('title', 'No Title Available')
            publisher = n.get('publisher', 'Unknown Source')
            link = n.get('link', 'No Link')
            
            # We provide the Link so Gemini knows it's a real source
            formatted_news.append(f"TITLE: {title}\nPUBLISHER: {publisher}\nLINK: {link}\n")
            
        return "Recent Financial Headlines:\n\n" + "\n".join(formatted_news)
    except Exception as e:
        return f"Error during search: {str(e)}"


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
    print(f"🚀 Researching: {user_query}\n")
    
    # We keep the raw messages in a list for our logic
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    # Turn loop (Safety limit of 5 turns)
    for turn in range(5):
        print(f"--- Turn {turn + 1} ---")
        
        # Prepare history for Gemini: Join all past messages into one prompt        
        full_history = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        
        # Call the Brain (Gemini)
        response = client_gemini.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=full_history,
            config={
                'temperature': 0,
                'stop_sequences': ["PAUSE", "Observation:"] # Stops the AI from making up its own facts
            }
        )
        
        agent_text = response.text.strip()
        
        # Print ONLY the new output to the terminal
        print(f"🤖 Agent thought:\n{agent_text}\n")
        
        # Add the agent's response to the real history
        messages.append({"role": "assistant", "content": agent_text})
        
        # 1. Check if the task is finished
        if "Final Answer:" in agent_text:
            print("✅ Task Complete!")
            return agent_text
            
        # 2. Try to parse an Action: tool_name[input]
        action_match = re.search(r"Action: (\w+)\[(.*)\]", agent_text)
        
        if action_match:
            tool_name = action_match.group(1)
            tool_input = action_match.group(2)
            
            if tool_name in tools:
                print(f"⚙️ Running tool: {tool_name}({tool_input})...")
                
                # EXECUTE THE ACTUAL TOOL
                observation = tools[tool_name](tool_input)
                
                print(f"👀 Observation: {observation}\n")
                
                # Feed the result back into the history for the next turn
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                error_msg = f"Error: Tool '{tool_name}' is not in my registry."
                print(f"❌ {error_msg}")
                messages.append({"role": "user", "content": error_msg})
        else:
            # If the model didn't call a tool and didn't give a final answer, nudge it
            nudge = "Please continue using a tool or provide your Final Answer."
            messages.append({"role": "user", "content": nudge})

# --- 4. EXECUTION ---
if __name__ == "__main__":
    # Ensure you have OPENAI_API_KEY in your .env file
    run_agent("Is Netflix a good stock to invest money in? Check price, news, and analyze the market sentiment.")
    