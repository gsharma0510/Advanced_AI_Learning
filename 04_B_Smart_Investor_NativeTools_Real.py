# A Smart Investor Agent that uses tools to fetch stock prices and news.
# This example demonstrates how to build an agent that can use tools natively with Google Gemini.
# Import necessary libraries
import os
import json
import yfinance as yf
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Load Google Gemini API key from environment variable
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


def get_news(ticker: str):
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

# Map string names to actual functions for execution
available_functions = {
    "get_stock_price": get_stock_price,
    "get_news": get_news
}

# ---2.  Create a list of the actual Python functions fore gemini use ---
gemini_tools = [get_stock_price, get_news]

# --- 3. THE AGENT LOOP ---

def run_native_agent(query):
    print(f"🚀 Starting Native Agent Research for: {query}\n")
    
    # Start the history with the user's question
    messages = [
        types.Content(role="user", parts=[types.Part.from_text(text=query)])
    ]
    
    for turn in range(5):
        print(f"--- Turn {turn + 1} ---")
        
        response = client_gemini.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=messages,
            config=types.GenerateContentConfig(                
                tools=gemini_tools,
                temperature=0,
                system_instruction="You are a helpful financial assistant. Use tools to answer questions."
            )
        )
        
        # Extract the content from Gemini's response
        response_content = response.candidates[0].content
        msg_parts = response_content.parts
        
        # Detect any tool calls
        function_calls = [p.function_call for p in msg_parts if p.function_call]

        if function_calls:
            print(f"🤖 Agent wants to call {len(function_calls)} tool(s)...")
            
            # Step 1: Add the Model's Tool Call request to history
            messages.append(response_content)

            # Step 2: Execute each tool call
            for fc in function_calls:
                function_name = fc.name
                function_args = fc.args 
                
                print(f"⚙️ Calling: {function_name}({function_args})")
                
                function_to_call = available_functions[function_name]
                observation = function_to_call(**function_args)
                
                # Step 3: Add the Tool Result back to history
                messages.append(
                    types.Content(
                        role="user", 
                        parts=[types.Part.from_function_response(
                            name=function_name,
                            response={'result': observation}
                        )]
                    )
                )
                print(f"👀 Observation: {observation}\n")
        
        else:
            # If no tool calls, this is the final answer
            # Using getattr to handle potential empty text safely
            final_text = getattr(response, 'text', "I couldn't generate a final response.")
            print(f"✅ Final Answer: {final_text}")
            break


if __name__ == "__main__":
    print("--- 🤖 Financial Research Agent (Native Tooling) ---")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        # Get user input from the terminal
        user_input = input("💡 What would you like to research? (e.g., AAPL price and news): ")
        
        # Check if the user wants to stop
        if user_input.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye! Happy investing.")
            break
        
        # Skip empty inputs
        if not user_input.strip():
            print("⚠️  Please enter a valid query.")
            continue
            
        # Run the agent with the user's specific question
        try:
            run_native_agent(user_input)
            print("\n" + "="*50 + "\n") # Visual separator between queries
        except Exception as e:
            print(f"❌ An error occurred: {e}")