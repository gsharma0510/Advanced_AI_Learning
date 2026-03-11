# A Smart Investor Agent that uses tools to fetch stock prices and news.
# This example demonstrates how to build an agent that can use tools natively.
# Import necessary libraries
import os
import json
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
"""
# Load OpenAI API key from environment variable 
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
"""

# Load Google Gemini API key from environment variable
client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 1. DEFINE TOOLS (The "Hands") ---
def get_stock_price(ticker: str):
    """Fetches the current stock price."""
    # Mock data - in prod this hits an API
    return json.dumps({"ticker": ticker, "price": 215.40, "currency": "USD"})

def get_news(ticker: str):
    """Searches for recent news about a company."""
    return json.dumps({"ticker": ticker, "news": "Earnings beat expectations. Analysts bullish."})

# We do not need tools schema for gemini as google-genai can directly use the Python function definitions and create the schema automatically
"""
# --- 2. DEFINE TOOL SCHEMAS (The "API Definition" for OpenAI) ---
# This JSON schema tells OpenAI exactly what functions are available
# It replaces the need for a complex System Prompt description
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a given ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker (e.g., AAPL)"}
                },
                "required": ["ticker"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get recent news for a given stock ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "The stock ticker (e.g., NVDA)"}
                },
                "required": ["ticker"]
            }
        }
    }
]
"""
# Map string names to actual functions for execution
available_functions = {
    "get_stock_price": get_stock_price,
    "get_news": get_news
}

# ---2.  Create a list of the actual Python functions fore gemini use ---
gemini_tools = [get_stock_price, get_news]

# --- 3. THE AGENT LOOP ---
def run_native_agent(query):
    print(f"User: {query}\n")
    # We define the system instruction in the config later, 
    # so we only start with the user's question here.
    messages = [
        types.Content(
            role="user", 
            parts=[types.Part.from_text(text=query)]
        )
    ]
    
    # Using Open AI
    """
    for turn in range(5): # Max 5 turns safety limit
        print(f"--- Turn {turn + 1} ---")
        
        # 1. Call LLM with Tools enabled
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=tools_schema,
            tool_choice="auto" # Let the model decide whether to use a tool or chat
        )
        
        msg = response.choices[0].message
        tool_calls = msg.tool_calls

        # 2. Check if the model wants to call a tool
        if tool_calls:
            print(f"🤖 Agent wants to call {len(tool_calls)} tool(s)...")
            
            # Important: Add the assistant's request to history
            messages.append(msg) 
            
            # 3. Execute Tools
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print(f"⚙️ Calling: {function_name}({function_args})")
                
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                # 4. Feed Output back to LLM
                # We must include the 'tool_call_id' so the LLM knows which request this answer belongs to
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
                print(f"👀 Observation: {function_response}")
        
        else:
            # No tool calls? The model has finished its task.
            print(f"✅ Final Answer: {msg.content}")
            break
    """
    # For Gemini
    for turn in range(5):
        print(f"--- Turn {turn + 1} ---")
        
        # 1. Call Gemini with Tools enabled
        # Notice we don't need a separate schema; we just pass the function list!
        response = client_gemini.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=messages,
            config=types.GenerateContentConfig(
                tools=gemini_tools,
                temperature=0,
                system_instruction="You are a helpful financial assistant. Use tools to answer questions."
            )
        )
        
        # In Gemini, we check if the response contains a function_call
        # We look at the first "part" of the message
        msg_parts = response.candidates[0].content.parts
        function_calls = [p.function_call for p in msg_parts if p.function_call]

        if function_calls:
            print(f"🤖 Agent wants to call {len(function_calls)} tool(s)...")
            
            # Important: Add Gemini's request to our message history
            messages.append(response.candidates[0].content)

            for fc in function_calls:
                # Gemini gives us the name and the dictionary of arguments automatically
                function_name = fc.name
                function_args = fc.args 
                
                print(f"⚙️ Calling: {function_name}({function_args})")
                
                # Execute the actual Python function
                function_to_call = available_functions[function_name]
                function_response = function_to_call(**function_args)
                
                # 4. Feed Output back to Gemini
                # We use a 'FunctionResponse' part to keep the history clean
                messages.append(
                    types.Content(
                        role="user", 
                        parts=[types.Part.from_function_response(
                            name=function_name,
                            response={'result': function_response}
                        )]
                    )
                )
                print(f"👀 Observation: {function_response}")
        
        else:
            # No function calls? We found the final text response.
            final_text = response.text
            print(f"✅ Final Answer: {final_text}")
            break

if __name__ == "__main__":
    # This query forces the agent to use BOTH tools
    run_native_agent("What is the price of NVDA and is there any good news?")