import os
import asyncio
import random  # New import for generating random seeds
from google import genai
from google.genai import types # Helper for configuration
from dotenv import load_dotenv
from colorama import Fore, Style, init

init(autoreset=True)
load_dotenv()

client_gemini = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 1. THE WORKER ---
async def ask_agent(name, prompt, color):
    print(f"{color}🤖 {name} is thinking...{Style.RESET_ALL}")
    
    # Generate a unique random seed for THIS specific agent call
    agent_seed = random.randint(1, 10000)
    
    response = await client_gemini.aio.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.8,
            seed=agent_seed,          # Forces different "paths" for each agent
            max_output_tokens=200     # Allows for ~5-8 lines of text
        )
    )
    
    content = response.text
    print(f"{color}✅ {name} finished (Seed: {agent_seed}).{Style.RESET_ALL}")
    return f"{name}: {content}"

# --- 2. THE ORCHESTRATOR ---
async def run_consensus():
    # ASK FOR USER INPUT
    topic = input(f"{Fore.WHITE}Enter a topic for the agents to discuss: {Style.RESET_ALL}")
    print(f"\nProcessing Topic: {topic}\n")
    
    # UPDATED PROMPT: Asking for more detail
    prompt = f"Provide a concise but detailed 4-5 line paragraph about: {topic}"
    
    results = await asyncio.gather(
        ask_agent("Agent A", prompt, Fore.YELLOW),
        ask_agent("Agent B", prompt, Fore.CYAN),
        ask_agent("Agent C", prompt, Fore.MAGENTA)
    )
    
    print(f"\n{Fore.WHITE}--- AGGREGATING RESULTS ---{Style.RESET_ALL}")
    combined_text = "\n\n".join(results)
    print(combined_text)
    
    print(f"\n{Fore.GREEN}--- FINAL JUDGMENT ---{Style.RESET_ALL}")
    final_verdict = await client_gemini.aio.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=f"Synthesize these 3 viewpoints into one definitive 5-line summary:\n\n{combined_text}",
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=300)
    )
    print(final_verdict.text)

if __name__ == "__main__":
    asyncio.run(run_consensus())