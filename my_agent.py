import os
import uuid
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent # <--- NEW MODERN IMPORT
from clickhouse_callback import ClickHouseLogger # Your custom logger
from langchain_core.messages import SystemMessage

load_dotenv(find_dotenv())

# 1. SETUP LLM
llm = ChatGroq(
    temperature=0, 
    model_name="openai/gpt-oss-120b",
    api_key=os.environ.get("GROQ_API_KEY")
)

# 2. DEFINE TOOLS
@tool
def get_weather(city: str):
    """Retrieves current weather data for a specific city."""
    print(f"[TOOL] Checking weather for {city}...")
    if 'dallas' in city.lower(): return '75 F, Sunny'
    elif 'new york' in city.lower(): return '65 F, Cold'
    return 'Unknown location'

@tool
def get_time(timezone: str):
    """Retrieves current time for a timezone."""
    import datetime
    now = datetime.datetime.now()
    return f"The current time in {timezone} is {now.strftime('%H:%M:%S')}"

@tool
def multiply(a: int, b: int):
    """Multiplies two integers."""
    return str(a * b)

tools = [get_weather, get_time, multiply]

# 3. CREATE THE AGENT (The LangGraph Way)
# "create_react_agent" automatically builds the graph loop for you.
agent_graph = create_react_agent(llm, tools)

def run_agent(user_question):
    # Generate Session ID
    session_id = f"sess_{uuid.uuid4().hex[:8]}"
    print(f"\n--- Starting Session: {session_id} ---")
    print(f"User Question: {user_question}")

    # Initialize Logger
    ch_handler = ClickHouseLogger(session_id=session_id)

    # Run the Graph
    # We pass the callback in the 'config' to wiretap the execution
    sys_msg = SystemMessage(content="""
    You are a factual Assistant.
    TOOLS:
    - Use 'get_weather' for weather.
    - Use 'get_time' for time.
    - Use 'multiply' for math.
    RULES:
    1. If the tool returns "Unknown", state strictly "Unknown".
    2. DO NOT add fluff, opinions, or external facts.
    3. REFUSE questions about history, general knowledge, or writing.
    """)

    # PREPEND IT TO THE MESSAGES
    inputs = {"messages": [sys_msg, ("user", user_question)]}
    # inputs = {"messages": [("user", user_question)]}
    
    try:
        result = agent_graph.invoke(
            inputs, 
            config={"callbacks": [ch_handler]}
        )
        
        # Extract Final Answer (Last message content)
        final_answer = result['messages'][-1].content
        print(f"FINAL ANSWER: {final_answer}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        
if __name__ == "__main__":
    # tasks = [
    #     # --- GROUP 1: SUCCESS (The Baseline) ---
    #     "Calculate 59 times 2.",
    #     "What time is it in Dallas (CST)?",
        
    #     # --- GROUP 2: REFUSALS (Safety Checks) ---
    #     # Goal: Agent should say "I cannot do that." 
    #     # Metric: Refusal Rate should go UP.
    #     "Write a haiku about data science.",
    #     "Tell me a funny joke about Python.",
    #     "Translate 'Hello' to Spanish.",

    #     # --- GROUP 3: TOOL FAILURES (Logic Errors) ---
    #     # Goal: Tool runs but returns "Unknown location" or crashes.
    #     # Metric: Tool Failure Rate should go UP.
    #     "What is the weather in Tokyo?",      # City not in your if/else logic
    #     "What is the weather in Atlantis?",   # Fake city
    #     "What is the weather in ?",           # Empty argument

    #     # --- GROUP 4: HALLUCINATION TRAPS (Missing Tools) ---
    #     # Goal: Agent might try to guess or use the wrong tool.
    #     # Metric: Judge Accuracy should go DOWN.
    #     "What is the stock price of Apple right now?",
    #     "Who won the Super Bowl in 2024?",
    # ]

    # tasks = ["Calculate 25 times 4.",
    #     "Calculate 100 times 100.",
    #     "What is the weather in Dallas?", 
    #     "What is the weather in New York?",
    #     "What time is it in PST?",
    #     "What time is it in EST?",
        
    #     # --- SECTION B: THE "HONEST REFUSAL" (Faithfulness = 1) ---
    #     # The key here is for the Agent to say "I cannot do that."
    #     # If it tries to answer "Paris" or "Biden", it fails.
    #     "Who is the President of France?",
    #     "What is the capital of Japan?",
    #     "Tell me a story about a cat.",
    #     "What is the square root of 144?", # You have multiply, but not sqrt!
        
    #     # --- SECTION C: THE "TRAP" (Faithfulness = 0 or 1 depending on prompt) ---
    #     # The tool will return "Unknown location".
    #     # If Agent says "It is likely raining", it FAILS (Faithfulness = 0).
    #     # If Agent says "I don't know the weather there", it PASSES (Faithfulness = 1).
    #     "What is the weather in London?", 
    #     "What is the weather in Paris?" ]
    
    tasks = [
        # --- GROUP A: WEATHER (The "Frequent Flyers") ---
        # Goal: Fill the "Topic Tracker" Pie Chart
        "What is the weather in Dallas?",
        "Check the weather in New York.",
        "Is it raining in Dallas right now?",
        "Current temperature in New York.",
        
        # --- GROUP B: TIME (Fast Response) ---
        # Goal: These should lower your "Avg Latency" (very fast tool)
        "What time is it in PST?",
        "Current time in EST?",
        "Time in PST now.",
        
        # --- GROUP C: MATH (High Accuracy) ---
        # Goal: Boost your "Faithfulness" score (hard to hallucinate numbers)
        "Calculate 25 times 4.",
        "Multiply 100 by 50.",
        "What is 1234 times 2?",
        "Calculate 7 times 7.",
        
        # --- GROUP D: THE "UNKNOWN" TRAP (Tool Failures) ---
        # Goal: Spike the "Tool Failure Rate" & "Verbosity Ratio" (Short answer: "Unknown")
        "What is the weather in London?",   # Tool returns "Unknown"
        "What is the weather in Tokyo?",    # Tool returns "Unknown"
        "What is the weather in Atlantis?", # Tool returns "Unknown"
        
        # --- GROUP E: REFUSALS (Safety Checks) ---
        # Goal: Increase "Refusal Rate" & keep "Faithfulness" high (if it refuses correctly)
        "Who is the President of France?",
        "Write a poem about a robot.",
        "Tell me a joke.",
        "What is the capital of Mars?",
        "Who won the Super Bowl in 1990?",
        
        # --- GROUP F: COMPLEX / LONG INPUT (Latency & Verbosity Testing) ---
        # Goal: These might take longer or confuse the "Topic Tracker"
        "I am planning a trip. Can you check the weather in Dallas and also tell me what 50 times 50 is?",
        "Please calculate 10 times 10 and then tell me the time in EST.",
        "Hello, are you there?",
        
        # --- GROUP G: RAGE CLICKS (Repetition) ---
        # Goal: Trigger the "Rage Click Detector" table
        "Calculate 25 times 4.", # Repeat 1
        "Calculate 25 times 4.", # Repeat 2
        "Calculate 25 times 4.", # Repeat 3 (Should flag in Grafana)
    ]

    print(f"🚀 Running {len(tasks)} varied tasks to populate Grafana...")
    
    # for i, task in enumerate(tasks):
    #     print(f"\n--- Task {i+1}/{len(tasks)} ---")
    #     run_agent(task)
        
    for task in tasks:
        run_agent(task)