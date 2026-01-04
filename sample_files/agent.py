import base64
from groq import Groq
import os
from dotenv import load_dotenv, find_dotenv
import json
import datetime
import time
import uuid
from tools_def import TOOL_SYSTEM_PROMPT

load_dotenv(find_dotenv())

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

LOG_FILE = "agent_logs.jsonl"

def save_log(session_id, turn_number, event_type, content):
    '''
    Saves a single event to the log file
    Args:
        session_id: Unique ID for this specific user question run.
        turn_number: Which loop iteration are we on?
        event_type: 'input', 'decision', 'tool_output', 'final_answer', 'error'
        content: The actual data (prompt, tool name, result, etc.)
    '''
    entry = {
        'timestamp':datetime.datetime.now().isoformat(),
        'session_id':session_id,
        'turn_number':turn_number,
        'event_type':event_type,
        'content':content
    }
    
    with open(LOG_FILE,'a') as f:
        f.write(json.dumps(entry)+ '\n')
    
def get_weather(city):
    """
    Retrieves current weather data (temperature, condition) for a specific city.
    Args:
        city_name (str): The name of the city (e.g., 'Dallas', 'London').
    
    IMPORTANT: This tool CANNOT send emails, book flights, or do math. 
    It ONLY checks weather.
    """
    
    print(f'\n[Tool Execution] Connecting weather tool for {city}')
    
    if 'dallas' in city.lower():
        return '75 F, Sunny'
    elif 'new york' in city.lower():
        return '65 F, Cold'
    else:
        return 'Unknown location'
    
def get_time(timezone):
    """
    Docstring for get_time
    
    :param timezone: Description
    """
    now = datetime.datetime.now()
    return f"The current time in {timezone} is {now.strftime('%H:%M:%S')}"

def multiply(a, b):
    """
    Docstring for multiply
    
    :param a: int/ float
    :param b: int/float
    """
    return str(a*b)
    
system_prompt = '''
    You are a helpful AI assistant. You have access to THREE tools.

TOOLS:
1. function: get_weather
   description: Checks weather for a city.
   arguments: {"city": "string"}

2. function: get_time
   description: Checks current time for a timezone.
   arguments: {"timezone": "string"}

3. function: multiply
   description: Multiplies two numbers.
   arguments: {"a": "integer", "b": "integer"}

RULES:
- You must pick the ONE tool that best fits the NEXT step of the user's request.
- If a request requires multiple tools, solve them ONE BY ONE. 
- Do not guess answers. Use tools for everything.
- Format your answer as JSON.
- The JSON must follow this EXACT structure:
  {
    "tool": "tool_name",
    "arguments": { ... key-value pairs ... }
  }
- You may ONLY answer questions that use the provided tools.
- If the user asks for anything else (like essays, jokes, or general knowledge), you must REFUSE.
- Refusal format: {"tool": null, "message": "I cannot do that. I only handle weather, time, and math."}
'''

def run_agent(user_question):
    
    session_id = f'sess_{str(uuid.uuid4())[:8]}'
    
    messages = [
        {'role':'system', 'content':system_prompt},
        {'role':'user','content': user_question}
    ]
    
    print(f'---Starting agent task {user_question} (ID: {session_id}) ---')
    save_log(session_id, 0, 'user_input', user_question)
            
    turn_count = 0
    max_count = 5
    
    while turn_count < max_count:
        turn_count +=1
        try:
            response = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format = {'type':'json_object'},
                messages=messages
            )
        
            ai_message =response.choices[0].message.content
            print(f'[LLM MESSAGE] AI message: {ai_message}')

            data = json.loads(ai_message)
            
        except Exception as e:
            error_msg = f'LLM crash : {str(e)}'
            print(error_msg)
            save_log(session_id, turn_count, 'error', error_msg)
            continue
        
        tool_name = data['tool']
        tool_args = data.get("arguments", {})
            
        if tool_name:
            print(f"[DECISION] Agent wants to use: {tool_name}")
            save_log(session_id, turn_count, 'decision', {"tool":tool_name, "args": tool_args})
            
            observation = "Error: Unknown Tool"
            if tool_name == 'get_weather':
                city = tool_args.get("city")
                observation = get_weather(city)
                
            elif tool_name == 'get_time':
                tz = tool_args.get('timezone')
                observation = get_time(tz)
            
            elif tool_name == 'multiply':
                val_a = tool_args.get('a')
                val_b = tool_args.get('b')
                if val_a is not None and val_b is not None:
                    observation = multiply(val_a, val_b)
                else:
                    observation = "Error: Missing 'a' or 'b' arguments."
                
            # else:
            #     print(f'[ERROR] Agent tried to call unknown tool: {tool_name}')
                
                # break
            
            print(f'[OBSERVATION] {observation}, turn count: {turn_count}')
            save_log(session_id, turn_count,"tool_output", observation)
            
            messages.append({'role':'assistant', 'content':ai_message})
            messages.append({'role':'user', 'content':f"Tool returned {observation}"})
            
        else:
            final_msg = data.get('message', ai_message)
            print(f"\n[FINAL ANSWER]: {data.get('message', ai_message)}")
            save_log(session_id, turn_count,"final_answer", final_msg)
            break
        
# run_agent("What is the weather in Dallas right now?")
# run_agent('Calculate 50 times 12.')
# run_agent("What time is it in Dallas (CST)?")
# run_agent("What is the weather in Dallas and what is 5 times 5?")
# run_agent("Write me an essay about AI agents")

# 1. The Good (Should pass)
run_agent("What is the weather in Dallas?")

# 2. The Bad (Should trigger Refusal)
run_agent("Write a poem about rust.")

# 3. The Ugly (Should trigger Tool Error)
# We force an error by asking for a city that doesn't exist in your code
run_agent("What is the weather in Germany?")