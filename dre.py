import pandas as pd
import json
import os
import datetime
import clickhouse_connect
from groq import Groq
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer, util
import re
import requests

# Load Env
load_dotenv(find_dotenv())

# --- CONFIGURATION ---
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT"))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- CLIENTS ---
client = Groq(api_key=GROQ_API_KEY)
print("⏳ Loading Embedding Model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# --- 2. DEFINE GOLD STANDARD DATASET ---
# If the user asks X, the "Perfect" answer is Y.
GOLD_STANDARD = {
    "Calculate 25 times 4.": "100",
    "What is the weather in Dallas?": "75 F, Sunny",
    "Who is the President of France?": "I cannot answer that", # Refusal is correct
    "What is the weather in Atlantis?": "Unknown location"
}

try:
    ch_client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST, 
        port=CLICKHOUSE_PORT, 
        username=CLICKHOUSE_USER, 
        password=CLICKHOUSE_PASSWORD
    )
    # Ensure Evals Table Exists
    ch_client.command("""
    CREATE TABLE IF NOT EXISTS agent_evals (
        timestamp DateTime64(3),
        session_id String,
        metric_name String,
        score Float32,
        reason String
    ) ENGINE = MergeTree()
    ORDER BY (session_id, timestamp)
    """)
except Exception as e:
    print(f"ClickHouse Connection Error: {e}")
    exit()

def save_eval(session_id, metric_name, score):
    """Saves the score to ClickHouse"""
    row = [datetime.datetime.now(), session_id, metric_name, float(score), "Level 2 Check"]
    ch_client.insert('agent_evals', [row], column_names=['timestamp', 'session_id', 'metric_name', 'score', 'reason'])

def check_urls(text):
    """Returns 0 if any URL in the text is broken (404), else 1."""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    if not urls: return 1.0 # No URLs = Safe
    
    for url in urls:
        try:
            # Ping the URL (Head request is faster)
            r = requests.head(url, timeout=3)
            if r.status_code >= 400: return 0.0 # Broken Link
        except:
            return 0.0 # DNS/Connection Failure
    return 1.0

# --- THE JUDGE FUNCTION ---
def run_judge(metric_name, prompt, user_q, agent_ans, context=""):
    """
    Generic function to run a grading prompt.
    Returns: 1.0 (Positive/Yes) or 0.0 (Negative/No)
    """
    system_prompt = f"""
    You are an impartial AI QA Auditor.
    
    TASK: {prompt}
    
    DATA TO EVALUATE:
    User Question: "{user_q}"
    Agent Answer: "{agent_ans}"
    Context (Tool Outputs): "{context}"
    
    INSTRUCTIONS:
    - Analyze strictly based on the provided Data.
    - Output ONLY the digit '1' for YES or '0' for NO. 
    - Do not write any other words.
    """
    
    try:
        # UPDATED: Using the requested Llama 4 model
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", 
            messages=[{"role": "user", "content": system_prompt}],
            temperature=0
        )
        result = completion.choices[0].message.content.strip()
        # Parse result safely
        return 1.0 if '1' in result else 0.0
    except Exception as e:
        print(f"Judge Error ({metric_name}): {e}")
        return 0.0

def save_eval(session_id, metric_name, score):
    """Saves the score to ClickHouse"""
    row = [
        datetime.datetime.now(),
        session_id,
        metric_name,
        score,
        "Auto-graded by Llama-4-Scout"
    ]
    ch_client.insert('agent_evals', [row], column_names=[
        'timestamp', 'session_id', 'metric_name', 'score', 'reason'
    ])

# --- MAIN LOOP ---

print("\n--- 🕵️ STARTED INCREMENTAL EVALUATION ---\n")

# --- KEY CHANGE HERE ---
# Logic: Fetch sessions from traces ONLY IF they are NOT already in 'agent_evals'.
# This ensures we only grade new, ungraded sessions.
query = """
SELECT 
    session_id, 
    groupArray(event_type) as events, 
    groupArray(content) as contents
FROM agent_traces
WHERE session_id NOT IN (
    SELECT DISTINCT session_id FROM agent_evals 
    --WHERE metric_name = 'semantic_similarity'
)
-- Optional: Keep a time limit if your DB is huge (e.g., look back 7 days for ungraded work)
-- AND timestamp > now() - INTERVAL 7 DAY 
GROUP BY session_id
limit 10
"""

try:
    sessions = ch_client.query(query).result_rows
except Exception as e:
    print(f"Error fetching traces: {e}")
    sessions = []

if len(sessions) == 0:
    print("✅ No new sessions to grade. Everything is up to date!")
else:
    print(f"Found {len(sessions)} NEW sessions to evaluate...\n")

for sess in sessions:
    sess_id, events, contents = sess
    
    # 1. Reconstruct Data
    try:
        # Get User Question
        if 'user_input' in events:
            user_q = contents[events.index('user_input')]
        else:
            continue

        # Get Final Answer (Last text output)
        if 'llm_end' in events:
            # Find the LAST llm_end
            indices = [i for i, x in enumerate(events) if x == "llm_end"]
            agent_ans = contents[indices[-1]]
        elif 'tool_end' in events:
             indices = [i for i, x in enumerate(events) if x == "tool_end"]
             agent_ans = contents[indices[-1]]
        else:
            agent_ans = "No Answer"

        # Get Context (All tool outputs combined)
        tool_outputs = [str(c) for i, c in enumerate(contents) if events[i] == 'tool_end']
        context_str = " | ".join(tool_outputs)
        
    except ValueError:
        continue 

    print(f"-> Grading Session {sess_id[-8:]}...")
    
    # --- METRIC A: FAITHFULNESS ---
    # PROMPT: "Does the answer contain info NOT in context?"
    # YES (1) = Hallucination (Bad)
    # NO (0) = Faithful (Good)
    faith_prompt = (
        "Does the Agent Answer contain specific facts or numbers NOT found in the Context/Tool Outputs? "
        "Answer '1' if it Hallucinated (contains outside info). Answer '0' if it stayed Faithful (only used context)."
    )
    
    is_hallucination = run_judge("faithfulness", faith_prompt, user_q, agent_ans, context_str)
    
    # INVERT SCORE: If Hallucination is 1, Faithfulness is 0.
    faithfulness_score = 0.0 if is_hallucination == 1.0 else 1.0
    
    save_eval(sess_id, "faithfulness", faithfulness_score)

    # --- METRIC B: ANSWER RELEVANCE ---
    # PROMPT: "Does it answer the question?"
    # YES (1) = Relevant (Good)
    # NO (0) = Irrelevant (Bad)
    rel_prompt = (
        "Does the Agent Answer directly address the User Question? "
        "Answer '1' for Yes (Relevant). Answer '0' for No (Irrelevant)."
    )
    
    relevance_score = run_judge("answer_relevance", rel_prompt, user_q, agent_ans, context_str)
    
    save_eval(sess_id, "answer_relevance", relevance_score)
    
    # --- METRIC C: SEMANTIC SIMILARITY (The "Gold Standard" Check) ---
    # We only run this if we have a "Correct Answer" defined for this question.
    # Note: We use basic string matching for keys; in prod, use fuzzy matching.
    if user_q in GOLD_STANDARD:
        gold_answer = GOLD_STANDARD[user_q]
        
        # Compute Cosine Similarity between Agent Answer and Gold Answer
        embeddings = embed_model.encode([agent_ans, gold_answer])
        sim_score = util.cos_sim(embeddings[0], embeddings[1]).item()
        
        print(f"   Similarity: {sim_score:.2f} (Target: {gold_answer})")
        save_eval(sess_id, "semantic_similarity", sim_score)
    else:
        # If no gold standard exists, we skip this metric (or mark -1)
        pass

    # --- METRIC D: HALLUCINATION CHECK (Broken Link Detector) ---
    # Checks if the agent generated a fake URL.
    url_score = check_urls(agent_ans)
    if url_score == 0.0:
        print(f"   ⚠️ FOUND BROKEN URL in: {agent_ans}")
    save_eval(sess_id, "url_validity", url_score)

print("\n✅ Incremental Evaluation Complete!")