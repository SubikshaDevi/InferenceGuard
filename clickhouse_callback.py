import clickhouse_connect
import datetime
import uuid
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from typing import Dict, Any, List
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
# --- CONFIGURATION ---
CLICKHOUSE_HOST = os.environ.get("CLICKHOUSE_HOST")
CLICKHOUSE_PORT = int(os.environ.get("CLICKHOUSE_PORT"))
CLICKHOUSE_USER = os.environ.get("CLICKHOUSE_USER")
CLICKHOUSE_PASSWORD = os.environ.get("CLICKHOUSE_PASSWORD")

class ClickHouseLogger(BaseCallbackHandler):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = clickhouse_connect.get_client(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            username=CLICKHOUSE_USER,  # Added
            password=CLICKHOUSE_PASSWORD, # Added
            secure=False # Set to True if using Cloud/HTTPS
        )
        # Ensure Table Exists (Run this once or in setup)
        self.client.command("""
        CREATE TABLE IF NOT EXISTS agent_traces (
            timestamp DateTime64(3),
            session_id String,
            event_type Enum('user_input', 'tool_start', 'tool_end', 'llm_end', 'error'),
            content String,
            tool_name String,
            latency_ms UInt32
        ) ENGINE = MergeTree()
        ORDER BY (session_id, timestamp)
        """)

    def _insert_log(self, event_type, content, tool_name="", latency_ms=0):
        """Helper to push row to ClickHouse"""
        row = [
            datetime.datetime.now(),
            self.session_id,
            event_type,
            str(content), # Ensure string format for DB
            tool_name,
            latency_ms
        ]
        # Insert a single row (In prod, you might batch this)
        self.client.insert('agent_traces', [row], column_names=[
            'timestamp', 'session_id', 'event_type', 'content', 'tool_name', 'latency_ms'
        ])

    # --- EVENT HOOKS ---
    
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs):
        """Captures the User Input"""
        # Usually the input is inside a key like 'input' or 'chat_history'
        user_input = inputs.get("input", str(inputs))
        self._insert_log("user_input", user_input)

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs):
        """Captures when the Agent calls a tool"""
        tool_name = serialized.get("name", "unknown")
        self._insert_log("tool_start", input_str, tool_name=tool_name)

    def on_tool_end(self, output: str, **kwargs):
        """Captures what the tool returned"""
        # Note: We don't easily get tool_name here without state, 
        # but for simple traces, just logging the output is enough.
        self._insert_log("tool_end", output)

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Captures the Final Answer (or intermediate thought)"""
        # The LLM output is nested in the response object
        text_response = response.generations[0][0].text
        self._insert_log("llm_end", text_response)

    def on_chain_error(self, error: BaseException, **kwargs):
        """Captures Crashes"""
        self._insert_log("error", str(error))