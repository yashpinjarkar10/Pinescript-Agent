from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st
import json
import datetime
import logfire
from supabase import Client
from pydantic_ai.models.gemini import GeminiModel
from supabase import create_client

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_agent import PineScript_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL1"),
    os.getenv("SUPABASE_SERVICE_KEY1")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='always')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts,
    tool calls, tool returns, etc.
    """
    # system-prompt
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    # user-prompt
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    # text
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    # Prepare dependencies for the agent
    deps = PydanticAIDeps(
        supabase=supabase,
        gemini=GeminiModel('gemini-1.5-flash', api_key=GOOGLE_API_KEY)
    )
    message_placeholder = st.empty()
    partial_text = ""
    try:
        # Run the agent in a stream
        async with PineScript_expert.run_stream(
            user_input,
            deps=deps,
            message_history=st.session_state.messages[:-1],  # pass entire conversation so far
        ) as result:
            # Render partial text as it arrives
            async for chunk in result.stream_text(delta=True):
                partial_text += chunk
                message_placeholder.markdown(partial_text)
    except Exception as e:
        # Instead of displaying the raw error, display a friendly message.
        message_placeholder.markdown("**Could you please ask again?**")
        st.error("Could you please ask again?")
        # Also, add the friendly message to the conversation history.
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content="Could you please ask again?")])
        )
        return

    # Now that the stream is finished, add new messages from this run
    filtered_messages = [
        msg for msg in result.new_messages()
        if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
    ]
    st.session_state.messages.extend(filtered_messages)
    st.session_state.messages.append(
        ModelResponse(parts=[TextPart(content=partial_text)])
    )


def store_user_query(user_query: str):
    """
    Stores the user's query in the 'user_queries' table.
    """
    try:
        result = supabase.table('user_queries').insert({
            'query': user_query,
            'timestamp': datetime.datetime.now().isoformat()
        }).execute()
        print("User query stored successfully.")
    except Exception as e:
        print(f"Error storing user query: {e}")

async def main():
    st.title("PineScript Agentic RAG")
    st.write("Ask any question about PineScript, the hidden truths of its beauty lie within.")

    # Initialize chat history in session state if not present
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation so far
    # Each message is either a ModelRequest or ModelResponse.
    # We iterate over their parts to decide how to display them.
    for msg in st.session_state.messages:
        if isinstance(msg, ModelRequest) or isinstance(msg, ModelResponse):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user
    user_input = st.chat_input("What questions do you have about PineScript?")

    if user_input:
        # We append a new request to the conversation explicitly
        store_user_query(user_input)
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )
        
        # Display user prompt in the UI
        with st.chat_message("user"):
            st.markdown(user_input)

        # Display the assistant's partial response while streaming
        with st.chat_message("assistant"):
            # Actually run the agent now, streaming the text
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())