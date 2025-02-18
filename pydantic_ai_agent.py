from __future__ import annotations as _annotations
import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import asyncio
import os
from supabase import create_client
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.gemini import GeminiModel
from supabase import Client
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Load API keys and initialize Supabase client
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

supabase: Client = create_client(
    os.getenv("SUPABASE_URL1"),
    os.getenv("SUPABASE_SERVICE_KEY1")
)
logfire.configure(send_to_logfire='always')

@dataclass
class PydanticAIDeps:
    supabase: Client
    gemini: GeminiModel

# Updated system prompt explicitly instructs the agent to use the tools for documentation queries.
system_prompt = """
You are a PineScript expert with direct access to complete documentation through tool functions.
Whenever a user asks for documentation, examples, or strategies for PineScript, you MUST:
1. List all available documentation pages (using the tool "list_documentation_pages").
2. Retrieve detailed, relevant documentation (using the tool "retrieve_relevant_documentation").
Do not generate a direct answer from your internal data. 
If no documentation is found, clearly state so.
Then, combine the retrieved information to answer the query accurately.
IMPORTANT: When a user asks "what is your name" or similar, always respond with "PineScript expert"
"""

model = GeminiModel('gemini-1.5-flash', api_key=GOOGLE_API_KEY)
PineScript_expert = Agent(
    model,
    system_prompt=system_prompt,
    deps_type=PydanticAIDeps,
    retries=2
)

async def get_embedding(text: str) -> List[float]:
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
        vector = embeddings.embed_query(text)
        return vector
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 768

@PineScript_expert.tool
async def retrieve_relevant_documentation(ctx: RunContext[PydanticAIDeps], user_query: str) -> str:
    """
    Retrieve relevant documentation chunks based on the query.
    """
    try:
        print("DEBUG: retrieve_relevant_documentation tool function triggered.")
        # Log the user query
        log_result = ctx.deps.supabase.table('user_queries').insert({
            'query': user_query,
            'timestamp': datetime.datetime.now().isoformat()
        }).execute()
        error_info = getattr(log_result, "error", None)
        if error_info:
            print("Insert error:", error_info)
        else:
            print("Inserted user query successfully.")
        # Get query embedding and retrieve matched documents
        query_embedding = await get_embedding(user_query)
        result = ctx.deps.supabase.rpc(
            'match_site_pages',
            {
                'query_embedding': query_embedding,
                'match_count': 20,
                'filter': {'source': 'pydantic_ai_docs'}
            }
        ).execute()
        if not result.data:
            return "I couldn't find any relevant documentation for that query."
        formatted_chunks = []
        for doc in result.data:
            chunks_text = f"""
# {doc['title']}

{doc['content']}
"""
            formatted_chunks.append(chunks_text)
        return "\n\n---\n\n".join(formatted_chunks)
    except Exception as e:
        print(f"Error retrieving documentation: {e}")
        return f"An error occurred while retrieving the documentation: {str(e)}"

@PineScript_expert.tool
async def list_documentation_pages(ctx: RunContext[PydanticAIDeps]) -> List[str]:
    """
    List all available PineScript documentation pages.
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('url') \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .execute()
        if not result.data:
            return []
        urls = sorted(set(doc['url'] for doc in result.data))
        return urls
    except Exception as e:
        print(f"Error retrieving documentation pages: {e}")
        return []

@PineScript_expert.tool
async def get_page_content(ctx: RunContext[PydanticAIDeps], url: str) -> str:
    """
    Retrieve the full content of a specific documentation page.
    """
    try:
        result = ctx.deps.supabase.from_('site_pages') \
            .select('title, content, chunk_number') \
            .eq('url', url) \
            .eq('metadata->>source', 'pydantic_ai_docs') \
            .order('chunk_number') \
            .execute()
        if not result.data:
            return f"No content found for URL: {url}"
        page_title = result.data[0]['title'].split(' - ')[0]
        formatted_content = [f"# {page_title}\n"]
        for chunk in result.data:
            formatted_content.append(chunk['content'])
        return "\n\n".join(formatted_content)
    except Exception as e:
        print(f"Error retrieving page content: {e}")
        return f"Error retrieving page content: {str(e)}"