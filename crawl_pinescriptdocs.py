import os
import sys
import psutil
import asyncio
import requests
import time
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timezone
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Ensure the API key is loaded from the .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(
    os.getenv("SUPABASE_URL1"),
    os.getenv("SUPABASE_SERVICE_KEY1")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

# Global RateLimiter for GROQ API calls
class RateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = 0
        self.start_time = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.start_time
            if elapsed > self.period:
                # Reset the counter once the period has passed
                self.calls = 0
                self.start_time = now
            if self.calls >= self.max_calls:
                # Wait for the remaining time in the period
                wait_time = self.period - elapsed
                await asyncio.sleep(wait_time)
                self.calls = 0
                self.start_time = time.monotonic()
            self.calls += 1

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass

# Create a global GROQ rate limiter to allow a maximum of 30 calls per 60 seconds.
global_groq_rate_limiter = RateLimiter(max_calls=30, period=60)

def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break
        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:
                end = start + last_break
        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:
                end = start + last_period + 1
        # Extract and clean chunk
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move start position for next chunk
        start = max(start + 1, end)
    return chunks

# Constants for rate limiting (adjust as needed)
GROQ_API_DELAY = 3  # per-request delay (in addition to the global rate limiter)
GOOGLE_API_DELAY = 4  # Rate limiting delay between Google API calls

# Groq TPM Limit
GROQ_TPM_LIMIT = 6000

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using Groq."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
You MUST return ONLY a valid JSON object with 'title' and 'summary' keys.
Do not include any explanations, thinking steps, or markdown formatting.
Do not wrap the response in ```json``` tags.
Bad example:
<think>This is a document about...</think>
{"title": "Example", "summary": "Text"}

Good example:
{"title": "Example Title", "summary": "Concise summary of the content"}

For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
For the summary: Create a concise summary of the main points in this chunk."""
    try:
        # Initialize the model with the correct parameters
        model = ChatGroq(model="gemma2-9b-it", groq_api_key=GROQ_API_KEY)
        messages = [
            ("system", system_prompt),
            ("human", f"URL: {url}\n\nContent:\n{chunk[:1000]}...")
        ]
        # Use the global GROQ rate limiter
        async with global_groq_rate_limiter:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, model.invoke, messages)
        # Clean up the response to ensure it's valid JSON
        content = response.content.strip()
        content = content.replace('```json', '').replace('```', '').strip()
        try:
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                content = content[json_start:json_end]
        except:
            pass
        result = json.loads(content)
        if not all(key in result for key in ['title', 'summary']):
            raise ValueError("Missing required keys in JSON response")
        await asyncio.sleep(GROQ_API_DELAY)  # Additional per-request delay
        return result
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {"title": "Error processing title", "summary": "Error processing summary"}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from gemini with retry logic."""
    max_retries = 3
    base_delay = 2  # seconds
    for attempt in range(max_retries):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)
            vector = embeddings.embed_query(text)
            await asyncio.sleep(GOOGLE_API_DELAY)  # Rate limiting delay
            return vector
        except Exception as e:
            print(f"Attempt {attempt + 1} failed to get embedding: {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                print("Max retries reached. Returning zero vector.")
                return [0] * 768  # Return zero vector on final error

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    extracted = await get_title_and_summary(chunk, url)
    embedding = await get_embedding(chunk)
    metadata = {
        "source": "pydantic_ai_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    chunks = chunk_text(markdown)
    tasks = [process_chunk(chunk, i, url) for i, chunk in enumerate(chunks)]
    processed_chunks = await asyncio.gather(*tasks)
    insert_tasks = [insert_chunk(chunk) for chunk in processed_chunks]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 3):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(url=url, config=crawl_config, session_id="session1")
                if result.success:
                    print(f"Successfully crawled: {url}")
                    # Uncomment the following line to process and store the document
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

def get_page_urls():
    """
    Extract all URLs from a given webpage.
    
    Args:
        url (str): The URL of the webpage to extract links from
        
    Returns:
        list: A list of unique, absolute URLs found on the page
    """
    url = "https://www.tradingview.com/pine-script-docs/welcome/" 
    try:
        parsed = urlparse(url)
        if not all([parsed.scheme, parsed.netloc]):
            print("Invalid URL provided")
            return []
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        urls = set()
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(url, href)
            if absolute_url.startswith(('http://', 'https://')):
                urls.add(absolute_url)
                print(f"Found URL: {absolute_url}")
        return sorted(list(urls))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the page: {e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

async def main():
    urls = get_page_urls()
    if not urls:
        print("No URLs found to crawl")
        return
    print(f"Found {len(urls)} URLs to crawl")
    await crawl_parallel(urls)

if __name__ == "__main__":
    asyncio.run(main())