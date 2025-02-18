# PineScript Agentic RAG

PineScript Agentic RAG is a Retrieval-Augmented Generation (RAG) system built to help users interactively query and explore PineScript documentation. The project provides a Streamlit-based chat interface where users can ask questions about PineScript and receive dynamic, streaming responses powered by advanced AI models. The system leverages a custom PineScript expert agent that retrieves, processes, and summarizes documentation stored in a Supabase database.

The project is deployed on Hugging Face Spaces and can be accessed at:  
https://yashpinjarkar10-pinescript-agent.hf.space

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Documentation Crawler](#documentation-crawler)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project is designed to serve as a comprehensive tool for PineScript enthusiasts and developers. It integrates:
- A **Streamlit chat interface** that enables real-time conversation with a PineScript expert agent.
- A **RAG-based agent** that uses advanced language models (such as Google's Gemini) to fetch and synthesize documentation from a Supabase backend.
- An **asynchronous documentation crawler** that scrapes PineScript documentation pages (from TradingView), processes them (including text chunking, title/summary extraction, and embedding computation), and stores them in a database.

---

## Features

- **Interactive Chat Interface:** A user-friendly interface built with Streamlit to ask questions and receive streaming responses.
- **Asynchronous Agent:** Uses Python’s asyncio to stream responses in real time while preserving conversation history.
- **Documentation Retrieval:** Incorporates custom tool functions (`list_documentation_pages`, `retrieve_relevant_documentation`, and `get_page_content`) to access and retrieve relevant documentation stored in Supabase.
- **Web Crawler:** A dedicated crawler script that leverages BeautifulSoup and an asynchronous web crawler (crawl4ai) to scrape and process PineScript documentation from TradingView.
- **Robust Error Handling:** Uses logfire for logging and has built-in error handling and retry mechanisms.
- **Deployment Ready:** Deployed on Hugging Face Spaces for easy access and demonstration.

---

## Project Structure

```
├── streamlit.py
│   ├── Contains the main Streamlit app that:
│   │   ├── Displays a chat interface
│   │   ├── Streams responses from the PineScript expert agent
│   │   └── Stores conversation history and user queries in Supabase
│
├── pydantic_ai_agent.py
│   ├── Defines the PineScript expert agent with:
│   │   ├── A custom system prompt instructing the agent to use specific documentation tools
│   │   └── Tool functions for retrieving and listing documentation pages and content from Supabase
│
├── crawl_pinescriptdocs.py
│   ├── A crawler script that:
│   │   ├── Extracts URLs from TradingView’s PineScript documentation
│   │   ├── Processes and chunks documentation pages
│   │   ├── Uses ChatGroq for title and summary extraction
│   │   └── Computes embeddings via GoogleGenerativeAIEmbeddings before storing data in Supabase
│
└── .env (Not included)
    ├── Must be configured with your API keys and Supabase credentials
```

---

## Technologies Used

- **Python 3.x:** Core programming language.
- **Streamlit:** For building the interactive web interface.
- **Asyncio:** To handle asynchronous tasks and real-time streaming.
- **Supabase:** Serves as the backend database for storing user queries and documentation pages.
- **GeminiModel (pydantic_ai):** Utilized for language processing and powering the agent.
- **GoogleGenerativeAIEmbeddings:** Computes text embeddings.
- **ChatGroq (langchain_groq):** Extracts titles and summaries from documentation chunks.
- **BeautifulSoup & Requests:** For web scraping the PineScript documentation from TradingView.
- **Crawl4ai:** Provides asynchronous crawling capabilities.
- **Python-dotenv:** Loads environment variables from a .env file.
- **Logfire:** Configured for logging and error handling.

---

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/pinescript-agentic-rag.git
   cd pinescript-agentic-rag
   ```

2. **Create a Virtual Environment and Activate It:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Make sure to install all required packages. You can create a `requirements.txt` file with entries such as:

   ```
   streamlit
   supabase
   pydantic-ai
   python-dotenv
   logfire
   beautifulsoup4
   requests
   crawl4ai
   langchain-google-genai-embeddings
   langchain-groq
   ```
   
   Then run:

   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

Create a `.env` file in the project root directory with the following keys (replace placeholder values with your actual credentials):

```env
GOOGLE_API_KEY=your_google_api_key
SUPABASE_URL1=your_supabase_url
SUPABASE_SERVICE_KEY1=your_supabase_service_key
GROQ_API_KEY=your_groq_api_key
```

This file is used by both the Streamlit app and the crawler script to load the necessary API keys and configuration parameters.

---

## Running the Project

### Chat Interface (Streamlit)

To launch the interactive chat interface:

```bash
streamlit run streamlit.py
```

This will open a browser window (or provide a local URL) where you can ask questions about PineScript. The agent will stream its responses, fetching relevant documentation from Supabase.

### Documentation Crawler

To populate the documentation database by crawling PineScript docs:

```bash
python crawl_pinescriptdocs.py
```

This script will:
- Scrape URLs from TradingView’s PineScript documentation landing page.
- Process and split the content into manageable chunks.
- Use AI models to extract titles, summaries, and compute embeddings.
- Insert processed chunks into the Supabase `site_pages` table.

---

## Documentation Crawler Details

The crawler script (`crawl_pinescriptdocs.py`) performs the following steps:
- **URL Extraction:** Uses BeautifulSoup and Requests to extract absolute URLs from the PineScript documentation page.
- **Text Chunking:** Splits large markdown documents into smaller chunks while respecting code blocks and paragraphs.
- **Title & Summary Extraction:** Uses a ChatGroq model to derive a title and summary for each chunk.
- **Embedding Computation:** Computes text embeddings via the GoogleGenerativeAIEmbeddings API.
- **Data Insertion:** Stores the processed chunks along with metadata (e.g., URL, chunk number, timestamp) into Supabase.

A rate limiter is implemented to ensure compliance with API limits for both GROQ and Google API calls.

---

## Deployment

The project is deployed on [Hugging Face Spaces](https://yashpinjarkar10-pinescript-agent.hf.space), making it easily accessible for demonstration and usage. The Hugging Face Spaces deployment runs the Streamlit app, allowing users to interact with the agent directly from the browser.

---

## Contributing

Contributions are welcome! If you have suggestions, bug fixes, or improvements, please:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes.
4. Open a pull request explaining your changes.

Feel free to open issues for any bugs or feature requests.

---

## License

[MIT License](LICENSE)  
*(Replace with your chosen license.)*

---

This README provides a comprehensive guide to the PineScript Agentic RAG project, outlining its purpose, structure, and the technologies that power it. Enjoy exploring PineScript like never before!

---
