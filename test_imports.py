import json
import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

logging.basicConfig(level=logging.INFO)
