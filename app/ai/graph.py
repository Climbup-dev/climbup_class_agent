import asyncio
import re
from typing import TypedDict, List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import json
import logging
import os

logging.basicConfig(level=logging.INFO)

class ClassroomState(TypedDict):
    retriever: Any
    classroom_id: str
    subject_name: str
    topic_name: str
    lecture_date: str
    active_students: str
    student_name: str
    student_profile: str
    chat_history: str
    question: str
    context: str
    used_analogies: List[str]
    strike_count: int
    is_disruptive: bool
    is_abusive: bool
    awarded_xp: int
    board_content: str
    chat_content: str

# LLMs are now dynamically loaded per-request for Load Balancing

def extract_text(content: Any) -> str:
    if isinstance(content, str): return content
    elif isinstance(content, list): return "".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
    return str(content)

def clean_json(content: Any) -> str:
    text = extract_text(content)
    try:
        # Find the outermost JSON object
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        return text
    except Exception:
        return text

# Initialize FlashRank globally so it doesn't reload the model every time
try:
    from flashrank import Ranker, RerankRequest
    # cache_dir ensures it doesn't redownload models constantly
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="flashrank_cache")
    FLASHRANK_AVAILABLE = True
except Exception as e:
    logging.error(f"FlashRank failed to initialize: {e}")
    FLASHRANK_AVAILABLE = False


class SingleShotApp:
    async def generate_multi_queries(self, question: str, chat_history: str = "") -> Dict[str, Any]:
        prompt = PromptTemplate.from_template("""
        You are an expert at analyzing a student's question for an Educational AI Tutor.
        
        Recent Chat History: {chat_history}
        Latest Question: {question}
        
        TASK 1: Intent Classification
        Determine if the student is asking a question that requires searching the course textbook (PDF) or if they are asking a general knowledge question/making casual chat.
        - Set "source_needed" to "PDF" if they ask about concepts, definitions, assignments, or course material (even if referencing history).
        - Set "source_needed" to "General_Knowledge" if they ask for a joke, coding help, general facts outside the syllabus, or say hello.
        
        TASK 2: User Angle Detection
        Determine HOW the student wants the answer formatted based on their tone and request:
        - Provide a 1-sentence instruction for the main AI on how to format the answer (e.g. "Use simple analogies", "Extract exact points", "Standard detailed explanation").
        
        TASK 3: Search Query Generation (Only if PDF needed)
        If PDF is needed, generate the SINGLE best search query to find this information in the textbook. If the user says "explain it more", use the chat history to know what "it" is and write a complete search query (e.g. "TCP protocol details").
        
        Respond STRICTLY in JSON format:
        {{
            "source_needed": "PDF or General_Knowledge",
            "user_angle_instruction": "Your 1-sentence instruction",
            "search_query": "The optimal search query string"
        }}
        """)
        try:
            from app.core.llm_balancer import get_balanced_fast_llm
            llm = get_balanced_fast_llm()
            response = await llm.ainvoke(prompt.format(question=question, chat_history=chat_history[-500:] if chat_history else "None"))
            raw = clean_json(response.content)
            data = json.loads(raw)
            return data
        except Exception as e:
            logging.error(f"Multi-query generation failed: {e}")
            return {"source_needed": "PDF", "user_angle_instruction": "Standard detailed educational explanation.", "search_query": question}

    async def ainvoke(self, state: ClassroomState) -> Dict[str, Any]:
        question = state.get("question", "")
        
        # 1. Handle SYSTEM_INIT gracefully (Zero hallucination, ultra fast)
        if "[SYSTEM_INIT]" in question:
            return {
                "board_content": "# Welcome to the Class!\n\nI am your AI Professor. We will be studying **" + state.get("subject_name", "this topic") + "**. Ask me anything!",
                "chat_content": "SILENCE"
            }
            
        # 2. Smart Routing, Angle Detection & Multi-Query Retrieval
        retriever = state.get("retriever")
        context = "No context available."
        chat_history = state.get("chat_history", "")
        
        logging.info(f"Analyzing Intent and Generating search query for: {question}")
        analysis = await self.generate_multi_queries(question, chat_history)
        source_needed = analysis.get("source_needed", "PDF")
        user_angle_instruction = analysis.get("user_angle_instruction", "Standard detailed educational explanation.")
        search_query = analysis.get("search_query", question)
        
        if source_needed == "General_Knowledge":
            logging.info("Routing to General Knowledge (Skipping PDF Search)")
            context = "GENERAL_KNOWLEDGE_MODE"
        elif retriever and search_query.strip():
            logging.info(f"Executing PDF search for: '{search_query}'")
            
            all_docs = []
            seen_content = set()
            
            try:
                docs = retriever.invoke(search_query)
                for d in docs:
                    if d.page_content not in seen_content:
                        all_docs.append(d)
                        seen_content.add(d.page_content)
            except Exception as e:
                logging.error(f"Retriever error on query '{search_query}': {e}")
            
            if all_docs:
                if FLASHRANK_AVAILABLE:
                    logging.info(f"Applying FlashRank Compression on {len(all_docs)} unique chunks...")
                    try:
                        passages = [{"id": i, "text": doc.page_content, "meta": doc.metadata} for i, doc in enumerate(all_docs)]
                        rerankrequest = RerankRequest(query=question, passages=passages)
                        results = ranker.rerank(rerankrequest)
                        
                        # Take top 6 most relevant chunks so we don't miss any assignment questions
                        top_6_chunks = [f"[Page {res['meta'].get('page_label', 'Unknown')}]\n{res['text']}" for res in results[:6]]
                        context = "\n\n---\n\n".join(top_6_chunks)
                        logging.info("FlashRank successfully reduced context size.")
                    except Exception as e:
                        logging.error(f"FlashRank Error: {e}. Falling back to normal context.")
                        context = "\n\n---\n\n".join([f"[Page {doc.metadata.get('page_label', 'Unknown')}]\n{doc.page_content}" for doc in docs[:6]])
                else:
                    context = "\n\n---\n\n".join([f"[Page {doc.metadata.get('page_label', 'Unknown')}]\n{doc.page_content}" for doc in docs[:6]])
            else:
                context = "DATA NOT AVAILABLE IN PDF"
        
        # 3. The Super Prompt (Single Shot reasoning + formatting)
        super_prompt = PromptTemplate.from_template("""
        You are an incredibly smart, friendly human tutor helping a student study from their PDF notes.
        Student Name: {student_name}
        
        Student's Question: {question}
        Chat History: {chat_history}
        
        PDF CONTEXT (Source of Truth):
        {context}
        
        YOUR TASK & RULES:
        1. MIRROR THE USER'S TONE: If the student speaks in formal English, reply in Professional English. If they use casual Hinglish, use casual/professional Hinglish.
        2. EASY & FUNNY EXPLANATIONS: When explaining complex concepts from the PDF, use simple, everyday analogies, real-world examples, and a bit of humor to make it fun and easy to understand.
        3. KEEP IT REAL AND SHORT: Never write massive essays. Give crisp, to-the-point answers. If they ask a specific question (like "page number batao" or "assignment questions do"), directly give them exactly what they asked in 1-2 sentences. 
        4. BE ACCURATE TO THE PDF: Your answers MUST be based strictly on the PDF CONTEXT provided. Mention page numbers naturally (e.g., "Yeh topic tumhe Page X par mil jayega!"). The Page Number is indicated in the Context block as [Page X].
        5. IF NOT IN PDF: If the answer isn't in the context, be honest like a real person: "I couldn't find this in the PDF notes. Are you sure it's in here?"
        6. Put your entire response inside "chat_content". Set "board_content" to "".
        
        RESPOND STRICTLY IN JSON FORMAT:
        {{
            "chat_content": "Your natural, tone-matched, and accurate chat response here.",
            "board_content": ""
        }}
        """)
        
        try:
            safe_question = question.replace("{", "{{").replace("}", "}}")
            safe_history = state.get("chat_history", "").replace("{", "{{").replace("}", "}}")
            safe_user_angle = user_angle_instruction.replace("{", "{{").replace("}", "}}")
            
            formatted = super_prompt.format(
                student_name=state.get("student_name", "Student"),
                subject_name=state.get("subject_name", ""),
                topic_name=state.get("topic_name", ""),
                question=safe_question,
                chat_history=safe_history,
                user_angle_instruction=safe_user_angle,
                context=context
            )
            from app.core.llm_balancer import get_balanced_text_llm
            llm_text = get_balanced_text_llm().bind(response_format={"type": "json_object"})
            response = await llm_text.ainvoke(formatted)
            raw_content = clean_json(response.content)
            result = json.loads(raw_content)
            
            return {
                "board_content": result.get("board_content", "Error generating board."),
                "chat_content": result.get("chat_content", "SILENCE"),
                "awarded_xp": 10,
                "used_analogies": []
            }
        except Exception as e:
            logging.error(f"SingleShotApp Error: {e}")
            return {
                "board_content": f"Sorry, there was an error processing your request: {e}",
                "chat_content": "SILENCE",
                "awarded_xp": 0,
                "used_analogies": []
            }

classroom_app = SingleShotApp()
