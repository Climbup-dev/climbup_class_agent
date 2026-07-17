from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from duckduckgo_search import DDGS
import json
import logging

logging.basicConfig(level=logging.INFO)

class ClassroomState(TypedDict):
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
    
    # Moderation State
    strike_count: int
    is_disruptive: bool
    is_abusive: bool
    
    # Gamification
    awarded_xp: int
    
    should_intervene: bool
    teaching_strategy: str
import os

import os

# 1. Primary: Groq Llama 3 70B (Fast, High Quality)
llm_groq = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
llm_groq_json = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7).bind(response_format={"type": "json_object"})

# 2. Fallback 1: OpenRouter Llama 3 8B (Safety net, highly reliable)
llm_openrouter = ChatOpenAI(
    model="meta-llama/llama-3-8b-instruct:free", 
    openai_api_key=os.environ.get("OPENROUTER_API_KEY"), 
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.7
)

# 3. Fallback 2: Gemini 3.1 Flash Lite
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

# Build the Ultimate Unbreakable Brain (Groq -> OpenRouter -> Gemini)
llm = llm_groq.with_fallbacks([llm_openrouter, llm_gemini])
llm_json = llm_groq_json.with_fallbacks([llm_openrouter, llm_gemini])

# Helper function to extract text safely (handles both str and list responses)
def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "".join([str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content])
    return str(content)

# Helper function to clean JSON from any model output
def clean_json(content: Any) -> str:
    text = extract_text(content)
    return text.replace("```json", "").replace("```", "").strip()



def router_node(state: ClassroomState) -> Dict[str, Any]:
    """Decides if the AI should intervene based on student messages."""
    prompt = PromptTemplate.from_template("""
    You are the Router Agent for an AI Teacher. 
    Analyze the following classroom chat to decide if the AI should intervene.
    
    Rules:
    - Intervene (True) IF: The student asks a question, seems confused, is completely off-topic, or directly addresses the teacher/AI.
    - DO NOT Intervene (False) IF: Students are just saying "hi", "ok", or having a quick casual chat among themselves that doesn't need a teacher.
    - is_disruptive (True) IF: The student is deliberately ignoring the topic, repeatedly talking about unrelated things (like video games, movies), or being disrespectful.
    - is_abusive (True) IF: The student uses explicit swear words, profanity, gali, or extreme disrespect.
    
    Student Speaking: {student_name}
    Message: {question}
    Chat History: {chat_history}
    
    Respond strictly in JSON format:
    {{
        "intervene": true/false,
        "is_disruptive": true/false,
        "is_abusive": true/false,
        "reason": "short reason"
    }}
    """)
    
    try:
        response = llm_json.invoke(prompt.format(**state))
        raw_content = clean_json(response.content)
        result = json.loads(raw_content)
        return {
            "should_intervene": result.get("intervene", True),
            "is_disruptive": result.get("is_disruptive", False),
            "is_abusive": result.get("is_abusive", False)
        }
    except Exception as e:
        logging.error(f"Router Error: {e}")
        return {"should_intervene": True, "is_disruptive": False, "is_abusive": False}

def teacher_node(state: ClassroomState) -> Dict[str, Any]:
    """Uses Chain of Thought to analyze the student and formulate an elite teaching strategy."""
    prompt = PromptTemplate.from_template("""
    You are the Master Teacher Agent. Your job is to decide HOW to teach the current topic based ONLY on the provided PDF context.
    You will use a Chain of Thought process to craft an extremely high-quality, targeted teaching strategy.
    
    Subject: {subject_name} ({topic_name})
    Context from PDF:
    {context}
    
    Student Message: {question}
    Chat History: {chat_history}
    Already Used Analogies (DO NOT REPEAT THESE): {used_analogies}
    
    MODERATION STATUS:
    Is the student being disruptive? {is_disruptive}
    Is the student being abusive/swearing? {is_abusive}
    Strike Count: {strike_count}/3
    
    TASK:
    Respond strictly in JSON format with the following keys:
    1. "student_analysis": Briefly analyze the student's current state. Are they confused, bored, curious, deliberately disruptive, abusive, or just wanting to chit-chat/greet?
    2. "pedagogical_decision": Decide the best approach. 
       - If they are greeting you ("how are you") or sharing feelings/stress, choose "Casual/Empathetic Chit-Chat". 
       - If they are abusive, choose "Angry Warning". 
       - If they are disruptive, choose "Strict Warning". 
       - Otherwise choose "Real-world Analogy", "Interactive MCQ Quiz", "Roleplay Scenario", or "Direct Encouraging Answer".
    3. "selected_concept": The exact technical fact/concept from the PDF context you will teach right now. If giving a warning OR doing "Casual/Empathetic Chit-Chat", leave this EMPTY ("").
    4. "awarded_xp": If the student correctly answered a previous technical question/challenge you gave them, award 10 XP. If it was an exceptionally brilliant answer, award 20 XP. If they gave a wrong answer, were off-topic, or just chit-chatting, award 0 XP. (Must be an integer: 0, 10, or 20).
    5. "strategy": The final instructions (2-3 sentences) for the Persona Agent on what exactly to say. 
       - If awarded_xp > 0: Instruct the Persona to enthusiastically congratulate them for earning XP before continuing.
       - If "Casual/Empathetic Chit-Chat": Instruct the Persona to act like a cool, caring mentor. Validate their feelings, relieve their stress, and DO NOT force any PDF teaching in this message.
       - If abusive: Instruct the Persona to react like an EXTREMELY ANGRY MAN (Bhai, tameez se baat kar!).
       - If disruptive: Instruct the Persona to give a very strict but slightly funny Hinglish warning. If Strike Count is 2, mention it's their LAST warning before getting kicked out.
    
    JSON Output:
    """)
    
    formatted_prompt = prompt.format(
        subject_name=state["subject_name"],
        topic_name=state["topic_name"],
        context=state["context"],
        question=state["question"],
        chat_history=state["chat_history"],
        used_analogies=", ".join(state["used_analogies"]) if state.get("used_analogies") else "None",
        is_disruptive=state.get("is_disruptive", False),
        is_abusive=state.get("is_abusive", False),
        strike_count=state.get("strike_count", 0)
    )
    
    try:
        response = llm_json.invoke(formatted_prompt)
        raw_content = clean_json(response.content)
        result = json.loads(raw_content)
        strategy = result.get("strategy", "Teach the concept beautifully.")
        pedagogy = result.get("pedagogical_decision", "")
        awarded_xp = result.get("awarded_xp", 0)
        
        # Save a snippet of the strategy to avoid repeating
        new_analogies = state.get("used_analogies", [])
        if "analogy" in pedagogy.lower() or "roleplay" in pedagogy.lower():
            new_analogies.append(strategy[:50] + "...") 
            
        logging.info(f"Teacher CoT: {result}")
        return {"teaching_strategy": strategy, "used_analogies": new_analogies, "awarded_xp": awarded_xp}
    except Exception as e:
        logging.error(f"Teacher Error: {e}")
        return {"teaching_strategy": "Explain the concept from the PDF in a fun way.", "used_analogies": state.get("used_analogies", []), "awarded_xp": 0}

def visualizer_node(state: ClassroomState) -> Dict[str, Any]:
    """Fetches an image related to the teaching strategy using DuckDuckGo."""
    strategy = state.get("teaching_strategy", "")
    
    keyword_prompt = PromptTemplate.from_template("""
    Extract ONE short, highly visual search keyword (2-3 words max) from the following teaching strategy. 
    It should be an object or scenario (e.g., "bank locker", "hospital database", "shopping mall").
    Strategy: {strategy}
    Keyword only:
    """)
    
    try:
        raw_text = extract_text(llm.invoke(keyword_prompt.format(strategy=strategy)).content)
        keyword = raw_text.strip().strip('"')
        results = DDGS().images(keyword, max_results=1)
        image_url = results[0].get("image", "") if results else ""
    except Exception as e:
        logging.error(f"Image Search failed: {e}")
        image_url = ""
        
    return {"image_url": image_url}

def persona_node(state: ClassroomState) -> Dict[str, Any]:
    """Formats the final response into highly emotionally intelligent (High EQ), engaging Hinglish with emojis."""
    prompt = PromptTemplate.from_template("""
    You are the Actor/Persona Agent. You have EXTREME Emotional Intelligence (High EQ). 
    Take the Teaching Strategy and format it into the final response for the student.
    
    Teaching Strategy: {teaching_strategy}
    Student Name: {student_name}
    Student Profile: {student_profile}
    Awarded XP: {awarded_xp} XP
    Image URL to include (if any): {image_url}
    
    HIGH EQ RULES:
    1. Write in a flawless, natural mix of Hinglish and English. You are their favorite, cool, stress-relieving engineering teacher.
    2. Read the Room: 
       - If Awarded XP > 0: Start by celebrating their correct answer and telling them they earned {awarded_xp} XP! Use party emojis 🎉🔥!
       - If it's a "Casual/Empathetic Chit-Chat", be extremely warm and friendly. Do NOT teach anything technical. Just connect with them human-to-human.
       - If the strategy is an "Angry Warning", be EXTREMELY ANGRY. No emojis, just pure scolding ("Tameez mein rehna seekho!").
       - If it's a normal warning, be strict but keep a tiny bit of humor ("Class se bahar nikal dunga!"). 
       - If it's a game, format it beautifully with emojis (A 🟢, B 🔴, C 🔵).
    
    3. THE 3-STEP TEACHING FRAMEWORK (Only apply this if you are teaching a concept):
       - Step 1: THE HOOK 🪝 - Never start with a boring definition. Start with a relatable problem, a shocking question, or connect it to their life/apps (Insta, PubG, College).
       - Step 2: THE STORY 📖 - Explain the core concept using a real-world example (Marvel, hacking a bank, etc.). Connect emotions to the logic.
       - Step 3: THE MICRO-CHALLENGE 🎯 - NEVER end with "Did you understand?". End with a fun scenario-based question that forces them to apply what they just learned.
    
    4. Praise Naturally: ONLY praise them if they actually answered a technical question correctly. Give context-aware, genuine compliments.
    5. Keep paragraphs short (1-2 lines). Break up large text.
    6. If an Image URL is provided, include it exactly like this at the very end of the board_content: ![Visual]({image_url})
    
    RESPOND STRICTLY IN JSON FORMAT WITH THESE KEYS:
    {{
        "board_content": "The main technical teaching, markdown, deep analogies, code, and Image URL. Leave EMPTY if this is just casual chat, warning, or greeting.",
        "chat_content": "Short, highly emotional Hinglish chat response for the live chat. Includes jokes, warnings, or XP celebrations."
    }}
    """)
    
    try:
        formatted_prompt = prompt.format(
            teaching_strategy=state["teaching_strategy"],
            student_name=state["student_name"],
            student_profile=state["student_profile"],
            awarded_xp=state.get("awarded_xp", 0),
            image_url=state.get("image_url", "")
        )
        response = llm_json.invoke(formatted_prompt)
        raw_content = clean_json(response.content)
        result = json.loads(raw_content)
        return {
            "board_content": result.get("board_content", ""),
            "chat_content": result.get("chat_content", "")
        }
    except Exception as e:
        logging.error(f"Persona Error: {e}")
        return {"board_content": "", "chat_content": "I'm having a little trouble thinking right now. Could you repeat that?"}

# Define routing logic
def route_after_router(state: ClassroomState) -> str:
    if state.get("should_intervene"):
        return "teacher_node"
    return "end"

# Build Graph
workflow = StateGraph(ClassroomState)

workflow.add_node("router_node", router_node)
workflow.add_node("teacher_node", teacher_node)
workflow.add_node("visualizer_node", visualizer_node)
workflow.add_node("persona_node", persona_node)

workflow.set_entry_point("router_node")
workflow.add_conditional_edges("router_node", route_after_router, {
    "teacher_node": "teacher_node",
    "end": END
})
workflow.add_edge("teacher_node", "visualizer_node")
workflow.add_edge("visualizer_node", "persona_node")
workflow.add_edge("persona_node", END)

classroom_app = workflow.compile()

