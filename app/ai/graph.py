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
    image_url: str
    requires_image: bool
    
    # Final Output
    board_content: str
    chat_content: str
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
    1. "student_analysis": Briefly analyze the student's current state and intent.
    2. "pedagogical_decision": Choose EXACTLY ONE from the following options:
       - CRITICAL RULE 1: If the student asks about a topic NOT found in the 'Context from PDF' below, you MUST choose 'Out of Syllabus'.
       - CRITICAL RULE 2: If the student asks for 'questions', 'assignment', 'practice problems', 'exam questions', or anything similar, you MUST choose 'Assignment/Question Extraction'. Extract and list ALL questions, case studies, and practice problems VERBATIM from the PDF context. Do NOT make up new questions.
       - If the student is greeting or sharing feelings/stress, choose 'Casual/Empathetic Chit-Chat'.
       - If the student is abusive, choose 'Angry Warning'.
       - If the student is disruptive, choose 'Strict Warning'.
       - For normal concept questions, choose one of: 'First-Principles Breakdown', 'Real-world Analogy', 'Interactive MCQ Quiz', 'Roleplay Scenario', 'Direct Encouraging Answer'.
    3. "selected_concept": The exact concept/questions from the PDF context you will address. MUST BE STRICTLY FROM THE PDF.
    4. "awarded_xp": Award 10 or 20 XP for correct answers, 0 otherwise. Always 0 for 'Assignment/Question Extraction' or 'Casual/Empathetic Chit-Chat'.
    5. "requires_image": true/false. Set to true ONLY if a visual analogy helps understand the concept. ALWAYS false for 'Assignment/Question Extraction'.
    6. "strategy": The final instructions (3-4 sentences) for the Persona Agent.
       - If 'Out of Syllabus': Tell the Persona to politely inform the student that this topic is not in the current PDF.
       - If 'Assignment/Question Extraction': Tell the Persona to list ALL the extracted questions VERBATIM and clearly numbered on the board. Do not add any extra teaching.
       - For casual chat: Act like a cool caring mentor, no technical content.
       - For warnings: Be strict/angry as required.
       - For conceptual teaching: Give detailed structural breakdown instructions.
    
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
        requires_image = result.get("requires_image", False)
        
        # Save a snippet of the strategy to avoid repeating
        new_analogies = state.get("used_analogies", [])
        if "analogy" in pedagogy.lower() or "roleplay" in pedagogy.lower():
            new_analogies.append(strategy[:50] + "...") 
            
        logging.info(f"Teacher CoT: {result}")
        return {"teaching_strategy": strategy, "used_analogies": new_analogies, "awarded_xp": awarded_xp, "requires_image": requires_image}
    except Exception as e:
        logging.error(f"Teacher Error: {e}")
        return {"teaching_strategy": "Explain the concept from the PDF in a fun way.", "used_analogies": state.get("used_analogies", []), "awarded_xp": 0, "requires_image": False}

def visualizer_node(state: ClassroomState) -> Dict[str, Any]:
    """Generates an image related to the teaching strategy using Pollinations AI."""
    if not state.get("requires_image", False):
        return {"image_url": ""}
        
    strategy = state.get("teaching_strategy", "")
    
    keyword_prompt = PromptTemplate.from_template("""
    You are an expert prompt engineer for an AI Image Generator.
    Read the following teaching strategy and extract the core concept or analogy.
    Generate a highly descriptive, visually stunning digital art prompt (10-15 words) representing this concept. 
    Make it look premium, educational, and cinematic. Do not include text in the image.
    
    Strategy: {strategy}
    Image Prompt only:
    """)
    
    try:
        raw_text = extract_text(llm.invoke(keyword_prompt.format(strategy=strategy)).content)
        keyword = raw_text.strip().strip('"')
        
        import urllib.parse
        encoded_keyword = urllib.parse.quote(keyword)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_keyword}?width=800&height=400&nologo=true"
        
    except Exception as e:
        logging.error(f"Image Generation failed: {e}")
        image_url = ""
        
    return {"image_url": image_url}

def persona_node(state: ClassroomState) -> Dict[str, Any]:
    """Formats the final response into highly emotionally intelligent (High EQ), engaging Hinglish with emojis."""
    prompt = PromptTemplate.from_template("""
    You are the Ultimate 1-on-1 Personal AI Tutor. You have EXTREME Emotional Intelligence (High EQ) and a knack for making complex things painfully simple.
    Take the Teaching Strategy and format it into the final response for the student.
    
    Teaching Strategy: {teaching_strategy}
    Student Name: {student_name}
    Student Profile: {student_profile}
    Awarded XP: {awarded_xp} XP
    Image URL to include (if any): {image_url}
    
    HIGH EQ RULES:
    1. Write in a flawless, natural mix of Hinglish and English. You are their favorite, cool, ultra-smart mentor for ANY subject.
    2. Read the Room based on the Teaching Strategy:
       - If Awarded XP > 0: Start by celebrating their correct answer and telling them they earned {awarded_xp} XP! Use party emojis 🎉🔥!
       - If it says 'Assignment/Question Extraction': Your ONLY job is to list ALL the questions from the strategy VERBATIM in the board_content. Number them clearly. Chat message should be short (e.g., "Yeh raha tera assignment! 📋 Board check kar."). Do NOT teach, explain, or add analogies.
       - If it's a 'Casual/Empathetic Chit-Chat', be extremely warm and friendly. Do NOT teach anything technical.
       - If the strategy is an 'Angry Warning', be EXTREMELY ANGRY. No emojis, just pure scolding.
       - If it's a normal warning, be strict but keep a tiny bit of humor.
       - If it's a game, format it beautifully with emojis.
    
    3. THE WOW-FACTOR TEACHING FRAMEWORK (Only apply this if you are teaching a concept):
       - Step 1: THE HOOK 🪝 - Start with a relatable problem or a shocking question.
       - Step 2: THE DEEP BREAKDOWN 🧠 - Break the concept into bite-sized steps (Step 1, Step 2, Step 3). 
       - Step 3: THE ANALOGY 📖 - Invent a fresh, highly relatable real-world analogy. 
       - Step 4: THE MICRO-CHALLENGE 🎯 - End with a fun scenario-based question to test them.
    
    4. Code & Technical Diagrams: 
       - If the topic involves PROGRAMMING/CODING, you MUST write the code using Markdown fenced code blocks (e.g., ```python ... ```). Keep code snippets short, optimized, and beautifully formatted.
       - If the topic requires a flowchart, architecture diagram, or table (like TCP/IP layers), DO NOT rely on the Image URL. You MUST draw it yourself using Markdown tables or ASCII art in the `board_content`.
       
    5. Board Formatting: The `board_content` MUST BE STUNNING. Use `# Headers`, `**Bold text**`, `>` quotes, and clear spacing. Make it look like a beautifully designed study note.
    6. If an Image URL is provided, include it exactly like this at the very end of the board_content: ![Visual]({image_url})
    
    RESPOND STRICTLY IN JSON FORMAT WITH THESE KEYS:
    {{
        "board_content": "The main technical teaching. MUST USE BEAUTIFUL MARKDOWN. Include technical diagrams (ASCII/Tables) if needed, and the Image URL at the end. Leave EMPTY if this is casual chat.",
        "chat_content": "Short, highly emotional Hinglish chat response for the live chat. If board_content is heavy, chat_content MUST just be a short pointer like 'Look at the board!'. Includes jokes, warnings, or XP celebrations."
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

