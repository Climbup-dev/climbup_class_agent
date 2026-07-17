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
    
    # Smart Router Output
    should_intervene: bool
    intent_type: str      # technical_question | assignment_extraction | casual_chat | concept_explanation | summary_request | comparison_request | out_of_scope
    student_emotion: str  # curious | confused | stressed | bored | confident | frustrated
    specific_need: str    # Precise description of what the student actually wants
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
    """Mind-Reader Agent: Deeply classifies the student's intent, emotion, and exact need before routing."""
    
    # HARDCODED BYPASS FOR SYSTEM STARTUP
    # This guarantees zero hallucination and blazing fast load times for the initial syllabus overview.
    if "[SYSTEM_INIT]" in state.get("question", ""):
        return {
            "should_intervene": True,
            "intent_type": "summary_request",
            "student_emotion": "curious",
            "specific_need": "Student has just opened the chat. Extract a highly engaging Syllabus Overview and list the most important exam topics from the PDF.",
            "is_disruptive": False,
            "is_abusive": False
        }
        
    prompt = PromptTemplate.from_template("""
    You are an expert Student Psychology Analyst and Intent Classifier for an AI Tutor.
    Your job is to deeply understand what the student ACTUALLY wants — their intent, their emotion, and their specific need.
    Do NOT answer the question. Just analyze it with precision.
    
    Subject: {subject_name} | Topic: {topic_name}
    Student Name: {student_name}
    Student Message: {question}
    Recent Chat History: {chat_history}
    
    ANALYSIS TASK:
    Respond strictly in JSON format with these keys:
    
    1. "intent_type": Classify the student's intent into EXACTLY ONE of:
       - "assignment_extraction": Student wants questions, assignments, or practice problems listed.
       - "concept_explanation": Student wants a concept explained simply (e.g., "what is X", "explain X").
       - "comparison_request": Student wants to compare two things (e.g., "difference between A and B").
       - "summary_request": Student wants a brief summary or overview of a topic.
       - "technical_question": Student has a specific deep technical question from the syllabus.
       - "casual_chat": Student is greeting, venting stress, or making small talk.
       - "abusive": Student is using swear words, profanity, or extreme disrespect.
       - "disruptive": Student is deliberately ignoring the topic or being disrespectful.
    
    2. "student_emotion": Classify into ONE of: "curious", "confused", "stressed", "bored", "confident", "frustrated", "happy".
    
    3. "specific_need": In one precise sentence, state EXACTLY what the student wants. This will be passed directly to the Teacher Agent. Be very specific.
       Examples: 
       - "Student wants all assignment questions listed from the PDF verbatim."
       - "Student wants a simple, beginner-friendly explanation of Cybercrime definition."
       - "Student wants to compare Doctrinal vs Consensual approach to contract law."
       - "Student is stressed about exams and needs emotional support, not a lesson."
    
    4. "should_intervene": true if the AI should respond, false only if the message is a simple "ok", "hmm", or doesn't need a response.
    
    JSON Output:
    """)
    
    try:
        response = llm_json.invoke(prompt.format(
            subject_name=state["subject_name"],
            topic_name=state["topic_name"],
            student_name=state["student_name"],
            question=state["question"],
            chat_history=state["chat_history"]
        ))
        raw_content = clean_json(response.content)
        result = json.loads(raw_content)
        logging.info(f"Router Intent: {result}")
        return {
            "should_intervene": result.get("should_intervene", True),
            "intent_type": result.get("intent_type", "technical_question"),
            "student_emotion": result.get("student_emotion", "curious"),
            "specific_need": result.get("specific_need", state["question"]),
            "is_disruptive": result.get("intent_type") == "disruptive",
            "is_abusive": result.get("intent_type") == "abusive"
        }
    except Exception as e:
        logging.error(f"Router Error: {e}")
        return {
            "should_intervene": True,
            "intent_type": "technical_question",
            "student_emotion": "curious",
            "specific_need": state["question"],
            "is_disruptive": False,
            "is_abusive": False
        }

def teacher_node(state: ClassroomState) -> Dict[str, Any]:
    """Laser-Focused Teacher: Uses Router's classified intent to build a surgically precise teaching strategy."""
    prompt = PromptTemplate(
        template="""You are the Brain (Teacher Agent) of a highly advanced intelligent classroom.
    You have been provided with the ENTIRE FULL TEXT of the student's study material below. You have perfect visibility of every page and line.
    
    Student: {student_name}
    Emotion/State: {student_emotion}
    Class: {subject_name} ({topic_name})
    
    FULL STUDY MATERIAL CONTEXT (Pay attention to the [--- PAGE X ---] markers):
    {context}
    
    You are the 'Brain' of a World-Class Master Pedagogue.
    The Router Agent has already analyzed the student deeply. Use this analysis to craft a LASER-FOCUSED teaching strategy.
    DO NOT waste time re-analyzing. Act on the classified intent immediately.
    
    Subject: {subject_name} | Topic: {topic_name}
    
    --- ROUTER INTELLIGENCE (Act on this) ---
    Student's Classified Intent: {intent_type}
    Student's Emotion Right Now: {student_emotion}
    What Student ACTUALLY Needs: {specific_need}
    ----------------------------------------
    
    Context from PDF (Use ONLY this. Never hallucinate):
    {context}
    
    Full Student Message (for nuance): {question}
    Chat History: {chat_history}
    Already Used Analogies (DO NOT REPEAT): {used_analogies}
    
    TASK: Respond in JSON format:
    1. "awarded_xp": 10 or 20 if student correctly answered a previous challenge, 0 otherwise.
    2. "requires_image": true ONLY if a visual analogy would genuinely help. False for assignments, comparisons, summaries.
    3. "strategy": 3-5 sentence precise teaching plan for the Persona Agent. MUST directly address "{specific_need}". 
       - Be FLUID and NATURAL. Give the student EXACTLY what they asked for.
       - HIGH-PRECISION GROUNDING: If the student asks about a specific concept, locate it in the text. You MUST cite the EXACT PAGE (e.g. "As seen on Page 4...") and quote the exact sentence verbatim before explaining it.
       - EXTERNAL/APPLICATION QUESTIONS: If the student asks to solve a specific case study (e.g., Suhas Katti) that is NOT in the PDF, DO NOT say "Out of syllabus" and refuse. Instead, acknowledge it's an external case, but use the core theories from the PDF (e.g., Jurisdiction approaches) to brilliantly solve and analyze their exact question.
       - If they asked for a LIST of questions, just list them verbatim with their page numbers.
       - Match your depth/tone to the student's emotion: If confused → extra simple. If stressed → gentle. If curious → exciting.
       - For conceptual teaching: Plan a Hook → Simple Breakdown → Fresh Analogy → Micro-Challenge.
    
    JSON Output:
    """)
    
    formatted_prompt = prompt.format(
        subject_name=state["subject_name"],
        topic_name=state["topic_name"],
        intent_type=state.get("intent_type", "technical_question"),
        student_emotion=state.get("student_emotion", "curious"),
        specific_need=state.get("specific_need", state["question"]),
        context=state["context"],
        question=state["question"],
        chat_history=state["chat_history"],
        used_analogies=", ".join(state["used_analogies"]) if state.get("used_analogies") else "None"
    )
    
    try:
        response = llm_json.invoke(formatted_prompt)
        raw_content = clean_json(response.content)
        result = json.loads(raw_content)
        strategy = result.get("strategy", "Teach the concept beautifully.")
        awarded_xp = result.get("awarded_xp", 0)
        requires_image = result.get("requires_image", False)
        
        # Track used analogies to avoid repetition
        new_analogies = state.get("used_analogies", [])
        if "analogy" in strategy.lower():
            new_analogies.append(strategy[:60] + "...")
            
        logging.info(f"Teacher Strategy: {strategy[:100]}")
        return {"teaching_strategy": strategy, "used_analogies": new_analogies, "awarded_xp": awarded_xp, "requires_image": requires_image}
    except Exception as e:
        logging.error(f"Teacher Error: {e}")
        return {"teaching_strategy": "Explain the concept from the PDF in a fun and simple way.", "used_analogies": state.get("used_analogies", []), "awarded_xp": 0, "requires_image": False}

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
    """The Actor: Emotionally calibrated response writer that fulfills the student's exact expectation."""
    prompt = PromptTemplate.from_template("""
    You are the Ultimate 1-on-1 Personal AI Tutor — the Actor who brings the Teaching Strategy to life.
    Your voice must be emotionally calibrated to the student's current state and laser-focused on their exact need.
    
    Teaching Strategy (from Teacher Agent): {teaching_strategy}
    Student's Intent Type: {intent_type}
    Student's Current Emotion: {student_emotion}
    Student Name: {student_name}
    Awarded XP: {awarded_xp}
    Image URL (if any): {image_url}
    
    HIGH EQ RULES:
    1. Write in a flawless, natural mix of Hinglish and English. You are their favorite, cool, ultra-smart mentor.
    2. Be FLUID and DIRECT. Give the student exactly what the Teaching Strategy dictates without sounding robotic.
       - If the student asked for an answer to a question, explain it clearly on the board and say something encouraging in the chat (e.g., "Ye raha tera answer, ekdum simple words mein!").
       - If the student asked for a list of assignments, list them nicely on the board.
       - If Awarded XP > 0: Celebrate their correct answer! 🎉🔥
    
    3. THE WOW-FACTOR TEACHING FRAMEWORK (Only apply this if teaching a new, deep concept):
       - Step 1: THE HOOK 🪝 - Start with a relatable problem.
       - Step 2: THE DEEP BREAKDOWN 🧠 - Break it into bite-sized steps.
       - Step 3: THE ANALOGY 📖 - Invent a fresh real-world analogy.
       - Step 4: THE MICRO-CHALLENGE 🎯 - Ask a quick test question.
    
    4. Code, Quotes & Technical Diagrams: 
       - If the Teaching Strategy quotes a specific line from the PDF, you MUST display that quote prominently on the board using Markdown quote blocks (e.g., `> "The quoted text..."`) before explaining it.
       - If the topic involves PROGRAMMING/CODING, write the code using Markdown fenced code blocks (```python ... ```).
       - If the topic requires a flowchart or table, draw it using Markdown tables or ASCII art in the `board_content`.
       
    5. Board Formatting: The `board_content` MUST BE STUNNING. Use `# Headers`, `**Bold text**`, `>` quotes.
    6. If an Image URL is provided, include it at the very end of the board_content: ![Visual]({image_url})
    
    RESPOND STRICTLY IN JSON FORMAT WITH THESE KEYS:
    {{
        "board_content": "The main technical teaching. MUST USE BEAUTIFUL MARKDOWN. Include technical diagrams (ASCII/Tables) if needed, and the Image URL at the end. Leave EMPTY if this is casual chat.",
        "chat_content": "Short, highly emotional Hinglish chat response for the live chat. If board_content is heavy, chat_content MUST just be a short pointer like 'Look at the board!'. Includes jokes, warnings, or XP celebrations."
    }}
    """)
    
    try:
        formatted_prompt = prompt.format(
            teaching_strategy=state["teaching_strategy"],
            intent_type=state.get("intent_type", "technical_question"),
            student_emotion=state.get("student_emotion", "curious"),
            student_name=state["student_name"],
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

