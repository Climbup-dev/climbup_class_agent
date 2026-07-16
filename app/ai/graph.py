from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
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
    
    should_intervene: bool
    teaching_strategy: str
    image_url: str
    final_response: str

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
llm_json = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1).bind(response_format={"type": "json_object"})

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
        result = json.loads(response.content)
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
    1. "student_analysis": Briefly analyze the student's current state. Are they confused, bored, curious, deliberately disruptive, or abusive?
    2. "pedagogical_decision": Decide the best approach. If they are abusive, choose "Angry Warning". If they are disruptive, choose "Strict Warning". Otherwise choose "Real-world Analogy", "Interactive MCQ Quiz", "Roleplay Scenario", or "Direct Encouraging Answer".
    3. "selected_concept": The exact technical fact/concept from the PDF context you will teach right now. If giving a warning, this can be empty.
    4. "strategy": The final instructions (2-3 sentences) for the Persona Agent on what exactly to say. 
       - If abusive: Instruct the Persona to react like an EXTREMELY ANGRY MAN (Bhai, tameez se baat kar!).
       - If disruptive: Instruct the Persona to give a very strict but slightly funny Hinglish warning. 
       - If Strike Count is 2, mention it's their LAST warning before getting kicked out.
    
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
        result = json.loads(response.content)
        strategy = result.get("strategy", "Teach the concept beautifully.")
        pedagogy = result.get("pedagogical_decision", "")
        
        # Save a snippet of the strategy to avoid repeating
        new_analogies = state.get("used_analogies", [])
        if "analogy" in pedagogy.lower() or "roleplay" in pedagogy.lower():
            new_analogies.append(strategy[:50] + "...") 
            
        logging.info(f"Teacher CoT: {result}")
        return {"teaching_strategy": strategy, "used_analogies": new_analogies}
    except Exception as e:
        logging.error(f"Teacher Error: {e}")
        return {"teaching_strategy": "Explain the concept from the PDF in a fun way.", "used_analogies": state.get("used_analogies", [])}

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
        keyword = llm.invoke(keyword_prompt.format(strategy=strategy)).content.strip().strip('"')
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
    Image URL to include (if any): {image_url}
    
    HIGH EQ RULES:
    1. Write in a flawless, natural mix of Hinglish and English. Treat them like a younger sibling you are mentoring.
    2. Read the Room: 
       - If the strategy is an "Angry Warning" because they swore, be EXTREMELY ANGRY like a furious elder brother/teacher. No emojis, just pure scolding ("Tameez mein rehna seekho!").
       - If it's a normal warning, be strict but keep a tiny bit of humor ("Class se bahar nikal dunga!"). 
       - If it's a game, format it beautifully with emojis (A 🟢, B 🔴, C 🔵).
    3. Make them feel smart when they are right! "Arre waah Amir, ekdum Hacker wali soch hai tumhari!"
    4. Keep paragraphs short (1-2 lines). Break up large text.
    5. End with ONE clear, punchy question or statement to keep them hooked. NEVER bombard them with multiple trailing questions.
    6. If an Image URL is provided, include it exactly like this at the very end: ![Visual]({image_url})
    
    Final Response:
    """)
    
    try:
        formatted_prompt = prompt.format(
            teaching_strategy=state["teaching_strategy"],
            student_name=state["student_name"],
            student_profile=state["student_profile"],
            image_url=state.get("image_url", "")
        )
        response = llm.invoke(formatted_prompt)
        return {"final_response": response.content.strip()}
    except Exception as e:
        logging.error(f"Persona Error: {e}")
        return {"final_response": "I'm having a little trouble thinking right now. Could you repeat that?"}

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

