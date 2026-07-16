from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS
import json

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
    
    should_intervene: bool
    teaching_strategy: str
    image_url: str
    final_response: str

# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
llm_json = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1).bind(response_format={"type": "json_object"})

def router_node(state: ClassroomState) -> Dict[str, Any]:
    """Decides if the AI should intervene and what it should focus on."""
    prompt = PromptTemplate.from_template("""
    You are the Router Agent for an AI Teacher. 
    Analyze the following classroom chat to decide if the AI should intervene.
    
    Rules:
    - Intervene (True) IF: The student asks a question, seems confused, is completely off-topic, or directly addresses the teacher/AI.
    - DO NOT Intervene (False) IF: Students are just saying "hi", "ok", or having a quick casual chat among themselves that doesn't need a teacher.
    
    Student Speaking: {student_name}
    Message: {question}
    Chat History: {chat_history}
    
    Respond strictly in JSON format:
    {{
        "intervene": true/false,
        "reason": "short reason"
    }}
    """)
    
    formatted_prompt = prompt.format(
        student_name=state["student_name"],
        question=state["question"],
        chat_history=state["chat_history"]
    )
    
    try:
        response = llm_json.invoke(formatted_prompt)
        result = json.loads(response.content)
        return {"should_intervene": result.get("intervene", True)}
    except:
        return {"should_intervene": True} # Default to intervene if parsing fails

def teacher_node(state: ClassroomState) -> Dict[str, Any]:
    """Formulates the teaching strategy strictly using PDF context and avoiding past analogies."""
    prompt = PromptTemplate.from_template("""
    You are the Master Teacher Agent. Your job is to decide HOW to teach the current topic based ONLY on the provided PDF context.
    
    Subject: {subject_name} ({topic_name})
    Context from PDF:
    {context}
    
    Student Message: {question}
    Chat History: {chat_history}
    Already Used Analogies (DO NOT REPEAT THESE): {used_analogies}
    
    Task:
    Draft a short teaching strategy (2-3 sentences). 
    - If the user asks for a new topic, pick a NEW un-discussed topic from the Context.
    - Pick a completely FRESH real-world analogy to explain it.
    - If context is empty, state that the data is missing.
    """)
    
    formatted_prompt = prompt.format(
        subject_name=state["subject_name"],
        topic_name=state["topic_name"],
        context=state["context"],
        question=state["question"],
        chat_history=state["chat_history"],
        used_analogies=", ".join(state["used_analogies"]) if state.get("used_analogies") else "None"
    )
    
    response = llm.invoke(formatted_prompt)
    strategy = response.content.strip()
    
    # Simple extraction of analogy for memory
    new_analogies = state.get("used_analogies", [])
    if "analogy" in strategy.lower() or "example" in strategy.lower():
        new_analogies.append(strategy[:50] + "...") 
        
    return {"teaching_strategy": strategy, "used_analogies": new_analogies}

def visualizer_node(state: ClassroomState) -> Dict[str, Any]:
    """Fetches an image related to the teaching strategy."""
    strategy = state.get("teaching_strategy", "")
    
    # Use a cheap fast LLM call to extract a search keyword from strategy
    keyword_prompt = PromptTemplate.from_template("""
    Extract ONE short, highly visual search keyword (2-3 words max) from the following teaching strategy. 
    It should be an object or scenario (e.g., "bank locker", "hospital database", "shopping mall").
    Strategy: {strategy}
    Keyword only:
    """)
    keyword = llm.invoke(keyword_prompt.format(strategy=strategy)).content.strip().strip('"')
    
    image_url = ""
    try:
        results = DDGS().images(keyword, max_results=1)
        if results and len(results) > 0:
            image_url = results[0].get("image", "")
    except Exception as e:
        print(f"DuckDuckGo Image Search failed: {e}")
        pass
        
    return {"image_url": image_url}

def persona_node(state: ClassroomState) -> Dict[str, Any]:
    """Formats the final response into engaging Hinglish with emojis."""
    prompt = PromptTemplate.from_template("""
    You are the Actor/Persona Agent. Take the Teaching Strategy and format it into the final response for the student.
    
    Teaching Strategy: {teaching_strategy}
    Student Name: {student_name}
    Student Profile: {student_profile}
    Image URL to include (if any): {image_url}
    
    Rules:
    1. Write in a flawless, natural mix of Hinglish and English.
    2. Be super enthusiastic, witty, and encouraging. Use relevant emojis! 🎉🔥🚀
    3. Keep it conversational. Break into short paragraphs.
    4. End with ONE clear, engaging question to keep them hooked. DO NOT bombard them with multiple questions.
    5. If an Image URL is provided, include it at the end using markdown format: ![Visual Example]({image_url})
    
    Final Response:
    """)
    
    formatted_prompt = prompt.format(
        teaching_strategy=state["teaching_strategy"],
        student_name=state["student_name"],
        student_profile=state["student_profile"],
        image_url=state.get("image_url", "")
    )
    
    response = llm.invoke(formatted_prompt)
    return {"final_response": response.content.strip()}

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
