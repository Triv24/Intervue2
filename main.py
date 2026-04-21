from fastapi import FastAPI
import os
import json
import uuid
from typing import List, Dict, Optional, Literal, TypedDict


from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

from google import genai
import wave
from google.genai import types

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI")

# Initialize app
app = FastAPI(
    title="AI Interviewer Backend",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:5173", "http://localhost:5175", "http://localhost:5173/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class CreateSessionRequest(BaseModel):
    job_role: str = Field(..., example="React Developer")
    experience: int = Field(..., ge=0, le=50, example=2)

class CreateSessionResponse(BaseModel):
    session_id: str
    job_role: str
    experience: int
    questions: List[str]
    current_question_idx: int

class SessionState(BaseModel):
    job_role: str
    experience: int
    data: List[dict]
    current_question_idx: int
    final_report: str

class SubmitAnswerRequest(BaseModel):
    answer: str

class SubmitAnswerResponse(BaseModel):
    question_idx: int
    question: str
    feedback: str
    next_question_idx: Optional[int] = None
    next_question: Optional[str] = None

class TTSRequest(BaseModel) :
    text: str = Field(..., example="Hello there")
    voice : str = Field(..., examples=["Kore", "Gacrux", "charon"])

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Hello from AI Interviewer Backend!"}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/sessions", response_model=CreateSessionResponse, status_code=201)
async def create_session(payload: CreateSessionRequest):
    """Create a new interview session using LangGraph workflow."""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    # Initial state for the workflow
    initial_state: InterviewState = {
        "job_role": payload.job_role,
        "experience": payload.experience,
        "data": [],
        "current_question_idx": 0,
        "interview_complete": False,
        "final_report": "",
        "last_question": "",
        "last_answer": None
    }
    
    # Run the workflow up to ask_question
    result = compiled_graph.invoke(initial_state, config=config)
    
    # Get the current state
    state = compiled_graph.get_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=500, detail="Failed to initialize session")
    
    values = state.values
    questions = [row["question"] for row in values.get("data", [])]
    
    return CreateSessionResponse(
        session_id=session_id,
        job_role=values["job_role"],
        experience=values["experience"],
        questions=questions,
        current_question_idx=values.get("current_question_idx", 0),
    )

@app.get("/sessions/{session_id}", response_model=SessionState)
async def get_session(session_id: str):
    """Get session state from LangGraph checkpointer."""
    config = {"configurable": {"thread_id": session_id}}
    state = compiled_graph.get_state(config)
    
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session not found")
    
    values = state.values
    return SessionState(
        job_role=values["job_role"],
        experience=values["experience"],
        data=values["data"],
        current_question_idx=values["current_question_idx"],
        final_report=values.get("final_report", "")
    )

@app.post("/sessions/{session_id}/answers", response_model=SubmitAnswerResponse)
async def submit_answer(session_id: str, payload: SubmitAnswerRequest):
    """Submit an answer using LangGraph node functions."""
    config = {"configurable": {"thread_id": session_id}}
    
    # Get current state
    current_state = compiled_graph.get_state(config)
    if not current_state or not current_state.values:
        raise HTTPException(status_code=404, detail="Session not found")
    
    values = current_state.values
    idx = values.get("current_question_idx", 0)
    
    if idx >= 5:
        raise HTTPException(status_code=400, detail="All questions already answered")

    question = values["data"][idx]["question"]

    # Create state with the answer
    eval_state = {
        **values,
        "last_answer": payload.answer.strip()
    }
    
    # Call evaluate_answer node function directly
    updated_state = evaluate_answer(eval_state)
    
    # Update the graph state
    compiled_graph.update_state(config, updated_state)
    
    # Prepare response
    feedback = updated_state["data"][idx]["feedback"]
    next_q_idx = None
    next_q = None
    
    if updated_state.get("current_question_idx", 0) < 5:
        next_q_idx = updated_state["current_question_idx"]
        next_q = updated_state["data"][next_q_idx]["question"]
    else:
        # All questions answered, generate final report
        final_state = generate_final_report(updated_state)
        compiled_graph.update_state(config, final_state)

    return SubmitAnswerResponse(
        question_idx=idx,
        question=question,
        feedback=feedback,
        next_question_idx=next_q_idx,
        next_question=next_q,
    )

@app.get("/sessions/{session_id}/report")
async def get_report(session_id: str):
    """Get final report from LangGraph checkpointer."""
    config = {"configurable": {"thread_id": session_id}}
    state = compiled_graph.get_state(config)
    
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Session not found")
    
    values = state.values
    if values.get("current_question_idx", 0) < 5:
        raise HTTPException(status_code=400, detail="Interview not yet complete")
    
    return {
        "job_role": values["job_role"],
        "experience": values["experience"],
        "final_report": values.get("final_report", ""),
    }

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        # Initialize OpenAI client
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Generate speech using OpenAI TTS
        def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(rate)
                wf.writeframes(pcm)

        client = genai.Client(api_key=GOOGLE_API_KEY)

        response = client.models.generate_content(
        model="gemini-3.1-flash-tts-preview",
        contents=request.text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=request.voice,
                    )
                )
            ),
        )
        )

        data = response.candidates[0].content.parts[0].inline_data.data

        file_name='speech.wav'
        
                
        return Response(
                content=wave_file(file_name, data),
                media_type="audio/wav",
                headers={
                    "Content-Disposition": "inline; filename=speech.wav"
                }
            )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS Error: {str(e)}")

# ---------------------------
# LLM Setup (use env var OPENAI_API_KEY)
# ---------------------------


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, api_key=GOOGLE_API_KEY)
parser = StrOutputParser()


# ---------------------------
# Prompts
# ---------------------------
GENERATE_QUESTIONS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer. Based on the job role and experience level, generate exactly 5 relevant technical questions.


Job Role: {job_role}
Experience Level: {experience} years


Generate 5 questions that are:
1. Appropriate for the experience level
2. Technical and role-specific
3. Progressive in difficulty
4. Cover different aspects of the role


Return ONLY a JSON array of 5 questions, no explanations:
[
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5"
]
""")


EVALUATE_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer evaluating a candidate's answer.


Job Role: {job_role}
Experience Level: {experience} years
Question: {question}
Candidate's Answer: {answer}


Provide a detailed evaluation including:
1. Technical accuracy (1-10)
2. Completeness of answer (1-10)
3. Clarity of explanation (1-10)
4. Specific feedback and suggestions
5. Overall score (1-10)


Format your response as:
Score: X/10
Technical Accuracy: X/10
Completeness: X/10
Clarity: X/10
Feedback: [Your detailed feedback here]
""")


FINAL_REPORT_PROMPT = ChatPromptTemplate.from_template("""
You are an expert technical interviewer creating a comprehensive interview report.


Job Role: {job_role}
Experience Level: {experience} years


Interview Results:
{interview_results}


Create a professional final report including:
1. Overall assessment
2. Strengths identified
3. Areas for improvement
4. Technical competency score (average of all scores)
5. Recommendation (Pass/Fail/Consider with conditions)
6. Detailed breakdown of each question


Format as a professional report.
""")


# ---------------------------
# LangGraph State Definition
# ---------------------------
class InterviewQA(TypedDict):
    question: str
    answer: str
    feedback: str


class InterviewState(TypedDict):
    job_role: str
    experience: int
    data: List[InterviewQA]
    current_question_idx: int
    interview_complete: bool
    final_report: str
    last_question: str
    last_answer: Optional[str]  # For API input


# ---------------------------
# LangGraph Nodes
# ---------------------------
def get_job_role_and_experience(state: InterviewState) -> InterviewState:
    """Validate job role and experience are provided."""
    if not state.get("job_role") or state.get("experience") is None:
        raise ValueError("job_role and experience must be provided.")
    return state


def generate_questions(state: InterviewState) -> InterviewState:
    """Generate interview questions based on job role and experience."""
    chain = GENERATE_QUESTIONS_PROMPT | llm | parser
    questions_json = chain.invoke({
        "job_role": state["job_role"],
        "experience": state["experience"]
    })
    questions = json.loads(questions_json)
    question_list: List[InterviewQA] = [{"question": q, "answer": "", "feedback": ""} for q in questions]
    
    return {
        **state,
        "data": question_list,
        "current_question_idx": 0,
        "interview_complete": False,
        "final_report": "",
        "last_question": question_list[0]["question"] if question_list else ""
    }


def ask_question(state: InterviewState) -> InterviewState:
    """Get the current question."""
    idx = state.get("current_question_idx", 0)
    if idx >= 5:
        return state
    q_text = state["data"][idx]["question"]
    return {**state, "last_question": q_text}


def evaluate_answer(state: InterviewState) -> InterviewState:
    """Evaluate the candidate's answer and generate feedback."""
    idx = state.get("current_question_idx", 0)
    if idx >= 5:
        return state
    
    # Check if we have an answer to evaluate
    last_answer = state.get("last_answer")
    if not last_answer:
        # No answer provided, just return current state
        return state
    
    question_text = state["data"][idx]["question"]
    answer_text = last_answer.strip()
    
    if not answer_text:
        raise ValueError("No answer provided for current question.")


    chain = EVALUATE_ANSWER_PROMPT | llm | parser
    feedback = chain.invoke({
        "job_role": state["job_role"],
        "experience": state["experience"],
        "question": question_text,
        "answer": answer_text
    })
    
    data_copy = state["data"].copy()
    data_copy[idx]["answer"] = answer_text
    data_copy[idx]["feedback"] = feedback


    new_idx = idx + 1
    next_question = data_copy[new_idx]["question"] if new_idx < 5 else ""
    
    return {
        **state,
        "data": data_copy,
        "current_question_idx": new_idx,
        "last_question": next_question,
        "last_answer": None  # Clear the answer after processing
    }


def generate_final_report(state: InterviewState) -> InterviewState:
    """Generate the final interview report."""
    interview_results = ""
    for i in range(5):
        qd = state["data"][i]
        interview_results += f"\nQuestion {i+1}: {qd['question']}\n"
        interview_results += f"Answer: {qd['answer']}\n"
        interview_results += f"Feedback: {qd['feedback']}\n"
        interview_results += "-" * 40 + "\n"


    chain = FINAL_REPORT_PROMPT | llm | parser
    final_report = chain.invoke({
        "job_role": state["job_role"],
        "experience": state["experience"],
        "interview_results": interview_results
    })
    
    return {
        **state, 
        "final_report": final_report, 
        "interview_complete": True
    }


def should_continue_interview(state: InterviewState) -> Literal["continue", "complete"]:
    """Determine if interview should continue or complete."""
    current_idx = state.get("current_question_idx", 0)
    return "continue" if current_idx < 5 else "complete"


# ---------------------------
# LangGraph Workflow Setup
# ---------------------------
workflow = StateGraph(InterviewState)
workflow.add_node("get_job_role_and_experience", get_job_role_and_experience)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("ask_question", ask_question)
workflow.add_node("evaluate_answer", evaluate_answer)
workflow.add_node("generate_final_report", generate_final_report)


# Simple workflow: just generate questions and ask first question
workflow.add_edge(START, "get_job_role_and_experience")
workflow.add_edge("get_job_role_and_experience", "generate_questions")
workflow.add_edge("generate_questions", "ask_question")
workflow.add_edge("ask_question", END)


# Compile with memory checkpointer for session persistence
checkpointer = MemorySaver()
compiled_graph = workflow.compile(checkpointer=checkpointer)