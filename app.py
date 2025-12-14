import streamlit as st
import os
import warnings
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Dict
import json

# Load environment
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY", "")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from tavily import TavilyClient
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Page config
st.set_page_config(page_title="Neural Networks Study System", page_icon="NN", layout="wide")

# MILESTONE 1: CHECKPOINT STRUCTURE & CONTEXT GATHERING
@dataclass
class Checkpoint:
    id: str
    topic: str
    objectives: List[str]
    pass_mark: float = 0.7  # MILESTONE 2: 70% threshold as per requirements

# MILESTONE 1: Define structured learning checkpoints with specific objectives
CHECKPOINTS: List[Checkpoint] = [
    Checkpoint(
        id="cp1",
        topic="Basics of Neural Networks",
        objectives=[
            "Explain what a neuron is in a neural network",
            "Describe how weights and bias affect the output",
            "Define an activation function and its purpose",
        ]
    ),
    Checkpoint(
        id="cp2",
        topic="Forward Propagation",
        objectives=[
            "How input data go through all the layers?",
            "Calculate output for simple 2-layer network?"
        ]
    ),
    Checkpoint(
        id="cp3",
        topic="Loss Function",
        objectives=[
            "What loss function actually measure?",
            "Difference between training loss and accuracy explain?"
        ]
    )
]

# MILESTONE 1: User-provided notes
USER_NOTES: Dict[str, str] = {
    "cp1": """
Neural network have many neurons in layers. Each neuron take inputs, multiply
them with weights, add bias, then put through activation function like ReLU
or sigmoid. Weights tell how much each input important, bias give little
adjustment. Activation function make it non-linear so model can learn
complex patterns not just straight lines.
    """
}

# MILESTONE 4: STATE MANAGEMENT FOR SEAMLESS MULTI-CHECKPOINT PROGRESSION

if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.current_checkpoint = None
    st.session_state.study_material = None
    st.session_state.search_index = None
    st.session_state.questions = []
    st.session_state.answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.score = 0
    st.session_state.weak_areas = []
    st.session_state.completed_checkpoints = []  # MILESTONE 4: Track completed checkpoints
    st.session_state.show_hint = {}
    st.session_state.hints_cache = {}
    st.session_state.stage = "select"  # select, study, quiz, results, feynman

    # MILESTONE 3: Feynman Teaching Module state
    st.session_state.feynman_explanations = {}
    st.session_state.incorrect_answers = {}
    st.session_state.retry_count = {}  # Per-checkpoint retry tracking
    st.session_state.max_retries = 3

# LLM INTEGRATION: Core Large Language Model for reasoning and generation
@st.cache_resource(show_spinner="Loading Qwen2.5-1.5B model...")
def load_llm():
    try:
        # Allow overriding to a smaller model to fit memory; default to 0.5B to avoid OOM kills
        model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            padding_side='left'
        )
        
        # Better dtype for CPU/GPU
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model.eval()
        
        # Move to device if CPU
        if device == "cpu":
            model = model.to(device)
        
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_web_searcher():
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise ValueError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=api_key)

@st.cache_resource
def load_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)

# Helper functions
def invoke_llm(message, max_tokens=1000):
    tokenizer, model = load_llm()
    
    if tokenizer is None or model is None:
        return "Error: Model not loaded"
    
    messages = [
        {"role": "system", "content": "You are a helpful AI teacher assistant."},
        {"role": "user", "content": message}
    ]
    
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=384,
            padding=False
        )
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                top_k=40,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                num_beams=1,
                use_cache=True
            )
        
        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return generated if generated else ""
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

def get_study_material(checkpoint_obj):
    # MILESTONE 1: Prioritize user-provided notes first
    if checkpoint_obj.id in USER_NOTES and USER_NOTES[checkpoint_obj.id].strip():
        return USER_NOTES[checkpoint_obj.id], "Using saved notes"
    
    # MILESTONE 1: Fallback to web search for dynamic content retrieval
    search_string = f"{checkpoint_obj.topic}: " + "; ".join(checkpoint_obj.objectives)
    
    try:
        tavily_client = load_web_searcher()
        web_results = tavily_client.search(search_string, max_results=3)
    except Exception as e:
        return f"Topic: {checkpoint_obj.topic}\n\nLearning objectives:\n" + "\n".join([f"- {obj}" for obj in checkpoint_obj.objectives]), f"Error: {str(e)}"
    
    all_content = []
    
    # Tavily returns a dict with a "results" list
    results_list = web_results.get("results", web_results) if isinstance(web_results, dict) else web_results

    if isinstance(results_list, list):
        for result in results_list:
            if isinstance(result, dict):
                text_content = result.get("content") or result.get("snippet") or ""
            elif isinstance(result, str):
                text_content = result
            else:
                text_content = str(result)

            if text_content and len(text_content.strip()) > 0:
                all_content.append(text_content)
    elif isinstance(results_list, dict):
        text_content = results_list.get("content", str(results_list))
        if text_content:
            all_content.append(text_content)
    else:
        all_content.append(str(results_list))
    
    if all_content:
        return "\n\n---\n\n".join(all_content), f"Retrieved {len(all_content)} web results"
    else:
        return f"Topic: {checkpoint_obj.topic}\n\nLearning objectives:\n" + "\n".join([f"- {obj}" for obj in checkpoint_obj.objectives]), "No web results; using objectives"

# MILESTONE 2: CONTEXT PROCESSING (Chunking, Embedding, Vector Store)
def make_search_index(material_text):
    embeddings = load_embeddings()  # Load embedding model
    text_splitter = load_text_splitter()  # Chunk size 800, overlap 150
    one_doc = [Document(page_content=material_text)]
    small_pieces = text_splitter.split_documents(one_doc)  # Text chunking
    search_index = FAISS.from_documents(small_pieces, embeddings)  # Vector store
    return search_index

# MILESTONE 2: QUESTION GENERATION
def generate_questions(checkpoint_obj, material, num_questions=2):
    # Generate targeted questions based on checkpoint objectives
    message = f"Generate {num_questions} numbered questions about: {checkpoint_obj.topic}\n\nQuestions:"
    
    reply = invoke_llm(message, max_tokens=80)
    
    questions = []
    lines = reply.split("\n")
    
    for line in lines:
        line = line.strip()
        if not line or len(line) < 10:
            continue
        
        if line[0].isdigit() and ('.' in line[:4] or ')' in line[:4]):
            for sep in ['. ', ') ', ': ']:
                if sep in line[:5]:
                    question = line.split(sep, 1)[1].strip()
                    break
            else:
                question = line
            
            if not question.endswith('?'):
                question += '?'
            
            if len(question) > 15 and question not in questions:
                questions.append(question)
                if len(questions) >= num_questions:
                    break
    
    if len(questions) < 1:
        questions = [obj if obj.endswith('?') else obj + '?' 
                    for obj in checkpoint_obj.objectives[:num_questions]]
    
    return questions[:num_questions]

def get_rag_hint(search_index, question, topic):
    if search_index is None:
        return "Hint not available (no study material indexed)"
    
    try:
        relevant_docs = search_index.similarity_search(question, k=2)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    except Exception:
        return "Unable to retrieve hint"
    
    if not context:
        return f"Unable to retrieve relevant information for: {question}"
    
    prompt = f"Context: {context[:600]}\n\nQuestion: {question}\n\nBrief answer:"
    
    answer = invoke_llm(prompt, max_tokens=1000)
    return answer if answer else "Unable to generate hint"

# MILESTONE 3: FEYNMAN TEACHING MODULE
def generate_feynman_explanation(question, incorrect_answer, search_index):
    """Generate simplified Feynman-style explanation with analogies for incorrect answers."""
    
    # Get relevant context from study material
    context = ""
    if search_index:
        try:
            relevant_docs = search_index.similarity_search(question, k=2)
            context = "\n".join([doc.page_content for doc in relevant_docs])[:500]
        except:
            context = "General neural network concepts"
    
    # Feynman-style prompt: Simple terms + Analogies + Avoid jargon
    feynman_prompt = f"""Explain this concept in the simplest way possible, like teaching a 10-year-old:

Question: {question}

Their confused answer: {incorrect_answer}

Context: {context}

Rules:
1. Use simple everyday analogies (like comparing to cooking, building blocks, etc.)
2. Avoid technical jargon - if you must use it, explain it immediately
3. Use concrete examples
4. Keep it short (2-3 sentences)

Simple explanation:"""
    
    explanation = invoke_llm(feynman_prompt, max_tokens=1000)
    return explanation if explanation else "Let me explain this more simply: " + context[:200]

# MILESTONE 2 & 3: UNDERSTANDING VERIFICATION & KNOWLEDGE GAP IDENTIFICATION
def grade_answers(checkpoint_obj, material, questions, answers):
    all_marks = []
    weak_areas = []
    incorrect_qa = {}  
    for idx, (question, answer) in enumerate(zip(questions, answers)):
        if not answer or len(answer.strip()) < 3:
            all_marks.append(0.0)
            weak_areas.append(question[:100])
            incorrect_qa[idx] = {"question": question, "answer": answer, "score": 0.0}
            continue
        
        # Keyword grading
        answer_lower = answer.lower()
        question_lower = question.lower()
        
        score = 0.0
        word_count = len(answer.split())
        if word_count >= 10:
            score += 0.3
        elif word_count >= 5:
            score += 0.2
        elif word_count >= 3:
            score += 0.1
        
        # Topic-specific keywords
        key_terms = []
        if "neuron" in question_lower:
            key_terms = ["neuron", "input", "weight", "bias", "activation"]
        elif "weight" in question_lower or "bias" in question_lower:
            key_terms = ["weight", "bias", "multiply", "important", "adjust"]
        elif "activation" in question_lower:
            key_terms = ["activation", "function", "relu", "sigmoid", "non-linear"]
        elif "forward" in question_lower or "propagation" in question_lower:
            key_terms = ["layer", "forward", "propagation", "input", "output"]
        elif "loss" in question_lower:
            key_terms = ["loss", "error", "measure", "training", "accuracy"]
        else:
            key_terms = ["neural", "network", "learn", "data", "model"]
        
        terms_found = sum(1 for term in key_terms if term in answer_lower)
        term_score = min(0.5, terms_found * 0.15)
        score += term_score
        
        if any(word in answer_lower for word in ["process", "compute", "calculate"]):
            score += 0.1
        
        if any(word in answer_lower for word in ["multiply", "add", "apply", "function"]):
            score += 0.1
        
        score = max(0.0, min(1.0, score))
        all_marks.append(score)
        
        # MILESTONE 2: Apply 70% threshold for understanding verification
        # MILESTONE 3: Track incorrect answers (< 70%) for Feynman teaching
        if score < 0.7:
            weak_areas.append(question[:100])
            incorrect_qa[idx] = {"question": question, "answer": answer, "score": score}
    
    avg_score = sum(all_marks) / len(all_marks) if all_marks else 0.0
    return avg_score, weak_areas, incorrect_qa

# UI Components
def render_header():
    st.title("Neural Networks Study System")
    st.markdown("---")

# MILESTONE 4: CHECKPOINT SELECTION & SEQUENTIAL PROGRESSION
def render_checkpoint_selection():
    st.header("Select a Checkpoint")
    
    # MILESTONE 4: Filter out completed checkpoints for sequential progression
    available = [cp for cp in CHECKPOINTS if cp.id not in st.session_state.completed_checkpoints]
    
    if not available:
        st.success("Congratulations! You've completed all checkpoints!")
        if st.button("Start Over"):
            st.session_state.completed_checkpoints = []
            st.rerun()
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for cp in available:
            if st.button(f"{cp.topic}", key=f"select_{cp.id}", use_container_width=True):
                st.session_state.current_checkpoint = cp
                st.session_state.stage = "loading"
                # Reset checkpoint-specific state
                st.session_state.study_material = None
                st.session_state.search_index = None
                st.session_state.questions = []
                st.session_state.answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.show_hint = {}
                st.session_state.hints_cache = {}
                st.rerun()
    
    with col2:
        st.metric("Completed", f"{len(st.session_state.completed_checkpoints)}/{len(CHECKPOINTS)}")

# MILESTONE 1 & 2: STUDY MATERIAL DISPLAY & CONTEXT PROCESSING
def render_study_material():
    # Only render study material if explicitly in study stage, not quiz
    if st.session_state.stage != "study":
        return
    
    cp = st.session_state.current_checkpoint
    st.header(f"Study Material: {cp.topic}")
    
    if st.session_state.study_material is None:
        with st.spinner("Fetching study material..."):
            material, source = get_study_material(cp)  # MILESTONE 1: Context gathering
            st.session_state.study_material = material
            st.info(f"{source}")
            
            # MILESTONE 2: Process context (chunk, embed, store in vector DB)
            st.session_state.search_index = make_search_index(material)
    
    with st.expander("View Study Material", expanded=True):
        st.markdown(st.session_state.study_material)
    
    st.markdown("---")

    # Only show start button when explicitly in study stage
    if st.session_state.stage == "study" and not st.session_state.questions:
        if st.button("Start Quiz", key=f"start_quiz_btn_{cp.id}", type="primary", use_container_width=True):
            st.session_state.stage = "quiz"  # Change stage FIRST
            st.session_state.questions = ["Generating..."]  # Mark as generating
            st.rerun()

# MILESTONE 2: QUIZ ASSESSMENT
def render_quiz():
    cp = st.session_state.current_checkpoint
    
    # Ensure study material and search index exist
    if st.session_state.study_material is None or st.session_state.search_index is None:
        with st.spinner("Fetching study material..."):
            material, source = get_study_material(cp)
            st.session_state.study_material = material
            st.session_state.search_index = make_search_index(material)
    
    # Generate questions if marked as generating
    if st.session_state.questions == ["Generating..."]:
        with st.spinner("Generating quiz questions..."):
            st.session_state.questions = generate_questions(cp, st.session_state.study_material)
            st.session_state.answers = {}
            st.session_state.show_hint = {}
            st.session_state.hints_cache = {}
        st.rerun()
    
    st.header(f"Quiz: {cp.topic}")
    st.info(f"Pass mark: {int(cp.pass_mark * 100)}%")  # MILESTONE 2: Display threshold
    
    for idx, question in enumerate(st.session_state.questions):
        st.markdown(f"**Question {idx + 1}/{len(st.session_state.questions)}**")
        st.write(question)
        
        answer = st.text_area(
            "Your Answer:",
            key=f"answer_{idx}",
            value=st.session_state.answers.get(idx, ""),
            height=100
        )
        st.session_state.answers[idx] = answer
        
        # Hint button
        if st.button("Get Hint", key=f"hint_btn_{idx}"):
            st.session_state.show_hint[idx] = True
            st.rerun()
        
        # Display hint if requested
        if st.session_state.show_hint.get(idx, False):
            with st.expander("Hint", expanded=True):
                # Check if hint is already cached
                if idx not in st.session_state.hints_cache:
                    with st.spinner("Generating hint..."):
                        hint = get_rag_hint(st.session_state.search_index, question, cp.topic)
                        st.session_state.hints_cache[idx] = hint
                else:
                    hint = st.session_state.hints_cache[idx]
                
                st.info(hint)
        
        st.markdown("---")
    
    if st.button("Submit Quiz", type="primary", use_container_width=True):
        # Check all answers provided and not empty
        unanswered = []
        for i in range(len(st.session_state.questions)):
            answer = st.session_state.answers.get(i, "").strip()
            if not answer:
                unanswered.append(i + 1)
        
        if unanswered:
            st.error(f"Please answer all questions before submitting! Missing answers for question(s): {', '.join(map(str, unanswered))}")
        else:
            answers_list = [st.session_state.answers.get(i, "") for i in range(len(st.session_state.questions))]
            score, weak, incorrect_qa = grade_answers(
                cp,
                st.session_state.study_material,
                st.session_state.questions,
                answers_list
            )
            st.session_state.score = score
            st.session_state.weak_areas = weak
            st.session_state.incorrect_answers = incorrect_qa
            st.session_state.quiz_submitted = True
            
            # MILESTONE 2: Evaluate score against 70% threshold
            # MILESTONE 3: Trigger Feynman teaching if score < 70%
            current_retry = st.session_state.retry_count.get(cp.id, 0)
            if score < 0.7 and current_retry < st.session_state.max_retries:
                st.session_state.stage = "feynman"  # MILESTONE 3: Route to Feynman node
            else:
                st.session_state.stage = "results"  # MILESTONE 2: Proceed if passed
            st.rerun()

# MILESTONE 3: FEYNMAN TEACHING MODULE
def render_feynman_teaching():
    """Render Feynman-style simplified explanations for incorrect answers."""
    cp = st.session_state.current_checkpoint
    current_retry = st.session_state.retry_count.get(cp.id, 0)
    st.header("Let's Learn Together - Simplified Explanations")
    
    score_pct = st.session_state.score * 100
    st.info(f"Your score: {score_pct:.1f}% - Let me explain the tricky parts in simpler terms!")
    
    st.markdown("---")
    
    # Generate and display Feynman explanations for each incorrect answer
    if st.session_state.incorrect_answers:
        st.markdown("### Understanding Your Mistakes")
        
        for idx, qa_data in st.session_state.incorrect_answers.items():
            question = qa_data["question"]
            user_answer = qa_data["answer"]
            
            with st.expander(f"Question {idx + 1}: {question[:80]}...", expanded=True):
                st.markdown(f"**Your answer:** {user_answer}")
                st.markdown(f"**Score:** {qa_data['score']*100:.0f}%")
                
                # Generate Feynman explanation if not cached
                if idx not in st.session_state.feynman_explanations:
                    with st.spinner("Creating a simple explanation..."):
                        explanation = generate_feynman_explanation(
                            question,
                            user_answer,
                            st.session_state.search_index
                        )
                        st.session_state.feynman_explanations[idx] = explanation
                else:
                    explanation = st.session_state.feynman_explanations[idx]
                
                st.success(f"**Simple Explanation:**\n\n{explanation}")
    
    st.markdown("---")
    
    # MILESTONE 3: LOOP-BACK MECHANISM
    # After Feynman explanation, workflow returns to question generation/verification
    st.markdown("### Ready to Try Again?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(" Retake Quiz (with new questions)", type="primary", use_container_width=True):
            # MILESTONE 3: Track retry attempts per checkpoint
            st.session_state.retry_count[cp.id] = current_retry + 1
            st.session_state.answers = {}
            st.session_state.show_hint = {}
            st.session_state.hints_cache = {}
            st.session_state.feynman_explanations = {}
            st.session_state.quiz_submitted = False
            
            # MILESTONE 3: Generate new questions for adaptive re-assessment
            with st.spinner("Generating new questions based on your weak areas..."):
                st.session_state.questions = generate_questions(cp, st.session_state.study_material)
            
            # MILESTONE 3: Loop back to quiz stage
            st.session_state.stage = "quiz"
            st.rerun()
    
    with col2:
        if st.button("See Final Results", use_container_width=True):
            st.session_state.stage = "results"
            st.rerun()
    
    # Display retry information
    retries_left = st.session_state.max_retries - current_retry
    if retries_left > 0:
        st.info(f"You have {retries_left} more attempt(s) to improve your score!")
    else:
        st.warning("This is your final attempt. Review the explanations carefully!")

# MILESTONE 4: RESULTS & CHECKPOINT PROGRESSION
def render_results():
    cp = st.session_state.current_checkpoint
    st.header("Quiz Results")
    
    score_pct = st.session_state.score * 100
    passed = st.session_state.score >= cp.pass_mark  # MILESTONE 2: Threshold check
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Score", f"{score_pct:.1f}%")
    with col2:
        st.metric("Pass Mark", f"{int(cp.pass_mark * 100)}%")
    with col3:
        if passed:
            st.success("PASSED!")
        else:
            st.error("Not Passed")
    
    if st.session_state.weak_areas:
        st.markdown("### Areas to Review:")
        for area in st.session_state.weak_areas:
            st.markdown(f"- {area}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if passed:
            # MILESTONE 4: Mark checkpoint complete and progress to next
            if st.button("Complete Checkpoint", type="primary", use_container_width=True):
                if cp.id not in st.session_state.completed_checkpoints:
                    st.session_state.completed_checkpoints.append(cp.id)  # MILESTONE 4: Track completion
                # Reset retry count for this checkpoint since it's completed
                if cp.id in st.session_state.retry_count:
                    st.session_state.retry_count[cp.id] = 0
                st.session_state.stage = "select"
                st.session_state.current_checkpoint = None
                st.session_state.study_material = None
                st.session_state.search_index = None
                st.session_state.questions = []
                st.session_state.answers = {}
                st.session_state.quiz_submitted = False
                # MILESTONE 3: Reset Feynman state
                st.session_state.feynman_explanations = {}
                st.session_state.incorrect_answers = {}
                st.rerun()
        else:
            # MILESTONE 3: Loop-back to Feynman teaching for failed attempts
            current_retry = st.session_state.retry_count.get(cp.id, 0)
            if current_retry < st.session_state.max_retries:
                if st.button("Get Simplified Explanations", type="primary", use_container_width=True):
                    st.session_state.stage = "feynman"
                    st.rerun()
            else:
                if st.button("Retake Quiz", type="primary", use_container_width=True):
                    st.session_state.retry_count[cp.id] = 0
                    st.session_state.answers = {}
                    st.session_state.show_hint = {}
                    st.session_state.hints_cache = {}
                    st.session_state.feynman_explanations = {}
                    st.session_state.quiz_submitted = False
                    st.session_state.stage = "quiz"
                    st.rerun()
    
    with col2:
        if st.button("Back to Menu", use_container_width=True):
            st.session_state.stage = "select"
            st.session_state.current_checkpoint = None
            st.session_state.study_material = None
            st.session_state.search_index = None
            st.session_state.questions = []
            st.session_state.answers = {}
            st.session_state.hints_cache = {}
            st.session_state.quiz_submitted = False
            # MILESTONE 3: Reset Feynman state (but preserve retry_count per checkpoint)
            st.session_state.feynman_explanations = {}
            st.session_state.incorrect_answers = {}
            st.rerun()

# MILESTONE 4: MAIN WORKFLOW ORCHESTRATION
def main():
    render_header()
    
    # MILESTONE 4: State-based workflow routing
    if st.session_state.stage == "select":
        render_checkpoint_selection()  # MILESTONE 1: Define checkpoint
    elif st.session_state.stage == "loading":
        st.session_state.stage = "study"
        st.rerun()
    elif st.session_state.stage == "study":
        render_study_material()  # MILESTONE 1 & 2: Gather & process context
    elif st.session_state.stage == "quiz":
        render_quiz()  # MILESTONE 2: Generate questions & assess
    elif st.session_state.stage == "feynman":
        render_feynman_teaching()  # MILESTONE 3: Adaptive teaching
    elif st.session_state.stage == "results":
        render_results()  # MILESTONE 4: Show results & progress

if __name__ == "__main__":
    main()