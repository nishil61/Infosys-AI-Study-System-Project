# Neural Networks Study System

An intelligent AI-powered learning system that helps you understand neural network concepts step-by-step. This project uses advanced AI technology to create personalized quizzes, provide simplified explanations, and track your learning progress.

## What is This Project?

This is a complete educational application that teaches you about Neural Networks in an interactive way. Think of it as your personal AI tutor that:
- Gives you study material automatically
- Creates quiz questions to test your understanding
- Explains difficult concepts in very simple language (like explaining to a 10-year-old)
- Tracks your progress across different topics
- Gives you multiple chances to improve your score

## Key Features (What Makes This Special)

### Milestone 1: Smart Study Material Collection
- **Saved Notes First**: Uses pre-written notes if available for faster learning
- **Web Search Backup**: Automatically searches the internet (using Tavily API) if notes are not available
- **Topic-Focused**: Collects information specifically related to your current checkpoint

### Milestone 2: Intelligent Quiz System
- **Auto Question Generation**: AI creates questions based on study material (no manual question creation needed)
- **Vector Database (FAISS)**: Stores study material in a smart way for quick retrieval
- **Smart Grading**: Evaluates your answers automatically using keyword matching
- **70% Pass Mark**: You need 70% or more to pass each checkpoint
- **Hint System**: Get helpful hints during quiz by clicking "Get Hint" button

### Milestone 3: Feynman Teaching Method
- **Simple Explanations**: When you score below 70%, the AI explains concepts in very simple language
- **Everyday Examples**: Uses real-life analogies (like cooking, building blocks) to explain technical terms
- **Loop-Back Learning**: After explanations, you can retake the quiz with new questions
- **Multiple Attempts**: Get up to 3 chances to improve your score per checkpoint
- **Progress Tracking**: System remembers your retry count even if you go back to main menu

### Milestone 4: Complete Learning Journey
- **Sequential Progression**: Move from one checkpoint to next after passing
- **State Management**: Your progress is saved throughout the session
- **Completion Tracking**: See how many checkpoints you have completed
- **End-to-End Workflow**: Smooth flow from study → quiz → teaching → results

## Topics Covered (Checkpoints)

1. **Basics of Neural Networks** - Learn about neurons, weights, bias, and activation functions
2. **Forward Propagation** - Understand how data moves through network layers
3. **Loss Function** - Know how to measure model performance and accuracy

## Technology Used

- **Python 3.13** - Programming language
- **Streamlit** - For creating beautiful web interface
- **Qwen 2.5-1.5B AI Model** - Fast and smart AI model for generating questions and explanations
- **FAISS Vector Database** - For storing and searching study material efficiently
- **Tavily API** - For searching relevant information from the web
- **LangChain** - For processing text and managing AI workflows
- **PyTorch** - Deep learning library for running AI model

## How to Install and Run

### Step 1: Install Required Software
First, install all necessary Python packages:
```bash
pip install -r requirements.txt
```

### Step 2: Get Tavily API Key
1. Go to [Tavily.com](https://tavily.com) and create a free account
2. Copy your API key
3. Create a file named `.env` in the project folder
4. Add this line in the `.env` file:
```
TAVILY_API_KEY=your_api_key_here
```

### Step 3: Run the Application
Open your terminal/command prompt in the project folder and type:
```bash
streamlit run app_qwen.py
```

The application will open automatically in your web browser!

## How to Use (Step-by-Step Guide)

### First Time Users:

1. **Select a Checkpoint**
   - When you start, you will see three checkpoints
   - Click on any checkpoint button to begin learning
   - Example: Click "Basics of Neural Networks"

2. **Study the Material**
   - Study material will appear automatically
   - Read it carefully to understand the concepts
   - Click "Start Quiz" when you feel ready

3. **Take the Quiz**
   - You will get 2 questions about the topic
   - Type your answers in the text box
   - Click "Get Hint" if you need help
   - Click "Submit Quiz" when done

4. **Check Your Score**
   - If you score **70% or more**: You pass! Click "Complete Checkpoint" to move to next topic
   - If you score **below 70%**: Don't worry! Click "Get Simplified Explanations"

5. **Learn from Mistakes (If score < 70%)**
   - You will see simple explanations for questions you got wrong
   - Read the explanations carefully
   - Click "Retake Quiz" to try again with new questions
   - You get 3 attempts to improve

6. **Track Progress**
   - Top right shows how many checkpoints you completed
   - Complete all 3 checkpoints to finish the course!

## Project Files Explanation

- `app_qwen.py` - Main application with Qwen AI model (fastest and recommended)
- `app.py` - Original version with TinyLlama model
- `infosys.py` - Command-line version (basic, for learning purposes)
- `.env` - Contains your API keys (keep this secret!)
- `README.md` - This instruction file
- `Details.md` - Full project requirements and milestone details

## How It Works Behind the Scenes

### The Complete Learning Flow:
```
Select Checkpoint → Fetch Study Material → Create Vector Database → 
Generate Quiz Questions → You Take Quiz → AI Grades Answers → 
Score ≥ 70%? → Yes: Move to Next Checkpoint
            → No: Get Feynman Explanations → Retake Quiz (max 3 times)
```

### Technical Process (Simplified):

1. **Study Material Collection (Milestone 1)**
   - Checks if notes exist for the topic
   - If not, searches web using Tavily API
   - Collects relevant information

2. **Content Processing (Milestone 2)**
   - Breaks study material into small chunks (800 characters each)
   - Converts chunks into numerical vectors (embeddings)
   - Stores vectors in FAISS database for fast searching

3. **Question Generation (Milestone 2)**
   - AI model reads study material
   - Creates 2 relevant questions automatically
   - Questions are saved for the quiz

4. **Answer Grading (Milestone 2 & 3)**
   - Checks if your answer contains important keywords
   - Gives points based on word count and relevant terms
   - Identifies weak areas (questions with score < 70%)

5. **Feynman Teaching (Milestone 3)**
   - For wrong answers, AI creates simple explanations
   - Uses everyday examples and avoids technical jargon
   - Gives you option to retake quiz with better understanding

6. **Progress Management (Milestone 4)**
   - Tracks completed checkpoints
   - Remembers retry count for each checkpoint
   - Guides you through sequential learning path

## Important Notes

- **First Run Takes Time**: AI model downloads automatically first time (around 3GB). Be patient!
- **Internet Required**: For web search feature and downloading the AI model
- **Pass Mark**: 70% is required to pass each checkpoint
- **Retry Limit**: 3 attempts per checkpoint (count doesn't reset even if you go back to menu)
- **Progress Persistence**: Your retry count and completed checkpoints are saved during the session
- **AI Model**: Qwen 2.5-1.5B runs on your computer (no data sent to cloud)

## Troubleshooting (Common Problems)

**Problem**: Application is slow
- **Solution**: First time is slow because AI model is downloading. Next time will be faster.

**Problem**: "API key error" message
- **Solution**: Make sure you created `.env` file with correct Tavily API key

**Problem**: No study material appears
- **Solution**: Check your internet connection. Web search needs internet.

**Problem**: Retry count not saving
- **Solution**: This is now fixed! Retry count is saved per checkpoint.

## For Developers and Presentations

- See `Details.md` for complete milestone requirements
- All code is well-commented with milestone labels
- Search for "MILESTONE 1", "MILESTONE 2", etc. in code to find specific features

## Credits

**Developed By**: Nishil (Infosys Internship Project)
**AI Model**: Qwen 2.5-1.5B by Alibaba Cloud
**Search API**: Tavily
**Framework**: Streamlit, LangChain, PyTorch

---

**Note for Freshers**: Don't worry if some technical terms sound complicated. Just follow the "How to Use" section step-by-step, and you will be able to run the application successfully. The application is designed to be simple and easy to understand!
