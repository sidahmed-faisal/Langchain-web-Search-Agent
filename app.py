from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import re , logging

from searchtool import summarize_url, latest_summary, answer_followup, topic_from_summary, get_agent, get_auothor

app = FastAPI(title="Web Summarizer Agent")

# Configure logging on log.txt
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("log.txt"),
                        logging.StreamHandler()
                    ])

class InputRequest(BaseModel):
    input: str

# --- URL detection ---
URL_RE = re.compile(
    r"""(?ix)
    \bhttps?://
    (?:[a-z0-9-]+\.)+[a-z]{2,}
    (?:[/?#][^\s]*)?
    """
)
def is_valid_url(s: str) -> bool:
    return bool(URL_RE.search(s.strip()))

agent = get_agent()  # Initialize the agent with memory

@app.post("/summarize")
async def summarize_or_answer(req: InputRequest):
    user_input = req.input.strip()

    # 1) URL -> summarize (follow-up memory is cleared inside summarize_url)
    if is_valid_url(user_input):
        try:
            logging.info(f"Summarizing URL: {user_input}")
            summary = agent.run(user_input)

            logging.info(f"creating topic from summary")
            topic = topic_from_summary(summary)

            logging.info(f"Extracting author from summary")
            author = get_auothor(summary, user_input)

            return {
                "summary": summary,
                "main_topic": topic,
                "author": author,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

    # 2) Not a URL -> answer from cached summary using the chain+memory from searchtool.py
    if latest_summary.get("content", "").strip():
        try:
            logging.info(f"Answering follow-up question: {user_input}")
            response_text = answer_followup(user_input)
            return {
                "question": user_input,
                "response": response_text,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Follow-up failed: {str(e)}")

    # 3) No summary context yet
    raise HTTPException(status_code=400, detail="No context available. Please provide a URL first.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
