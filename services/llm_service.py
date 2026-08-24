import requests
from config import GROQ_API_KEY

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "openai/gpt-oss-120b"


def generate_answer(question: str, contexts):
    """
    Send user question + retrieved context to Groq.
    Return the final AI-generated answer.
    """
    if not contexts:
        context_block = "No relevant context found."
    else:
        context_block = "\n\n".join(contexts)

    prompt = f"""Use the following context to answer the user's question.
If the answer is not in the context, say you don't know.

Context:
{context_block}

User question: {question}

Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 300,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()

    data = response.json()
    answer = data["choices"][0]["message"]["content"].strip()

    return answer