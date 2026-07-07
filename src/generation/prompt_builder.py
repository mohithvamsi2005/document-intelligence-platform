"""
Prompt Builder Module
----------------------
Builds structured prompts for the RAG pipeline.

WHY a separate module for prompts:
- Prompts are the #1 lever for LLM output quality
- Keeping them separate makes them easy to version and test
- Different use cases need different prompt structures
- This is standard practice in production RAG systems

The prompt structure we use:
  1. System prompt  → tells LLM its role and rules
  2. Context block  → retrieved document chunks
  3. Chat history   → previous conversation turns
  4. User question  → current question
"""

from typing import List, Dict


# ── System Prompts ────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are an expert Document Intelligence Assistant.
Your job is to answer questions based STRICTLY on the provided document context.

RULES you must follow:
1. Answer ONLY from the context provided below — never from prior knowledge
2. If the answer is not in the context, say exactly:
   "I could not find this information in the provided documents."
3. Always cite your sources using [Source N] notation
4. Be concise but complete — do not pad your answer
5. If multiple sources support the answer, cite all of them
6. Preserve any numbers, dates, or technical terms exactly as they appear
7. If the question is vague (like "what is this about"), summarize
   the main topics you can find across ALL the provided context chunks.
   Do not say you cannot answer — instead summarize what you see.
You are grounded in facts. You do not guess. You do not hallucinate."""


STANDALONE_QUESTION_PROMPT = """Given the conversation history below and a follow-up question,
rephrase the follow-up question to be a complete standalone question.
The standalone question must contain all necessary context from the history.

Conversation History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""


# ── Prompt Builders ───────────────────────────────────────────────────────────

def build_rag_prompt(
    question: str,
    context: str,
    chat_history: List[Dict] = None
) -> List[Dict]:
    """
    Builds the complete message list for a RAG query.
    Returns OpenAI-compatible message format.

    Args:
        question     : Current user question
        context      : Formatted context string from retriever
        chat_history : List of previous messages
                       [{"role": "user", "content": "..."},
                        {"role": "assistant", "content": "..."}]

    Returns:
        List of message dicts ready for LLM API call:
        [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},  # history
            {"role": "assistant", "content": "..."},  # history
            {"role": "user",      "content": "..."}   # current question
        ]
    """
    messages = []

    # 1. System message — sets the LLM's behavior
    messages.append({
        "role"   : "system",
        "content": RAG_SYSTEM_PROMPT
    })

    # 2. Chat history — gives the LLM memory of the conversation
    if chat_history:
        for turn in chat_history:
            messages.append({
                "role"   : turn["role"],
                "content": turn["content"]
            })

    # 3. Current user message — context + question combined
    user_message = f"""Here is the relevant context from the documents:

{context}

---

Based ONLY on the context above, please answer this question:
{question}

Remember to cite sources using [Source N] notation."""

    messages.append({
        "role"   : "user",
        "content": user_message
    })

    return messages


def build_standalone_question_prompt(
    question: str,
    chat_history: List[Dict]
) -> str:
    """
    Builds a prompt to rewrite a follow-up question into
    a standalone question that includes all necessary context.

    WHY: If a user asks "What about the refund policy?" after
    asking about returns, we need to rewrite this as
    "What is the refund policy for returned items?"
    before sending to the retriever. Otherwise the retriever
    gets a vague query and returns bad results.

    This is Feature #13 — Query Rewriting.

    Args:
        question     : The follow-up question
        chat_history : Previous conversation turns

    Returns:
        A prompt string for the rewriting LLM call
    """
    # Format chat history as readable text
    history_text = ""
    for turn in chat_history[-4:]:  # only last 4 turns for brevity
        role    = "Human" if turn["role"] == "user" else "Assistant"
        history_text += f"{role}: {turn['content']}\n"

    prompt = STANDALONE_QUESTION_PROMPT.format(
        chat_history=history_text,
        question=question
    )

    return prompt


def count_tokens_approximate(text: str) -> int:
    """
    Approximates token count without calling the tokenizer.
    Rule of thumb: 1 token ≈ 4 characters in English.
    Used for cost estimation (Feature #19).

    Args:
        text: Any string

    Returns:
        Approximate token count
    """
    return len(text) // 4


def build_prompt_stats(messages: List[Dict]) -> Dict:
    """
    Returns stats about a built prompt.
    Used for token tracking (Feature #18) and cost estimation.

    Args:
        messages: The message list built by build_rag_prompt

    Returns:
        Dictionary with token estimates and cost estimates
    """
    # Sum up characters across all message content strings
    total_chars   = sum(len(m["content"]) for m in messages)

    # Approximate tokens directly from the integer — 1 token ≈ 4 chars
    approx_tokens = total_chars // 4

    # GPT-3.5-turbo pricing (2026)
    input_cost  = (approx_tokens / 1000) * 0.0005
    output_cost = (200 / 1000) * 0.0015

    stats = {
        "message_count"        : len(messages),
        "total_chars"          : total_chars,
        "approx_input_tokens"  : approx_tokens,
        "approx_output_tokens" : 200,
        "estimated_cost_usd"   : round(input_cost + output_cost, 6)
    }

    return stats