
import re
from sentence_transformers import SentenceTransformer, util


STOP = set("""
the a an and or but if in on at to of for with is are was were
be been being it this that as by from into about using
""".split())


# Load the embedding model once and reuse it
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


def tokens(text):
    return {
        token
        for token in re.findall(r"[a-z0-9]+", (text or "").lower())
        if token not in STOP and len(token) > 1
    }


def evaluate(question, context, answer, reference):
    question_tokens = tokens(question)
    answer_tokens = tokens(answer)
    context_tokens = tokens(context)
    reference_tokens = tokens(reference)

    # 1. Keyword relevance
    keyword_relevance = (
        len(question_tokens & answer_tokens) / len(question_tokens)
        if question_tokens else 0.0
    )

    # 2. Context overlap
    context_overlap = (
        len(answer_tokens & context_tokens) / len(answer_tokens)
        if answer_tokens and context_tokens else 0.0
    )

    # 3. Reference precision and recall
    matched = len(answer_tokens & reference_tokens)

    reference_precision = (
        matched / len(answer_tokens) if answer_tokens else 0.0
    )

    reference_recall = (
        matched / len(reference_tokens) if reference_tokens else 0.0
    )

    # 4. Reference F1
    reference_f1 = (
        2 * reference_precision * reference_recall
        / (reference_precision + reference_recall)
        if reference_precision + reference_recall else 0.0
    )

    # 5. Semantic similarity using sentence embeddings
    if answer.strip() and reference.strip():
        embeddings = semantic_model.encode(
            [answer, reference],
            convert_to_tensor=True
        )

        semantic_similarity = float(
            util.cos_sim(embeddings[0], embeddings[1]).item()
        )

        semantic_similarity = max(0.0, semantic_similarity)
    else:
        semantic_similarity = 0.0

    # 6. Groundedness proxy
    # Measures how much of the answer's meaningful vocabulary
    # is also present in the supplied context.
    if answer_tokens and context_tokens:
        groundedness = (
            len(answer_tokens & context_tokens) / len(answer_tokens)
        )
    else:
        groundedness = 0.0

    return {
        "keyword_relevance": round(keyword_relevance, 4),
        "context_overlap_proxy": round(context_overlap, 4),
        "reference_overlap_proxy": round(reference_recall, 4),
        "reference_precision_proxy": round(reference_precision, 4),
        "reference_f1_proxy": round(reference_f1, 4),
        "semantic_similarity": round(semantic_similarity, 4),
        "groundedness_proxy": round(groundedness, 4),
        "completeness_proxy": round(reference_recall, 4),
        "answer_token_count": len(answer_tokens)
    }
