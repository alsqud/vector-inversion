import re
import numpy as np


MIL_KEYWORDS = [
    "군", "국방", "작전", "전술", "전략", "부대", "편성", "지휘", "통제",
    "항공", "해군", "육군", "공군", "함정", "전차", "포", "탄", "탄약",
    "기폭", "신관", "폭발", "소이", "화생방", "방독면", "미사일", "레이더",
    "수송", "군수", "보급", "정비", "기동", "방어", "공격", "첩보", "정보",
    "위성", "무기", "장비", "전투", "기뢰", "상륙", "진지", "사격", "병영",
    "국방기술", "방산", "방위", "전력", "군수품"
]


def normalize_text(text):
    """
    Normalize whitespace in text.
    """
    return re.sub(r"\s+", " ", str(text)).strip()


def remove_spaces(text):
    """
    Remove all whitespace characters.
    """
    return "".join(str(text).split())


def get_term(reference_text):
    """
    Extract the term from a term-definition text.

    Expected format:
        term: definition
    """
    reference_text = str(reference_text)

    if ":" in reference_text:
        return reference_text.split(":", 1)[0].strip()

    return reference_text[:30].strip()


def truncate_keep_space(reference_text, predicted_text, ratio=1.2):
    """
    Truncate predicted text based on the non-space length of the reference text.
    Spaces are preserved in the output.
    """
    reference_length = len(remove_spaces(reference_text))
    cut_length = max(1, int(reference_length * ratio))

    output_chars = []
    non_space_count = 0

    for char in str(predicted_text):
        output_chars.append(char)

        if not char.isspace():
            non_space_count += 1

        if non_space_count >= cut_length:
            break

    return "".join(output_chars).strip()


def truncate_no_space(reference_text, predicted_text, ratio=1.2):
    """
    Truncate predicted text after removing spaces.
    """
    reference_length = len(remove_spaces(reference_text))
    cut_length = max(1, int(reference_length * ratio))

    return remove_spaces(predicted_text)[:cut_length]


def lcs_length(a, b):
    """
    Compute the length of the longest common subsequence.
    """
    dp = [0] * (len(b) + 1)

    for x in a:
        previous = 0

        for j, y in enumerate(b, 1):
            temp = dp[j]

            if x == y:
                dp[j] = previous + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])

            previous = temp

    return dp[-1]


def char_rouge_l_f1(reference_text, predicted_text):
    """
    Compute character-level ROUGE-L F1.
    """
    reference_text = remove_spaces(reference_text)
    predicted_text = remove_spaces(predicted_text)

    if len(reference_text) == 0 or len(predicted_text) == 0:
        return 0.0

    lcs = lcs_length(list(reference_text), list(predicted_text))

    recall = lcs / len(reference_text)
    precision = lcs / len(predicted_text)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def keyword_recall_one(reference_text, predicted_text, keywords=None):
    """
    Compute keyword recall for one reference-prediction pair.
    """
    if keywords is None:
        keywords = MIL_KEYWORDS

    reference_keywords = [keyword for keyword in keywords if keyword in reference_text]

    if len(reference_keywords) == 0:
        return np.nan

    matched_count = sum(1 for keyword in reference_keywords if keyword in predicted_text)

    return matched_count / len(reference_keywords)


def term_hit_one(reference_text, predicted_text):
    """
    Check whether the original term appears in the predicted text.
    """
    term = get_term(reference_text)

    if term is None or len(term) < 2:
        return np.nan

    return 1.0 if term in predicted_text else 0.0


def evaluate_text_pairs(reference_texts, predicted_texts, truncate_ratio=1.2):
    """
    Evaluate multiple reference-prediction text pairs.

    Returns:
        Char_ROUGE_L_F1
        Keyword_Recall
        Term_Hit_Rate
    """
    rouge_scores = []
    keyword_scores = []
    term_hit_scores = []

    for reference_text, predicted_text in zip(reference_texts, predicted_texts):
        reference_text = normalize_text(reference_text)
        predicted_text = normalize_text(predicted_text)

        predicted_cut_for_rouge = truncate_no_space(
            reference_text,
            predicted_text,
            ratio=truncate_ratio,
        )

        rouge_scores.append(
            char_rouge_l_f1(reference_text, predicted_cut_for_rouge)
        )

        keyword_scores.append(
            keyword_recall_one(reference_text, predicted_text)
        )

        term_hit_scores.append(
            term_hit_one(reference_text, predicted_text)
        )

    return {
        "Char_ROUGE_L_F1": float(np.nanmean(rouge_scores)),
        "Keyword_Recall": float(np.nanmean(keyword_scores)),
        "Term_Hit_Rate": float(np.nanmean(term_hit_scores)),
    }