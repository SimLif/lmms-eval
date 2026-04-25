# Default prompts for different judge types
#
# BINARY_JUDGE_PROMPT: lenient semantic-match prompt inspired by MedEvalKit
# (DAMO / Lingshu), with a normal 0/1 convention (1 = correct).
# See /afs_.../memory/references/2026-04-24-open-ended-judge-analysis.md
# for the analysis that motivated switching from the old "strict" prompt
# to this lenient version — strict was rejecting semantically-equivalent
# medical answers like "Lymphoma" vs "hematologic".
BINARY_JUDGE_PROMPT = """Your task is to determine whether the user's answer is correct based on the provided question and standard answer.

Be LENIENT: consider the user's answer correct ({positive}) if any of these hold:
- The user's answer expresses a similar meaning as the standard answer.
- The user's answer is another valid interpretation (e.g., a specific disease when the standard is the organ system it belongs to; a synonym; a clinically equivalent phrasing).
- Only formatting / capitalization / spacing differs from the standard answer.
- A numerical answer matches within reasonable precision (value and unit when required).

Score {negative} only when the user's answer is clearly wrong — different organ, different disease, different finding, or semantically unrelated.

# Input
Question: {question}
Standard answer: {answer}
User's answer: {prediction}

# Output format (strict)
<think>one-line reasoning</think>
<judge>{positive} for correct, {negative} for incorrect</judge>"""


COMPARATIVE_JUDGE_PROMPT = """We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of {min_score} to {max_score}, where a higher score indicates better overall performance.
Please first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space.
In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Question]
{question}

{context_section}

[Assistant 1]
{response1}
[End of Assistant 1]

[Assistant 2]
{response2}
[End of Assistant 2]

[System]
{evaluation_instruction}"""


CORRECTNESS_JUDGE_PROMPT = """You are given a question, the solution and the correct answer. Please determine if the solution matches the correct answer.
Focus only on the mathematical or semantic correctness of the content. Ignore any differences in formatting, such as LaTeX syntax, symbols, styles, or additional wrappers (e.g., \\boxed, $...$, or similar). Compare only the core mathematical or textual meaning of the solution and the correct answer.
The process or reasoning leading to the Solution is irrelevant, ONLY the correctness of the result matters.
Return only "{positive}" if the solution is correct or "{negative}" if it is incorrect.
Only return "{positive}" or "{negative}" with no additional text or formatting.

Question:
{question}
--------------------------------
Correct Answer:
{answer}
--------------------------------
Solution:
{prediction}
--------------------------------"""
