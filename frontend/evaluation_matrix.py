#-----------------Evaluation Matrix for NovaCart AI Chatbot-----------------#
"""
evaluation_matrix.py

Pure evaluation logic for NovaCart AI Chatbot
No UI, no backend dependencies
"""

def intent_accuracy(logs):
    """
    Percentage of correctly identified intents
    """
    if not logs:
        return 0.0
    correct = sum(1 for l in logs if l.get("intent_correct") == "Yes")
    return correct / len(logs)


def average_response_rating(logs):
    """
    Average human rating (1–5)
    """
    ratings = [l["response_rating"] for l in logs if l.get("response_rating")]
    if not ratings:
        return 0.0
    return sum(ratings) / len(ratings)


def task_success_rate(logs):
    """
    Percentage of successful task completion
    """
    if not logs:
        return 0.0
    success = sum(1 for l in logs if l.get("task_success") == "Yes")
    return success / len(logs)


