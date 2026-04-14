# training/rl_rewards.py

import torch


def compute_rewards(responses: list[str]) -> torch.Tensor:
    rewards = []

    for text in responses:
        score = 0.0
        lower = text.lower()

        # ✅ Encourage Shakespeare tone
        if any(word in lower for word in ["thou", "thy", "thee","doth"]):
            score += 1.0

        # Trying to cut back on the large nonsense
        if len(text.split()) < 20:
            score += 0.5

        # ❌ Penalize modern tone
        if any(word in lower for word in ["lol", "bro", "dude"]):
            score -= 1.0

        # ❌ Penalize AI-breaking character
        if "as an ai" in lower:
            score -= 2.0

        rewards.append(score)

    return torch.tensor(rewards)
