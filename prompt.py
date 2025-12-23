# prompt.py
from __future__ import annotations

import json
from typing import List, Dict, Tuple


MODE_SAME_SUBDOMAIN_SAME_WEBSITE = "same_subdomain_same_website"
MODE_SAME_SUBDOMAIN_DIFF_WEBSITE = "same_subdomain_diff_website"
MODE_DIFF_SUBDOMAIN_DIFF_WEBSITE = "diff_subdomain_diff_website"


def build_prompts(
    candidate_blocks: List[Dict],
    mode: str,
) -> Tuple[str, str]:
    """
    candidate_blocks: LLM에 보여줄 후보 블록 리스트
      - same_website_same_subdomain: [{"website": "...", "tasks": [...], [...] }]
      - diff_website_same_subdomain: [{"website": "...", "tasks": [...] }, {"website": "...", "tasks": [...]} ...]
      - diff_website_diff_subdomain: [{"subdomain": "...", "website": "...", "tasks": [...] }, ...]
    """

    # mode별 요구사항(최소 서로 다른 website/subdomain 개수)
    if mode == MODE_SAME_SUBDOMAIN_SAME_WEBSITE:
        req = (
            "- Select 2 to 4 subtasks from the SAME website.\n"
            "- All selected subtasks must come from that single website.\n"
        )
        header = "Candidate subtasks from ONE website (same subdomain):\n"

    elif mode == MODE_SAME_SUBDOMAIN_DIFF_WEBSITE:
        req = (
            "- Select 2 to 4 subtasks.\n"
            "- Prefer selecting subtasks from at least TWO DIFFERENT websites.\n"
            "- All subtasks must be within the SAME subdomain.\n"
        )
        header = "Candidate websites within ONE subdomain:\n"

    elif mode == MODE_DIFF_SUBDOMAIN_DIFF_WEBSITE:
        req = (
            "- Select 2 to 4 subtasks.\n"
            "- Prefer selecting subtasks from at least TWO DIFFERENT websites.\n"
            "- Prefer selecting subtasks spanning at least TWO DIFFERENT subdomains.\n"
        )
        header = "Candidate websites across MULTIPLE subdomains:\n"

    else:
        raise ValueError(f"Unknown mode: {mode}")

    system_prompt = (
        "You are designing realistic user scenarios for a web agent benchmark.\n"
        "You will be given several concrete subtasks that a web agent could perform.\n"
        "Your job is to:\n"
        "1) Select a subset of 2 to 4 subtasks that could realistically be performed together by a single user.\n"
        "2) Imagine a plausible real-world situation where a user would naturally want to perform ONLY these selected subtasks together.\n"
        "3) Write a short, natural-language description of this situation as 'scenario'.\n"
        "4) Write a single high-level objective that unifies the selected subtasks as 'combined_task'.\n\n"
        "Constraints:\n"
        "- You do NOT need to use all subtasks; selecting a coherent subset of 2–4 is encouraged.\n"
        "- The scenario must sound realistic and coherent for the selected subtasks.\n"
        "- It is better to select fewer subtasks (2 instead of 3–4) than to include a loosely related one.\n"
        f"{req}"
        "- Do NOT rewrite, paraphrase, or summarize the subtasks.\n"
        "- You MUST include the exact selected subtasks verbatim in a list called 'selected_subtasks'.\n"
        "- If it is NOT realistically possible to find ANY subset of 2–4 subtasks that can be naturally combined,\n"
        '  return ONLY the string "PASS" and nothing else.\n'
        "- If you return a scenario, return ONLY valid JSON with exactly these keys:\n"
        '  {"scenario": "...", "combined_task": "...", "selected_subtasks": [...]}'
    )

    user_prompt = (
        header
        + json.dumps(candidate_blocks, indent=2, ensure_ascii=False)
    )

    return mode, system_prompt, user_prompt

candidate_blocks = [
    {
        "website": "example.com",
        "tasks": [
            "Task A",
            "Task B"
        ]
    }
]
mode = MODE_SAME_SUBDOMAIN_SAME_WEBSITE


if __name__ == "__main__":
    mode, system_prompt, user_prompt= build_prompts(candidate_blocks, mode)
    print("mode: ", mode)
    print("system_prompt : ", system_prompt)
    print("user_prompt: ", user_prompt)