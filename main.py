# main.py
from __future__ import annotations
import os

import json
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

from openai import OpenAI
from prompt import (
    build_prompts,
    MODE_SAME_SUBDOMAIN_SAME_WEBSITE,
    MODE_SAME_SUBDOMAIN_DIFF_WEBSITE,
    MODE_DIFF_SUBDOMAIN_DIFF_WEBSITE,
)
from build_candidates import (
    build_candidates_same_website_same_subdomain,
    build_candidates_same_subdomain_diff_website,
    build_candidates__diff_subdomain_diff_website
)
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# -----------------------
# Config
# -----------------------
TASK_FILE = "/mnt/raid5/parksh/Mind2web/online_mind2web/matched_only.json"
MODEL_NAME = "gpt-5-mini"

# 아래 3개 중 하나로 설정
MODE = MODE_SAME_SUBDOMAIN_SAME_WEBSITE
#MODE = MODE_SAME_SUBDOMAIN_DIFF_WEBSITE
#MODE = MODE_DIFF_SUBDOMAIN_DIFF_WEBSITE


client = OpenAI(api_key=api_key)


# -----------------------
# 1) Load & group
# -----------------------
def load_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def group_by_subdomain_website(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, List[str]]]:
    # subdomain -> website -> tasks
    subdomain_groups = defaultdict(lambda: defaultdict(list))
    for item in data:
        sd = item["sub_domain"]
        w = item["website"]
        t = item["confirmed_task"]
        subdomain_groups[sd][w].append(t)
    return subdomain_groups




# -----------------------
# 3) LLM call & validation
# -----------------------
def call_llm(system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


# raw_output = original output from llm
def validate_output(
    raw_output: str,
    mode: str,
    task_to_website: Dict[str, str],
    task_to_subdomain: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    
    if raw_output == "PASS":
        return {"status": "PASS"}

    scenario_data = json.loads(raw_output)
    selected_subtasks = scenario_data["selected_subtasks"]

    used_websites: Set[str] = {
        task_to_website[t] for t in selected_subtasks if t in task_to_website
    }

    if mode == MODE_SAME_SUBDOMAIN_SAME_WEBSITE:
        # 반드시 website 1개
        if len(used_websites) != 1:
            return {"status": "PASS", "reason": "expected exactly 1 website"}
        return {
            "status": "OK",
            "selected_websites": list(used_websites),
            "scenario": scenario_data["scenario"],
            "combined_task": scenario_data["combined_task"],
            "selected_subtasks": selected_subtasks,
        }

    if mode == MODE_SAME_SUBDOMAIN_DIFF_WEBSITE:
        # website 2개 이상
        if len(used_websites) < 2:
            return {"status": "PASS", "reason": "selected_subtasks use < 2 websites"}
        return {
            "status": "OK",
            "selected_websites": list(used_websites),
            "scenario": scenario_data["scenario"],
            "combined_task": scenario_data["combined_task"],
            "selected_subtasks": selected_subtasks,
        }

    if mode == MODE_DIFF_SUBDOMAIN_DIFF_WEBSITE:
        used_subdomains: Set[str] = {
            task_to_subdomain[t] for t in selected_subtasks if t in task_to_subdomain
        }

        if len(used_websites) < 2:
            return {"status": "PASS", "reason": "selected_subtasks use < 2 websites"}
        if len(used_subdomains) < 2:
            return {"status": "PASS", "reason": "selected_subtasks use < 2 subdomains"}

        return {
            "status": "OK",
            "selected_websites": list(used_websites),
            "selected_subdomains": list(used_subdomains),
            "scenario": scenario_data["scenario"],
            "combined_task": scenario_data["combined_task"],
            "selected_subtasks": selected_subtasks,
        }

    return {"status": "PASS", "reason": f"unknown mode {mode}"}


# -----------------------
# 4) Main
# -----------------------
def main():
    data = load_data(TASK_FILE)
    subdomain_groups = group_by_subdomain_website(data)

    if MODE == MODE_SAME_SUBDOMAIN_SAME_WEBSITE:
        selected_subdomain, candidate_blocks, task_to_website = \
            build_candidates_same_website_same_subdomain(subdomain_groups)

        system_prompt, user_prompt = build_prompts(candidate_blocks, MODE)
        raw_output = call_llm(system_prompt, user_prompt)

        validated = validate_output(raw_output, MODE, task_to_website)
        final_result = {"mode": MODE, "subdomain": selected_subdomain, **validated}

    elif MODE == MODE_SAME_SUBDOMAIN_DIFF_WEBSITE:
        selected_subdomain, candidate_blocks, task_to_website = \
            build_candidates_same_subdomain_diff_website(
                subdomain_groups,
            )

        system_prompt, user_prompt = build_prompts(candidate_blocks, MODE)
        raw_output = call_llm(system_prompt, user_prompt)

        validated = validate_output(raw_output, MODE, task_to_website)
        final_result = {"mode": MODE, "subdomain": selected_subdomain, **validated}

    elif MODE == MODE_DIFF_SUBDOMAIN_DIFF_WEBSITE:
        candidate_blocks, task_to_website, task_to_subdomain = \
            build_candidates__diff_subdomain_diff_website(
                subdomain_groups
            )
        system_prompt, user_prompt = build_prompts(candidate_blocks, MODE)
        raw_output = call_llm(system_prompt, user_prompt)

        validated = validate_output(raw_output, MODE, task_to_website, task_to_subdomain)
        final_result = {"mode": MODE, **validated}

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print(json.dumps(final_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
