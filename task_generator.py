# main.py
from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

from openai import OpenAI
from prompt import (
    build_prompts,
    MODE_SAME_WEBSITE_SAME_SUBDOMAIN,
    MODE_DIFF_WEBSITE_SAME_SUBDOMAIN,
    MODE_DIFF_WEBSITE_DIFF_SUBDOMAIN,
)

# -----------------------
# Config
# -----------------------
TASK_FILE = "/mnt/raid5/parksh/Mind2web/online_mind2web/matched_only.json"
MODEL_NAME = "gpt-5-mini"

# 아래 3개 중 하나로 설정
#MODE = MODE_SAME_WEBSITE_SAME_SUBDOMAIN
MODE = MODE_DIFF_WEBSITE_SAME_SUBDOMAIN
# MODE = MODE_DIFF_WEBSITE_DIFF_SUBDOMAIN

MIN_WEBSITES_PER_SUBDOMAIN = 2
MIN_SUBDOMAINS_FOR_DIFF_MODE = 2  # diff_website_diff_subdomain에서 사용할 최소 subdomain 수

#client = OpenAI()


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
# 2) Candidate builders
# -----------------------
def build_candidates_same_website_same_subdomain(
    subdomain_groups: Dict[str, Dict[str, List[str]]],
    min_tasks_in_website: int = 2,
) -> Tuple[str, List[Dict], Dict[str, str]]:
    """
    같은 subdomain 안에서 'website 하나'를 골라 그 website의 task만 LLM에 제공.
    반환:
      selected_subdomain, candidate_blocks, task_to_website
    """
    # website에 task가 충분히 있는 (sd, website) 조합들 수집
    pairs = []
    for sd, web_map in subdomain_groups.items():
        for w, tasks in web_map.items():
            if len(tasks) >= min_tasks_in_website:
                pairs.append((sd, w))

    if not pairs:
        raise RuntimeError("No (subdomain, website) pair has enough tasks.")

    selected_subdomain, selected_website = random.choice(pairs)
    tasks = subdomain_groups[selected_subdomain][selected_website]

    candidate_blocks = [{"website": selected_website, "tasks": tasks}]
    task_to_website = {t: selected_website for t in tasks}
    return selected_subdomain, candidate_blocks, task_to_website


def build_candidates_same_subdomain_diff_website(
    subdomain_groups: Dict[str, Dict[str, List[str]]],
    min_websites_per_subdomain = MIN_WEBSITES_PER_SUBDOMAIN,
) -> Tuple[str, List[Dict], Dict[str, str]]:
    """
    같은 subdomain 하나를 고르고, 그 안의 여러 website들을 LLM에 제공.
    
    eligible_subdomains: Dict[
        subdomain: str,
        Dict[
            website: str,
            List[task: str]
        ]]
    """
    eligible_subdomains = {
        sd: web_map
        for sd, web_map in subdomain_groups.items()
        if len(web_map) >= min_websites_per_subdomain
    }

    #print("eligible_subdomains : ", eligible_subdomains)
    if not eligible_subdomains:
        raise RuntimeError("No eligible subdomain has enough websites.")

    selected_subdomain = random.choice(list(eligible_subdomains.keys()))
    web_map = subdomain_groups[selected_subdomain]

    candidate_blocks = []
    task_to_website = {}

    for website, tasks in web_map.items():
        chosen_task = random.choice(tasks)
        candidate_blocks.append({"website": website, "tasks": [chosen_task]})
        task_to_website[chosen_task] = website

    return selected_subdomain, candidate_blocks, task_to_website


def build_candidates__diff_subdomain_diff_website(
    subdomain_groups: Dict[str, Dict[str, List[str]]],
    min_websites_per_subdomain = MIN_WEBSITES_PER_SUBDOMAIN,
    min_subdomains_total = MIN_SUBDOMAINS_FOR_DIFF_MODE):

    # 1) subdomain 중 website가 충분한 것만
    eligible_subdomains = [
        sd for sd, web_map in subdomain_groups.items()
        if len(web_map) >= min_websites_per_subdomain
    ]
    if len(eligible_subdomains) < min_subdomains_total:
        raise RuntimeError("Not enough eligible subdomains to sample from.")

    # 2) 서로 다른 subdomain 여러 개 뽑기
    selected_subdomains = random.sample(eligible_subdomains, k=min_subdomains_total)

    candidate_blocks = []
    task_to_website = {}
    task_to_subdomain = {}

    # 3) 각 subdomain에서 website/task를 후보에 추가 (전부 추가해도 되고, 샘플링해도 됨)
    for sd in selected_subdomains:
        web_map = subdomain_groups[sd]
        for website, tasks in web_map.items():
            candidate_blocks.append({"subdomain": sd, "website": website, "tasks": tasks})
            for t in tasks:
                task_to_website[t] = website
                task_to_subdomain[t] = sd

    return selected_subdomains, candidate_blocks, task_to_website, task_to_subdomain


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

    if mode == MODE_SAME_WEBSITE_SAME_SUBDOMAIN:
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

    if mode == MODE_DIFF_WEBSITE_SAME_SUBDOMAIN:
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

    if mode == MODE_DIFF_WEBSITE_DIFF_SUBDOMAIN:
        # website 2개 이상 + subdomain 2개 이상(가능하면)
        if task_to_subdomain is None:
            return {"status": "PASS", "reason": "task_to_subdomain missing"}

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

    if MODE == MODE_SAME_WEBSITE_SAME_SUBDOMAIN:
        selected_subdomain, candidate_blocks, task_to_website = \
            build_candidates_same_website_same_subdomain(subdomain_groups)
        
        print("same website : ", selected_subdomain)
        print(candidate_blocks)

        system_prompt, user_prompt = build_prompts(candidate_blocks, MODE)
        #raw_output = call_llm(system_prompt, user_prompt)

        #validated = validate_output(raw_output, MODE, task_to_website)
        #final_result = {"mode": MODE, "subdomain": selected_subdomain, **validated}

    elif MODE == MODE_DIFF_WEBSITE_SAME_SUBDOMAIN:
        selected_subdomain, candidate_blocks, task_to_website = \
            build_candidates_same_subdomain_diff_website(
                subdomain_groups,
                min_websites_per_subdomain=MIN_WEBSITES_PER_SUBDOMAIN
            )
        print("diff website : ", selected_subdomain)
        print(candidate_blocks)

        system_prompt, user_prompt = build_prompts(candidate_blocks, MODE)
        #raw_output = call_llm(system_prompt, user_prompt)

        # validated = validate_output(raw_output, MODE, task_to_website)
        # final_result = {"mode": MODE, "subdomain": selected_subdomain, **validated}

    elif MODE == MODE_DIFF_WEBSITE_DIFF_SUBDOMAIN:
        selected_subdomains, candidate_blocks, task_to_website, task_to_subdomain = \
            build_candidates__diff_subdomain_diff_website(
                subdomain_groups,
                min_websites_per_subdomain=MIN_WEBSITES_PER_SUBDOMAIN,
                min_subdomains_total=MIN_SUBDOMAINS_FOR_DIFF_MODE,
            )

        system_prompt, user_prompt = build_prompts(candidate_blocks, MODE)
        #raw_output = call_llm(system_prompt, user_prompt)

        # validated = validate_output(raw_output, MODE, task_to_website, task_to_subdomain)
        # final_result = {"mode": MODE, "subdomains_pool": selected_subdomains, **validated}

    else:
        raise ValueError(f"Unknown MODE: {MODE}")

    print(json.dumps(final_result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
