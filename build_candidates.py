import random
from typing import Dict, List, Tuple, Any, Set

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
    subdomain_groups: Dict[str, Dict[str, List[str]]]
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
    selected_subdomain = random.choice(list(subdomain_groups.keys()))
    web_map = subdomain_groups[selected_subdomain]

    candidate_blocks = []
    task_to_website = {}

    for website, tasks in web_map.items():
        chosen_task = random.choice(tasks)
        candidate_blocks.append({"website": website, "tasks": [chosen_task]})
        task_to_website[chosen_task] = website

    return selected_subdomain, candidate_blocks, task_to_website


def build_candidates__diff_subdomain_diff_website(
    subdomain_groups: Dict[str, Dict[str, List[str]]]):

    candidate_blocks = []
    task_to_website = {}
    task_to_subdomain = {}

    for sd, web_map in subdomain_groups.items():

        # 1) subdomain 안에서 website 하나 랜덤 선택
        selected_website = random.choice(list(web_map.keys()))
        tasks = web_map[selected_website]

        # 2) website 안에서 task 하나 랜덤 선택
        selected_task = random.choice(tasks)

        candidate_blocks.append({
            "subdomain": sd,
            "website": selected_website,
            "tasks": [selected_task],
        })

        task_to_website[selected_task] = selected_website
        task_to_subdomain[selected_task] = sd

    return candidate_blocks, task_to_website, task_to_subdomain
