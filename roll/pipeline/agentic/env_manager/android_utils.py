import json
import string
from typing import Any


def get_format_fields(step: str) -> list[str]:
    return {field for _, field, _, _ in string.Formatter().parse(step) if field}


def read_memory(memory_path: str) -> dict[str, Any]:
    with open(memory_path, "r") as f:
        memory = json.load(f)
    return memory


def get_skill(memory: dict[str, Any], skill_name: str | None, kwargs: dict | None = None) -> dict[str, Any]:
    if skill_name is None:
        return {}
    skill = memory.get(skill_name, None)
    if skill is None:
        return {}
    steps = skill["steps"]
    real_steps = []
    if kwargs:
        for step in steps:
            format_fields = get_format_fields(step)
            need_format_name = format_fields & set(kwargs.keys())
            need = True
            for name in need_format_name:
                if kwargs[name] != "":
                    step = step.replace("{" + name + "}", kwargs[name])
                    need = True
                else:
                    need = False
                    break
            if need:
                real_steps.append(step)
        steps = real_steps
    steps = "\n".join([f"Step {idx + 1}: {plan}" for idx, plan in enumerate(steps)])
    tips = "\n".join([f"{idx + 1}: {tip}" for idx, tip in enumerate(skill["tips"])])
    # print(steps)
    skill_template = """{skill_name}:
    General Steps:
    {steps}
    Useful Tips:
    {tips}
    """
    skill_str = skill_template.format(
        skill_name=skill_name,
        steps=steps,
        tips=tips,
    )
    return skill_str
