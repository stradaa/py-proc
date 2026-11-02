"""Task-type aliases and trial classification shared by py_proc and pyCheck.

The Rust-executor delegates log their own ``task_type`` (``joystick_intro_rust``,
``simple_touch_rust``) but produce the same states and behav_result as the
Python task they port, so the pipeline processes them as that task.
"""

from typing import Any, Dict

TASK_TYPE_ALIASES = {
    "joystick_intro_rust": "joystick_intro",
    "simple_touch_rust": "simple_touch_task",
}

TOUCH_INPUT_FAILURE_REASON = "touch_input"


def canonical_task_type(task_type: Any) -> str:
    """The task the pipeline should process ``task_type`` as."""
    tt = str(task_type or "").strip()
    return TASK_TYPE_ALIASES.get(tt.lower(), tt)


def is_touch_input_fail(behav_result: Dict[str, Any]) -> bool:
    """True if a joystick trial ended because the subject touched the screen
    (``fail_on_touch_input``). These are not joystick attempts, so performance
    metrics exclude them and report them separately."""
    if not isinstance(behav_result, dict):
        return False
    attempt = behav_result.get("final_attempt")
    if not isinstance(attempt, dict):
        attempts = behav_result.get("attempts") or []
        attempt = attempts[-1] if attempts and isinstance(attempts[-1], dict) else {}
    return str(attempt.get("failure_reason") or "").strip().lower() == TOUCH_INPUT_FAILURE_REASON
