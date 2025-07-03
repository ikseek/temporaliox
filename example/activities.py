from datetime import timedelta

from temporaliox.activity import decl


@decl(task_queue="playwright", start_to_close_timeout=timedelta(seconds=60))
def generate_report(user_name: str, action: str) -> str:
    pass
