from .activities import generate_report


@generate_report.defn
def generate_report_def(user_name: str, action: str) -> str:
    return f"{action}, {user_name}!"
