from datetime import timedelta

from temporalio import workflow

from .activities import generate_report


@workflow.defn()
class GenerateReport:
    @workflow.run
    async def run(self):
        report = await generate_report.with_options(
            start_to_close_timeout=timedelta(seconds=30)
        )("Alice", "Hello")
        # report will be "Hello, Alice!"
        return report
