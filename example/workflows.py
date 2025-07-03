from temporalio import workflow

from .activities import generate_report


@workflow.defn()
class GenerateReport:
    @workflow.run
    async def run(self):
        report = await generate_report("Alice", "Hello")
        # report should become 'Hello, Alice!'. temporalio.workflow.execute_activity
        # should be called under the hood
        return report
