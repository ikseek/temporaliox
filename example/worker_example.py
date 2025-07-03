"""
Example demonstrating how to use activities_for_queue to set up a Temporal worker.

This example shows:
1. Declaring activities for different queues
2. Implementing the activities
3. Using activities_for_queue to get implementations for worker setup
"""

from temporaliox.activity import activities_for_queue, decl


# Declare activities for different task queues
@decl(task_queue="email-queue")
def send_email(recipient: str, subject: str, body: str) -> bool:
    """Send an email activity declaration."""
    pass


@decl(task_queue="email-queue")
def send_sms(phone: str, message: str) -> bool:
    """Send SMS activity declaration."""
    pass


@decl(task_queue="processing-queue")
def process_data(data: dict) -> dict:
    """Process data activity declaration."""
    pass


# Implement the activities
@send_email.defn
def send_email_impl(recipient: str, subject: str, body: str) -> bool:
    print(f"Sending email to {recipient}: {subject}")
    print(f"Body: {body}")
    return True


@send_sms.defn
def send_sms_impl(phone: str, message: str) -> bool:
    print(f"Sending SMS to {phone}: {message}")
    return True


@process_data.defn
def process_data_impl(data: dict) -> dict:
    print(f"Processing data: {data}")
    return {"processed": True, "result": data}


def setup_email_worker():
    """Example of setting up a worker for email queue activities."""
    try:
        email_activities = activities_for_queue("email-queue")
        print(f"Email queue activities: {len(email_activities)} functions")

        # In a real application, you would pass email_activities to your worker:
        # worker = Worker(
        #     client,
        #     task_queue="email-queue",
        #     activities=email_activities,
        # )
        return email_activities
    except ValueError as e:
        print(f"Error setting up email worker: {e}")
        return []


def setup_processing_worker():
    """Example of setting up a worker for processing queue activities."""
    try:
        processing_activities = activities_for_queue("processing-queue")
        print(f"Processing queue activities: {len(processing_activities)} functions")
        return processing_activities
    except ValueError as e:
        print(f"Error setting up processing worker: {e}")
        return []


if __name__ == "__main__":
    print("Setting up workers...")

    email_worker_activities = setup_email_worker()
    processing_worker_activities = setup_processing_worker()

    print(f"Email worker has {len(email_worker_activities)} activities")
    print(f"Processing worker has {len(processing_worker_activities)} activities")

    # This would fail because no activities are declared for "unknown-queue"
    unknown_activities = activities_for_queue("unknown-queue")
    print(f"Unknown queue activities: {len(unknown_activities)}")
