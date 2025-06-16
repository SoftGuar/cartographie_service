import httpx 
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

def send_notification(notification_payload, notif_url):
    notify_url = os.getenv("Gateway_URL")+notif_url
    if not notify_url:
        raise ValueError("NOTIFY_URL environment variable is not set")

    try:
        response = httpx.post(notify_url, json=notification_payload)
        response.raise_for_status()  # Raise an error for bad responses
        return response
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to send notification: {e}")
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Notification service returned an error: {e.response.status_code} - {e.response.text}")