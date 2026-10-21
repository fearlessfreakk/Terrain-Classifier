import uuid
from datetime import datetime

def create_session():
    return {
        "id": str(uuid.uuid4())[:8],
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
