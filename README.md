# Cartographie Service 

This project provides a backend service for a cartography system. It exposes APIs for managing and accessing cartographic data using FastAPI and PostgreSQL.

---

## Requirements

- **Python**: `3.9`
- **Virtual environment** (optional but recommended)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/SoftGuar/cartographie_service.git
   cd cartographie_service
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙Configuration
create a ".env" file and paste in it the content you find in the ".env.example" file

## Running the Application

To start the FastAPI server:

```bash
python main.py
```

Then open your browser and go to:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Testing

To test the API endpoints, use:

- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- Postman or any other API client

---

