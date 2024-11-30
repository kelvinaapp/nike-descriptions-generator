from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

from .generate import generate_description

app = FastAPI(
    title="Nike Product Description Generator",
    description="Generate Nike product descriptions using AI",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTML content
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Nike Product Description Generator</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #000;
            text-align: center;
        }}
        form {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        input[type="text"] {{
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            box-sizing: border-box;
        }}
        button {{
            background-color: #007bff;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background-color 0.2s;
        }}
        button:hover {{
            background-color: #0056b3;
        }}
        button:disabled {{
            background-color: #80bdff;
            cursor: not-allowed;
        }}
        #result {{
            margin-top: 20px;
            padding: 15px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background-color: #fff;
            min-height: 100px;
        }}
        .loading {{
            opacity: 0.7;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Nike Product Description Generator</h1>
        <form id="generateForm">
            <input type="text" name="input_text" id="input_text" placeholder="Enter your product details here..." />
            <button type="submit" id="submitBtn">Generate Description</button>
        </form>
        <div id="result">
            {result}
        </div>
    </div>
    
    <script>
        document.getElementById('generateForm').addEventListener('submit', async function(e) {{
            e.preventDefault();
            
            const form = e.target;
            const submitBtn = form.querySelector('button[type="submit"]');
            const resultDiv = document.getElementById('result');
            
            // Disable button and show loading state
            submitBtn.disabled = true;
            submitBtn.textContent = 'Generating...';
            resultDiv.classList.add('loading');
            
            try {{
                const response = await fetch('/generate', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/x-www-form-urlencoded',
                    }},
                    body: new URLSearchParams(new FormData(form))
                }});
                
                const data = await response.json();
                resultDiv.textContent = data.result;
            }} catch (error) {{
                resultDiv.textContent = 'Error: ' + error.message;
            }} finally {{
                // Reset button and loading state
                submitBtn.disabled = false;
                submitBtn.textContent = 'Generate Description';
                resultDiv.classList.remove('loading');
            }}
        }});
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return HTML_CONTENT.format(result="Generated description will appear here...")

@app.post("/generate")
async def generate(input_text: str = Form(...)):
    try:
        generated_text = generate_description(input_text)
        return JSONResponse(content={"result": generated_text})
    except Exception as e:
        return JSONResponse(content={"result": f"Error: {str(e)}"}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
