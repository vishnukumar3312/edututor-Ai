!pip install transformers torch gradio google-auth google-auth-oauthlib google-api-python-client --quiet

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import gradio as gr
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import pickle, os

# === Google Classroom Setup ===
SCOPES = ['https://www.googleapis.com/auth/classroom.courses.readonly']
def authenticate_google():
    creds = None
    if os.path.exists("token.pkl"):
        with open("token.pkl", "rb") as token:
            creds = pickle.load(token)
    else:
        flow = InstalledAppFlow.from_client_secrets_file("client_secrets.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.pkl", "wb") as token:
            pickle.dump(creds, token)
    return build('classroom', 'v1', credentials=creds)

def sync_courses(user_email):
    service = authenticate_google()
    results = service.courses().list(pageSize=10).execute()
    courses = results.get('courses', [])
    return {
        "email": user_email,
        "courses": [{"name": c["name"], "topics": ["Topic 1", "Topic 2"]} for c in courses]
    }

# === Granite Model ===
class GraniteModel:
    def init(self, model_id="ibm-granite/granite-3.3-2b-instruct"):
        try:
            print(f"🔄 Loading model: {model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
            print("✅ Granite model loaded.")
        except Exception as e:
            print(f"⚠ Error: {e}. Falling back to GPT-2.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")

    def generate_quiz(self, topic, difficulty="medium"):
        prompt = f"Generate a {difficulty} quiz on the topic: {topic}. Provide questions with answers."
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=300, do_sample=True, top_k=50, top_p=0.95)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

granite = GraniteModel()

# === Gradio Interface ===
def student_ui(email, course, topic, difficulty):
    return granite.generate_quiz(topic, difficulty)

def educator_ui():
    dash = {
        "students": [
            {"name": "Alice", "last_score": 85, "last_topic": "Algebra"},
            {"name": "Bob", "last_score": 60, "last_topic": "Kinematics"},
        ],
        "insights": "Students are struggling with Kinematics. Recommend focused revision."
    }
    return f"{dash['insights']}\n\n" + "\n".join([f"{s['name']} - {s['last_topic']} - Score: {s['last_score']}" for s in dash["students"]])

with gr.Blocks() as demo:
    gr.Markdown("# 🎓 EduTutor AI (Colab + Google Classroom)")

    with gr.Tab("Student"):
        email = gr.Textbox(label="Student Email")
        course = gr.Textbox(label="Course Name")
        topic = gr.Textbox(label="Topic")
        difficulty = gr.Radio(["easy", "medium", "hard"], value="medium", label="Difficulty")
        output = gr.Textbox(label="Generated Quiz", lines=10)

        btn = gr.Button("Generate Quiz")
        btn.click(student_ui, inputs=[email, course, topic, difficulty], outputs=output)

    with gr.Tab("Educator"):
        insights = gr.Textbox(label="Educator Insights", lines=10)
        gr.Button("Load Dashboard").click(educator_ui, inputs=[], outputs=insights)

demo.launch(share=True, debug=True)
