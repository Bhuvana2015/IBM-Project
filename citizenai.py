# Install dependencies
!pip install transformers torch gradio -q

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Core response generator
def generate_response(prompt, max_length=1024):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    return response

# ---------------------- Features ----------------------

# 1. City Analysis
def city_analysis(city_name):
    prompt = f"""
    Provide a comprehensive, well-structured analysis of {city_name} including:

    1. Crime Index and public safety statistics
    2. Common types of crimes reported
    3. Accident rates, road safety, and traffic management
    4. Availability of emergency services (police, ambulance, fire department)
    5. Healthcare and hospital accessibility in emergencies
    6. Overall quality of infrastructure related to public safety (CCTV, traffic signals, lighting, etc.)
    7. Final overall safety assessment with pros & cons

    City: {city_name}
    Analysis:
    """
    return generate_response(prompt, max_length=1000)

# 2. Citizen Services
def citizen_interaction(query):
    prompt = f"""
    You are a virtual government helpdesk assistant. Answer the citizen's query in a clear, 
    reliable, and structured manner. Provide useful references if relevant. 

    Format your response as:
    - ✅ Summary Answer
    - 📌 Key Details (step-by-step explanation or guidelines)
    - 📞 Useful Contacts / Official Websites if applicable
    - ⚖️ Note on rules, policies, or exceptions

    Query: {query}
    Response:
    """
    return generate_response(prompt, max_length=1000)

# 3. City Comparison
def city_comparison(city1, city2):
    prompt = f"""
    Compare the safety, crime index, accident rates, emergency services, and overall 
    living conditions between {city1} and {city2}. 

    Provide the comparison in a clear table-like structure with pros & cons for each city.
    """
    return generate_response(prompt, max_length=1200)

# 4. Travel Safety Tips
def travel_tips(city_name):
    prompt = f"""
    Provide important travel safety tips for tourists visiting {city_name}. 
    Include advice on:
    - Common scams to avoid
    - Safe transport methods
    - Nightlife safety
    - Emergency contacts
    - General do's and don'ts
    """
    return generate_response(prompt, max_length=800)

# 5. Emergency Contacts
def emergency_contacts(country):
    prompt = f"""
    Provide the emergency contact numbers for police, ambulance, and fire services in {country}.
    If possible, also include a government helpline for tourists or citizens.
    """
    return generate_response(prompt, max_length=500)

# ---------------------- Gradio UI ----------------------

with gr.Blocks() as app:
    gr.Markdown("## 🏙️ City Safety & Citizen Services AI")
    gr.Markdown("An AI-powered tool for analyzing city safety, accessing government info, and finding emergency resources.")

    with gr.Tabs():
        # Tab 1: City Analysis
        with gr.TabItem("🌆 City Analysis"):
            with gr.Row():
                with gr.Column():
                    city_input = gr.Textbox(
                        label="Enter City Name",
                        placeholder="e.g., New York, London, Mumbai, Tokyo..."
                    )
                    analyze_btn = gr.Button("🔍 Analyze City")

                with gr.Column():
                    city_output = gr.Textbox(
                        label="📊 City Analysis Report",
                        placeholder="The analysis will appear here...",
                        lines=20
                    )
            analyze_btn.click(city_analysis, inputs=city_input, outputs=city_output)

        # Tab 2: Citizen Services
        with gr.TabItem("🛂 Citizen Services"):
            with gr.Row():
                with gr.Column():
                    citizen_query = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about public services, government policies, or civic issues...",
                        lines=4
                    )
                    query_btn = gr.Button("📢 Get Government Info")

                with gr.Column():
                    citizen_output = gr.Textbox(
                        label="📖 Government Response",
                        placeholder="You will get a structured official-like answer here...",
                        lines=20
                    )
            query_btn.click(citizen_interaction, inputs=citizen_query, outputs=citizen_output)

        # Tab 3: City Comparison
        with gr.TabItem("🏙️ City Comparison"):
            with gr.Row():
                with gr.Column():
                    city1 = gr.Textbox(label="City 1", placeholder="e.g., New York")
                    city2 = gr.Textbox(label="City 2", placeholder="e.g., London")
                    compare_btn = gr.Button("⚖️ Compare Cities")

                with gr.Column():
                    comparison_output = gr.Textbox(
                        label="Comparison Report",
                        placeholder="Side-by-side analysis will appear here...",
                        lines=20
                    )
            compare_btn.click(city_comparison, inputs=[city1, city2], outputs=comparison_output)

        # Tab 4: Travel Safety
        with gr.TabItem("✈️ Travel Safety"):
            with gr.Row():
                with gr.Column():
                    travel_city = gr.Textbox(
                        label="Destination City",
                        placeholder="e.g., Bangkok, Paris, Dubai..."
                    )
                    tips_btn = gr.Button("🧳 Get Travel Tips")

                with gr.Column():
                    tips_output = gr.Textbox(
                        label="Travel Safety Guide",
                        placeholder="Safety tips will appear here...",
                        lines=15
                    )
            tips_btn.click(travel_tips, inputs=travel_city, outputs=tips_output)

        # Tab 5: Emergency Contacts
        with gr.TabItem("🚨 Emergency Contacts"):
            with gr.Row():
                with gr.Column():
                    country_input = gr.Textbox(
                        label="Enter Country",
                        placeholder="e.g., USA, India, Germany..."
                    )
                    contact_btn = gr.Button("📞 Get Emergency Numbers")

                with gr.Column():
                    contact_output = gr.Textbox(
                        label="Emergency Numbers",
                        placeholder="Police, Ambulance, Fire contacts will appear here...",
                        lines=10
                    )
            contact_btn.click(emergency_contacts, inputs=country_input, outputs=contact_output)

# Launch app with public link
app.launch(share=True)

