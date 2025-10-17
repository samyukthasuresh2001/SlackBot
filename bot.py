import os
from pathlib import Path
import requests
import pandas as pd
import numpy as np # Added numpy import for consistency with original code
from flask import Flask, request, jsonify
from slackeventsapi import SlackEventAdapter
import slack
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from threading import Thread
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from serpapi import GoogleSearch
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
from dotenv import load_dotenv
import base64
import textwrap

# ---------------------------
# Load environment variables
# ---------------------------
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# --- Configuration Constants ---
# Assuming a verified sender email is set in your environment
SENDER_EMAIL = os.environ.get('SENDGRID_SENDER_EMAIL', 'default@example.com')

# ---------------------------
# Flask & Slack Setup
# ---------------------------
app = Flask(__name__)
slack_event_adapter = SlackEventAdapter(os.environ['SIGNING_SECRET'], '/slack/events', app)
client = slack.WebClient(token=os.environ['SLACK_TOKEN'])
# BOT_ID is fetched later in the thread for better practice, but kept here for existing code structure
BOT_ID = client.api_call("auth.test")['user_id'] 

# ---------------------------
# Embeddings Model (Local)
# ---------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(keywords):
    return model.encode(keywords, convert_to_numpy=True, show_progress_bar=False)

# ---------------------------
# File Processing (No changes needed)
# ---------------------------
def process_keywords(file_name):
    try:
        if file_name.endswith('.csv'):
            df = pd.read_csv(file_name)
        elif file_name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_name)
        else:
            return []
        if 'keywords' not in df.columns:
            return []
        # Return raw keywords for report: raw_keywords, cleaned_keywords
        return df['keywords'].dropna().astype(str).tolist()
    except Exception as e:
        print(f"Error reading file: {e}")
        return []

# ---------------------------
# Clustering (No changes needed)
# ---------------------------
def cluster_keywords(keywords, embeddings, n_clusters=3):
    n_clusters = min(n_clusters, len(keywords))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    clustered = {}
    for label, keyword in zip(labels, keywords):
        clustered.setdefault(label, []).append(keyword)
    return clustered

# ---------------------------
# SerpAPI: Fetch Top Web Snippets (Refined)
# ---------------------------
def get_top_snippets(cluster_keywords, num_results=3):
    """Fetches top snippets for the cluster's main keyword."""
    if not cluster_keywords:
        return []
    
    # Use the top keyword as the search query
    main_kw = cluster_keywords[0]
    search_query = f"guide to {main_kw}"
    
    try:
        params = {
            "engine": "google",
            "q": search_query,
            "api_key": os.environ['SERPAPI_KEY'],
            "num": num_results # Fetch the top 3 results
        }
        search_results = GoogleSearch(params).get_dict()
        snippets = []
        for r in search_results.get("organic_results", []):
            snippet = r.get("snippet")
            url = r.get("link")
            # Only use snippets for outline generation
            if snippet:
                snippets.append(snippet)
        
        # Return a list of snippets and the top URL for the report
        top_url = search_results.get("organic_results", [{}])[0].get("link", "N/A")
        return snippets, top_url
        
    except Exception as e:
        print(f"Error fetching snippets for '{main_kw}': {e}")
        return [], "N/A"

# ---------------------------
# Summarizer for Outline (Refined)
# ---------------------------
# Using the smaller, faster T5 for pipeline on a free-tier environment
# Note: facebook/bart-large-cnn is very large and may cause memory errors in Flask/Thread environment
summarizer_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_outline_from_snippets(snippets):
    """Generates a brief summary/outline based on web snippets."""
    if not snippets:
        return ["No outline available from web search."]
        
    text = " ".join(snippets)
    # A simplified, less resource-intensive outline generation
    try:
        summary_text = summarizer_pipeline(
            text, 
            max_length=60, 
            min_length=30, 
            do_sample=False
        )[0]['summary_text']
        
        # Break the summary into a structured outline
        outline = [
            "**Introduction:** Key Takeaways from Top Content",
            f"**Section 1:** {summary_text}"
        ]
        return outline
    except Exception as e:
        print(f"Summarizer error: {e}")
        return ["*Error generating structured outline from snippets.*"]

# ---------------------------
# Post Idea Generator (Fixed LLM Usage)
# ---------------------------
# NOTE: Using a text-generation model for creative ideas is highly resource-intensive.
# We will use the summarization model in a creative way, or a smaller T5 model 
# for stability in a free environment. Let's use T5 for stability.
idea_generator_pipeline = pipeline("text2text-generation", model="t5-small")

def generate_post_idea_llm(cluster_keywords, outline_summary):
    """Generates a single, catchy post idea using the LLM pipeline."""
    main_kw = cluster_keywords[0]
    
    prompt = (
        f"Generate one catchy blog post title/idea about '{main_kw}' "
        f"based on this summary: {outline_summary} "
        f"Focus on the main keyword: {main_kw}. Return only the title."
    )
    
    try:
        generated = idea_generator_pipeline(
            prompt,
            max_length=30, # Max length for a title
            do_sample=True,
            temperature=0.8
        )
        idea_text = generated[0]['generated_text'].strip()
        
        # Basic cleanup
        if idea_text.lower().startswith("generate"):
            return f"The Ultimate Guide to Mastering {main_kw} in 5 Steps"
        
        return idea_text
        
    except Exception as e:
        print(f"Idea generator error: {e}")
        return f"Fallback: Top 5 Secrets to {main_kw} Today!"


# ---------------------------
# PDF Report Generator (Refined for structure)
# ---------------------------
def generate_pdf(cluster_summaries, report_name="keyword_report.pdf"):
    c = canvas.Canvas(report_name, pagesize=letter)
    width, height = letter
    margin = 50
    y = height - margin
    line_height = 14
    wrap_width = 80  # Approx characters per line

    def draw_wrapped_text(text, x, y, font="Helvetica", font_size=10):
        c.setFont(font, font_size)
        lines = textwrap.wrap(text, width=wrap_width)
        for line in lines:
            if y < margin:
                c.showPage()
                y = height - margin
            c.drawString(x, y, line)
            y -= line_height
        return y

    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "Keyword Clustering Report")
    y -= 25
    c.setFont("Helvetica", 10)
    y = draw_wrapped_text(f"Total Clusters: {len(cluster_summaries)}", margin, y)

    for cluster_id, info in cluster_summaries.items():
        if y < margin + 100:
            c.showPage()
            y = height - margin
        
        y = draw_wrapped_text(f"Cluster {cluster_id + 1} ({len(info['keywords'])} Keywords)", margin, y, font="Helvetica-Bold", font_size=12)
        
        kws = ', '.join(info["keywords"][:10]) + ('...' if len(info["keywords"]) > 10 else '')
        y = draw_wrapped_text(f"Keywords: {kws}", margin + 10, y)
        y = draw_wrapped_text(f"Top Web URL: {info.get('top_url', 'N/A')}", margin + 10, y)
        
        y = draw_wrapped_text("Suggested Post Idea:", margin + 10, y, font="Helvetica-Bold", font_size=10)
        y = draw_wrapped_text(f"* {info['post_idea']}", margin + 20, y)

        y = draw_wrapped_text("Summary Outline:", margin + 10, y, font="Helvetica-Bold", font_size=10)
        for o in info["outline"]:
            y = draw_wrapped_text(f"- {o.replace('**','')}", margin + 20, y)
        
        y -= 10  # extra space between clusters

    c.save()
    return report_name
# ---------------------------
# Email Report (Implemented)
# ---------------------------
def send_report_email(report_file, recipient_email):
    """Mails the PDF report using SendGrid."""
    if not os.environ.get('SENDGRID_API_KEY'):
        print("SendGrid API key not found. Skipping email.")
        return
        
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        with open(report_file, 'rb') as f:
            data = f.read()
        encoded_file = base64.b64encode(data).decode()

        mail = Mail(
            from_email=SENDER_EMAIL,
            to_emails=recipient_email,
            subject='Your Keyword Clustering Report',
            html_content='<strong>Here is your detailed keyword report attached.</strong>'
        )
        
        attached_file = Attachment(
            FileContent(encoded_file),
            FileName('keyword_report.pdf'),
            FileType('application/pdf'),
            Disposition('attachment')
        )
        mail.attachment = attached_file
        
        response = sg.send(mail)
        print(f"Email sent to {recipient_email}. Status code: {response.status_code}")
    
    except Exception as e:
        print(f"Error sending email: {e}")


# ---------------------------
# Slack Event Handlers (Refactored to be asynchronous and complete)
# ---------------------------
@slack_event_adapter.on('file_shared')
def handle_file_shared(event_data):
    event = event_data.get('event', {})
    file_id = event.get('file_id')
    user_id = event.get('user_id')
    
    if not file_id:
        return "", 200

    def process_file_async():
        try:
            # --- Setup ---
            file_info = client.files_info(file=file_id)
            file_url = file_info['file']['url_private_download']
            file_name = file_info['file']['name']
            channels = file_info['file'].get('channels', [])
            channel_id = channels[0] if channels else None
            user_info = client.users_info(user=user_id)
            user_email = user_info['user']['profile'].get('email', 'N/A')
            
            # --- Download File ---
            headers = {'Authorization': f"Bearer {os.environ['SLACK_TOKEN']}"}
            response = requests.get(file_url, headers=headers)
            with open(file_name, 'wb') as f:
                f.write(response.content)
            
            keywords = process_keywords(file_name)

            if channel_id and keywords:
                client.chat_postMessage(channel=channel_id, text=f"✅ Received `{file_name}` with {len(keywords)} keywords. Starting semantic clustering and content generation...")
                
                embeddings = generate_embeddings(keywords)
                clustered = cluster_keywords(keywords, embeddings, n_clusters=3)
                cluster_summaries = {}
                message_text = "Here are your clustered keywords and *LLM-generated* content ideas:\n"
                
                # --- Core Processing Loop ---
                for cluster_id, kws in clustered.items():
                    # Get snippets for the top keyword in the cluster
                    snippets, top_url = get_top_snippets(kws)
                    outline = generate_outline_from_snippets(snippets)
                    
                    # Generate post idea based on the summary
                    post_idea = generate_post_idea_llm(kws, " ".join(outline))
                    
                    cluster_summaries[cluster_id] = {
                        "keywords": kws, 
                        "outline": outline, 
                        "post_idea": post_idea,
                        "top_url": top_url
                    }
                    
                    message_text += f"\n*Cluster {cluster_id + 1} ({len(kws)} kws)*: {', '.join(kws[:5])}...\n"
                    message_text += f"**Idea:** {post_idea}\n"
                
                # --- Report Generation & Delivery ---
                report_file = generate_pdf(cluster_summaries)
                
                # Upload PDF to Slack
                with open(report_file, "rb") as f:
                    client.files_upload_v2(
                        channel=channel_id, file=f, filename="keyword_report.pdf",
                        title="Keyword Report", initial_comment="Here is your detailed keyword report!"
                    )
                
                # Send email (if configuration exists)
                send_report_email(report_file, user_email)

                final_message = message_text + f"\n\n**Processing Complete!** Check the attached report for full details. A copy has been emailed to `{user_email}`."
                client.chat_postMessage(channel=channel_id, text=final_message)
            
            else:
                client.chat_postMessage(channel=channel_id, text=f"⚠️ Could not find keywords in `{file_name}` or file could not be accessed.")
        except Exception as e:
            print(f"Error in file processing thread: {e}")
            if channel_id:
                client.chat_postMessage(channel=channel_id, text=f"🔥 A critical error occurred during processing: {e}")
    
    # Start thread immediately to ensure fast Slack response
    Thread(target=process_file_async).start()
    return "", 200

# ---------------------------
# Slash Command /keywords
# ---------------------------
@app.route('/commands', methods=['POST'])
def handle_command():
    data = request.form
    channel_id = data.get('channel_id')
    user_id = data.get('user_id') 
    text = data.get('text')
    response_url = data.get('response_url')
    
    if not text:
        return jsonify({"response_type": "ephemeral", "text": "Please provide keywords like: `/keywords AI, chatbot, natural language processing`"})
    
    # Send immediate acknowledgement
    requests.post(response_url, json={"text": "✅ Received your keywords. Starting semantic clustering and content generation..."})
    
    def process_keywords_async():
        try:
            user_info = client.users_info(user=user_id)
            user_email = user_info['user']['profile'].get('email', 'N/A')
            keywords = [kw.strip() for kw in text.split(',')]
            
            embeddings = generate_embeddings(keywords)
            clustered = cluster_keywords(keywords, embeddings, n_clusters=3)
            cluster_summaries = {}
            message_text = "Here are your clustered keywords and *LLM-generated* content ideas:\n"
            
            # --- Core Processing Loop ---
            for cluster_id, kws in clustered.items():
                snippets, top_url = get_top_snippets(kws)
                outline = generate_outline_from_snippets(snippets)
                
                post_idea = generate_post_idea_llm(kws, " ".join(outline))
                
                cluster_summaries[cluster_id] = {
                    "keywords": kws, "outline": outline, "post_idea": post_idea, "top_url": top_url
                }
                
                message_text += f"\n*Cluster {cluster_id + 1} ({len(kws)} kws)*: {', '.join(kws[:5])}...\n"
                message_text += f"**Idea:** {post_idea}\n"
            
            # --- Report Generation & Delivery ---
            report_file = generate_pdf(cluster_summaries)
            
            # Upload PDF to Slack
            with open(report_file, "rb") as f:
                client.files_upload_v2(
                    channel=channel_id, file=f, filename="keyword_report.pdf",
                    title="Keyword Report", initial_comment="Here is your detailed keyword report!"
                )
            
            # Send email
            send_report_email(report_file, user_email)

            final_message = message_text + f"\n\n**Processing Complete!** Check the attached report for full details. A copy has been emailed to `{user_email}`."
            client.chat_postMessage(channel=channel_id, text=final_message)
            
        except Exception as e:
            client.chat_postMessage(channel=channel_id, text=f"⚠️ Error processing keywords: {e}")
    
    Thread(target=process_keywords_async).start()
    return "", 200

# ---------------------------
# Root Route
# ---------------------------
@app.route('/', methods=['GET'])
def home():
    return "Slackbot is running!", 200


# ---------------------------
# Run Flask App
# ---------------------------
if __name__ == "__main__":
    # Get port from environment or default to 5000 (good for local testing)
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, port=port) # <--- Uses the dynamic PORT variable