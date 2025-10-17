🧠 Slackbot Content Pipeline
📘 Overview

Slackbot Content Pipeline is an intelligent Slack integration designed to help content teams streamline keyword-based content creation.
The bot automates the entire workflow — from uploading raw keyword lists to generating final content strategy reports — directly within Slack.

🎯 Goal

To build a Slackbot that:

Accepts keyword lists from users.

Automatically segments keywords into relevant clusters.

Analyzes top-ranking web content for each cluster.

Generates post ideas and outlines.

Compiles a comprehensive PDF report of the results for easy sharing.

⚙️ Tech Stack

Component	Description

Python - 	Core programming language
Flask	- Web framework to handle Slack events
Slack SDK -	Connects and interacts with Slack workspace
SendGrid - Sends email notifications and reports
ReportLab -	Generates PDF reports
SentenceTransformers / Transformers -	Embeddings and NLP-based clustering
KMeans (scikit-learn) -	Keyword grouping
SerpAPI / GoogleSearch -	Retrieves top-ranking content for keyword analysis

🚀 Features

✅ Keyword Upload via Slack – Upload Excel keyword lists directly in a Slack channel.
✅ Automatic Keyword Clustering – Groups related keywords using embeddings.
✅ Top Web Results Analysis – Fetches insights from top-ranking web pages.
✅ Content Outline Generation – Suggests post ideas and outlines based on keyword clusters.
✅ PDF Report Generation – Generates a detailed report summarizing all results.
✅ Email Integration (SendGrid) – Option to email the final report to the user.

⚙️ Setup Instructions
1️⃣ Clone the Repository

git clone https://github.com/yourusername/Slackbot-Content-Pipeline.git
cd Slackbot-Content-Pipeline

2️⃣ Create a Virtual Environment

python -m venv venv
source venv/bin/activate       # On Mac/Linux
venv\Scripts\activate          # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Configure Environment Variables

Create a .env file in the project root with:

SLACK_BOT_TOKEN=your_slack_bot_token
SIGNING_SECRET=your_slack_signing_secret
SENDGRID_API_KEY=your_sendgrid_api_key
SERPAPI_KEY=your_serpapi_key

5️⃣ Run the Application

python bot.py

6️⃣ Expose Your Local Server (for Slack)

Use ngrok or Render to expose your Flask app:

ngrok http 5000


Copy the generated public URL and update it in your Slack App’s Event Subscriptions.

Slack Setup

Go to Slack API – Your Apps

Create a new app and enable Event Subscriptions.

Add your ngrok/Render URL followed by /slack/events.

Under OAuth & Permissions, add scopes like:

chat:write
files:read
files:write
commands


Install the app to your workspace and note the Bot Token and Signing Secret.

📄 Output

Slack messages showing clustered keywords, outlines, and suggestions.

PDF Report automatically generated with keyword insights and post ideas.

Optional email delivery of reports through SendGrid.

💡 Future Improvements

Integrate OpenAI / Gemini API for advanced content generation.

Add dashboard analytics for keyword trends.

Enable multi-user support for team workflows.

Deploy with a fully cloud-hosted backend (Render, AWS, or Azure).

👩‍💻 Author

Samyuktha Sureshkumar
💼 Data Analyst | Aspiring AI, ML & Automation Engineer
📧 samyukthasuresh01@gmail.com
