# 🚀 Deploying TruthGuard AI to Streamlit Cloud

Follow these steps to host your **TruthGuard AI** application online for free.

## 1. Prepare your GitHub Repository
Streamlit Cloud deploys directly from GitHub.
1. Create a new **Public** repository on [GitHub](https://github.com/new).
2. Upload all files from your desktop folder (`TruthGuard-AI-Final`) to this repository.
   - **Crucial**: Include the `model/` folder with your `.pkl` files.
   - **Note**: Do **NOT** upload your `.env` file (containing your Groq key) to GitHub for security.

## 2. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
2. Click **"Create app"**.
3. Select your repository, the **main** branch, and set the main file path to `app.py`.
4. Click **"Deploy"**.

## 3. Set Up Your API Keys (Secrets)
Since we didn't upload the `.env` file, we need to tell Streamlit Cloud about your Groq key:
1. In your Streamlit Cloud dashboard, click on your app.
2. Go to **Settings** > **Secrets**.
3. Paste the following into the text area:
   ```toml
   GROQ_API_KEY = "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"
   ```
4. Click **Save**. The app will automatically restart and use this key.

---
**Your app will now be live at a public URL (e.g., `https://truthguard-ai.streamlit.app`)!**
