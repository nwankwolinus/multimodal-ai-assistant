# 🤖 Multimodal AI Assistant

An intelligent assistant built with **GPT-4o**, capable of:

- 🗣️ Understanding and responding to **voice input** (using Whisper)
- 🧠 Generating natural replies with **OpenAI's GPT-4o**
- 🖼️ Analyzing uploaded **images**
- 🌐 Performing **real-time web search** (via Google CSE)
- 🔊 Replying back with **voice output** (using gTTS)

## 🚀 Live Demo
Coming soon...

## 🔧 Features

- 🎤 **Voice-to-Text**: Users can speak, and Whisper will transcribe the speech in real-time.
- 💬 **Chat Interface**: Powered by GPT-4o for fluent, natural conversation.
- 🖼️ **Image Analysis**: Upload an image and receive contextual visual analysis.
- 🔍 **Web Search**: Automatically detects when current data is needed and performs a web search using Google Custom Search API.
- 🔊 **Text-to-Speech**: AI responses are read aloud using Google Text-to-Speech (gTTS).

## 🛠️ Tech Stack

- **Frontend/UI**: Gradio
- **Backend**: Python
- **AI Models**:
  - [OpenAI GPT-4o](https://platform.openai.com/)
  - [Whisper (Speech-to-Text)](https://github.com/openai/whisper)
  - [gTTS (Text-to-Speech)](https://pypi.org/project/gTTS/)
- **Real-time Search**: Google Custom Search API
- **Deployment Options**: Localhost / Hugging Face Spaces / Render / Colab

## 🔐 Environment Variables

Create a `.env` or use secret manager with the following keys:

```bash
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_custom_search_engine_id
