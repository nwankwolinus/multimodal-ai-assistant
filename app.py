# === Imports and Configurations ===
import os
import gradio as gr
from openai import OpenAI
from gtts import gTTS
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from googleapiclient.discovery import build
import whisper
import base64

# Load Whisper model
model = whisper.load_model("base")

# === Load API keys from Hugging Face Secrets ===
openai_api_key = os.getenv("OPENAI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

client = OpenAI(api_key=openai_api_key)

# === Real-Time Web Search Function ===
def search_web(query: str) -> str:
    try:
        service = build("customsearch", "v1", developerKey=google_api_key)
        result = service.cse().list(q=query, cx=google_cse_id, num=3).execute()
        items = result.get("items", [])
        if not items:
            return "🔍 No results found for your query."

        formatted_results = []
        for item in items:
            title = item.get("title", "No Title")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            formatted_results.append(f"**{title}**\n{snippet}\n🔗 {link}")

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"❌ Error during web search: {e}"

# === Fixed Image Analysis Handler ===
def analyze_image(image_data: str, prompt: str) -> str:
    """Handle image analysis from Gradio's filepath or base64 input"""
    try:
        # Gradio passes either a filepath or base64 string
        if isinstance(image_data, str) and (image_data.startswith("/") or "\\" in image_data):
            # Handle filepath input
            with open(image_data, "rb") as f:
                img_data = f.read()
                b64_image = base64.b64encode(img_data).decode("utf-8")
        else:
            # Handle base64 input (Gradio's in-memory images)
            b64_image = image_data.split(",")[-1] if "," in image_data else image_data
            
        # Create the OpenAI chat with image input
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]}
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"❌ Image analysis error: {str(e)}"

# === AI Text Response Handler with Real-Time Trigger ===
# === AI Text Response Handler with Real-Time Trigger ===
def should_trigger_search(text: str) -> bool:
    keywords = [
        "latest", "current", "recent", "now", "today", "upcoming", 
        "new version", "update", "version number", "release date",
        "as of today", "as of now", "currently", "right now", "present",
        "version", "release", "announcement", "newest"
    ]
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in keywords)

def generate_response(user_input: str, messages: List[Dict]) -> str:
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages + [{"role": "user", "content": user_input}],
            max_tokens=1000
        )
        reply = response.choices[0].message.content.strip()

        # Enhanced fallback detection
        fallback_phrases = [
            "as of my last", "cannot provide real-time", "don't have access to current",
            "recommend checking", "might want to check", "suggest visiting", 
            "unable to browse", "no real-time", "not have live", "as an AI"
        ]
        
        # Fixed condition syntax
        if any(phrase in reply.lower() for phrase in fallback_phrases) or should_trigger_search(user_input):
            print(f"[DEBUG] Triggering real-time search for: {user_input}")
            search_results = search_web(user_input)

            # Only summarize if we have valid search results
            if not search_results.startswith("❌") and not search_results.startswith("🔍"):
                summary_prompt = (
                    "Provide a concise, conversational summary of these search results. "
                    "Include key facts and end with reference links. Be direct and helpful.\n\n"
                    f"{search_results}"
                )

                summary_response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You provide real-time information updates based on web searches."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=1000
                )
                return summary_response.choices[0].message.content.strip()
            
            # Return raw results if summary fails
            return search_results

        return reply

    except Exception as e:
        return f"Error generating response: {e}"

# === Core Chat Handler ===
def chat_interface(
    user_input: str,
    history: List[Dict],
    image: Optional[str] = None,
    voice: Optional[str] = None
) -> Tuple[List[Dict], str, Optional[str], Optional[str], Optional[str]]:

    if history is None:
        history = []

    audio_path = None
    transcribed_input = None
    image_analysis_result = None  # Store image analysis separately

    # Handle voice input
    if voice:
        try:
            print("[DEBUG] Transcribing voice input...")
            result = model.transcribe(voice)
            transcribed_input = result.get("text", "").strip()
            print(f"[DEBUG] Transcribed: {transcribed_input}")
            user_input = transcribed_input
            history.append({"role": "user", "content": user_input})
        except Exception as e:
            print("[ERROR] Failed to transcribe audio:", e)
            history.append({"role": "user", "content": "Sorry, I couldn't process that audio"})

    # Handle text input
    elif user_input.strip():
        history.append({"role": "user", "content": user_input})
    else:
        # Handle image-only input
        if image:
            user_input = "Please describe this image"
            history.append({"role": "user", "content": user_input})
        else:
            return history, "", image, voice, None  # Empty input fallback

    # Build message context
    messages = [{"role": "system", "content": "You are a helpful assistant that can analyze images and fetch real-time information."}]
    messages.extend(history)

    # Handle image input - FIXED SECTION
    if image:
        try:
            print("[DEBUG] Analyzing image...")
            img_prompt = user_input if user_input.strip() else "Please describe this image"
            analysis = analyze_image(image, img_prompt)
            image_analysis_result = f"🖼️ **Image Analysis**\n{analysis}"
            history.append({"role": "assistant", "content": image_analysis_result})
        except Exception as e:
            print("[ERROR] Image analysis failed:", e)
            history.append({"role": "assistant", "content": f"❌ Failed to analyze image: {e}"})

    # Generate GPT-4o reply (skip if we only did image analysis)
    ai_reply = None
    if user_input.strip() or not image:  # Only generate if there's text input or no image
        ai_reply = generate_response(user_input, messages)
        history.append({"role": "assistant", "content": ai_reply})

    # Voice Output (TTS)
    if voice:
        try:
            # Determine what to speak
            text_to_speak = ai_reply if ai_reply else image_analysis_result
            if text_to_speak:
                tts = gTTS(text=text_to_speak)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                    tts.save(tmpfile.name)
                    audio_path = tmpfile.name
        except Exception as e:
            print("[ERROR] TTS generation failed:", e)
            audio_path = None

    return history, "", None, None, audio_path

# === Gradio UI Layout ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🤖 AI Assistant with Voice, Vision & Web Search")

    chatbot = gr.Chatbot(
        height=500,
        avatar_images=("https://example.com/user.png", "https://example.com/bot.png"),
        show_label=False,
        show_copy_button=True,
        type="messages"
    )

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here...", container=False, scale=7)
        send_btn = gr.Button(value="➤", variant="primary", scale=1)

    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image", height=200)
        voice_input = gr.Audio(type="filepath", label="Speak", scale=1)

    audio_output = gr.Audio(label="AI Response", autoplay=True, visible=False)
    clear_btn = gr.Button("Clear Chat")
    state = gr.State([])

    send_btn.click(
        fn=chat_interface,
        inputs=[msg, state, image_input, voice_input],
        outputs=[chatbot, msg, image_input, voice_input, audio_output]
    )

    voice_input.change(
        fn=chat_interface,
        inputs=[msg, state, image_input, voice_input],
        outputs=[chatbot, msg, image_input, voice_input, audio_output]
    )

    clear_btn.click(
        fn=lambda: ([], [], "", None, None),
        outputs=[chatbot, state, msg, voice_input, audio_output]
    )

# === Launch App ===
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)