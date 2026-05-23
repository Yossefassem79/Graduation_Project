
import requests
import time

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"


def generate_local(prompt: str, retries: int = 2) -> str:
    """
    Stable + fast local LLM call (Ollama)
    """

    # 🔥 cut prompt (مهم جدًا للسرعة)
    prompt = prompt[:1200]

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 150,   # يقلل الوقت
            "top_p": 0.9
        }
    }

    for attempt in range(retries):
        try:
            start_time = time.time()

            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=60
            )

            duration = round(time.time() - start_time, 2)

            # ❌ bad status
            if response.status_code != 200:
                print(f"❌ Local status {response.status_code}")
                time.sleep(1)
                continue

            # ❌ invalid json
            try:
                data = response.json()
            except Exception:
                print("❌ Invalid JSON from local")
                time.sleep(1)
                continue

            result = data.get("response", "").strip()

            # ❌ empty response
            if not result or len(result) < 10:
                print(f"⚠️ Empty/short response ({duration}s)")
                time.sleep(1)
                continue

            print(f"✅ Local OK ({duration}s)")
            return result

        except requests.exceptions.Timeout:
            print(f"⏱️ Timeout ({attempt+1}/{retries})")
            time.sleep(1)

        except Exception as e:
            print("❌ Local error:", e)
            time.sleep(1)

    # ❌ failed completely
    print("🚨 Local LLM failed → fallback")
    return ""

