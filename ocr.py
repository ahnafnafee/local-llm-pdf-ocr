from openai import AsyncOpenAI
import os
import sys
from dotenv import load_dotenv
import logging

load_dotenv()

class OCRProcessor:
    def __init__(self):
        base_url = os.getenv("LLM_API_BASE", "http://localhost:1234/v1")
        model = os.getenv("LLM_MODEL", "allenai/olmocr-2-7b")
        
        self.client = AsyncOpenAI(base_url=base_url, api_key="lm-studio")
        self.model = model

    async def perform_ocr(self, image_base64):
        """
        Sends the image to the local LLM for OCR.
        Returns a list of strings (lines) representing the text.
        """
        # logging.debug(f"  - Transcribing with {self.model}...")

        text = await self._transcribe(image_base64)
        if not text:
            return []
        
        # Split into lines
        return [line.strip() for line in text.split('\n') if line.strip()]

    async def _transcribe(self, image_base64):
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the text in this image accurately. Preserve line breaks. Return only the plain text."},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                },
                            },
                        ],
                    }
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.debug(f"DEBUG: Transcription error: {e}")
            return ""
