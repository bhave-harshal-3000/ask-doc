from openrouter import OpenRouter
import os
from dotenv import load_dotenv
load_dotenv()
with OpenRouter(
  api_key=os.getenv("OPENROUTER_API_KEY",""),
) as client:
  response = client.chat.send(
    model="openai/gpt-5",
    max_tokens=3000,
    messages=[
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "essay on chh. shivaji maharaj"
          }
        ]
      }
    ]
  )

  print(response.choices[0].message.content)