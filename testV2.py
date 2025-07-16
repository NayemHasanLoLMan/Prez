from openai import OpenAI
from openai import AuthenticationError as OpenAIAuthError
from typing import List, Dict, Optional
from dotenv import load_dotenv
import os
import base64
import mimetypes
import PyPDF2
import io


load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


def encode_image_to_base64(image_path: str) -> str:
    """Encode image to base64 string."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: str) -> str:
    """Get the MIME type of an image file."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type and mime_type.startswith('image/'):
        return mime_type
    # Default to jpeg if cannot determine
    return "image/jpeg"


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from PDF file."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")


def chat(text_input: str, image_input: Optional[str] = None, pdf_path: Optional[str] = None, conversation_history: List[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Chat with OpenAI API supporting text, images, and PDF content.
    
    PDF content is extracted as text and included in the conversation.
    """
    if conversation_history is None:
        conversation_history = []

    if not openai_api_key:
        return {"error": "OpenAI API key not found. Please set OPENAI_API_KEY environment variable."}

    try:
        messages = []

        # Add conversation history
        for conversation in conversation_history:
            if conversation.get("user", "").strip():
                messages.append({"role": "user", "content": conversation["user"]})
            if conversation.get("assistant", "").strip():
                messages.append({"role": "assistant", "content": conversation["assistant"]})

        # Prepare current user message
        user_content = [{"type": "text", "text": text_input}]

        # Add image if provided
        if image_input and image_input.strip():
            if not os.path.exists(image_input):
                return {"error": f"Image file not found: {image_input}"}
            
            try:
                image_base64 = encode_image_to_base64(image_input)
                mime_type = get_image_mime_type(image_input)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{image_base64}"
                    }
                })
            except Exception as e:
                return {"error": f"Error processing image: {str(e)}"}

        messages.append({"role": "user", "content": user_content})

        # Handle PDF by extracting text content
        if pdf_path and pdf_path.strip():
            if not os.path.exists(pdf_path):
                return {"error": f"PDF file not found: {pdf_path}"}
            
            try:
                pdf_text = extract_text_from_pdf(pdf_path)
                if pdf_text:
                    # Add PDF content as a separate message
                    messages.append({
                        "role": "user",
                        "content": f"Here is the content from the PDF file '{os.path.basename(pdf_path)}':\n\n{pdf_text}"
                    })
                else:
                    return {"error": "Could not extract text from PDF or PDF is empty."}
            except Exception as e:
                return {"error": f"Error processing PDF: {str(e)}"}

        # Initialize OpenAI client
        client = OpenAI(api_key=openai_api_key)

        # Make API call
        completion = client.chat.completions.create(
            model="gpt-4-turbo",  # Use gpt-4o for better performance
            messages=messages,
            max_tokens=2000
        )

        reply = completion.choices[0].message.content if completion.choices else "No response generated."
        return {"response": reply}

    except OpenAIAuthError:
        return {"error": "Invalid API key or authentication failed."}
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}


if __name__ == "__main__":
    # Test the function
    text_input = "Please analyze this contract and summarize the key terms."
    image_path = ""  # Leave empty if no image
    pdf_path = r"C:\Users\hasan\Downloads\Hasan Files\Prez\Contract-for-Sale-of-Real-Estate.pdf"  # Update with actual PDF path
    conversation_history = []
    
    result = chat(text_input, image_input=image_path, pdf_path=pdf_path, conversation_history=conversation_history)
    print(result)