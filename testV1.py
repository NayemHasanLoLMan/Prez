from openai import OpenAI
from openai import AuthenticationError as OpenAIAuthError
from typing import Dict, Optional, Union
from dotenv import load_dotenv
import os
import PyPDF2
import docx
import json
from dataclasses import dataclass

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class ContractRisk:
    risk_type: str
    description: str
    clause_reference: str
    severity: str  # Low/Medium/High/Critical

@dataclass
class ExecutiveSummary:
    """Executive summary of the contract."""
    overview: str



class ContractAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    text += f"[Page {page_num + 1}]\n"
                    text += page.extract_text() + "\n\n"
                return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    def extract_text_from_docx(self, docx_path: str) -> str:
        try:
            doc = docx.Document(docx_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            return text.strip()
        except Exception as e:
            raise Exception(f"Error extracting text from Word document: {str(e)}")

    def extract_document_text(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext in ['.docx', '.doc']:
            return self.extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def get_risk_analysis_prompt(self) -> str:
        return """
        You are a property contract analysis expert. Analyze the following real estate contract and provide your response in JSON format with the following structure:

        {
            "executive_summary": {
                "overview": "Brief overview of the contract in plain English.Break down each section into well-structured paragraphs with detailed explanations."
                }
            },
            "critical_risk_analysis": {
                "cooling_off_periods": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "What could happen if this risk materializes"
                    }
                ],
                "finance_clauses": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Impact if finance conditions aren't met"
                    }
                ],
                "penalties_and_fees": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Financial impact and other consequences"

                    }
                ],
                "special_conditions": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Impact of not meeting these conditions"
                    }
                ],
                "settlement_risks": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Impact of settlement issues"
                    }
                ],
                "legal_obligations": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Legal ramifications of non-compliance"
                    }
                ],
                "property_condition_risks": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Financial and legal implications"
                    }
                ],
                "title_and_ownership_risks": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Impact on ownership rights"
                    }
                ],
                "insurance_and_liability_risks": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section",
                        "severity": "Financial exposure if uninsured"
                    }
                ],
                "disclosure_and_representation_risks": [
                    {
                        "risk_level": "Low/Medium/High/Critical",
                        "description": "Description of the clause or section",
                        "clause_reference": "Specific clause or section"
                        "severity": "Legal and financial ramifications"
                    }
                ]
            }


        Instructions:
        - Analyze each section of the contract thoroughly
        - Each risk category can have multiple entries (0 to many)
        - Only include risk categories that are actually present in the contract
        - Use clear, non-legal language where possible (for the executive summary)
        - Provide specific clause references when available
        - Prioritize risks based on likelihood and impact
        - Ensure all JSON is properly formatted and valid
        """
    def analyze_contract(self, file_path: str) -> Dict[str, Union[str, None]]:
        if not self.client:
            return {"error": "OpenAI API key not found. Please set OPENAI_API_KEY."}
        try:
            document_text = self.extract_document_text(file_path)
            if not document_text.strip():
                return {"error": "No text found in the document."}

            messages = [
                {"role": "system", "content": self.get_risk_analysis_prompt()},
                {"role": "user", "content": f"Analyze this property contract:\n\n{document_text}"}
            ]

            completion = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                max_tokens=3500,
                temperature=0.1
            )

            response = completion.choices[0].message.content if completion.choices else "No response generated."

                        # Try to parse the JSON response
            try:
                # Clean the response in case there are markdown code blocks
                cleaned_response = response.strip()
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]  # Remove ```json
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]  # Remove ```
                cleaned_response = cleaned_response.strip()
                
                # Parse JSON
                json_response = json.loads(cleaned_response)

                return {
                    "response": json_response,
                }
        
            except json.JSONDecodeError as e:
                    return {
                        "error": f"Invalid JSON response from AI: {str(e)}"
                    }

        except OpenAIAuthError:
            return {"error": "Invalid API key or authentication failed."}
        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


if __name__ == "__main__":
    analyzer = ContractAnalyzer()

    # Path to the contract document
    contract_path = r"C:\Users\hasan\Downloads\Hasan Files\Prez\Contract-for-Sale-of-Real-Estate.pdf"

    result = analyzer.analyze_contract(contract_path)
    print(json.dumps(result, indent=2))
