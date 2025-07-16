from openai import OpenAI
from openai import AuthenticationError as OpenAIAuthError
from typing import Dict, List, Optional, Union, Any
from dotenv import load_dotenv
import os
import PyPDF2
import docx
import json
import logging
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class RiskItem:
    """Represents a single risk item within a category."""
    risk_level: str
    clause_reference: str
    potential_consequences: str


@dataclass
class ExecutiveSummary:
    """Executive summary of the contract."""
    overview: str


@dataclass
class CriticalRiskAnalysis:
    """Critical risk analysis containing all risk categories."""
    cooling_off_periods: List[RiskItem]
    finance_clauses: List[RiskItem]
    penalties_and_fees: List[RiskItem]
    special_conditions: List[RiskItem]
    settlement_risks: List[RiskItem]
    legal_obligations: List[RiskItem]
    property_condition_risks: List[RiskItem]
    title_and_ownership_risks: List[RiskItem]
    insurance_and_liability_risks: List[RiskItem]
    disclosure_and_representation_risks: List[RiskItem]


@dataclass
class ContractAnalysisResult:
    """Complete contract analysis result."""
    executive_summary: Optional[ExecutiveSummary] = None
    critical_risk_analysis: Optional[CriticalRiskAnalysis] = None
    error: Optional[str] = None


class DocumentExtractor:
    """Handles document text extraction from various file formats."""
    
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_parts.append(f"[Page {page_num + 1}]\n{page_text}")
                
                return "\n\n".join(text_parts)
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise Exception(f"Error extracting text from PDF: {str(e)}")

    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """Extract text from Word document."""
        try:
            doc = docx.Document(docx_path)
            paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error extracting text from Word document: {str(e)}")
            raise Exception(f"Error extracting text from Word document: {str(e)}")

    @staticmethod
    def extract_text_from_txt(txt_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(txt_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from text file: {str(e)}")
            raise Exception(f"Error extracting text from text file: {str(e)}")

    def extract_document_text(self, file_path: str) -> str:
        """Extract text from document based on file extension."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        extractors = {
            '.pdf': self.extract_text_from_pdf,
            '.docx': self.extract_text_from_docx,
            '.doc': self.extract_text_from_docx,
            '.txt': self.extract_text_from_txt
        }
        
        if ext not in extractors:
            raise ValueError(f"Unsupported file format: {ext}")
        
        return extractors[ext](str(file_path))


class PromptGenerator:
    """Generates prompts for contract analysis."""
    
    @staticmethod
    def get_risk_analysis_prompt() -> str:
        """Generate the risk analysis prompt for the AI model."""
        return """
You are a property contract analysis expert. Analyze the following real estate contract and provide your response in JSON format with the following structure:

{
    "executive_summary": {
        "overview": "Brief overview of the contract in plain English. Break down each section into well-structured paragraphs with detailed explanations."
    },
    "critical_risk_analysis": {
        "cooling_off_periods": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "What could happen if this risk materializes"
            }
        ],
        "finance_clauses": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Impact if finance conditions aren't met"
            }
        ],
        "penalties_and_fees": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Financial impact and other consequences"
            }
        ],
        "special_conditions": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Impact of not meeting these conditions"
            }
        ],
        "settlement_risks": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Impact of settlement issues"
            }
        ],
        "legal_obligations": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Legal ramifications of non-compliance"
            }
        ],
        "property_condition_risks": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Financial and legal implications"
            }
        ],
        "title_and_ownership_risks": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Impact on ownership rights"
            }
        ],
        "insurance_and_liability_risks": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Financial exposure if uninsured"
            }
        ],
        "disclosure_and_representation_risks": [
            {
                "risk_level": "Low/Medium/High/Critical",
                "clause_reference": "Specific clause or section reference",
                "potential_consequences": "Legal and financial ramifications"
            }
        ]
    }
}

IMPORTANT INSTRUCTIONS:
- Analyze each section of the contract thoroughly
- Each risk category is an ARRAY and can have multiple entries (0 to many items)
- Only include risk categories that are actually present in the contract
- If a category has no risks, include it as an empty array: []
- Use clear, non-legal language where possible
- Provide specific clause references when available
- Prioritize risks based on likelihood and impact
- Risk levels must be exactly one of: "Low", "Medium", "High", or "Critical"
- Ensure all JSON is properly formatted and valid
- Do not include any text outside the JSON structure
- Each risk item must have all three fields: risk_level, clause_reference, and potential_consequences
"""


class ContractAnalyzer:
    """Main contract analyzer class."""
    
    def __init__(self):
        self.client = OpenAI(api_key=openai_api_key) if openai_api_key else None
        self.extractor = DocumentExtractor()
        self.prompt_generator = PromptGenerator()

    def _clean_json_response(self, response: str) -> str:
        """Clean the JSON response by removing markdown code blocks."""
        cleaned_response = response.strip()
        
        # Remove markdown code blocks
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        elif cleaned_response.startswith('```'):
            cleaned_response = cleaned_response[3:]
            
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]
            
        return cleaned_response.strip()

    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response with error handling."""
        try:
            cleaned_response = self._clean_json_response(response)
            return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            logger.error(f"Response content: {response[:500]}...")
            raise json.JSONDecodeError(f"Invalid JSON response from AI: {str(e)}", response, 0)

    def _validate_response_structure(self, json_response: Dict[str, Any]) -> bool:
        """Validate the structure of the JSON response."""
        required_keys = ['executive_summary', 'critical_risk_analysis']
        
        for key in required_keys:
            if key not in json_response:
                logger.warning(f"Missing required key: {key}")
                return False
        
        # Validate executive summary
        if 'overview' not in json_response['executive_summary']:
            logger.warning("Missing overview in executive_summary")
            return False
        
        # Validate critical risk analysis structure
        risk_categories = [
            'cooling_off_periods', 'finance_clauses', 'penalties_and_fees',
            'special_conditions', 'settlement_risks', 'legal_obligations',
            'property_condition_risks', 'title_and_ownership_risks',
            'insurance_and_liability_risks', 'disclosure_and_representation_risks'
        ]
        
        for category in risk_categories:
            if category not in json_response['critical_risk_analysis']:
                logger.warning(f"Missing risk category: {category}")
                # Initialize missing categories as empty arrays
                json_response['critical_risk_analysis'][category] = []
        
        return True

    def analyze_contract(self, file_path: str) -> ContractAnalysisResult:
        """Analyze a contract document and return structured risk analysis."""
        
        # Initialize result with None values
        result = ContractAnalysisResult()
        
        if not self.client:
            result.error = "OpenAI API key not found. Please set OPENAI_API_KEY."
            return result
        
        try:
            # Extract text from document
            document_text = self.extractor.extract_document_text(file_path)
            
            if not document_text.strip():
                result.error = "No text found in the document."
                return result
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": self.prompt_generator.get_risk_analysis_prompt()},
                {"role": "user", "content": f"Analyze this property contract:\n\n{document_text}"}
            ]
            
            # Get response from OpenAI
            completion = self.client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                max_tokens=4000,
                temperature=0.1
            )
            
            if not completion.choices:
                result.error = "No response generated from AI."
                return result
            
            response = completion.choices[0].message.content
            
            # Parse JSON response
            json_response = self._parse_json_response(response)
            
            # Validate response structure
            if not self._validate_response_structure(json_response):
                result.error = "Invalid response structure from AI."
                return result
            
            # Create structured result
            result.executive_summary = ExecutiveSummary(
                overview=json_response['executive_summary']['overview']
            )
            
            # Create risk analysis with proper structure
            risk_data = json_response['critical_risk_analysis']
            result.critical_risk_analysis = CriticalRiskAnalysis(
                cooling_off_periods=[RiskItem(**item) for item in risk_data.get('cooling_off_periods', [])],
                finance_clauses=[RiskItem(**item) for item in risk_data.get('finance_clauses', [])],
                penalties_and_fees=[RiskItem(**item) for item in risk_data.get('penalties_and_fees', [])],
                special_conditions=[RiskItem(**item) for item in risk_data.get('special_conditions', [])],
                settlement_risks=[RiskItem(**item) for item in risk_data.get('settlement_risks', [])],
                legal_obligations=[RiskItem(**item) for item in risk_data.get('legal_obligations', [])],
                property_condition_risks=[RiskItem(**item) for item in risk_data.get('property_condition_risks', [])],
                title_and_ownership_risks=[RiskItem(**item) for item in risk_data.get('title_and_ownership_risks', [])],
                insurance_and_liability_risks=[RiskItem(**item) for item in risk_data.get('insurance_and_liability_risks', [])],
                disclosure_and_representation_risks=[RiskItem(**item) for item in risk_data.get('disclosure_and_representation_risks', [])]
            )
            
            logger.info("Contract analysis completed successfully")
            return result
            
        except OpenAIAuthError:
            result.error = "Invalid API key or authentication failed."
            return result
        except FileNotFoundError as e:
            result.error = str(e)
            return result
        except json.JSONDecodeError as e:
            result.error = f"Invalid JSON response from AI: {str(e)}"
            return result
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            result.error = f"An unexpected error occurred: {str(e)}"
            return result

    def print_analysis_result(self, result: ContractAnalysisResult) -> None:
        """Print the analysis result in a formatted way."""
        if result.error:
            print(f"Error: {result.error}")
            return
        
        print("="*60)
        print("CONTRACT ANALYSIS REPORT")
        print("="*60)
        
        if result.executive_summary:
            print("\nEXECUTIVE SUMMARY:")
            print("-" * 20)
            print(result.executive_summary.overview)
        
        if result.critical_risk_analysis:
            print("\n\nCRITICAL RISK ANALYSIS:")
            print("-" * 25)
            
            risk_categories = [
                ('Cooling Off Periods', result.critical_risk_analysis.cooling_off_periods),
                ('Finance Clauses', result.critical_risk_analysis.finance_clauses),
                ('Penalties and Fees', result.critical_risk_analysis.penalties_and_fees),
                ('Special Conditions', result.critical_risk_analysis.special_conditions),
                ('Settlement Risks', result.critical_risk_analysis.settlement_risks),
                ('Legal Obligations', result.critical_risk_analysis.legal_obligations),
                ('Property Condition Risks', result.critical_risk_analysis.property_condition_risks),
                ('Title and Ownership Risks', result.critical_risk_analysis.title_and_ownership_risks),
                ('Insurance and Liability Risks', result.critical_risk_analysis.insurance_and_liability_risks),
                ('Disclosure and Representation Risks', result.critical_risk_analysis.disclosure_and_representation_risks)
            ]
            
            for category_name, risks in risk_categories:
                if risks:  # Only print categories with risks
                    print(f"\n{category_name.upper()}:")
                    for i, risk in enumerate(risks, 1):
                        print(f"  {i}. Risk Level: {risk.risk_level}")
                        print(f"     Clause Reference: {risk.clause_reference}")
                        print(f"     Potential Consequences: {risk.potential_consequences}")
                        print()


def main():
    """Main function to run the contract analyzer."""
    analyzer = ContractAnalyzer()

    # Path to the contract document
    contract_path = r"C:\Users\hasan\Downloads\Hasan Files\Prez\Contract-for-Sale-of-Real-Estate.pdf"
    
    # Check if file exists
    if not os.path.exists(contract_path):
        print(f"Error: File not found at {contract_path}")
        return
    
    print("Analyzing contract...")
    result = analyzer.analyze_contract(contract_path)
    
    # Print formatted result
    analyzer.print_analysis_result(result)
    
    # Optionally save to JSON file
    if not result.error:
        output_file = "contract_analysis_result.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "executive_summary": {"overview": result.executive_summary.overview},
                "critical_risk_analysis": {
                    "cooling_off_periods": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.cooling_off_periods],
                    "finance_clauses": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.finance_clauses],
                    "penalties_and_fees": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.penalties_and_fees],
                    "special_conditions": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.special_conditions],
                    "settlement_risks": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.settlement_risks],
                    "legal_obligations": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.legal_obligations],
                    "property_condition_risks": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.property_condition_risks],
                    "title_and_ownership_risks": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.title_and_ownership_risks],
                    "insurance_and_liability_risks": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.insurance_and_liability_risks],
                    "disclosure_and_representation_risks": [{"risk_level": r.risk_level, "clause_reference": r.clause_reference, "potential_consequences": r.potential_consequences} for r in result.critical_risk_analysis.disclosure_and_representation_risks]
                }
            }, f, indent=2, ensure_ascii=False)
        print(f"\nAnalysis saved to {output_file}")


if __name__ == "__main__":
    main()