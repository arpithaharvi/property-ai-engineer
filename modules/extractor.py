# ============================================================
# modules/extractor.py — Structured data extraction
# Pydantic schema + AI analysis for Excel property data
# Used by: app.py (Risk Analysis page)
# ============================================================

import json
from pydantic import BaseModel, Field
from modules.llm import load_groq_client
from config import GROQ_LLM_MODEL

groq_client = load_groq_client()


class PropertyAnalysisSchema(BaseModel):
    risk_score: str = Field(
        description="Must be exactly: 'Low Risk', 'Medium Risk', or 'High Risk'"
    )
    investment_score: int = Field(
        description="Investment rating from 1 (poor) to 100 (excellent)"
    )
    market_liquidity: str = Field(
        description="Liquidity: 'High', 'Moderate', or 'Low'"
    )
    price_range: str = Field(
        description="Estimated valuation price range"
    )
    key_risks: list[str] = Field(
        description="List of identified risk factors"
    )


def analyze_property_row(property_data: dict) -> dict:
    # Analyze one Excel row and return structured risk assessment
    system_prompt = (
        "You are a senior property valuation analyst working for a bank. "
        "Analyze the property data and evaluate its risk profile. "
        "Respond ONLY in valid JSON matching the schema provided."
    )

    schema = PropertyAnalysisSchema.model_json_schema()
    schema_str = json.dumps(schema, indent=2)

    response = groq_client.chat.completions.create(
        model=GROQ_LLM_MODEL,
        messages=[
            {"role": "system", "content": f"{system_prompt}\n\nSchema:\n{schema_str}"},
            {"role": "user", "content": f"Analyze this property:\n{property_data}"}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "risk_score": "High Risk (Parsing Error)",
            "investment_score": 0,
            "market_liquidity": "Low",
            "price_range": "Unknown",
            "key_risks": [f"Error: {str(e)}"]
        }