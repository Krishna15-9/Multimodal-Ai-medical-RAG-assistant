"""
OpenRouter Client for Healthcare Q&A Tool.

Replaces Euri AI with OpenRouter API while maintaining all healthcare-specific optimizations.
"""

from typing import Any, Dict, List, Optional
import requests
from loguru import logger
from ..config import get_settings


class OpenRouterClientError(Exception):
    """Custom exception for OpenRouter client errors."""
    pass


class OpenRouterClient:
    """Client for OpenRouter API with healthcare optimizations."""

    def __init__(self):
        """Initialize the OpenRouter client."""
        self.settings = get_settings()
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"  # Correct endpoint
        self.headers = {
            "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            "Referer": "https://github.com/your-repo",  # Required by OpenRouter
            "X-Title": "Healthcare Q&A Tool"  # Optional
        }
        # Set default model to a free/open model for testing
        self.default_model = "openai/gpt-4o-mini"
        logger.info(f"Initialized OpenRouter client with model: {self.default_model}")
        logger.debug(f"OpenRouter API Key Loaded: {self.settings.openrouter_api_key}")

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = 512,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a chat completion using OpenRouter API with proper response validation.
        """
        
        try:
            logger.debug(f"OpenRouter API request payload: model={model or self.default_model}, temperature={temperature or self.settings.temperature}, max_tokens={max_tokens or self.settings.max_tokens}, messages={messages[:2]}...")
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": model or self.default_model,
                    "messages": messages,
                    "temperature": temperature or self.settings.temperature,
                    "max_tokens": max_tokens or self.settings.max_tokens
                }
            )

            logger.debug(f"OpenRouter raw response status: {response.status_code}")
            logger.debug(f"OpenRouter raw response text: {response.text}")

            if response.status_code != 200:
                raise OpenRouterClientError(f"API request failed with status {response.status_code}: {response.text}")

            if not response.text.strip():
                raise OpenRouterClientError("API response is empty.")

            try:
                json_response = response.json()
                return json_response
            except ValueError:
                raise OpenRouterClientError(f"Failed to parse JSON. Raw response: {response.text}")

        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise OpenRouterClientError(f"API request failed: {e}")

    def generate_healthcare_response(self, query: str, context: str) -> str:
        """Generates a healthcare-specific response."""
        messages = [
            {"role": "system", "content": self._get_healthcare_system_prompt()},
            {"role": "user", "content": self._create_healthcare_prompt(query, context)}
        ]
        response = self.chat_completion(messages, max_tokens=512)
        return response['choices'][0]['message']['content']

    def _get_healthcare_system_prompt(self) -> str:
        """Returns system prompt for healthcare context."""
        return """You are a knowledgeable healthcare research assistant specializing in evidence-based medicine. 
Your role is to provide accurate, well-researched answers based on peer-reviewed scientific literature.

Guidelines:
- Always base your answers on the provided research articles
- Be precise and factual, avoiding speculation
- Acknowledge limitations and uncertainties in the research
- Distinguish between established facts and preliminary findings
- Use clear, professional language accessible to healthcare professionals
- When discussing medical topics, always recommend consulting healthcare professionals for clinical decisions
- If the provided articles don't contain sufficient information, state this clearly
- Focus on intermittent fasting, obesity, diabetes, and metabolic disorders when relevant
- Cite specific studies or findings when possible"""

    def _create_healthcare_prompt(self, query: str, context: str) -> str:
        """Creates user prompt for healthcare context."""
        return f"""Based on the following research articles about healthcare and medical topics, please answer the user's question. 
Focus on providing evidence-based information from the provided sources.

Research Articles:
{context}

User Question: {query}

Please provide a comprehensive answer that:
1. Directly addresses the question with evidence from the research
2. Cites relevant findings from the provided articles
3. Mentions any limitations, conflicting findings, or areas of uncertainty
4. Provides practical implications when appropriate
5. Indicates if more research is needed
6. Recommends consulting healthcare professionals for clinical decisions

Answer:"""

    def generate_research_summary(self, topic: str, context: str) -> str:
        """Generates a research summary."""
        messages = [
            {"role": "system", "content": "You are a research analyst specializing in healthcare literature reviews. Provide comprehensive, evidence-based summaries of research topics based on peer-reviewed literature."},
            {"role": "user", "content": f"Based on the following research articles about {topic}, provide a comprehensive research summary that includes:\n\n1. Overview of the current state of research\n2. Key findings and areas of consensus\n3. Areas of disagreement or uncertainty\n4. Methodological considerations and study quality\n5. Clinical implications and practical applications\n6. Gaps in research and future directions\n7. Recommendations for healthcare practice\n\nResearch Articles:\n{context}\n\nPlease provide a structured, evidence-based summary:"}
        ]
        response = self.chat_completion(messages, temperature=0.1, max_tokens=800)
        return response['choices'][0]['message']['content']

    def test_connection(self) -> bool:
        """Test API connection."""
        try:
            test_response = self.chat_completion(
                messages=[{"role": "user", "content": "Connection test"}],
                max_tokens=256
            )
            return bool(test_response.get('choices'))
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
