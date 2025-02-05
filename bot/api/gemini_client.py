import os
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        
        # Initialize the Gemini client
        genai.configure(api_key=self.api_key)

    def analyze(self, text):
        """Analyze text using Gemini Flash 2.0."""
        try:
            # Define the model
            model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Replace with the correct model name
            
            # Define the prompt
            prompt = (
                "You are a financial sentiment analysis model. Analyze the provided market data and return ONLY a sentiment score between -1 (bearish) and +1 (bullish). "
                f"Market Data: {text}"
            )

            # Call the Gemini API
            response = model.generate_content(prompt)
            logger.debug(f"Gemini API raw response: {response.text}")  # Log the raw response for debugging

            # Parse the sentiment score from the response
            try:
                # Attempt to parse the sentiment score directly
                sentiment_score = float(response.text.strip())
            except ValueError:
                logger.warning("Failed to parse sentiment score. Using default value.")
                logger.debug(f"Raw response content: {response.text}")  # Log raw response for debugging
                sentiment_score = 0  # Default fallback

            return sentiment_score
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            return 0  # Default fallback