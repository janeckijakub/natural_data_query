from typing import Dict, Any, Optional
import openai
from anthropic import Anthropic
import requests
import os
import json
import logging
from datetime import datetime
 
# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('LLMClient')
 
class LLMClient:
    def __init__(self, provider: str, api_key: str):
        """
        Initialize LLM client with specified provider and API key.
        
        Args:
            provider (str): One of "OpenAI", "Anthropic", or "Ollama"
            api_key (str): API key (not needed for Ollama)
        """
        self.provider = provider
        self.api_key = api_key
        
        # Initialize specific client based on provider
        try:
            if provider == "OpenAI":
                if not api_key:
                    raise ValueError("OpenAI API key is required")
                openai.api_key = api_key
                self.model = os.getenv("OPENAI_MODEL", "gpt-4")
                logger.info(f"Initialized OpenAI client with model: {self.model}")
                
            elif provider == "Anthropic":
                if not api_key:
                    raise ValueError("Anthropic API key is required")
                self.anthropic = Anthropic(api_key=api_key)
                self.model = os.getenv("ANTHROPIC_MODEL", "claude-3-haiku-20240307")
                logger.info(f"Initialized Anthropic client with model: {self.model}")
                
            elif provider == "Ollama":
                self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
                self.model = os.getenv("OLLAMA_MODEL", "llama3.2")
                # Test connection to Ollama
                self._test_ollama_connection()
                logger.info(f"Initialized Ollama client with model: {self.model} at {self.ollama_host}")
                
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
                
        except Exception as e:
            logger.error(f"Error initializing {provider} client: {str(e)}")
            raise
            
    def _test_ollama_connection(self) -> None:
        """Test connection to Ollama server"""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags")
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            raise RuntimeError(f"Could not connect to Ollama server at {self.ollama_host}")
            
    def generate_prompt(self, query: str, context: Dict[str, Any], dataframe_info: str) -> str:
        """
        Generate a prompt for the LLM based on query and context.
        """
        # Detect if this is a visualization query
        viz_keywords = ['wykres', 'plot', 'visualize', 'chart', 'histogram', 'scatter', 'pie', 'violin']
        is_viz_query = any(keyword in query.lower() for keyword in viz_keywords)
     
        if is_viz_query:
            prompt = f"""You are a data analysis assistant specialized in creating visualizations using plotly express.
             
                        Please follow these rules strictly:
                        1. ALWAYS use plotly.express (px) for creating plots
                        2. NEVER use pandas plot or matplotlib
                        3. Store all plots in a variable named 'fig'
                        4. Use proper column names in single quotes
                        5. For aggregated data:
                            - ALWAYS create a helper DataFrame first if you need to calculate averages, sums, or counts
                            - Name the helper DataFrame 'df_agg'
                            - Use this aggregated DataFrame for visualization
                            - Make sure to reset_index() after groupby operations
             
                        Examples of correct plotly code:
             
                        Example 1 - Simple plot:
                        fig = px.scatter(df, x='column1', y='column2', title='Title')
             
                        Example 2 - Plot with aggregation:
                        df_agg = df.groupby('category')[['value1', 'value2']].mean().reset_index()
                        fig = px.bar(df_agg, x='category', y=['value1', 'value2'], title='Average Values by Category')
             
                        Example 3 - Multiple metrics:
                        df_agg = df.groupby('group').agg({
                            'income': 'mean',
                            'age': 'mean'
                        }).reset_index()
                        fig = px.bar(df_agg, x='group', y=['income', 'age'], barmode='group',
                                    title='Average Income and Age by Group')
             
                        Example 4 - Custom aggregations:
                        df_agg = df.groupby('department').agg({
                            'salary': ['mean', 'median', 'count']
                        }).reset_index()
                        df_agg.columns = ['department', 'avg_salary', 'median_salary', 'count']
                        fig = px.bar(df_agg, x='department', y='avg_salary',
                                    text='count', title='Average Salary by Department')
             
                        Remember:
                        - Always aggregate data first if showing averages, sums, or other statistics
                        - Use clear column names in aggregated DataFrame
                        - Reset index after groupby operations
                        - Use appropriate chart types for the data being shown"""
        else:
            prompt = f"""You are a data analysis assistant specialized in pandas operations.
             
                        Please follow these rules strictly:
                        1. For simple numerical results (mean, sum, count, etc.), store the result in a variable named 'result'
                        2. For DataFrame results, store the result in a variable named 'result'
                        3. Use proper column names in single quotes or double quotes
                        4. Return only executable Python code without any explanation
                        5. Make sure numeric results are not formatted as strings initially
                        6. Don't print results, just assign them to 'result'
             
                        Examples of correct code:
                        - Mean value: result = df['column'].mean()
                        - Count by category: result = df.groupby('category')['value'].count()
                        - Filter and aggregate: result = df[df['column'] > 100].groupby('category')['value'].sum()
                        - Multiple columns: result = df[['col1', 'col2', 'col3']].describe()"""
             
        prompt += f"""
             
                The DataFrame 'df' has the following structure and context:
             
                {dataframe_info}
                
                Additional Context:
                
                {context}
             
                User Query: {query}
             
                Generate only the Python code (without any explanation or formatting) to answer this query:"""
             
        return prompt
 
    def _clean_response(self, response: str) -> str:
        """Clean and validate the LLM response"""
        try:
            # Remove markdown formatting
            response = response.replace("```python", "").replace("```", "")
            
            # Remove leading/trailing whitespace
            response = response.strip()
            
            # Remove any print statements
            lines = []
            for line in response.split('\n'):
                if not line.strip().startswith('print('):
                    lines.append(line)
            response = '\n'.join(lines)
            
            # Ensure the response contains valid code
            if not response or 'df' not in response:
                raise ValueError("Invalid response: no pandas code found")
 
            # If there's no explicit result variable and it's not a visualization
            if 'result' not in response and 'fig' not in response:
                # Take the last line and assign it to result
                lines = response.split('\n')
                last_line = lines[-1]
                lines[-1] = f"result = {last_line}"
                response = '\n'.join(lines)
            
            return response
 
        except Exception as e:
            logger.error(f"Error cleaning response: {str(e)}")
            raise ValueError(f"Error cleaning response: {str(e)}")
 
    def _call_ollama(self, prompt: str) -> str:
        """Make a request to Ollama API"""
        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": "You are a data analysis assistant. Respond only with executable pandas code.",
                    "stream": False,
                    "options": {
                        "temperature": float(os.getenv("TEMPERATURE", 0.7))
                    }
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Extract and clean the response
            code = response.json()["response"]
            return code.strip()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            raise RuntimeError(f"Error calling Ollama API: {str(e)}")
 
    def get_response(self, prompt: str) -> str:
        """
        Get response from selected LLM provider
        
        Args:
            prompt (str): Formatted prompt for the LLM
            
        Returns:
            str: Generated pandas code
        """
        try:
            logger.info(f"Sending request to {self.provider}")
            
            if self.provider == "OpenAI":
                response = self._call_openai(prompt)
            elif self.provider == "Anthropic":
                response = self._call_anthropic(prompt)
            elif self.provider == "Ollama":
                response = self._call_ollama(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Clean and validate response
            cleaned_response = self._clean_response(response)
            
            logger.info(f"Successfully received and cleaned response from {self.provider}")
            logger.info(f"Cleaned code:\n{cleaned_response}")
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error getting response from {self.provider}: {str(e)}")
            raise


     
    @staticmethod
    def list_available_ollama_models() -> list:
        """
        List available Ollama models
     
        Returns:
            list: List of available model names
        """
        try:
            response = requests.get("http://localhost:11434/api/tags")
            response.raise_for_status()
            
            # Parsuj odpowiedź JSON
            data = response.json()
            
            # Sprawdź strukturę odpowiedzi
            if isinstance(data, dict) and "models" in data:
                models = [model["name"] for model in data["models"]]
            else:
                # Jeśli struktura jest inna, spróbuj pobrać nazwy modeli bezpośrednio
                models = [item.get("name") for item in data] if isinstance(data, list) else []
            
            logger.info(f"Found {len(models)} available Ollama models")
            return models if models else ["llama3.2"]  # Domyślny model jeśli lista jest pusta
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error listing Ollama models: {str(e)}. Using default model.")
            return ["llama3.2"]  # Zwróć domyślny model w przypadku błędu
        except Exception as e:
            logger.warning(f"Unexpected error while listing Ollama models: {str(e)}. Using default model.")
            return ["llama3.2"]  # Zwróć domyślny model w przypadku nieoczekiwanego błędu
     
     
        def get_model_info(self) -> Dict[str, Any]:
            """
            Get information about the current model
            
            Returns:
                Dict[str, Any]: Model information
            """
            return {
                "provider": self.provider,
                "model": self.model,
                "api_host": self.ollama_host if self.provider == "Ollama" else None,
                "timestamp": datetime.now().isoformat()
            }