import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Any, Union, Dict, List, Tuple
import ast
import re
import logging
from datetime import datetime
 
# Konfiguracja logowania
logging.basicConfig(
level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('QueryProcessor')
 
class QueryProcessor:
    def __init__(self, df: pd.DataFrame, context_manager, llm_client):
        self.df = df
        self.context_manager = context_manager
        self.llm_client = llm_client
        
        logger.info("Initializing QueryProcessor")
        logger.info(f"DataFrame shape: {df.shape}")
        
        # Definicje typów wizualizacji i ich słów kluczowych
        self.viz_types = {
            'histogram': {
                'keywords': ['histogram', 'rozkład', 'dystrybuacja'],
                'needs_numeric': True,
                'needs_categorical': False
            },
            'bar': {
                'keywords': ['słupkowy', 'słupki', 'bar'],
                'needs_numeric': True,
                'needs_categorical': True
            },
            'scatter': {
                'keywords': ['punktowy', 'scatter', 'zależność', 'korelacja'],
                'needs_numeric': True,
                'needs_categorical': False
            },
            'pie': {
                'keywords': ['kołowy', 'pie', 'tortowy'],
                'needs_numeric': False,
                'needs_categorical': True
            },
            'box': {
                'keywords': ['pudełkowy', 'box'],
                'needs_numeric': True,
                'needs_categorical': True
            }
        }
        
        # Słowa kluczowe dla tabel
        self.table_keywords = [
            'tabela', 'zestawienie', 'lista', 'pokaż dane', 'wyświetl dane',
            'przedstaw dane', 'pokaż zestawienie', 'wypisz', 'średnia', 'suma',
            'policz', 'count', 'mean', 'sum', 'max', 'min'
        ]
 
    def detect_viz_type(self, query: str) -> str:
        """Wykrywa typ wizualizacji na podstawie zapytania."""
        query = query.lower()
        logger.info(f"Detecting visualization type for query: {query}")
        
        for viz_type, config in self.viz_types.items():
            if any(keyword in query for keyword in config['keywords']):
                logger.info(f"Detected visualization type: {viz_type}")
                return viz_type
                
            logger.info("No visualization type detected")
        return None
 
    def create_visualization(self, query: str) -> Dict[str, Any]:
        """Tworzy wizualizację na podstawie zapytania."""
        logger.info(f"Creating visualization for query: {query}")
        
        viz_type = self.detect_viz_type(query)
        if not viz_type:
            logger.info("No visualization type detected, returning None")
            return None
 
        try:
            if viz_type == 'histogram':
                numeric_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    logger.info(f"Creating histogram for column: {numeric_cols[0]}")
                    fig = px.histogram(self.df, x=numeric_cols[0],
                                     title=f'Rozkład {numeric_cols[0]}')
 
            elif viz_type == 'bar':
                cat_col = self.df.select_dtypes(include=['object']).columns[0]
                num_col = self.df.select_dtypes(include=[np.number]).columns[0]
                logger.info(f"Creating bar chart for {num_col} by {cat_col}")
                df_grouped = self.df.groupby(cat_col)[num_col].mean().reset_index()
                fig = px.bar(df_grouped, x=cat_col, y=num_col,
                title=f'Średnie {num_col} według {cat_col}')
 
            elif viz_type == 'scatter':
                num_cols = self.df.select_dtypes(include=[np.number]).columns
                if len(num_cols) >= 2:
                    logger.info(f"Creating scatter plot for {num_cols[0]} vs {num_cols[1]}")
                    fig = px.scatter(self.df, x=num_cols[0], y=num_cols[1],
                                   title=f'Zależność {num_cols[1]} od {num_cols[0]}')
 
            elif viz_type == 'pie':
                cat_col = self.df.select_dtypes(include=['object']).columns[0]
                logger.info(f"Creating pie chart for {cat_col}")
                counts = self.df[cat_col].value_counts()
                fig = px.pie(values=counts.values, names=counts.index,
                           title=f'Rozkład {cat_col}')
 
            elif viz_type == 'box':
                num_col = self.df.select_dtypes(include=[np.number]).columns[0]
                cat_col = self.df.select_dtypes(include=['object']).columns[0]
                logger.info(f"Creating box plot for {num_col} by {cat_col}")
                fig = px.box(self.df, x=cat_col, y=num_col,
                           title=f'Rozkład {num_col} według {cat_col}')
 
            # Wspólne ustawienia dla wszystkich wykresów
            fig.update_layout(
                template='plotly_white',
                title_x=0.5,
                margin=dict(t=50, l=0, r=0, b=0),
                showlegend=True,
                height=600,
                width=800
            )
 
            logger.info("Visualization created successfully")
            return {
                'type': 'plot',
                'data': fig
            }
 
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            return None
 
    def create_table(self, data: Union[pd.DataFrame, pd.Series], query: str) -> Dict[str, Any]:
        """Tworzy interaktywną tabelę z danych."""
        logger.info(f"Creating table for query: {query}")
        try:
            if isinstance(data, pd.Series):
                data = data.to_frame()
                
                logger.info(f"Table data shape: {data.shape}")
 
            # Formatowanie liczb
            for col in data.select_dtypes(include=[np.number]).columns:
                data[col] = data[col].round(2)
 
            fig = go.Figure(data=[go.Table(
                header=dict(
                    values=list(data.columns),
                    fill_color='paleturquoise',
                    align='left',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[data[col] for col in data.columns],
                    fill_color='lavender',
                    align='left',
                    font=dict(size=11)
                )
            )])
 
            fig.update_layout(
                title=dict(
                    text=query,
                    x=0.5,
                    font=dict(size=14)
                ),
                margin=dict(t=50, l=0, r=0, b=0),
                height=min(600, 100 + 30 * len(data)),
                width=800
            )
 
            logger.info("Table created successfully")
            return {
                'type': 'plot',
                'data': fig
            }
        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            return None
    def get_dataframe_info(self) -> str:
        """Get information about DataFrame structure."""
        info = []
        info.append("DataFrame Information:")
        info.append(f"Rows: {len(self.df)}")
        info.append(f"Columns: {len(self.df.columns)}")
        
        info.append("\nColumn Details:")
        for col in self.df.columns:
            dtype = self.df[col].dtype
            unique_count = self.df[col].nunique()
            
            info.append(f"\n{col}:")
            info.append(f"- Type: {dtype}")
            info.append(f"- Unique Values: {unique_count}")
            
            if dtype == 'object' or unique_count < 10:
                unique_vals = self.df[col].unique()[:5]
                info.append(f"- Sample Values: {', '.join(map(str, unique_vals))}")
            elif np.issubdtype(dtype, np.number):
                info.append(f"- Min: {self.df[col].min()}")
                info.append(f"- Max: {self.df[col].max()}")
                info.append(f"- Mean: {self.df[col].mean():.2f}")
                
        return "\n".join(info)
        
    def clean_code(self, code: str) -> str:
        """Czyści i formatuje kod."""
        logger.info(f"Cleaning code: {code}")
        
        # Usuń znaczniki markdown, jeśli istnieją
        code = code.replace("```python", "").replace("```", "")
        
        # Usuń komentarze
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        
        # Usuń puste linie i nadmiarowe spacje
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        # Połącz linie z powrotem w string
        cleaned_code = '\n'.join(lines)
        
        logger.info(f"Executing cleaned code: {cleaned_code}")
        
        return cleaned_code  # Upewnij się, że zawsze zwracamy string
       
    def get_suggested_queries(self) -> List[str]:
        """Get suggested queries based on data structure."""
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns
        
        suggestions = []
        
        # Sugestie tabel
        suggestions.extend([
            "Pokaż tabelę z podstawowymi statystykami",
            "Wyświetl zestawienie liczebności według departamentów",
            "Zrób tabelę średnich zarobków dla stanowisk"
        ])
        
        # Sugestie wizualizacji
        if len(numeric_cols) > 0:
            suggestions.extend([
                f"Pokaż rozkład {numeric_cols[0]} na histogramie",
                f"Narysuj wykres pudełkowy dla {numeric_cols[0]}"
            ])
        
        if len(categorical_cols) > 0:
            suggestions.append(f"Zrób wykres kołowy dla {categorical_cols[0]}")
        
        if len(numeric_cols) >= 2:
            suggestions.append(
                f"Pokaż wykres punktowy zależności między {numeric_cols[0]} a {numeric_cols[1]}"
            )
        
        return suggestions[:10] # Limit do 10 sugestii
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """Przetwarza zapytanie i zwraca odpowiedni wynik."""
        logger.info(f"\n{'='*50}\nProcessing query: {query}\n{'='*50}")
        try:
            # Sprawdź czy to zapytanie o wizualizację
            if self.detect_viz_type(query):
                logger.info("Processing as visualization query")
                viz_result = self.create_visualization(query)
                if viz_result:
                    return viz_result
            
            # Standardowe przetwarzanie dla zapytań o dane
            logger.info("Processing as data query")
            context = self.context_manager.get_formatted_context()
            df_info = self.get_dataframe_info()
            prompt = self.llm_client.generate_prompt(query, context, df_info)
            
            logger.info(f"Generated prompt:\n{prompt}")
            
            code = self.llm_client.get_response(prompt)
            logger.info(f"Received code from LLM:\n{code}")
            
            result = self.execute_query(code)
            
            # Jeśli to zapytanie o tabelę, stwórz tabelę
            if any(keyword in query.lower() for keyword in self.table_keywords):
                if isinstance(result.get('value'), (pd.DataFrame, pd.Series)):
                    return self.create_table(result['value'], query)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise RuntimeError(f"Error processing query: {str(e)}")
 
    def execute_query(self, code: str) -> Dict[str, Any]:
        """Wykonuje kod i zwraca wynik."""
        try:
            # Wyczyść kod
            cleaned_code = self.clean_code(code)
            logger.info(f"Executing cleaned code:\n{cleaned_code}")
            
            # Utwórz przestrzeń nazw
            local_ns = {'df': self.df, 'pd': pd, 'np': np}
            
            # Wykonaj kod
            exec(cleaned_code, None, local_ns)
            
            # Sprawdź czy jest zmienna result
            if 'result' in local_ns:
                result = local_ns['result']
            else:
                # Jeśli nie, weź ostatnią linię
                last_line = cleaned_code.strip().split('\n')[-1]
                result = eval(last_line, None, local_ns)
            
            logger.info(f"Raw execution result type: {type(result)}")
            
            # Obsługa różnych typów wyników
            if isinstance(result, (pd.DataFrame, pd.Series)):
                return {
                    'type': 'data',
                    'value': result
                }
            elif isinstance(result, (int, float, np.integer, np.floating)):
                formatted_result = f"{float(result):,.2f}"
                return {
                    'type': 'data',
                    'value': formatted_result
                }
            elif isinstance(result, (str, bool)):
                return {
                    'type': 'data',
                    'value': str(result)
                }
            elif result is None:
                return {
                    'type': 'data',
                    'value': "No result"
                }
            else:
                try:
                    # Próba konwersji na DataFrame dla innych typów
                    df_result = pd.DataFrame(result)
                    return {
                        'type': 'data',
                        'value': df_result
                    }
                except:
                    return {
                        'type': 'data',
                        'value': str(result)
                    }
            
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise RuntimeError(f"Error executing query: {str(e)}")

