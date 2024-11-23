import streamlit as st
import pandas as pd
import json
from llm_client import LLMClient
from context_manager import ContextManager
from query_processor import QueryProcessor
import os
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

st.set_page_config(page_title="Natural Data Query", page_icon="📊", layout="wide")

def initialize_session_state():
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = None
    if 'context_manager' not in st.session_state:
        st.session_state.context_manager = None
    if 'query_processor' not in st.session_state:
        st.session_state.query_processor = None
    if 'df' not in st.session_state:
        st.session_state.df = None

def display_result(result):
    """Display query result based on its type"""
    try:
        if isinstance(result, dict):
            if result.get('type') == 'plot' and result.get('data') is not None:
                # Wyświetl wykres
                fig = result['data']
                st.plotly_chart(fig, use_container_width=True)
                
                # Jeśli są dodatkowe wartości, wyświetl je
                if result.get('value') is not None:
                    st.write("Calculated value:", result['value'])
            
            elif result.get('type') == 'data':
                if isinstance(result['value'], pd.DataFrame):
                    st.dataframe(result['value'])
                else:
                    st.write("Result:", result['value'])
        else:
            st.write("Result:", result)
    except Exception as e:
        st.error(f"Error displaying result: {str(e)}")

def main():
    initialize_session_state()
    
    st.title("Natural Data Query")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # LLM Model Selection
        llm_choice = st.radio("Select LLM Model:", ["OpenAI", "Anthropic", "Ollama"])
        
        if llm_choice == "Ollama":
            available_models = LLMClient.list_available_ollama_models()
            if available_models:
                selected_model = st.selectbox("Select Ollama Model:", available_models)
                os.environ["OLLAMA_MODEL"] = selected_model
            else:
                st.warning("No Ollama models found. Make sure Ollama is running locally.")
            api_key = "not_required"
        else:
            api_key = st.text_input(
                f"Enter {llm_choice} API Key:",
                type="password",
                value=os.getenv(f"{llm_choice.upper()}_API_KEY", "")
            )
        
        # File uploaders
        csv_file = st.file_uploader("Upload CSV Data File:", type=['csv'])
        dict_file = st.file_uploader("Upload Dictionary File:", type=['json'])
        
        if st.button("Initialize System"):
            try:
                # Initialize LLM Client
                st.session_state.llm_client = LLMClient(llm_choice, api_key)
                
                # Load data
                if csv_file is not None:
                    st.session_state.df = pd.read_csv(csv_file)
                    st.success("CSV file loaded successfully!")
                
                # Load dictionary
                if dict_file is not None:
                    dictionary = json.load(dict_file)
                    st.session_state.context_manager = ContextManager(dictionary)
                    st.success("Dictionary file loaded successfully!")
                
                if st.session_state.df is not None and st.session_state.context_manager is not None:
                    st.session_state.query_processor = QueryProcessor(
                        st.session_state.df,
                        st.session_state.context_manager,
                        st.session_state.llm_client
                    )
                    st.success("System initialized successfully!")
            except Exception as e:
                st.error(f"Error during initialization: {str(e)}")
    
    # Main content area
    if st.session_state.df is not None:
        st.header("Data Preview")
        st.dataframe(st.session_state.df.head())
    
    if st.session_state.query_processor is not None:
        st.header("Query Data")
        
        # Example queries for plots
        with st.expander("Show example visualization queries"):
            st.markdown("""
            Try these example queries:
            - "Pokaż rozkład dochodów na histogramie"
            - "Stwórz wykres słupkowy średnich zarobków w departamentach"
            - "Zrób wykres kołowy pokazujący podział pracowników według wykształcenia"
            - "Narysuj wykres punktowy zależności wieku od dochodu"
            - "Pokaż trend zarobków względem stażu pracy"
            """)
        
        # Query input
        query = st.text_input("Enter your question about the data:")
        
        if query:
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Ask", use_container_width=True):
                    try:
                        with st.spinner("Processing query..."):
                            # Debug info
                            st.session_state.debug_info = {}
                            
                            # Process query
                            result = st.session_state.query_processor.process_query(query)
                            
                            # Display debug info if needed
                            if os.getenv("DEBUG") == "True":
                                with st.expander("Debug Info"):
                                    st.write(st.session_state.debug_info)
                            
                            # Display results
                            with col2:
                                st.success("Query processed successfully!")
                                display_result(result)
                                
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
    else:
        st.info("Please initialize the system using the sidebar controls first.")

if __name__ == "__main__":
    main()
