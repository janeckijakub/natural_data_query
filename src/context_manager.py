from typing import Dict, Any, List

class ContextManager:
    def __init__(self, dictionary: Dict[str, Any]):
        self.dictionary = dictionary
        self.columns = dictionary.get("columns", {})
        self.context = dictionary.get("context", {})
        
        # Create reverse mapping for column aliases
        self.alias_mapping = {}
        for col_name, col_info in self.columns.items():
            for alias in col_info.get("aliases", []):
                self.alias_mapping[alias.lower()] = col_name
            # Add the original column name as its own alias
            self.alias_mapping[col_name.lower()] = col_name
    
    def get_column_info(self, column_name: str) -> Dict[str, Any]:
        """Get information about a specific column."""
        return self.columns.get(column_name, {})
    
    def get_column_by_alias(self, alias: str) -> str:
        """Get the actual column name from an alias."""
        return self.alias_mapping.get(alias.lower())
    
    def get_all_columns(self) -> List[str]:
        """Get all column names."""
        return list(self.columns.keys())
    
    def get_context_info(self) -> Dict[str, str]:
        """Get general context information about the dataset."""
        return self.context
    
    def get_formatted_context(self) -> str:
        """Get formatted context for LLM prompt."""
        context_str = "Dataset Information:\n"
        
        # Add general context
        for key, value in self.context.items():
            context_str += f"{key}: {value}\n"
        
        # Add column definitions
        context_str += "\nColumns:\n"
        for col_name, col_info in self.columns.items():
            context_str += f"- {col_name}:\n"
            context_str += f" Description: {col_info.get('description', 'No description')}\n"
            context_str += f" Type: {col_info.get('type', 'unknown')}\n"
            if col_info.get('aliases'):
                context_str += f" Aliases: {', '.join(col_info['aliases'])}\n"
        
        return context_str
