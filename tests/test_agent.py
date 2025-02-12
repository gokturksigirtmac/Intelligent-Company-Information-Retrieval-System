import unittest
from unittest.mock import patch, MagicMock
import json

from app.agent import (
    extract_company_intent,
    refine_intent,
    process_user_input,
    search_location,
    search_business_model,
    search_investments,
    search_timeframe,
    search_customers,
    SearchState
)

class TestAgent(unittest.TestCase):
    
    @patch("agent.llm.invoke")
    def test_extract_company_intent(self, mock_llm):
        mock_llm.return_value = MagicMock(content=json.dumps({
            "company": "Tesla",
            "intent": "location",
            "clarification_needed": False
        }))
        
        company, intent, clarification_needed = extract_company_intent("Where is Tesla headquartered?")
        
        self.assertEqual(company, "Tesla")
        self.assertEqual(intent, "location")
        self.assertFalse(clarification_needed)

    def test_refine_intent(self):
        self.assertEqual(refine_intent("revenue"), "business_model")
        self.assertEqual(refine_intent("profit"), "business_model")
        self.assertEqual(refine_intent("customers"), "customers")
        self.assertEqual(refine_intent("funding"), "investments")
        self.assertEqual(refine_intent("news"), "timeframe")
        self.assertEqual(refine_intent("unknown_intent"), "unknown_intent")
    
    @patch("agent.generic_info_graph.invoke")
    def test_process_user_input(self, mock_graph):
        mock_graph.return_value = SearchState(query="Where is Tesla headquartered?", company="Tesla", intent="location", response="Palo Alto, California")
        
        result = process_user_input("Where is Tesla headquartered?")
        
        self.assertEqual(result["company"], "Tesla")
        self.assertEqual(result["intent"], "location")
        self.assertEqual(result["response"], "Palo Alto, California")
    
    @patch("agent.wikipedia_tool.run")
    def test_search_location(self, mock_wikipedia):
        mock_wikipedia.return_value = "Tesla Inc. is headquartered in Palo Alto, California."
        
        result = search_location("Tesla headquarters")
        
        self.assertIn("Tesla", result)
        self.assertIn("Palo Alto", result)
    
    @patch("agent.search_tool.search")
    def test_search_business_model(self, mock_search):
        mock_search.return_value = {"results": [{"content": "Tesla's business model relies on direct sales and online orders.", "url": "https://example.com"}]}
        
        result = search_business_model("Tesla business model")
        
        self.assertIn("Tesla", result)
        self.assertIn("direct sales", result)
    
    @patch("agent.search_tool.search")
    def test_search_investments(self, mock_search):
        mock_search.return_value = {"results": [{"content": "Tesla has invested in battery technology.", "url": "https://example.com"}]}
        
        result = search_investments("Tesla investments")
        
        self.assertIn("Tesla", result)
        self.assertIn("battery technology", result)
    
    @patch("agent.search_tool.search")
    def test_search_timeframe(self, mock_search):
        mock_search.return_value = {"results": [{"content": "Tesla announced new models in 2024.", "url": "https://example.com"}]}
        
        result = search_timeframe("Tesla latest news")
        
        self.assertIn("Tesla", result)
        self.assertIn("2024", result)
    
    @patch("agent.search_tool.search")
    def test_search_customers(self, mock_search):
        mock_search.return_value = {"results": [{"content": "Tesla's customers include individuals and fleet buyers.", "url": "https://example.com"}]}
        
        result = search_customers("Tesla customers")
        
        self.assertIn("Tesla", result)
        self.assertIn("individuals and fleet buyers", result)

if __name__ == "__main__":
    unittest.main()
