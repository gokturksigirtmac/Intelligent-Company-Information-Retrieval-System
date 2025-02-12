import os
import re
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationStringBufferMemory
from tavily import TavilyClient
from langchain.tools import Tool
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from nltk.corpus import wordnet

load_dotenv()

search_tool = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
tavily_url = "https://api.tavily.com/search"
wikipedia_api_wrapper = WikipediaAPIWrapper()
wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY"))

class SearchState:
    def __init__(self, query: str, company: str = "", intent: str = "", response: str = "", 
                 clarification_needed: bool = False):
        self.query = query
        self.company = company
        self.intent = intent
        self.response = response
        self.clarification_needed = clarification_needed

def refine_intent(intent: str):
    synonyms = {
        "location": "location headquarters office hq address",
        "business_model": "business revenue profit monetization earnings",
        "investments": "investments funding venture capital backing",
        "timeframe": "news updates recent latest",
        "customers": "customers clients users buyers consumers"
    }
    
    for key, words in synonyms.items():
        if intent in words.split():
            return key
    return intent 

def extract_company_intent(user_input: str):
    prompt = PromptTemplate(
        input_variables=["user_input"],
        template="""
         Identify the company name and classify the user's intent into one of the following categories:
        - Location (e.g., "Where is Tesla headquartered?")
        - Business Model (e.g., "How does Tesla make money?")
        - Investments (e.g., "Which companies has Sequoia invested in?")
        - Timeframe (e.g., "What are the latest news articles about NVIDIA?")
        - Customers (e.g., "Who are the customers of Entrapeer?")
        
        If the question is too broad, suggest clarification by listing potential topics.
        
        Input: {user_input}
        
        Output format (valid JSON only, no additional text):
        {{
            "company": "<company_name>",
            "intent": "<intent>",
            "clarification_needed": <true/false>,
            "clarification_topics": ["topic1", "topic2", ...]
        }}
        """
    )
    response = llm.invoke(prompt.format(user_input=user_input))
    response_text = response.content if hasattr(response, 'content') else str(response)

     # Extract JSON if the response includes unwanted text
    json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if json_match:
        response_text = json_match.group(0)
    
    try:
        parsed_response = json.loads(response_text)
        company = parsed_response.get("company", "Unknown")
        intent = parsed_response.get("intent", "Unknown").strip().lower()
        intent = refine_intent(intent)  # Apply intent refinement
        clarification_needed = parsed_response.get("clarification_needed", False)
        topics = parsed_response.get("clarification_topics", [])
    except json.JSONDecodeError:
        print("Error: LLM did not return valid JSON.")
        return "Unknown", "Unknown", clarification_needed
    
    return company, intent, clarification_needed

def extract_relevant_page(wikipedia_output, query):
    company_keywords = ["Inc.", "Corporation", "Company", "LLC", "Ltd", "Business", "Firm", "Enterprise"]
    
    # Split results by page
    pages = wikipedia_output.split("\n\n")  # Assuming Wikipedia tool returns multiple results separated by double newlines
    
    for page in pages:
        # Prioritize pages where the title contains company-related keywords
        if any(keyword in page for keyword in company_keywords) or query.lower() in page.lower():
            return page  # Return the first matching company-related result

    return None

def summarize_company_info(text, query):
    summarization_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
        Given the following Wikipedia information about a company, also, human's query for the Wikipedia search:
        {query}. Generate reasonable and related summarize human's query
        {text}

        Output Format:
        - Summarized information about the company's {query}
        """
    )                    

    response = llm.invoke(summarization_prompt.format(text=text, query=query))
    return response.content if hasattr(response, 'content') else str(response)

def summarize_article(news_text):
    summarization_prompt = PromptTemplate(
        input_variables=["news_text"],
        template="""
            You are an expert news analyst. Your task is to summarize the latest news article concisely while retaining key details.

            ### News Article:
            {news_text}

            ### Summary Requirements:
            - Provide a **brief, engaging, and factual** summary.
            - Highlight **key events, dates, and people** mentioned.
            - Capture the **main idea** in **2-3 sentences**.
            - Avoid unnecessary details or speculation.
            - If available, include the **source or author** at the end.

            ### Output Format:
            Provide the output in valid **JSON format** as follows:
            {{
                "Title": "[News headline]",
                "Summary": "[Concise summary]",
                "Source": "[URL or source name]"
            }}

            Now, generate the JSON response.
        """
    )

    response = llm.invoke(summarization_prompt.format(news_text=news_text))
    return response.content if hasattr(response, 'content') else str(response)

def process_user_input(user_input: str):
    company, intent, clarification_needed = extract_company_intent(user_input)

    if clarification_needed:
        return {
            "company": company,
            "intent": intent,
            "clarification_needed": True,
            "message": f"Your query '{user_input}' is too broad. Please specify a topic.",
            "topics": ["Revenue Model", "Key Customers", "Latest News", "Investors", "Headquarters"]
        }

    state = SearchState(query=user_input, company=company, intent=intent, clarification_needed=clarification_needed)
    result = generic_info_graph.invoke(state)
    
    return {
        "company": state.company,
        "intent": state.intent,
        "response": state.response
    }

def search_location(query):
    try:
        search_results = wikipedia_tool.run(query)  # Fetch Wikipedia search results
        relevant_page = extract_relevant_page(search_results, query)
        if relevant_page:
            return summarize_company_info(relevant_page, query)
        else:
            return "Relevant company information not found."

    except Exception as e:
        return f"Error fetching Wikipedia data: {str(e)}"

def search_business_model(query):
    search_results = search_tool.search(query=f"{query}", search_depth="basic")
    print(query)
    top_results = search_results["results"][:1]
    response = []
    for result in top_results:
        content = result.get('content', 'N/A')
        url = result.get('url', '#')

        response.append({
            "content": summarize_article(content),  
            "url": url
        })

    response_json = json.dumps(response, indent=2)

    return response_json 

def search_investments(query):
    try:
        search_results = search_tool.search(query=f"{query} latest news", search_depth="basic")

        top_results = search_results["results"][:1]  

        response = []
        for result in top_results:
            content = result.get('content', 'N/A')
            url = result.get('url', '#')

            # Push formatted results to response list
            response.append({
                "content": summarize_article(content), 
                "url": url
            })

        response_json = json.dumps(response, indent=2)
        return response_json  

    except Exception as e:
        return f"Error fetching news articles: {str(e)}"

def search_timeframe(query):
    try:
        search_results = search_tool.search(query=f"{query} latest news", search_depth="basic")

        top_results = search_results["results"][:5]  

        response = []
        for result in top_results:
            content = result.get('content', 'N/A')
            url = result.get('url', '#')

            response.append({
                "content": summarize_article(content), 
                "url": url
            })

        return response

    except Exception as e:
        return f"Error fetching news articles: {str(e)}"

def search_customers(query):
    try:
        search_results = search_tool.search(query=f"{query} latest news", search_depth="basic")

        top_results = search_results["results"][:1]  

        response = []
        for result in top_results:
            content = result.get('content', 'N/A')
            url = result.get('url', '#')

            response.append({
                "content": summarize_article(content),  # Ensure summarize_timeframe is defined
                "url": url
            })

        return response

    except Exception as e:
        return f"Error fetching news articles: {str(e)}"

search_tools = {
    "location": Tool(name="Location Lookup", func=search_location, description="Finds company headquarters location."),
    "business_model": Tool(name="Business Model Analysis", func=search_business_model, description="Retrieves business model insights."),
    "investments": Tool(name="Investment Lookup", func=search_investments, description="Finds investment details."),
    "timeframe": Tool(name="News Lookup", func=search_timeframe, description="Finds latest news articles."),
    "customers": Tool(name="Customer Analysis", func=search_customers, description="Finds customer details.")
}

def search_company_info(state: SearchState):
    normalized_intent = state.intent.strip().lower().replace(" ", "_")
    print(f"Searching for {state.intent} information about {state.company}...")

    tool = search_tools.get(normalized_intent)
    
    if tool:
        result = tool.func(f"{state.intent} of {state.company}")
    else:
        print(f"Warning: No tool found for intent '{normalized_intent}'")
        result = "No relevant tool found for this intent."
    
    state.response = result
    return state

graph = StateGraph(SearchState)
graph.add_node("search", search_company_info)
graph.set_entry_point("search")
generic_info_graph = graph.compile()