import os
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# State definition
class AgentState(TypedDict):
    """State for the feedback classification agent."""
    query: str
    query_type: str  # "simple" or "complex"
    classification_result: dict | None
    llm_analysis: str | None
    final_response: dict | None


class FeedbackAgent:
    """LangGraph agent for intelligent feedback classification."""
    
    def __init__(
        self,
        classifier_url: str = "http://localhost:8000",
        openai_api_key: str | None = None,
    ):
        self.classifier_url = classifier_url
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if self.api_key:
            self.llm = ChatOpenAI(
                model="gpt-4o-mini",
                api_key=self.api_key,
                temperature=0,
            )
        else:
            self.llm = None
            logger.warning("No OpenAI API key provided. LLM features disabled.")
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("analyze_query", self._analyze_query)
        workflow.add_node("classify_simple", self._classify_simple)
        workflow.add_node("analyze_complex", self._analyze_complex)
        workflow.add_node("format_response", self._format_response)
        
        # Set entry point
        workflow.set_entry_point("analyze_query")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_query",
            self._route_query,
            {
                "simple": "classify_simple",
                "complex": "analyze_complex",
            },
        )
        
        # Add edges to format_response
        workflow.add_edge("classify_simple", "format_response")
        workflow.add_edge("analyze_complex", "format_response")
        workflow.add_edge("format_response", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the query to determine complexity."""
        query = state["query"]
        
        # Heuristics for complex queries
        complex_indicators = [
            "why", "explain", "analyze", "compare", "what if",
            "how does", "can you", "tell me more", "elaborate",
            len(query.split()) > 50,  # Long queries
            "?" in query and query.count("?") > 1,  # Multiple questions
        ]
        
        is_complex = any(
            indicator if isinstance(indicator, bool) 
            else indicator.lower() in query.lower()
            for indicator in complex_indicators
        )
        
        state["query_type"] = "complex" if is_complex and self.llm else "simple"
        return state
    
    def _route_query(self, state: AgentState) -> Literal["simple", "complex"]:
        """Route to appropriate handler based on query type."""
        return state["query_type"]
    
    def _classify_simple(self, state: AgentState) -> AgentState:
        """Use the classifier for simple sentiment classification."""
        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    f"{self.classifier_url}/predict",
                    json={"text": state["query"], "use_agent": False},
                )
                response.raise_for_status()
                state["classification_result"] = response.json()
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state["classification_result"] = {
                "error": str(e),
                "label": "unknown",
                "confidence": 0.0,
            }
        
        return state
    
    def _analyze_complex(self, state: AgentState) -> AgentState:
        """Use LLM for complex query analysis."""
        # First, get classification
        state = self._classify_simple(state)
        
        if self.llm is None:
            state["llm_analysis"] = "LLM analysis unavailable (no API key)"
            return state
        
        # Then enhance with LLM analysis
        classification = state.get("classification_result", {})
        
        messages = [
            SystemMessage(content="""You are a customer feedback analyst. 
            Analyze the feedback and provide:
            1. Key themes and concerns
            2. Actionable insights
            3. Suggested response or action
            Keep your response concise and structured."""),
            HumanMessage(content=f"""
            Customer Feedback: {state['query']}
            
            Initial Classification: {classification.get('label', 'unknown')}
            Confidence: {classification.get('confidence', 0):.2%}
            
            Please provide a detailed analysis.
            """),
        ]
        
        try:
            response = self.llm.invoke(messages)
            state["llm_analysis"] = response.content
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            state["llm_analysis"] = f"Analysis unavailable: {e}"
        
        return state
    
    def _format_response(self, state: AgentState) -> AgentState:
        """Format the final response."""
        classification = state.get("classification_result", {})
        
        response = {
            "text": state["query"],
            "label": classification.get("label", "unknown"),
            "confidence": classification.get("confidence", 0.0),
            "probabilities": classification.get("probabilities", {}),
            "query_type": state["query_type"],
        }
        
        if state.get("llm_analysis"):
            response["analysis"] = state["llm_analysis"]
        
        state["final_response"] = response
        return state
    
    async def run(self, query: str) -> dict:
        """Run the agent on a query."""
        initial_state: AgentState = {
            "query": query,
            "query_type": "",
            "classification_result": None,
            "llm_analysis": None,
            "final_response": None,
        }
        
        # Run synchronously (LangGraph's invoke is sync)
        result = self.graph.invoke(initial_state)
        return result["final_response"]


# Global agent instance
_agent: FeedbackAgent | None = None


def get_agent() -> FeedbackAgent:
    """Get or create the global agent instance."""
    global _agent
    if _agent is None:
        _agent = FeedbackAgent()
    return _agent


async def run_agent(query: str) -> dict:
    """Run the feedback agent on a query."""
    agent = get_agent()
    return await agent.run(query)

