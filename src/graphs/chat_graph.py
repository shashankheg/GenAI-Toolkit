from typing import Annotated, TypedDict, List #Better code writing for readabality 
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.utils.llm import get_llm



# === STATE DEFINITION ===

class ChatState(TypedDict):
    messages : Annotated[list, add_messages]



 
# === SYSTEM PROMPT ===

SYSTEM_PROMPT = SystemMessage(content="""You are a helpful assistant.

You provide clear, accurate, and concise responses.
If you don't know something, you say so honestly.
You maintain context throughout the conversation.""")

# === NODES ===
def chat_node(state: ChatState) -> ChatState:
    # Get the last message from the state
    last_message = state["messages"][-1]

    # Get the LLM
    llm = get_llm()

    # Generate a response
    response = llm.invoke([SYSTEM_PROMPT] + state["messages"])

    # Add the response to the state
    state["messages"].append(response)

    return state        


# === BUILD GRAPH ===
def build_chat_graph():
    graph = StateGraph(ChatState)
    graph.add_node("chat", chat_node)
    graph.set_entry_point("chat")
    graph.add_edge("chat", END)
    return graph.compile()

# === CHAT SESSION ===
class ChatSession:
    """Manages a multi-turn chat conversation."""

    def __init__(self):
        self.graph = build_chat_graph()
        self.history: List = []

    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.

        Args:
            user_message: User input string

        Returns:
            AI response string
        """
        if not user_message.strip():
            return "⚠️ Please enter a message."
        # Add user message to history
        self.history.append(HumanMessage(content=user_message))

        # Run graph
        result = self.graph.invoke({"messages": self.history})

        # Extract and store response
        ai_message = result["messages"][-1]
        self.history.append(ai_message)


        return ai_message.content    
    
    def clear_history(self):
        """Clears conversation history."""
        self.history = []
        return "✅ Conversation cleared."
    






