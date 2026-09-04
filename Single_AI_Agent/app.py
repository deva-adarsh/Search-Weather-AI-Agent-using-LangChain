import os
import requests
import certifi
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")


# ==========================================
# CHECK API KEYS
# ==========================================

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY is missing from .env")

if not WEATHERSTACK_API_KEY:
    st.error("WEATHERSTACK_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    st.error("TAVILY_API_KEY is missing from .env")


# ==========================================
# STREAMLIT PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agentic AI Assistant")

st.markdown(
    "Search + Weather AI Agent using LangChain"
)


# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearchResults(
    max_results=2
)


# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:
        response = requests.get(
            url,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:
        return f"Weather API request failed: {str(e)}"

    except ValueError:
        return "Weather API returned invalid JSON."

    # Weatherstack can return an error object
    if "error" in data:
        return (
            f"Could not fetch weather for {city}. "
            f"API error: {data['error'].get('info', 'Unknown error')}"
        )

    if "current" not in data:
        return f"Could not fetch weather data for {city}."

    current = data["current"]

    weather_description = current.get(
        "weather_descriptions",
        ["Unknown"]
    )[0]

    temperature = current.get(
        "temperature",
        "Unknown"
    )

    humidity = current.get(
        "humidity",
        "Unknown"
    )

    return (
        f"City: {city}\n"
        f"Temperature: {temperature}°C\n"
        f"Weather: {weather_description}\n"
        f"Humidity: {humidity}%"
    )


# ==========================================
# LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=OPENAI_API_KEY
)


# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data
]


# ==========================================
# CREATE AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful Agentic AI assistant. "
        "You have access to a web search tool and a weather tool. "
        "Use the web search tool when the user needs current or "
        "general information from the internet. "
        "Use the weather tool when the user asks for weather. "
        "If the user asks for multiple pieces of information, "
        "use the appropriate tools and combine the results into "
        "a clear final answer."
    )
)


# ==========================================
# UI INPUT
# ==========================================

user_query = st.text_input(
    "Enter your query:",
    placeholder=(
        "Example: Find the capital of India "
        "and current weather"
    )
)


# ==========================================
# RUN AGENT
# ==========================================

if st.button("Ask Agent"):

    if not user_query:
        st.warning("Please enter a query.")

    else:

        with st.spinner("Agent is thinking..."):

            try:

                response = agent.invoke(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": user_query
                            }
                        ]
                    }
                )

                # Get the final AI message
                final_message = response["messages"][-1]

                st.subheader("🤖 Agent Response")

                st.write(final_message.content)

            except Exception as e:

                st.error(
                    f"An error occurred: {str(e)}"
                )