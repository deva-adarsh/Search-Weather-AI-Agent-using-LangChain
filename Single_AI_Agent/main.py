import os
import certifi
import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.prebuilt import create_react_agent


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ==========================================
# CHECK API KEYS
# ==========================================

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing from .env")

if not TAVILY_API_KEY:
    raise ValueError("TAVILY_API_KEY is missing from .env")

if not WEATHERSTACK_API_KEY:
    raise ValueError("WEATHERSTACK_API_KEY is missing from .env")


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

    response = requests.get(url, timeout=10)

    data = response.json()

    # Check for API error
    if "current" not in data:

        error_message = data.get("error", {}).get(
            "info",
            "Unknown weather API error"
        )

        return f"Could not fetch weather data for {city}. Error: {error_message}"

    current_weather = data["current"]

    return (
        f"City: {city}\n"
        f"Temperature: {current_weather['temperature']}°C\n"
        f"Weather: {current_weather['weather_descriptions'][0]}\n"
        f"Humidity: {current_weather['humidity']}%"
    )


# ==========================================
# LLM
# ==========================================

llm = ChatOpenAI(
    model="gpt-5-nano",
    api_key=OPENAI_API_KEY
)


# ==========================================
# TOOLS
# ==========================================

tools_list = [
    search_tool,
    get_weather_data
]


# ==========================================
# CREATE REACT AGENT
# ==========================================

agent = create_react_agent(
    model=llm,
    tools=tools_list,
    prompt=(
        "You are a helpful assistant. "
        "Use the available tools whenever necessary. "
        "For questions about current weather, always use the weather tool. "
        "For factual information such as capitals, use the search tool when necessary. "
        "Give a clear final answer to the user."
    )
)


# ==========================================
# RUN AGENT
# ==========================================

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Find the capital of India "
                    "and then find its current weather and humidity."
                )
            }
        ]
    }
)


# ==========================================
# FINAL OUTPUT
# ==========================================

print("\n===========================")
print("FINAL OUTPUT")
print("===========================")

print(response["messages"][-1].content)