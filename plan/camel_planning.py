from colorama import Fore
from camel.societies import RolePlaying
from camel.utils import print_text_animated
from camel.models import ModelFactory
from camel.types import ModelPlatformType
from camel.toolkits import FunctionTool
from dotenv import load_dotenv
from tavily import TavilyClient
import os

load_dotenv()

# this model uses deepseek-flash
class CamelPlanning:
    def __init__(self,user_name,assistant_name,prompt):
        self.llm_api_key=os.getenv("LLM_API_KEY")
        self.llm_base_url=os.getenv("LLM_BASE_URL")
        self.llm_model="deepseek-flash"
        self.tavily_api_key=os.getenv("TAVILY_API_KEY")
        self.task_prompt = prompt
        self.user_name=user_name
        self.assistant_name=assistant_name
        
        self.tavily_client=self.create_tavily_client()
        self.tool_list=[
            FunctionTool(self.tavily_search)
        ]
        
        self.camel_model_client=self.create_openai_model_client()
        self.role_play_model=self.create_role_play_model()

    def create_openai_model_client(self):
        return ModelFactory.create(
            model_platform=ModelPlatformType.DEEPSEEK,
            model_type=self.llm_model,
            url=self.llm_base_url,
            api_key=self.llm_api_key
        )
    
    def create_role_play_model(self):
        return RolePlaying(
            assistant_role_name=self.assistant_name, 
            assistant_agent_kwargs=dict(
                tools=self.tool_list
            ),
            user_role_name=self.user_name, 
            user_agent_kwargs=dict(
                tools=self.tool_list
            ),
            task_prompt=self.task_prompt,
            model=self.camel_model_client
        )
    
    def create_tavily_client(self):
        return TavilyClient(api_key=self.tavily_api_key)
    
    def tavily_search(self,query:str)->str:
        """ Use Tavily Search API to search information for the given query.
            Args:
            query: search query

            Returns:
            Relevant web search results.
            """
        response = self.tavily_client.search(query=query, max_results=5)

        if response.get('answer'):
            result = f"Answer: {response['answer']}\n\n"
        else:
            result = ""

        result += "Related result:\n"
        for i, item in enumerate(response.get('results', [])[:3], 1):
            result += f"[{i}] {item.get('title', '')}\n"
            result += f"URL: {item.get('url', '')}\n"
            result += f"{item.get('content', '')}\n\n"
        return result

    def communication(self,chat_turn_limit=30):
        input_msg = self.role_play_model.init_chat()

        print(Fore.YELLOW + f"Prompted task:\n{self.task_prompt}\n")
        print(Fore.CYAN + f"Detailed task:\n{self.role_play_model.task_prompt}\n")
        
        n=0
        while n < chat_turn_limit:
            n += 1
            assistant_response, user_response = self.role_play_model.step(input_msg)
            print_text_animated(Fore.BLUE + f"{self.user_name}:\n\n{user_response.msg.content}\n")
            print_text_animated(Fore.GREEN + f"{self.assistant_name}:\n\n{assistant_response.msg.content}\n")
            
            if "CAMEL_TASK_DONE" in user_response.msg.content:
                print(Fore.MAGENTA + "Task completed")
                break
            input_msg = assistant_response.msg

        print(Fore.YELLOW + f"There is in total {n} turn(s) of communication")