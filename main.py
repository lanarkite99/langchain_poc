from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper, TextRequestsWrapper
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

google_api_key=os.getenv('GEMINI_API_KEY')
print("Gemini Key Loaded:", google_api_key[:5] + "..." if google_api_key else "not found")

serpapi_key=os.getenv('SERPAPI_API_KEY')
print("SerpAPI Key Loaded:", serpapi_key[:5] + "..." if serpapi_key else "not found")

llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash',temperature=0,google_api_key=google_api_key)

srch_tool=Tool(name='Search',func=SerpAPIWrapper().run,
               description="Search for recent research articles on topic")
read_tool=Tool(name='ReadURL',func=TextRequestsWrapper().get,
               description='Read and summarize the full article')

tools=[srch_tool,read_tool]

agent=initialize_agent(llm=llm,
                       tools=tools,
                       agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                       verbose=True)

srch_query='Recent research articles on AI agents 2024 site:arxiv.org'
response=agent.run(srch_query)

import re
urls = re.findall(r'https?://[^\s]+', response)[:3]
print("Top 3 URLs:", urls)

reader = TextRequestsWrapper()
summaries=[]
for url in urls:
    try:
        content=reader.get(url)
        if content:
            summary_prompt=f"summarize the research article from this content:\n{content[:4000]}"
            summary=llm.predict(summary_prompt)
            summaries.append((url,summary))
    except Exception as e:
        print(f"error reading/summarizing {url}: {e}")

if len(summaries)>=3:
    comparison_prompt=PromptTemplate.from_template(
        """you're a helpful research assistant. you'll receive summaries of 3 recent research articles
        on AI agents. compare their key contributions and determine which is the most novel and impactful.
        
        summaries:
        1. {s1}
        2. {s2}
        3. {s3}
        
        please provide: a short comparison, your judgement on which one is the most novel and why"""
    )
    final_analysis=llm.predict(comparison_prompt.format(s1=summaries[0][1],
                                                        s2=summaries[1][1],
                                                        s3=summaries[2][1]))
else:
    final_analysis = "Not enough summaries were generated to perform a comparison."

op_file = "ai_agent_analysis.txt"
with open(op_file,'w',encoding='utf-8') as f:
    for i,(url,summ) in enumerate(summaries,1):
        f.write(f"\n-- Article {i} --\nURL: {url}\nSummary:\n{summ}\n")
    f.write("\n final comparison \n")
    f.write(final_analysis)