from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SerpAPIWrapper, TextRequestsWrapper
from langchain.prompts import PromptTemplate
import os,feedparser,re
from dotenv import load_dotenv

load_dotenv()

google_api_key=os.getenv('GEMINI_API_KEY')
print("Gemini Key Loaded:", google_api_key[:5] + "..." if google_api_key else "not found")

# serpapi_key=os.getenv('SERPAPI_API_KEY')
# print("SerpAPI Key Loaded:", serpapi_key[:5] + "..." if serpapi_key else "not found")

llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash',temperature=0,google_api_key=google_api_key)

# srch_tool=Tool(name='Search',func=SerpAPIWrapper().run,
#                description="Search for recent research articles on topic")
# read_tool=Tool(name='ReadURL',func=TextRequestsWrapper().get,
#                description='Read and summarize the full article')
#
# tools=[srch_tool,read_tool]
#
# agent=initialize_agent(llm=llm,
#                        tools=tools,
#                        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#                        verbose=True)

srch_query='AI agents 2024'
max_res=5
base_url="http://export.arxiv.org/api/query?"
query = f"search_query=all:{srch_query.replace(' ', '+')}&start=0&max_results={max_res}"
url=base_url+query

feed=feedparser.parse(url)

summaries=[]
for entry in feed.entries:
    title = entry.title
    authors = ', '.join(author.name for author in entry.authors)
    abstract = entry.summary.replace('\n', ' ').strip()
    pdf_link = entry.link.replace('abs', 'pdf')

    print(f"\n summarizing: {title}")
    prompt = f"summarize this research abstract in simple terms:\n\n{abstract}"

    try:
        summary=llm.predict(prompt)
    except Exception as e:
        summary=f"error generating summary: {e}"

    summaries.append({'title':title,
                      "authors": authors,
                      "abstract": abstract,
                      'summary':summary,
                      "pdf_link": pdf_link
                      })

output_file='arxiv_ai_agent_summary.txt'
with open(output_file,'w',encoding='utf-8') as f:
    for i,paper in enumerate(summaries,1):
        f.write(f"--article{i}--\n")
        f.write(f"title:{paper['title']}\n")
        f.write(f"authots:{paper['authors']}\n")
        f.write(f"pdf:{paper['pdf_link']}\n\n")
        f.write(f"Abtract:\n{paper['abstract']}\n\n")
        f.write(f"Summary:\n{paper['summary']}\n\n")

print(f"\n saved {len(summaries)} paper summaries to:{output_file}")