import streamlit as st
import os
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun,DuckDuckGoSearchResults
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
access_key=st.secrets['GROQ_API_KEY']


api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search=DuckDuckGoSearchResults(name="Search")

st.title("Langchian-Search with chat")

st.sidebar.title("Settings")
api_key=st.sidebar.text_input("enter your open source model key",key="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {
            "role":"assistant","content":"hi,i am a chat bot who search the web. how can i help you"
        },

    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)

    llm=ChatGroq(model='Llama3-8b-8192',streaming=True,api_key=access_key)

    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_errors=True)

    with st.chat_message("assistant"):

        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"user","content":response})
        st.write(response)

