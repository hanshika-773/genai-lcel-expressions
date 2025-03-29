#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import openai

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


# In[10]:


from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser


# In[29]:


prompt = ChatPromptTemplate.from_template(
    "tell me about {topic}"
)
model = ChatOpenAI()
output_parser = StrOutputParser()


# In[30]:


chain = prompt | model | output_parser


# In[31]:


chain.invoke({"topic": "dog"})


# In[ ]:





# In[45]:


from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch   


# In[46]:


vectorstore = DocArrayInMemorySearch.from_texts(
    ["GGenerative AI is a type of artificial intelligence that creates new content, such as text, images, or code, based on learned patterns.","(LCEL)LangChain Expression Language (LCEL) is a declarative syntax for composing and chaining LangChain components efficiently."],
    embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()


# In[47]:


retriever.get_relevant_documents("what is generative ai?")


# In[48]:


retriever.get_relevant_documents("what is the full form of LCEL")


# In[49]:


template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# In[50]:


from langchain.schema.runnable import RunnableMap


# In[51]:


chain = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
}) | prompt | model | output_parser


# In[55]:


chain.invoke({"question": "what is the full form of LCEL?"})


# In[58]:


inputs = RunnableMap({
    "context": lambda x: retriever.get_relevant_documents(x["question"]),
    "question": lambda x: x["question"]
})


# In[59]:


inputs.invoke({"question": "what is the full form of LCEL?"})

