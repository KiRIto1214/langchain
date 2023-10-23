import os
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

from key import openaikey
import streamlit as st

os.environ["OPENAI_API_KEY"] = openaikey

st.title('Cool Science Search')

input_text = st.text_input("topic you want to know")

#openai llm
first_input_prompt=PromptTemplate(
    input_variables=['topic'],
    template="tell me some interesting things about {topic}"
)

topic_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
history_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
application_memory = ConversationBufferMemory(input_key='application', memory_key='description_history')



llm = OpenAI(temperature=0.8)
chain1=LLMChain(
    llm=llm,prompt=first_input_prompt,verbose=True,output_key='history',memory=topic_memory)


second_input_prompt=PromptTemplate(
    input_variables=['topic'],
    template="when was the significance and history of {topic} and impact on world "
)

chain2=LLMChain(
    llm=llm,prompt=second_input_prompt,verbose=True,output_key='application',memory=history_memory)
# Promp

third_input_prompt=PromptTemplate(
    input_variables=['application'],
    template="What is the {application} in the current world "
)
chain3=LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key='history_application',memory=application_memory)

parent_chain=SequentialChain(
    chains=[chain1,chain2,chain3],input_variables=['topic'],output_variables=['topic','history','history_application'],verbose=True)

if input_text:
    st.write(parent_chain({'topic':input_text}))

    with st.expander('Brief History'): 
        st.info(history_memory.buffer)

    with st.expander('Application'): 
        st.info(application_memory.buffer)