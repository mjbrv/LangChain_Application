import os
from apikey import apikey
import streamlit as st

from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

os.environ["OPENAI_API_KEY"] = apikey
st.title('Recommendations for Your Company')
product = st.text_input("Input the name of product or service")
language = st.text_input("Input Language")

llm = OpenAI(temperature=0.9)


template = "What is a good name for a company that makes {product}?"
first_prompt = PromptTemplate.from_template(template)
first_chain = LLMChain(llm=llm, prompt=first_prompt)

second_template = "Write a catch phrase for the following company: {company_name}?"
second_prompt = PromptTemplate.from_template(second_template)
second_chain = LLMChain(llm=llm, prompt=second_prompt)

third_template = "Write a vision for " \
                 "the following company: {company_name}?"
third_prompt = PromptTemplate.from_template(third_template)
third_chain = LLMChain(llm=llm, prompt=third_prompt)

# overall_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)


if product:
    response1 = first_chain.run(product)
    response2 = second_chain.run(response1)
    response3 = third_chain.run(response1)

    # catchphrase = overall_chain.run(product)
    st.write("Recommended name for company:")
    st.write(response1)
    st.write("Recommended catch phrase:")
    st.write(response2)
    st.write("Recommended vision:")
    st.write(response3)
    # st.write(catchphrase)

