import streamlit as st
import time  # Ensure this import is at the top of your file
import os, json
from typing import List, Tuple
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser

from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain_upstage import ChatUpstage as Chat

from util import limit_chat_history

class StepOutput(BaseModel):
    title: str = Field(description="Title of the reasoning step")
    content: str = Field(description="Content of the reasoning step")
    next_action: str = Field(
        description="Next action to take: 'continue' or 'final_answer'"
    )


step_parser = PydanticOutputParser(pydantic_object=StepOutput)


class FinalAnswer(BaseModel):
    title: str = Field(description="Final Answer")
    content: str = Field(description="Detaild explnation of the final answer")
    next_action: str = Field(description="done")


final_answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)

system_template = """You are an expert AI assistant that explains your reasoning step by step. 
For each step, provide a title that describes what you're doing in that step, along with the content. 
Decide if you need another step or if you're ready to give the final answer. 
---

USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST FIVE. DO NOT REPEAT SIMILAR STEPS. DO NOT CONCLUDE in STEPs.
BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. 
IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, 
AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. 
YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. 
DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST FIVE METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES.
BE RELEVANT TO THE GIVEN USER QUERY.
"""

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", system_template),
        ("human", "User query: {input}"),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            """next step please!
Please follow the format below:
{format_instructions}
""",
        ),
    ]
)

llm = Chat(model="solar-pro")


def generate_response(prompt: str) -> List[Tuple[str, str, float]]:
    chat_history: List[BaseMessage] = []
    chain = chat_prompt | llm | step_parser

    start_time = time.time()

    for step_count in range(1, 11):
        step_data = None
        with st.spinner(f"Generating step {step_count}..."):
            limited_chat_history = limit_chat_history(chat_history)
            for i in range(7):
                try:
                    step_data = chain.invoke(
                        {
                            "chat_history": limited_chat_history,
                            "input": prompt,
                            "format_instructions": step_parser.get_format_instructions(),
                        }
                    )
                    break
                except Exception as e:
                    st.warning(f"Attempt {i + 1} failed. Retrying...")

        if step_data is None:
            st.error("Failed to generate step after 3 attempts.")
            break

        with st.expander(step_data.title, expanded=True):
            st.markdown(step_data.content.replace("\n", "<br>"), unsafe_allow_html=True)
            chat_history.append(AIMessage(content=json.dumps(step_data.dict())))

        if step_data.next_action == "final_answer":
            break

    # Generate final answer

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                """
             Please provide the final answer based on your reasoning above.
             Think step by step and read user query and previous steps carefully.
             read again and provide best final answer.
             """,
            ),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                """
             User query: {input}
             
Please follow the format below and provide the final answer in the content:
{format_instructions}""",
            ),
        ]
    )

    total_thinking_time = time.time() - start_time
    st.info(f"Total thinking time: {total_thinking_time:.1f} seconds")

    final_chain = final_prompt | llm | final_answer_parser

    limited_chat_history = limit_chat_history(chat_history)
    final_response = None
    for i in range(7):
        try:
            final_response = final_chain.invoke(
                {
                    "chat_history": limited_chat_history,
                    "input": prompt,
                    "format_instructions": final_answer_parser.get_format_instructions(),
                }
            )
            break
        except Exception as e:
            st.warning(f"Attempt {i + 1} failed. Retrying...")

    if final_response is None:
        st.error("Failed to generate final answer after all attempts.")
        return

    st.markdown(f"### {final_response.title}")
    st.markdown(final_response.content.replace("\n", "<br>"), unsafe_allow_html=True)
    st.write()


# Streamlit UI
st.set_page_config(page_title="Solar o2", page_icon="🤔")
st.title("Solar Reasoning: o2")
st.write(
    """Inspired by STaR paper, openai o1, refection and https://github.com/bklieger-groq/g1. 
         Try Solar-Pro Preview at https://huggingface.co/upstage/solar-pro-preview-instruct"""
)

user_input = st.text_input(
    "Enter your question:", "Number 3.11 vs 3.9. Which one is bigger?"
)

if st.button("Generate Response"):
    if user_input:
        generate_response(user_input)
