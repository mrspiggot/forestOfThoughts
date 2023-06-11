from dotenv import load_dotenv
import os
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
import re
from langchain.docstore.wikipedia import Wikipedia
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer
import streamlit as st
from PIL import Image


class Strategy:
    def __init__(self, temperature=0.7, max_tokens=1500):
        self.llm = OpenAI(temperature=temperature, max_tokens=max_tokens)

    def run(self, question):
        raise NotImplementedError()


class React(Strategy):
    def __init__(self):
        super().__init__(temperature=0.7, max_tokens=1500)

    def run(self, question):
        print('Using ReAct')
        st.write("Using 'ReAct' - (Reasoning and Action)")

        docstore = DocstoreExplorer(Wikipedia())
        tools = [
            Tool(
                name="Search",
                func=docstore.search,
                description="Search for a term in the docstore.",
            ),
            Tool(
                name="Lookup",
                func=docstore.lookup,
                description="Lookup a term in the docstore.",
            )
        ]
        react = initialize_agent(tools, self.llm, agent="react-docstore", verbose=True)
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent=react.agent,
            tools=tools,
            verbose=True,
        )
        response_react = agent_executor.run(question)
        print(response_react)
        st.write(response_react)


class Tot(Strategy):
    def __init__(self):
        super().__init__(temperature=0.7, max_tokens=1500)

    def run(self, question):
        print('Using ToT')
        st.write("Using 'Tree of Thoughts'")

        template_tot = """Imagine three different experts are answering this question.
    They will brainstorm the answer step by step reasoning carefully and taking all facts into consideration
    All experts will write down 1 step of their thinking,
    then share it with the group.
    They will each critique their response, and the all the responses of others
    They will check their answer based on science and the laws of physics
    Then all experts will go on to the next step and write down this step of their thinking.
    They will keep going through steps until they reach their conclusion taking into account the thoughts of the other experts
    If at any time they realise that there is a flaw in their logic they will backtrack to where that flaw occurred 
    If any expert realises they're wrong at any point then they acknowledges this and start another train of thought
    Each expert will assign a likelihood of their current assertion being correct
    Continue until the experts agree on the single most likely answer and write out that answer along with any commentary to support that answer
    The question is {question}

    The experts reasoning, along with their final answer is...
    """
        prompt = PromptTemplate(template=template_tot, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response_tot = llm_chain.run(question)
        print(response_tot)
        st.write(response_tot)


class Cot(Strategy):
    def __init__(self):
        super().__init__(temperature=0.7, max_tokens=1500)

    def run(self, question):
        print('Using CoT')
        st.write("Using 'Chain of Thought'")

        template_cot = """You are asked a question and rather than simply guessing the right answer break down the solution into a series of staps
    The question is {question}

    Write out your step by step reasoning and after considering all of the facts and applying this reasoning write out your final answer
    """
        prompt = PromptTemplate(template=template_cot, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=self.llm)
        response_cot = llm_chain.run(question)
        print(response_cot)
        st.write(response_cot)


class StrategyDeterminer:
    def __init__(self, api_key, question):
        self.api_key = api_key
        self.question = question

        self.strategies = {
            1: React(),
            2: Tot(),
            3: Cot()
        }

        self.template = """Consider the following problem or puzzle: {question}. Based on the characteristics of the problem, identify the most suitable approach among the three techniques described below. consider each carefully in the context of the question, write out the likelihood of success of each, and then select the most appropriate approach:
1. The solution for this problem requires searching for further information online, generating reasoning traces and task-specific actions in an interleaved manner. Starting with incomplete information this technique will prompt for the need to get additional helpful information at each step. It allows for dynamic reasoning to create, maintain, and adjust high-level plans for acting, while also interacting with external sources to incorporate additional information into reasoning [1].
2. This problem is complex and the solution requires exploring multiple reasoning paths over thoughts. It treats the problem as a search over a tree structure, with each node representing a partial solution and the branches corresponding to operators that modify the solution. It involves thought decomposition, thought generation, state evaluation, and a search algorithm [2].
3. This problem is simple and the solution may be obtained by focusing on generating a coherent series of reasoning steps that lead to the final answer. The approach provides interpretability, decomposes multi-step problems into intermediate steps, and allows for additional computation allocation [3].
Based on the characteristics of the given problem or puzzle, select the technique that aligns most closely with the nature of the problem. It is important to first provide the number of the technique that best solves the problem, followed by a period. Then you may provide your reason why you have chosen this technique. 

The number of the selected technique is...
"""

    @staticmethod
    def find_first_integer(text):
        match = re.search(r'\d+', text)
        if match:
            return int(match.group())
        else:
            return None

    def determine_and_execute(self):
        llm_1 = OpenAI(temperature=0.6, max_tokens=3000)
        prompt = PromptTemplate(template=self.template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm_1)

        response_1 = llm_chain.run(self.question)
        print(response_1)
        st.write(response_1)
        n = self.find_first_integer(response_1)

        if n in self.strategies:
            self.strategies[n].run(self.question)
        else:
            print(f"Strategy number {n} is not recognized.")


def run_app():
    # load environment variables
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # create a text input in Streamlit
    icon = Image.open("Colour Logo.png")
    st.image(icon, width=100)
    st.title("Lucidate Prompt Strategy Demo. Powered by LangChain 🦜🔗 ")
    question = st.text_area('Enter your question here:', height=200)

    # run strategy determiner
    if question:
        determiner = StrategyDeterminer(openai_api_key, question)
        determiner.determine_and_execute()

if __name__ == "__main__":
    run_app()
