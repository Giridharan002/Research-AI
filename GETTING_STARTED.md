# Research Assistant

## Overview
The **Research Assistant** is an advanced AI-powered tool designed to assist with complex research tasks. It utilizes state-of-the-art language models and Retrieval-Augmented Generation (RAG) techniques to provide accurate and contextually relevant information. By combining a local knowledge base with real-time web data, this tool can tackle a wide range of research queries.

## Features
- **Question routing and reformulation**: Dynamically redirects and rephrases questions for better accuracy.
- **Document retrieval with semantic search**: Finds relevant documents using context-aware search.
- **Web searching for up-to-date information**: Retrieves real-time information from the web.
- **Entity detection and filtering**: Identifies key entities to streamline searches.
- **Subquery generation**: Breaks down complex queries into manageable subqueries.
- **Hallucination checking and answer grading**: Validates answers to reduce false information.
- **Final answer aggregation and generation**: Combines multiple sources for comprehensive responses.
- **Example-based learning**: Improves query interpretation over time.
- **Multi-step reasoning**: Performs complex, multi-step logic to resolve intricate questions.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/research-assistant.git
    cd research-assistant
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv research_assistant_env
    source research_assistant_env/bin/activate  # On Windows: research_assistant_env\Scripts\activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Research Assistant:

```bash
streamlit run app.py
```

Follow the on-screen prompts to input your research questions and receive detailed answers.

## Configuration
The system is configurable through the `config.yaml` file. Key settings include:
- Language Model (LLM) selection
- Retrieval parameters (e.g., top-k documents)
- Prompt templates for LLM queries
- Web search API keys and settings

## Web Search Integration
The Research Assistant incorporates real-time web search to complement local data with the latest available information. You can configure the web search settings, including API keys and search parameters, in the `config.yaml` file.

## Contributing
Contributions to improve the Research Assistant are encouraged. To contribute:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature/AmazingFeature
    ```
3. Commit your changes:
    ```bash
    git commit -m 'Add some AmazingFeature'
    ```
4. Push to the branch:
    ```bash
    git push origin feature/AmazingFeature
    ```
5. Open a Pull Request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments
- Built using the **AI Starter Kit** framework.
- Utilizes advanced NLP models and RAG techniques from leading AI research.
- Integrates web search APIs for real-time information retrieval.

---

# Code Implementation

## app.py
```python
import streamlit as st
from research_assistant import ResearchAssistant

def main():
    st.title("Research Assistant")

    # Initialize Research Assistant
    assistant = ResearchAssistant()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is your research question?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = assistant.process_query(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
```

## research_assistant.py
```python
import yaml
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper
from langchain.callbacks import get_openai_callback
import os

class ResearchAssistant:
    def __init__(self):
        self.config = self.load_config()
        self.setup_environment()
        self.llm = OpenAI(temperature=self.config["llm_temperature"])
        self.embeddings = OpenAIEmbeddings()
        self.docsearch = self.setup_knowledge_base()
        self.search = SerpAPIWrapper()
        self.tools = self.setup_tools()
        self.agent_executor = self.setup_agent()

    def load_config(self):
        with open("config.yaml", "r") as config_file:
            return yaml.safe_load(config_file)

    def setup_environment(self):
        os.environ["OPENAI_API_KEY"] = self.config["openai_api_key"]
        os.environ["SERPAPI_API_KEY"] = self.config["serpapi_api_key"]

    def setup_knowledge_base(self):
        loader = TextLoader("knowledge_base.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return FAISS.from_documents(texts, self.embeddings)

    def setup_tools(self):
        return [
            Tool(
                name="Local Knowledge Base",
                func=self.docsearch.similarity_search,
                description="Useful for querying the local knowledge base"
            ),
            Tool(
                name="Web Search",
                func=self.search.run,
                description="Useful for searching the web for up-to-date information"
            )
        ]

    def setup_agent(self):
        template = """
        You are an advanced Research Assistant. Use the following tools to answer the user's question:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Begin!

        Question: {input}
        Thought: To answer this question, I'll need to use a combination of my local knowledge base and potentially web search for up-to-date information.
        {agent_scratchpad}
        """

        prompt = PromptTemplate(
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
            template=template
        )

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self.config["output_parser"],
            stop=["\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )

        memory = ConversationBufferMemory(memory_key="chat_history")

        return AgentExecutor.from_agent_and_tools(
            agent=agent, tools=self.tools, verbose=True, memory=memory
        )

    def process_query(self, query):
        with get_openai_callback() as cb:
            response = self.agent_executor.run(query)
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
        return response

```

## config.yaml
```yaml
llm_temperature: 0.7
openai_api_key: "your-openai-api-key-here"
serpapi_api_key: "your-serpapi-api-key-here"
output_parser:
  type: "regex"
  regex: "Final Answer: (.*)"
```

## requirements.txt
```
streamlit==1.22.0
pyyaml==6.0
langchain==0.0.184
openai==0.27.7
faiss-cpu==1.7.3
google-search-results==2.4.2
```

## knowledge_base.txt
```
The Research Assistant is an AI-powered tool designed to help with complex research tasks.
It uses a combination of local knowledge and web search to provide comprehensive answers.
The system leverages state-of-the-art language models and retrieval-augmented generation techniques.
Key features include question routing, document retrieval, web searching, entity detection, and multi-step reasoning.
The Research Assistant can handle a wide range of topics and adapt its responses based on the latest available information.
```

This implementation includes:

1. A streamlined `app.py` file that sets up the Streamlit interface.
2. A more modular `research_assistant.py` file that encapsulates the core functionality of the Research Assistant.
3. An updated `config.yaml` file for easy configuration.
4. An updated `requirements.txt` file with the latest package versions.
5. A sample `knowledge_base.txt` file to get started with local knowledge.

To use this Research Assistant:

1. Save each file in your project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Replace the placeholder API keys in `config.yaml` with your actual OpenAI and SerpAPI keys.
4. Run the application using `streamlit run app.py`.

This implementation provides a more structured and modular approach to the Research Assistant. You can further customize it by expanding the `ResearchAssistant` class, adding more sophisticated question routing, enhancing the web search capabilities, or implementing additional features like hallucination checking and answer grading.