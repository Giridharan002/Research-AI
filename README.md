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

# Use the Research Assistant in your project 😉

## Example `app.py`
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

This implementation includes:


1. A more modular `research_assistant.py` file that encapsulates the core functionality of the Research Assistant.
2. An updated `config.yaml` file for easy configuration.
3. An updated `requirements.txt` file with the latest package versions.

To use this Research Assistant:

1. Save each file in your project directory.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. create and update the `.env` with your actual API keys and configurations.
4. Run the application using `streamlit run app.py`.

This implementation provides a more structured and modular approach to the Research Assistant. You can further customize it by expanding the `ResearchAssistant` class, adding more sophisticated question routing, enhancing the web search capabilities, or implementing additional features like hallucination checking and answer grading.