import json
import logging
import os
import re
import sys
from urllib.parse import urlparse

import requests
import streamlit as st
import weave
import yaml
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import load_prompt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader, UnstructuredURLLoader
from langchain_community.document_transformers import Html2TextTransformer

current_dir = os.path.dirname(os.path.abspath(__file__))
kit_dir = os.path.abspath(os.path.join(current_dir, '..'))
repo_dir = os.path.abspath(os.path.join(kit_dir, '..'))
sys.path.append(kit_dir)
sys.path.append(repo_dir)

from serpapi import GoogleSearch

from utils.model_wrappers.api_gateway import APIGateway
from utils.vectordb.vector_db import VectorDb
from utils.visual.env_utils import get_wandb_key

CONFIG_PATH = os.path.join(kit_dir, 'config.yaml')
PERSIST_DIRECTORY = os.path.join(kit_dir, 'data/my-vector-db')

load_dotenv(os.path.join(repo_dir, '.env'))

# Handle the WANDB_API_KEY resolution before importing weave
wandb_api_key = get_wandb_key()

# If WANDB_API_KEY is set, proceed with weave initialization
if wandb_api_key:
    import weave

    # Initialize Weave with your project name
    weave.init('sambanova_search_assistant')
else:
    print('WANDB_API_KEY is not set. Weave initialization skipped.')


class SearchAssistant:
    def __init__(self, config=None, vectorstore=None) -> None:
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if config is None:
            self.config = {}
        else:
            self.config = config
        config_info = self._get_config_info(CONFIG_PATH)
        self.api_info = config_info[0]
        self.embedding_model_info = config_info[1]
        self.llm_info = config_info[2]
        self.retrieval_info = config_info[3]
        self.web_crawling_params = config_info[4]
        self.extra_loaders = config_info[5]
        self.prod_mode = config_info[6]
        self.documents = None
        self.urls = None
        self.llm = self.init_llm_model()
        self.vectordb = VectorDb()
        self.vectorstore = vectorstore
        self.qa_chain = None
        self.memory = None


    def _get_config_info(self, config_path):
       
        with open(config_path, 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)
        api_info = config['api']
        embedding_model_info = config['embedding_model']
        llm_info = config['llm']
        retrieval_info = config['retrieval']
        web_crawling_params = config['web_crawling']
        extra_loaders = config['extra_loaders']
        prod_mode = config['prod_mode']

        return api_info, embedding_model_info, llm_info, retrieval_info, web_crawling_params, extra_loaders, prod_mode

    def init_memory(self):
        summary_prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-summary.yaml'))

        self.memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=100,
            buffer='The human and AI greet each other to start a conversation.',
            memory_key='chat_history',
            return_messages=True,
            output_key='answer',
            prompt=summary_prompt,
        )

    def init_llm_model(self) -> None:
       
        if self.prod_mode:
            sambanova_api_key = st.session_state.SAMBANOVA_API_KEY
        else:
            if 'SAMBANOVA_API_KEY' in st.session_state:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY') or st.session_state.SAMBANOVA_API_KEY
            else:
                sambanova_api_key = os.environ.get('SAMBANOVA_API_KEY')

        llm = APIGateway.load_llm(
            type=self.api_info,
            streaming=True,
            coe=self.llm_info['coe'],
            do_sample=self.llm_info['do_sample'],
            max_tokens_to_generate=self.llm_info['max_tokens_to_generate'],
            temperature=self.llm_info['temperature'],
            select_expert=self.llm_info['select_expert'],
            process_prompt=False,
            sambanova_api_key=sambanova_api_key,
        )
        return llm

    def reformulate_query_with_history(self, query):
    
        if self.memory is None:
            self.init_memory()
        custom_condensed_question_prompt = load_prompt(
            os.path.join(kit_dir, 'prompts', 'llama3-multiturn-custom_condensed_question.yaml')
        )
        history = self.memory.load_memory_variables({})
        reformulated_query = self.llm.invoke(
            custom_condensed_question_prompt.format(chat_history=history, question=query)
        )
        return reformulated_query

    def remove_links(self, text):
        url_pattern = r'https?://\S+|www\.\S+'
        return re.sub(url_pattern, '', text)

    def parse_serp_analysis_output(self, answer, links):
        for i, link in enumerate(links):
            answer = answer.replace(f'[reference:{i+1}]', f'[{i+1}]({link})')
            answer = answer.replace(f'[reference: {i+1}]', f'[{i+1}]({link})')
            answer = answer.replace(f'[Reference:{i+1}]', f'[{i+1}]({link})')
            answer = answer.replace(f'[Reference: {i+1}]', f'[{i+1}]({link})')
        return answer


    def querySerper(
        self,
        query: str,
        limit: int = 5,
        do_analysis: bool = True,
        include_site_links: bool = False,
        conversational: bool = False,
    ):
        url = 'https://google.serper.dev/search'
        payload = json.dumps({'q': query, 'num': limit})
        headers = {'X-API-KEY': os.environ.get('SERPER_API_KEY'), 'Content-Type': 'application/json'}

        try:
            response = requests.post(url, headers=headers, data=payload)
            if response.status_code == 200:
                results = response.json().get('organic', [])
                if len(results) > 0:
                    links = [r['link'] for r in results]
                    context = []
                    for i, result in enumerate(results):
                        context.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("snippet", "")}')
                    context = '\n\n'.join(context)
                    self.logger.info(f'Context found: {context}')
                    if include_site_links:
                        sitelinks = []
                        for r in [r.get('sitelinks', []) for r in results]:
                            sitelinks.extend([site.get('link', None) for site in r])
                        links.extend(sitelinks)
                    links = list(filter(lambda x: x is not None, links))
                else:
                    context = 'Answer not found'
                    links = []
                    self.logger.info(f'No answer found for query: {query}')
            else:
                context = 'Answer not found'
                links = []
                self.logger.error(f'Request failed with status code: {response.status_code}')
                self.logger.error(f'Error message: {response.text}')
        except Exception as e:
            context = 'Answer not found'
            links = []
            self.logger.error(f'Error message: {e}')

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return context, links

    def queryOpenSerp(
        self,
        query: str,
        limit: int = 5,
        do_analysis: bool = True,
        engine='google',
        conversational: bool = False,
    ) -> str:
        if engine not in ['google', 'yandex', 'baidu']:
            raise ValueError('engine must be either google, yandex or baidu')
        url = f'http://127.0.0.1:7000/{engine}/search'
        params = {'lang': 'EN', 'limit': limit, 'text': query}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                results = response.json()
                if len(results) > 0:
                    links = [r['url'] for r in results]
                    context = []
                    for i, result in enumerate(results):
                        context.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("description", "")}')
                    context = '\n\n'.join(context)
                    self.logger.info(f'Context found: {context}')
                else:
                    context = 'Answer not found'
                    links = []
                    self.logger.info(f'No answer found for query: {query}')
            else:
                context = 'Answer not found'
                links = []
                self.logger.error(f'Request failed with status code: {response.status_code}')
                self.logger.error(f'Error message: {response.text}')
        except Exception as e:
            context = 'Answer not found'
            links = []
            self.logger.error(f'Error message: {e}')

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return context, links

    def querySerpapi(
        self,
        query: str,
        limit: int = 1,
        do_analysis: bool = True,
        engine='google',
    ) -> str:
        if engine not in ['google', 'bing']:
            raise ValueError('engine must be either google or bing')
        params = {
            'q': query,
            'num': limit,
            'engine': engine,
            'api_key': st.session_state.SERPAPI_API_KEY if self.prod_mode else os.environ.get('SERPAPI_API_KEY'),
        }   

        try:
            search = GoogleSearch(params)
            response = search.get_dict()

            knowledge_graph = response.get('knowledge_graph', None)
            results = response.get('organic_results', [])

            links = []
            if len(results) > 0:
                links = [r['link'] for r in results]
                context = []
                for i, result in enumerate(results):
                    context.append(f'[reference:{i+1}] {result.get("title", "")}: {result.get("snippet", "")}')
                context = '\n\n'.join(context)
                self.logger.info(f'Context found: {context}')
            else:
                context = 'Answer not found'
                links = []
                self.logger.info(f'No answer found for query: {query}. Raw response: {response}')
        except Exception as e:
            context = 'Answer not found'
            links = []
            self.logger.error(f'Error message: {e}')

        if do_analysis:
            prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-serp_analysis.yaml'))
            formatted_prompt = prompt.format(question=query, context=context)
            answer = self.llm.invoke(formatted_prompt)
            return self.parse_serp_analysis_output(answer, links), links
        else:
            return context, links

    def load_remote_pdf(self, url):
        loader = UnstructuredURLLoader(urls=[url])
        docs = loader.load()
        return docs

    def load_htmls(self, urls, extra_loaders=None):
        if extra_loaders is None:
            extra_loaders = []
        docs = []
        for url in urls:
            if url.endswith('.pdf'):
                if 'pdf' in extra_loaders:
                    docs.extend(self.load_remote_pdf(url))
                else:
                    continue
            else:
                loader = AsyncHtmlLoader(url, verify_ssl=False)
                docs.extend(loader.load())
        return docs

    def link_filter(self, all_links, excluded_links):
        clean_excluded_links = set()
        for excluded_link in excluded_links:
            parsed_link = urlparse(excluded_link)
            clean_excluded_links.add(parsed_link.netloc + parsed_link.path)
        filtered_links = set()
        for link in all_links:
            # Check if the link contains any of the excluded links
            if not any(excluded_link in link for excluded_link in clean_excluded_links):
                filtered_links.add(link)
        return filtered_links

    def clean_docs(self, docs):
        html2text_transformer = Html2TextTransformer()
        docs = html2text_transformer.transform_documents(documents=docs)
        return docs

    def web_crawl(self, urls, excluded_links=None):
        if excluded_links is None:
            excluded_links = []
        excluded_links.extend(self.web_crawling_params['excluded_links'])
        excluded_link_suffixes = {'.ico', '.svg', '.jpg', '.png', '.jpeg', '.', '.docx', '.xls', '.xlsx'}
        scrapped_urls = []

        urls = [url for url in urls if not url.endswith(tuple(excluded_link_suffixes))]
        urls = self.link_filter(urls, set(excluded_links))
        print(f'{urls=}')
        if len(urls) == 0:
            raise ValueError(
                'not sites to scrape after filtering links, check the excluded_links config or increase Max number of results to retrieve'
            )
        urls = list(urls)[: self.web_crawling_params['max_scraped_websites']]

        scraped_docs = self.load_htmls(urls, self.extra_loaders)
        scrapped_urls.extend(urls)

        docs = self.clean_docs(scraped_docs)
        self.documents = docs
        self.urls = scrapped_urls

    def get_text_chunks_with_references(self, docs: list, chunk_size: int, chunk_overlap: int) -> list:

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )
        sources = {site: i + 1 for i, site in enumerate(self.urls)}
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            reference = chunk.metadata['source']  # get the number in the dict
            chunk.page_content = f'[reference:{sources[reference]}] {chunk.page_content}\n\n'

        return chunks

    def create_load_vector_store(self, force_reload: bool = False, update: bool = False):

        persist_directory = self.config.get('persist_directory', 'NoneDirectory')

        embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info['type'],
            batch_size=self.embedding_model_info['batch_size'],
            coe=self.embedding_model_info['coe'],
            select_expert=self.embedding_model_info['select_expert'],
        )

        if os.path.exists(persist_directory) and not force_reload and not update:
            self.vector_store = self.vectordb.load_vdb(
                persist_directory, embeddings, db_type=self.retrieval_info['db_type']
            )

        elif os.path.exists(persist_directory) and update:
            chunks = self.get_text_chunks_with_references(
                self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
            )
            self.vector_store = self.vectordb.load_vdb(
                persist_directory, embeddings, db_type=self.retrieval_info['db_type']
            )
            self.vector_store = self.vectordb.update_vdb(
                chunks, embeddings, self.retrieval_info['db_type'], persist_directory
            )

        else:
            chunks = self.get_text_chunks_with_references(
                self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
            )
            self.vector_store = self.vectordb.create_vector_store(
                chunks, embeddings, self.retrieval_info['db_type'], None
            )

    def create_and_save_local(self, input_directory=None, persist_directory=None, update=False):
        persist_directory = persist_directory or self.config.get('persist_directory', 'NoneDirectory')

        chunks = self.get_text_chunks_with_references(
            self.documents, self.retrieval_info['chunk_size'], self.retrieval_info['chunk_overlap']
        )
        embeddings = APIGateway.load_embedding_model(
            type=self.embedding_model_info['type'],
            batch_size=self.embedding_model_info['batch_size'],
            coe=self.embedding_model_info['coe'],
            select_expert=self.embedding_model_info['select_expert'],
        )
        if update and os.path.exists(persist_directory):
            self.config['update'] = True
            self.vector_store = self.vectordb.update_vdb(
                chunks, embeddings, self.retrieval_info['db_type'], input_directory, persist_directory
            )

        else:
            if os.path.exists(persist_directory):
                self.vector_store = self.vectordb.create_vector_store(
                    chunks, embeddings, self.retrieval_info['db_type'], persist_directory
                )
            else:
                self.vector_store = self.vectordb.create_vector_store(
                    chunks, embeddings, self.retrieval_info['db_type'], None
                )

    def basic_call(self, query, reformulated_query=None, search_method='serpapi', max_results=5, search_engine='google', conversational=False):
        if reformulated_query is None:
            reformulated_query = query

        local_context = ""
        if self.vectorstore:
            local_results = self.vectorstore.similarity_search(query, k=2)
            if local_results:
                local_context = "\n".join([doc.page_content for doc in local_results])

        if search_method == 'serpapi':
            answer, links = self.querySerpapi(query=reformulated_query, limit=max_results, engine=search_engine, do_analysis=True)
        elif search_method == 'serper':
            answer, links = self.querySerper(query=reformulated_query, limit=max_results, do_analysis=True)
        elif search_method == 'openserp':
            answer, links = self.queryOpenSerp(query=reformulated_query, limit=max_results, engine=search_engine, do_analysis=True)

        # Integrate local context into the answer
        if local_context:
            answer = f"{answer}\n\nAdditional information from local documents:\n{local_context}"

        if self.vectorstore:
            local_sources = [doc.metadata.get('source', 'Local PDF') for doc in local_results]
            links = local_sources + links

        if conversational:
            self.memory.save_context(inputs={'input': query}, outputs={'answer': answer})

        return {'answer': answer, 'sources': links}




    def set_retrieval_qa_chain(self, conversational=False):
        retrieval_qa_prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-web_scraped_data_retriever.yaml'))
        retriever = self.vector_store.as_retriever(
            search_type='similarity_score_threshold',
            search_kwargs={
                'score_threshold': self.retrieval_info['score_treshold'],
                'k': self.retrieval_info['k_retrieved_documents'],
            },
        )
        if conversational:
            self.init_memory()

            custom_condensed_question_prompt = load_prompt(
                os.path.join(kit_dir, 'prompts', 'llama3-multiturn-custom_condensed_question.yaml')
            )

            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=retriever,
                memory=self.memory,
                chain_type='stuff',
                return_source_documents=True,
                verbose=False,
                condense_question_prompt=custom_condensed_question_prompt,
                combine_docs_chain_kwargs={'prompt': retrieval_qa_prompt},
            )

        else:
            self.qa_chain = RetrievalQA.from_llm(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                verbose=False,
                input_key='question',
                output_key='answer',
                prompt=retrieval_qa_prompt,
            )

    def search_and_scrape(self, query, search_method='serpapi', max_results=5, search_engine='google'):

        if search_method == 'serpapi':
            _, links = self.querySerpapi(query=query, limit=max_results, engine=search_engine, do_analysis=False)
        elif search_method == 'serper':
            _, links = self.querySerper(query=query, limit=max_results, do_analysis=False)
        elif search_method == 'openserp':
            _, links = self.queryOpenSerp(query=query, limit=max_results, engine=search_engine, do_analysis=False)
        if len(links) > 0:
            self.web_crawl(urls=links)
            # self.create_load_vector_store()
            self.create_and_save_local()
            self.set_retrieval_qa_chain(conversational=True)
        else:
            return {'message': f"No links found for '{query}'. Try again"}

    def get_relevant_queries(self, query):
        prompt = load_prompt(os.path.join(kit_dir, 'prompts/llama3-related_questions.yaml'))
        response_schemas = [ResponseSchema(name='related_queries', description=f'related search queries', type='list')]
        list_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        list_format_instructions = list_output_parser.get_format_instructions()
        relevant_queries_chain = prompt | self.llm | list_output_parser
        input_variables = {'question': query, 'format_instructions': list_format_instructions}
        return relevant_queries_chain.invoke(input_variables).get('related_queries', [])

    def parse_retrieval_output(self, result):

        parsed_answer = self.parse_serp_analysis_output(result['answer'], self.urls)
        # mapping original sources order with question used sources order
        question_sources = set(f'{doc.metadata["source"]}' for doc in result['source_documents'])
        question_sources_map = {source: i + 1 for i, source in enumerate(question_sources)}
        for i, link in enumerate(self.urls):
            if link in parsed_answer:
                parsed_answer = parsed_answer.replace(
                    f'[<sup>{i+1}</sup>]({link})', f'[<sup>{question_sources_map[link]}</sup>]({link})'
                )
        return parsed_answer

    def retrieval_call(self, query):
        result = self.qa_chain.invoke(query)
        result['answer'] = self.parse_retrieval_output(result)
        return result
