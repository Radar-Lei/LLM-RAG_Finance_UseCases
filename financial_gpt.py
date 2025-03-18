import requests
import pandas as pd
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from IPython.display import display, Markdown
from langchain_deepseek import ChatDeepSeek
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def fetch_news(query, from_date, to_date, language='en', sort_by='relevancy', page_size=30, api_key='YOUR_API_KEY'):
    # Initialize the NewsAPI client
    newsapi = NewsApiClient(api_key=api_key)
    query = query.replace(' ','&')
    # Fetch all articles matching the query
    all_articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        to=to_date,
        language=language,
        sort_by=sort_by,
        page_size=page_size
    )

    # Extract articles
    articles = all_articles.get('articles', [])

    # Convert to DataFrame
    if articles:
        df = pd.DataFrame(articles)
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no articles are found


def preprocess_news_data(df):
    # Convert publishedAt to datetime
    df['publishedAt'] = pd.to_datetime(df['publishedAt'])
    df = df[~df['author'].isna()]
    
    # Keep the keyword column if it exists
    columns_to_keep = ['author', 'title', 'description', 'publishedAt']
    if 'keyword' in df.columns:
        columns_to_keep.append('keyword')
    
    df = df[columns_to_keep]
    return df

def build_prompt_for_summary(news_df, question):
    prompt = f"你是一位金融分析师，负责提供与金融行业相关的最新新闻文章的洞察。以下是一些关于{question}最新的新闻文章：\n\n"
    
    for index, row in news_df.iterrows():
        title = row['title']
        description = row['description']
        published_time = row['publishedAt']
        keyword = row.get('keyword', '')  # Get the keyword if it exists
        
        if 'keyword' in news_df.columns:
            prompt += f"**关键词:** {keyword}, **新闻标题:** {title}, **新闻内容:** {description}, **新闻发布时间:** {published_time}\n\n"
        else:
            prompt += f"**新闻标题:** {title}, **新闻内容:** {description}, **新闻发布时间:** {published_time}\n\n"
    
    prompt += f"请分析这些新闻，并提供关于这些新闻关于{question}可能对金融行业产生的影响，按照不同的关键词分别进行分析，注意标注新闻来源及时间"
    
    return prompt

if __name__ == "__main__":
    # Example usage with multiple keywords
    keywords = ["Alibaba(BABA)"]  # Example list of keywords, e.g., "Apple(AAPL)", "Tesla(TSLA)"
    
    # Check if preprocessed_news.csv exists
    csv_path = "preprocessed_news.csv"
    if os.path.exists(csv_path):
        print(f"Loading existing news data from {csv_path}")
        preprocessed_news_df = pd.read_csv(csv_path)
        # Convert publishedAt back to datetime if needed
        if 'publishedAt' in preprocessed_news_df.columns:
            preprocessed_news_df['publishedAt'] = pd.to_datetime(preprocessed_news_df['publishedAt'])
    else:
        print("No existing news data found. Fetching new data...")
        # Get the current time
        current_time = datetime.now()
        # Get the time 10 days ago
        time_10_days_ago = current_time - timedelta(days=29)
        api_key = '61792ba8adbd40b7bc86b5563eb41f87'
        
        # Create an empty list to store DataFrames for each keyword
        all_dfs = []
        
        # Fetch news for each keyword separately
        for keyword in keywords:
            print(f"Fetching news for: {keyword}")
            df = fetch_news(keyword, time_10_days_ago, current_time, api_key=api_key)
            
            if not df.empty:
                # Add a column to identify which keyword this data is for
                df['keyword'] = keyword
                all_dfs.append(df)
        
        # Merge all DataFrames into one
        if all_dfs:
            merged_df = pd.concat(all_dfs, ignore_index=True)
            df_news = merged_df.drop("source", axis=1)
            preprocessed_news_df = preprocess_news_data(df_news)
            # Save the preprocessed data
            preprocessed_news_df.to_csv(csv_path, index=False)
        else:
            print("No news found for any of the keywords.")
            preprocessed_news_df = pd.DataFrame()

    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = "sk-7c87ef2add054e439095db9b18c921e9"

    llm = ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )

    # 构建提示词 - using all keywords joined together for the prompt
    all_keywords = ", ".join(keywords)
    prompt_news_summary = build_prompt_for_summary(preprocessed_news_df, all_keywords)

    # news_summry_response = llm.invoke(prompt_news_summary)
    # print(news_summry_response.content)
    
    # Markdown(news_summry_response.content)

    ### 开始虚假金融财经信息检测
    loader_news = CSVLoader(csv_path)
    documents_news = loader_news.load()

    # Get your splitter ready
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=5)

    # Split your docs into texts
    texts_news = text_splitter.split_documents(documents_news)

    hg_embeddings = HuggingFaceEmbeddings()

    persist_directory = 'docs/chroma_rag/'
    
    # Check if the Chroma database already exists
    if os.path.exists(persist_directory):
        print("Loading existing vector database...")
        economic_langchain_chroma = Chroma(
            collection_name="economic_data",
            embedding_function=hg_embeddings,
            persist_directory=persist_directory
        )
    else:
        print("Creating new vector database...")
        economic_langchain_chroma = Chroma.from_documents(
            documents=texts_news,
            collection_name="economic_data",
            embedding=hg_embeddings,
            persist_directory=persist_directory
        )

    retriever_eco = economic_langchain_chroma.as_retriever(search_kwargs={"k": 10})
    user_query = "阿里巴巴集团控股有限公司已承诺投资超过3800亿元人民币（约530亿美元）用于人工智能基础设施，如数据中心等..."
    qs=f"{user_query}"
    template = """你是一名金融信息专家。
                请仅根据这些信息 {context} 并回答有关该公司的信息 {question}是否为真，并准确标注信息来源及时间"""

    PROMPT = PromptTemplate(input_variables=["context","question"], template=template)
    qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",chain_type_kwargs = {"prompt": PROMPT}, retriever=retriever_eco, return_source_documents=True)
    llm_response = qa_with_sources({"query": qs})
    print(llm_response['result'])