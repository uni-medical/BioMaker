from bs4 import BeautifulSoup
from langchain.docstore.document import Document
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from tqdm import tqdm  # Import tqdm

def parse_monai_docs(root_dir="docs.monai.io/en/1.4.0"):
    docs = []
    # Wrap the outer loop with tqdm to show progress
    for root, _, files in tqdm(list(os.walk(root_dir)), desc="Parsing HTML files"):
        for file in files:
            # 人工设计的无用信息筛除
            if file.startswith("whatsnew"):
                continue

            if file.endswith(".html"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    
                    # 移除无用元素
                    for elem in soup(["nav", "footer", "script", "style"]):
                        elem.decompose()
                    
                    # 提取核心内容
                    main_content = soup.find(["div", "main"], {"role": "main"})  # Search both tags
                    # OR use a CSS selector
                    main_content = soup.select_one(':is(div, main)[role="main"]')
                    if not main_content:
                        continue
                        
                    # 提取元数据
                    metadata = {
                        "source": path,
                        "version": "latest",
                        "title": soup.title.string if soup.title else "",
                        "api_class": ", ".join([h2.get_text() for h2 in main_content.find_all("h2", class_="sig")])
                    }
                    
                    # 转换为Markdown格式
                    from html2text import html2text
                    content = html2text(str(main_content))
                    
                    docs.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
    return docs


def parse_and_index_docs(persist_dir="./monai_rag_db"):
    # 文档解析和预处理
    monai_docs = parse_monai_docs()
    
    # 文本分块处理
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        is_separator_regex=False
    )

    chunks = []
    # Wrap the loop with tqdm to show progress
    for doc in tqdm(monai_docs, desc="Splitting documents into chunks"):
        chunk_meta = doc.metadata.copy()
        
        if "api_class" in chunk_meta and chunk_meta["api_class"]:
            chunk_meta["doc_type"] = "API"
            chunks.append(Document(
                page_content=doc.page_content,
                metadata=chunk_meta
            ))
        else:
            splits = text_splitter.split_text(doc.page_content)
            for split in splits:
                chunk_meta["split_id"] = hash(split) % 1000
                chunks.append(Document(
                    page_content=split,
                    metadata=chunk_meta
                ))

    # 过滤空内容文档
    valid_chunks = [
        chunk for chunk in chunks 
        if chunk.page_content.strip() != ""
    ]
    print(f"过滤后有效块数: {len(valid_chunks)}/{len(chunks)}")

    # # 使用本地缓存模型
    # model_kwargs = {'device': 'cuda'}
    # encode_kwargs = {'normalize_embeddings': True}
    
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs=model_kwargs,
    #     encode_kwargs=encode_kwargs,
    #     cache_folder="./hf_models"  # 本地模型缓存路径
    # )

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # 验证嵌入模型
    test_embedding = embeddings.embed_query("test embedding")
    print(f"嵌入维度: {len(test_embedding)}")

    vector_db = Chroma.from_documents(
        documents=valid_chunks,  # 使用过滤后的有效块
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    vector_db.persist()
    print(f"已创建 {vector_db._collection.count()} 个向量")

if __name__ == "__main__":
    # documents = parse_monai_docs("./test")
    # print("\n\n-------\n".join([d.page_content for d in documents]))
    # exit()
    # 当需要重建索引时执行
    parse_and_index_docs()
