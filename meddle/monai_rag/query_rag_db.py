import os.path as osp

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from meddle.backend import query

PERSIST_DIR = osp.join(osp.dirname(__file__), "monai_rag_db")

def query2doc_search(search_query, model="gpt-4o"):
    prompt = "Refer to this example:\n" + \
             "query: what's the usage of ImageDataset in MONAI\n" + \
             "passage: ### ImageDataset[#](data.html#imagedataset 'Link to this heading') _class _monai.data.ImageDataset(_image_files_ , _seg_files=None_ , _labels=None_ , _transform=None_ , _seg_transform=None_ , _label_transform=None_ , _image_only=True_ , _transform_with_metadata=False_ , _dtype= <class 'numpy.float32'>_, _reader=None_ , _*args_ , _**kwargs_)[[source]](https://github.com/Project- MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L26-L155)[#](data.html#monai.data.ImageDataset 'Link to this definition')" +"\n"+\
             "query: ImageDataset\n" + \
             "passage: ### ImageDataset[#](data.html#imagedataset 'Link to this heading') _class _monai.data.ImageDataset(_image_files_ , _seg_files=None_ , _labels=None_ , _transform=None_ , _seg_transform=None_ , _label_transform=None_ , _image_only=True_ , _transform_with_metadata=False_ , _dtype= <class 'numpy.float32'>_, _reader=None_ , _*args_ , _**kwargs_)[[source]](https://github.com/Project- MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L26-L155)[#](data.html#monai.data.ImageDataset 'Link to this definition')"
             
    pseudo_doc = query(
        system_message=prompt,
        user_message=f"Write a passage that answers the given query: {search_query}\n\n",
        model=model,
        temperature=0.0,
    )
    return pseudo_doc

class MED_KNOWLEDGE_RAG:
    def __init__(self, model="gpt-4o", persist_dir=PERSIST_DIR):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.model = model
    
    def _rerank_documents(self, original_query: str, documents: list[Document], k: int) -> list[Document]:
        """
        Reranks a list of documents based on their relevance to the original query
        using a language model.
        """
        if not documents:
            return []

        reranked_docs_with_scores = []
        for doc in documents:
            # Create a prompt for the LLM to score the relevance
            try:
                score_str = query(
                    system_message="You are an expert at evaluating the relevance of text."+\
                                   "Assign a relevance score between 0 and 10 to the provided document based on the given query."+\
                                   "Output only the score.",
                    user_message=f"Query: {original_query}\nDocument: {doc.page_content}\nRelevance Score:",
                    model=self.model,
                    temperature=0.0,
                )
                score = float(score_str.strip())
            except Exception as e:
                score = 0.0
            
            reranked_docs_with_scores.append((score, doc))
        
        # Sort documents by score in descending order and take the top k
        reranked_docs_with_scores.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in reranked_docs_with_scores[:k]]

    def search(self, query, k=3, enable_query2doc=False, convert_to_str=True):
        documents = self.db.similarity_search(query, k=k)
        
        if enable_query2doc:
            q2d_query = query2doc_search(query, model=self.model)
            q2d_documents = self.db.similarity_search(q2d_query, k=k)
            
            # Merge documents
            all_documents = []
            seen_page_contents = set()
            for doc in documents + q2d_documents:
                if doc.page_content not in seen_page_contents:
                    all_documents.append(doc)
                    seen_page_contents.add(doc.page_content)
            
            # Rerank to get top k
            final_documents = self._rerank_documents(query, all_documents, k)
        else:
            final_documents = documents
        if (convert_to_str):
            return "\n".join([d.page_content[:].replace("AddChanneld", "") for d in final_documents])+"AddChanneld is deprecated, please use EnsureChannelFirstd instead."
        return final_documents

if __name__ == "__main__":
    # 初始化RAG系统
    rag = MED_KNOWLEDGE_RAG()
    
    # 示例查询
    test_queries = [
        # ("MONAI中LoadImage的使用方法", 3),
        # ("MONAI中Transforms有哪些", 3),
        # ("MONAI中CacheDataset的使用方法", 3),
        ("The basic knowledge of MONAI Data Pipeline usage", 3),
        ("The basic knowledge of MONAI Network usage", 3),
        # (query2doc_search("MONAI中Transforms有哪些"), 3),
        # ("如何实现3D数据增强", 5),
        # ("医学影像的IO处理模块", 3)
    ]
    for enable_q2d in [False, True]:
        print("setting: enable_query2doc =", enable_q2d)
        for q, k in test_queries:
            print(f"\n=== 测试查询: {q} ===")
            results = rag.search(q, k=k, enable_query2doc=enable_q2d, convert_to_str=False)
            # print(results)
            for i, doc in enumerate(results):
                print(f"[结果 {i+1}] {doc.metadata['source']}")
                print(len(doc.page_content))
                print(f"内容摘要: {doc.page_content[:100]}...")
                print("-"*50)