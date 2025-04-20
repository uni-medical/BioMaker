from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from meddle.backend import query
import os.path as osp

PERSIST_DIR = osp.join(osp.dirname(__file__), "monai_rag_db")

def query2doc_search(search_query, model="gpt-4o"):
    prompt = "Refer to this example:\n" + \
             "query: what's the usage of ImageDataset in MONAI\n" + \
             "passage: ### ImageDataset[#](data.html#imagedataset 'Link to this heading') _class _monai.data.ImageDataset(_image_files_ , _seg_files=None_ , _labels=None_ , _transform=None_ , _seg_transform=None_ , _label_transform=None_ , _image_only=True_ , _transform_with_metadata=False_ , _dtype= <class 'numpy.float32'>_, _reader=None_ , _*args_ , _**kwargs_)[[source]](https://github.com/Project- MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L26-L155)[#](data.html#monai.data.ImageDataset 'Link to this definition')" +"\n"+\
             "query: ImageDataset\n" + \
             "passage: ### ImageDataset[#](data.html#imagedataset 'Link to this heading') _class _monai.data.ImageDataset(_image_files_ , _seg_files=None_ , _labels=None_ , _transform=None_ , _seg_transform=None_ , _label_transform=None_ , _image_only=True_ , _transform_with_metadata=False_ , _dtype= <class 'numpy.float32'>_, _reader=None_ , _*args_ , _**kwargs_)[[source]](https://github.com/Project- MONAI/MONAI/blob/dev/monai/data/image_dataset.py#L26-L155)[#](data.html#monai.data.ImageDataset 'Link to this definition')" +"\n\n"+\
             f"Write a passage that answers the given query: {search_query}\n\n"
    pseudo_doc = query(
        system_message=prompt,
        user_message=None,
        model=model,
        temperature=0.0,
        max_tokens=128,
    )
    return pseudo_doc

class MONAI_RAG:
    def __init__(self, model="gpt-4o", persist_dir=PERSIST_DIR):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        self.db = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embeddings
        )
        self.model = model
    
    def search(self, query, k=3, enable_query2doc=False):
        if enable_query2doc:
            query = query2doc_search(query, model=self.model)
        documents = self.db.similarity_search(query, k=k)
        # return concated documents
        return "\n".join([d.page_content[:] for d in documents])

if __name__ == "__main__":
    # 初始化RAG系统
    rag = MONAI_RAG()
    
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

    for q, k in test_queries:
        print(f"\n=== 测试查询: {q} ===")
        results = rag.search(q, k=k, enable_query2doc=False)
        print(results)
        # for i, doc in enumerate(results):
        #     print(f"[结果 {i+1}] {doc.metadata['source']}")
        #     print(len(doc.page_content))
        #     print(f"内容摘要: {doc.page_content[:100]}...")
        #     print("-"*50)