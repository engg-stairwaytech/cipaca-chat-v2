from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Pinecone as PineconeStore
from llm import embeddings_model, pinecone_client, pinecone_index
from dotenv import load_dotenv
import time
load_dotenv()

loader = DirectoryLoader(
    "./docs",
    glob="**/*.pdf",  # you can adjust for *.txt, *.docx etc.
    loader_cls=PyPDFLoader
)
documents = loader.load()
file_count = len(documents)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # smaller chunks for better context
    chunk_overlap=200,   # overlap preserves continuity
    length_function=len,
    separators=["\n\n", "\n", ".", " ", ""]
)

chunks = splitter.split_documents(documents)

print('index details - ', pinecone_index.describe_index_stats())
# previous chunk count - 13322
last_completed_id = 0
batch_size = 100

batch_texts = []
batch_metadatas = []
batch_ids = []

print("total chunks - {}".format(len(chunks)))
for i, doc in enumerate(chunks):
    if i > last_completed_id:
        
        batch_texts.append(doc.page_content)
        metadata_with_text = doc.metadata.copy()
        # 2. ADD the actual chunk content under the 'text' key
        metadata_with_text['text'] = doc.page_content 
        batch_metadatas.append(metadata_with_text)
        batch_ids.append(str(i))
        # when we reach batch_size, push the batch
        if len(batch_texts) >= batch_size:
            # Batch embedding (single API call)
            print('Starting to embed next batch')
            embeddings = embeddings_model.embed_documents(batch_texts)
            print('Completed embedding')
            # Create vector objects
            vectors = [
                {"id": batch_ids[j], "values": embeddings[j], "metadata": batch_metadatas[j]}
                for j in range(len(batch_texts))
            ]
            print(f"Upserting batch ending at id {i}")
            upsert_status = pinecone_index.upsert(vectors=vectors)
            print(upsert_status)
            print(f"Completed batch up to {i}")
            batch_texts, batch_metadatas, batch_ids = [], [], []
            time.sleep(1)  # optional: short pause to avoid rate limit
    else:
        print('skipping pushing to embeddings or pinecone. Completed in previous step')

if batch_texts:
    print(f"Processing final batch ({len(batch_texts)} docs remaining)...")
    embeddings = embeddings_model.embed_documents(batch_texts)
    vectors = [
        {"id": batch_ids[j], "values": embeddings[j], "metadata": batch_metadatas[j]}
        for j in range(len(batch_texts))
    ]
    pinecone_index.upsert(vectors=vectors)
    print(f"Final batch completed. All {len(chunks)} chunks processed.")