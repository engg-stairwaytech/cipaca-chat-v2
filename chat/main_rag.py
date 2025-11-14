
from .llm import embeddings_model, llm_model, pinecone_client, pinecone_index, raw_llm_model
from langchain_pinecone import PineconeVectorStore
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

vectorstore = PineconeVectorStore(
    index=pinecone_index,
    embedding=embeddings_model,
    text_key="text"
)
# Always answer clearly, with reasoning and references when available.
# You are a professional AI assistant specialized in medical and academic knowledge to serve the organisation CIPACA.
# You are an expert, board-certified Intensivist AI
# 2. **Bolding for Content:** Highlight critical data, key findings, or important medical terms within paragraphs by making them bold. **Crucially, do not use Markdown Heading syntax (e.g., #, ##, ###)**, as this results in large fonts that disrupt the clinical flow. 
CUSTOM_QA_PROMPT = PromptTemplate.from_template("""
You are a professional AI assistant specialized in medical and academic knowledge to serve the organisation CIPACA. Your goal is to provide highly accurate, concise, and evidence-based medical information that is immediately useful in a critical care organisation CIPACA for saving lives. 
**Use medical terminology, abbreviations (e.g., ABG, Hgb, SBP), and a direct, professional, and action-oriented clinical tone.**"
CIPACA (Chennai Interventional Pulmonology and Critical Care Associates) is a healthcare organization that sets up and manages affordable, 24×7 ICU and critical care units in rural and semi-urban hospitals across India.
                         
                                                
**Formatting Rules:** 
    1. **Bolding for Headers:** Always use only double asterisks (e.g., '**Summary:**', '**Reasoning:**') to create **bold section headings**. 
    2. **Extra Information:** Differentiate supplemental or extra clinical detail using *italics* or single-backtick code blocks (e.g., `` `lab value: 1.2` ``).
    3. **Section Separation:** End each major section with a markdown horizontal rule (e.g., '---') to show completion." # Adds lines
    4. **Visual Explanation:** Add relevant and professional emojis (e.g., 🌡️, 💊, ⚠️, 🩺) to make the answer more explanatory and visually engaging.
    5. **Disclaimer:** Conclude the entire answer with the following phrase in *italics* on a new line: '*Disclaimer: Always confirm findings with local protocol and clinical judgement. This information is for reference only and does not substitute for professional medical judgment.*'
    6. **Data Simplification:** Use tables wherever needed to simplify the data, compare options, or present structured protocols/dosing.

Always answer with reasoning and references when available. 
When providing an answering with a question always, provide a compliment at the beginning of the answer for the user how good they were in questioning about the query and it can help CIPACA to save lives.
After answering the question, provide a highlevel summary of the anwser either in a table or in a few sentences.
At the end, ask user if they want a followup question or more detailed explanation about that topic.

                                                
Conversation history:
{chat_history}

User question:
{question}

Relevant context from documents:
{context}

Now, based on the context above, provide a concise, factual, and well-structured answer.
If you don't know the answer, say "The available sources do not provide that information."

Answer:
""")

retriever = vectorstore.as_retriever(search_kwargs={
       "k": 3
       })

# qa_chain = RetrievalQA.from_chain_type(
#     llm=llm_model,
#     retriever=retriever,
#     chain_type="stuff",  # "map_reduce" or "refine" for long docs
#     return_source_documents=True
# )

memory = ConversationBufferMemory(
    llm=raw_llm_model,             # Pass the LLM to the memory to summarize/manage history
    memory_key="chat_history", # This must match the expected input key for the chain
    return_messages=True ,      # Use a list of message objects for cleaner history passing,
    output_key="answer"
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm_model,
    retriever=retriever,
    memory=memory,                     # Pass the initialized memory object
    chain_type="stuff",
    return_source_documents=True,
    # Optional: configure how the chat history is used to create a new query
    # The default prompt is usually good, but you can customize it here:
    # combine_docs_chain_kwargs={"prompt": YOUR_PROMPT} 
    combine_docs_chain_kwargs={
        "prompt": CUSTOM_QA_PROMPT
    }
)

if __name__ == "__main__":
    test_query = "Explain shock"
    retrieved_docs = retriever.invoke(test_query)
    if retrieved_docs:
            print(f"✅ RETRIEVER SUCCESS: Found {len(retrieved_docs)} documents.")
            print(f"Top Document Source: {retrieved_docs[0].metadata.get('source', 'N/A')}")
    else:
            print("❌ RETRIEVER FAILURE: Returned 0 documents. Check Pinecone index data and embeddings.")
    fetched_vector = pinecone_index.fetch(ids=[1]) 
    print(fetched_vector)
    # print(pinecone_index.describe_index_stats())
    # query_vector = embeddings_model.embed_query('Explain shock')
    # results = pinecone_index.query(
    #     vector=query_vector,
    #        top_k=3,
    # )

    # print(results)