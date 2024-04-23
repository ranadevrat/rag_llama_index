from flask import Flask, request, render_template, jsonify


app = Flask(__name__)

import os
from llama_index.core import ServiceContext, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.indices.postprocessor import MetadataReplacementPostProcessor
from llama_index.core.indices.postprocessor import SentenceTransformerRerank
from llama_index.core import load_index_from_storage
#from gpt4all import GPT4All
from langchain.llms import GPT4All


def build_sentence_window_index(documents,llm, embed_model="local:BAAI/bge-small-en-v1.5",sentence_window_size=3,
                                save_dir="sentence_index",):
    # create the sentence window node parser w/ default settings
    node_parser = SentenceWindowNodeParser.from_defaults(
        window_size=sentence_window_size,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    sentence_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embed_model,
        node_parser=node_parser,
    )
    if not os.path.exists(save_dir):
        sentence_index = VectorStoreIndex.from_documents(
            documents, service_context=sentence_context
        )
        sentence_index.storage_context.persist(persist_dir=save_dir)
    else:
        sentence_index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=save_dir),
            service_context=sentence_context,
        )

    return sentence_index


def get_sentence_window_query_engine(sentence_index, similarity_top_k=6, rerank_top_n=2):
    # define postprocessors
    postproc = MetadataReplacementPostProcessor(target_metadata_key="window")
    rerank = SentenceTransformerRerank(
        top_n=rerank_top_n, model="BAAI/bge-reranker-base"
    )

    sentence_window_engine = sentence_index.as_query_engine(
        similarity_top_k=similarity_top_k, node_postprocessors=[postproc, rerank]
    )
    return sentence_window_engine

from llama_index.llms.openai import OpenAI
from data.dataprovider import key
from llama_index.core import SimpleDirectoryReader
#OpenAI.api_key =  key

documents = SimpleDirectoryReader(
    input_files=[r"data/eBook-How-to-Build-a-Career-in-AI.pdf"]
).load_data()

from llama_index.core import Document

document = Document(text="\n\n".join([doc.text for doc in documents]))

index = build_sentence_window_index(
    [document],
    #llm=OpenAI(model="gpt-3.5-turbo", temperature=0.1,api_key=key),
    #llm = GPT4All("mistral-7b-openorca.gguf2.Q4_0.gguf"),
    llm = GPT4All(model=r'C:\Users\91941\.cache\gpt4all\mistral-7b-openorca.gguf2.Q4_0.gguf'), #Replace this path with your model path
    save_dir="./sentence_index",
)

query_engine = get_sentence_window_query_engine(index, similarity_top_k=6)

def chat_bot_rag(query):
    window_response = query_engine.query(
        query
    )

    return window_response



# Define your Flask routes
@app.route('/')
def home():
    return render_template('bot_1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_input']    
    bot_message = chat_bot_rag(user_message)    
    return jsonify({'response': str(bot_message)})

if __name__ == '__main__':
    app.run()
    #app.run(debug=True)
