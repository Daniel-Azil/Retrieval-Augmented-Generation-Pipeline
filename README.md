# README for Retrieval-Augmented Generation Pipeline

This repository contains a Jupyter Notebook demonstrating a **Retrieval-Augmented Generation (RAG) Pipeline** using LangChain and Hugging Face models. This pipeline is designed to retrieve relevant documents based on a query and generate context-rich answers by leveraging advanced embeddings and similarity search techniques. Below, you'll find an overview of the steps involved and instructions for setup.

---

## Table of Contents
1. [Overview](#overview)
2. [Setup Instructions](#setup-instructions)
3. [Pipeline Steps](#pipeline-steps)
4. [Usage](#usage)
5. [Notes](#notes)
6. [References](#references)

---

### Overview
The **Retrieval-Augmented Generation Pipeline** is built to:
- Load and chunk documents.
- Generate embeddings using a Hugging Face model.
- Store document embeddings in a vector database (Chroma) for fast retrieval.
- Perform similarity searches to find context-relevant documents.
- Generate responses to queries using Hugging Face's large language models.

### Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
    ```

2. **Install Dependencies**:

    Ensure that all necessary packages (LangChain, Chroma, and Hugging Face models) are installed.

    ```bash
    pip install langchain langchain_community chromadb transformers
    ```

3. **Set Hugging Face API Token**:

    Obtain your API token from Hugging Face and set it as an environment variable.

    ```python
    import os
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = "Your Huggingface API Token"
    ```

4. **Modify Path if Needed**:

    Update the Python path in the notebook if using a virtual environment:

    ```python
    import sys
    # sys.path.append('Your Virtual Environment Python-site-packages Path')
    ```


### Pipeline Steps

- **Load the Dataset**: The pipeline reads text files from the specified directory and stores them for processing.

- **Data Chunking**: Using `RecursiveCharacterTextSplitter`, documents are split into manageable chunks for better context handling.

- **Embed Data**: Each document chunk is converted into an embedding vector using `HuggingFaceBgeEmbeddings`. These vectors are stored in a Chroma database, enabling efficient similarity searches.

- **Perform Similarity Search**: A similarity search is conducted on the Chroma vector store, fetching documents that closely match a query.

- **Set Up Language Model for QA**: A Hugging Face model is configured to answer queries based solely on the retrieved context.

- **Run RetrievalQA Pipeline**: The `RetrievalQA` chain processes queries by retrieving relevant documents and providing precise answers.


### Usage

- **Load and Chunk Documents**: Define the path to the directory containing your `.txt` files, then run the notebook to load and chunk them.

- **Embed and Store in Chroma Vector Database**: Initialize and add your document chunks to the Chroma vector database, preparing it for similarity searches.

- **Query and Retrieve**: Provide a query, and the pipeline will fetch the most relevant documents using Chroma and answer the question based on the document's content.

- **Run RetrievalQA**: Run the pipeline by defining a question, then view the answer generated using the context retrieved.

### Notes

- **API Token**: Ensure your Hugging Face API token is valid and stored securely.
- **Model and Database Configuration**: Adjust the model and database settings if running on different hardware or with larger datasets.
- **Prompt Customization**: Modify the `PromptTemplate` as necessary to tailor the response generation to your specific application.

### References

- [LangChain Documentation](https://www.langchain.com)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Chroma Vector Database](https://www.chromadb.com)