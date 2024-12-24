# Vision & Text RAG Chatbot

## Overview

The **Vision & Text RAG Chatbot** is a multimodal Retrieval-Augmented Generation (RAG) application designed to provide accurate and context-aware responses to user queries. By combining advanced vision language models (VLMs) and text retrieval techniques, it achieves a robust, efficient, and interactive experience. This project leverages cutting-edge architecture, including ColPali, Byaldi, and Llama-based vision models, to streamline document retrieval and enhance response generation.

---

![](https://github.com/yYorky/VisionRAG/blob/main/video/Vision.gif?raw=true)

## Key Features

- **Document Upload and Indexing**: Supports multiple document formats (PDF, DOCX, images) for streamlined processing.
- **Multimodal Retrieval**: Combines text and vision-based retrieval to deliver enriched responses.
- **Conversational Memory**: Maintains context across sessions using LangChain’s conversational memory.
- **Interactive Interface**: Developed with Streamlit for an intuitive user experience.
- **Custom Embedding Models**: Uses Google Generative AI embeddings for document vectorization.

---

## Architectural Highlights

### Retrieval-Augmented Generation (RAG) Pipeline

The system integrates both **ColPali** and **Llama-based vision models** to form an end-to-end RAG pipeline, simplified and enhanced by **Byaldi**. The architecture includes:

1. **ColPali**:

   - Simplifies traditional retrieval processes by embedding images directly using vision encoders.
   - Incorporates Polygama VLM for efficient and accurate information extraction from document images.
   - Utilizes multi-vector representations for enhanced retrieval performance, capturing both local features and global context.
   - Benchmarked to outperform existing retrieval methods, including BM25 and dense embedding approaches.
   - Offers explainability through attention maps, allowing users to see which parts of the image influenced the response.

2. **Byaldi**:

   - Acts as a wrapper around ColPali, making it easy to integrate late-interaction multimodal models with a familiar API.
   - Supports all ColPali-engine models, including advanced checkpoints like `vidore/colpali`.
   - Facilitates indexing and retrieval with minimal setup.
   - Provides options for efficient document storage using base64 encoding.
   - Future updates will include advanced features like HNSW indexing and 2-bit quantization.

3. **Llama 3.2 Vision model**:

   - Uses the Llama 3.2 Vision model to integrate image understanding into RAG workflows.
   - Capable of performing image-based tasks like optical character recognition (OCR) and visual analysis.



The combination of ColPali, Byaldi, and Llama-based vision models ensures a nuanced and comprehensive understanding of documents and queries, enabling richer multimodal interactions.


## Setup Instructions

### Prerequisites

- Python 3.10+
- Required dependencies listed in `requirements.txt`
- **Poppler**: Required for PDF to image conversion. Install via their website: [https://poppler.freedesktop.org/](https://poppler.freedesktop.org/)



### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yYorky/VisionRAG.git
   cd VisionRAG
   ```

2. Create and activate a virtual environment:

   ```bash
   conda create -n visionrag python=3.10
   conda activate visionrag
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   - Create a `.env` file and add your API keys:
     ```plaintext
     GROQ_API_KEY=your_api_key_here
     GOOGLE_API_KEY=your_api_key_here
     ```

### Run the Application

1. Start the Streamlit app:
   ```bash
   streamlit run app_V2.py
   ```
2. Open the app in your browser at `http://localhost:8501`.

---

## File Structure

```plaintext
.
├── app_V2.py               # Main Streamlit application
├── logger.py  
├── models/
│   ├── indexer.py          # Handles document indexing
│   ├── retriever.py        # Handles document retrieval
│   ├── responder.py        # Generates responses
│   ├── model_loader.py     # Loads AI models
├── static/images/          # Stores static images
├── uploaded_documents/     # Folder for uploaded documents
├── .env                    # Environment variables
├── .gitignore              # Files and folders to ignore
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
```

---

## Workflow

Document Processing:

- Upload files via the Streamlit interface.
- Process documents using ColPali for direct image embedding and splitting into manageable chunks.
- Generate vector embeddings for efficient retrieval using Google Generative AI embeddings and FAISS as the vector database.

Indexing and Retrieval:

- Leverage Byaldi for simplified indexing and querying of multimodal documents.
- Create indexes with minimal configuration, storing metadata and base64-encoded documents for seamless LLM integration.
- Perform searches with results ranked by relevance using LangChain’s framework to orchestrate the retrieval pipeline.

Query and Response:

- Input queries via the conversational interface.

- Retrieve relevant information and images using multimodal retrieval techniques.

- Generate responses using conversational memory and Llama-based vision models.


## Technologies Used

- **Streamlit**: For building the interactive UI.

- **LangChain**: Manages text splitting and conversational memory.

- **FAISS**: Enables efficient similarity search for retrieval.

- **Google Generative AI**: Provides advanced embedding models.

- **Byaldi**: Simplifies the integration of ColPali-based retrieval pipelines.

- **ChatGroq**: Llama-3.1 for text response generation.

- **RAGMultiModalModel**: Combines vision and text retrieval.

---

## Limitations and Future Work

### Current Limitations:

- Difficulty handling low-resolution or complex images.
- Limited reasoning capabilities for intricate queries.
- Time-intensive to index image page by page

### Future Directions:

- Integrate other VLMs (e.g., OpenAI models) for enhanced accuracy.
- Adding Agentic workflow to orchestrate the two RAG pipelines

---

## Inspired by

Ollama with Vision - Enabling Multimodal RAG: [https://www.youtube.com/watch?v=45LJT-bt500](https://www.youtube.com/watch?v=45LJT-bt500)

ColPali: Efficient Document Retrieval with Vision Language Models - [https://arxiv.org/abs/2407.01449](https://arxiv.org/abs/2407.01449)

Byaldi: Simple wrapper around the ColPali repository - [https://github.com/AnswerDotAI/byaldi](https://github.com/AnswerDotAI/byaldi)

---

## Acknowledgements

Demo catalogue document extracted from: https://www.decathlon.sg/s/decathlon-pro

