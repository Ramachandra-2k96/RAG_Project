Abstract:
The project is a personal portfolio chatbot built with Streamlit, LangChain, and Cerebras GPT-OSS-120B. It acts as a conversational agent that represents **Ramachandra Udupa**, answering questions about his work, projects, services, and experience. The chatbot uses Retrieval-Augmented Generation (RAG) to pull accurate information from curated sources such as a PDF resume and personal websites, ensuring contextually correct responses while maintaining the persona of Ramachandra.

---

**Introduction**
Personal branding is critical in technology careers. Static resumes and portfolios limit interaction and personalization. This project introduces an interactive AI-powered chatbot that serves as a living portfolio, providing a conversational interface for recruiters, clients, or collaborators to explore professional details dynamically. The chatbot is designed to emulate the voice and style of Ramachandra, making the interaction more engaging and authentic.

---

**Objectives**

1. To build an AI chatbot that embodies Ramachandra’s persona.
2. To enable recruiters and clients to interactively explore portfolio information.
3. To ensure responses remain truthful, concise, and context-specific.
4. To integrate resume data and portfolio websites into a retrieval pipeline.
5. To deliver a smooth user experience through a custom Streamlit interface.

---

**System Architecture**

* **Frontend (Streamlit):** Provides a clean UI with chat bubbles, status indicators, and session control.
* **LLM Backbone (Cerebras GPT-OSS-120B):** Powers conversational responses.
* **Embedding Model (HuggingFace all-mpnet-base-v2):** Encodes text for semantic search.
* **Vector Store (Chroma):** Stores portfolio documents and enables similarity-based retrieval.
* **Retrieval Module:** Fetches relevant resume or portfolio sections.
* **LangGraph Agent:** Manages reasoning, tool usage, and conversational memory.
* **Memory & Caching:** In-memory cache avoids redundant lookups, while session memory maintains conversational flow.

---

**Implementation Details**

* **Document Sources:**

  * Resume (PDF).
  * Portfolio websites and service pages.
* **Text Processing:** Recursive text splitting into chunks for efficient retrieval.
* **Custom Tools:**

  * Retrieval tool fetches and formats relevant chunks.
* **Conversation Flow:**

  * System prompt enforces persona rules.
  * Agent streams reasoning and responses.
  * Messages displayed with custom CSS-styled UI.
* **Controls:**

  * Clear chat.
  * Reset session.
  * Sidebar with context and debug info.

---

**Features**

* AI speaks strictly as Ramachandra.
* Retrieval ensures factual grounding.
* User-friendly UI with live “thinking” indicator.
* Session persistence across multiple user queries.
* Debug info for monitoring behavior.

---

**Use Cases**

* Recruiters querying skills and experience.
* Clients exploring offered services.
* Collaborators reviewing projects and portfolio.
* Interactive alternative to static resumes.

---

**Results**
The system provides consistent, natural, and concise answers while remaining grounded in verified data. Tool usage ensures retrieval is only triggered when necessary. The bot never fabricates unrelated information, aligning strictly with professional portfolio details.

---

**Limitations**

* Dependent on curated sources; missing data limits answers.
* Requires Cerebras API access and GPU resources.
* No integration with real-time web search outside predefined portfolio.

---

**Conclusion**
The project demonstrates how AI can transform a portfolio into an interactive, conversational agent. By merging LLMs with RAG, Ramachandra’s professional identity is represented authentically and dynamically. This approach improves accessibility, engagement, and personalization in professional networking, offering a forward-looking alternative to resumes and static portfolio websites.
