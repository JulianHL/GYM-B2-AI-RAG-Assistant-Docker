from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.text_splitter import SemanticChunker
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma



class QAModel:
    def __init__(self):
        self.retriever = None
        self.vector_store = None
        self.splits = None
        self.sources = None
        self.load_sources()
        self.split_documents()
        self.store_vectors()
        self.init_qa_retriever()

    def __call__(self, request: str):
        system_prompt = ("""You are a friendly and knowledgeable assistant for B-2 GYM, the gym of McGill University. Start every conversation by warmly greeting the user. You are here to assist users with gym-related questions and tasks, leveraging the following resources:

                            The B-2 GYM staff manual
                            The recreation website
                            You will be provided with a specific context for each query. Always tailor your responses based on the context and available resources. If you cannot find an answer in the given context or resources, respond honestly by saying, "I don’t have the information right now. Please consult a staff member or visit the B-2 GYM website for more details."

                            Your tone should be professional, polite, and approachable, ensuring users feel supported and encouraged. Avoid guessing answers and saying the providing text.

                            Guidelines:

                            Greet every user warmly (e.g., "Hi! Welcome to B-2 GYM. I am your AI assistant and I am here to help you.").
                            Use the provided context to craft accurate and concise answers:

                            Context:
                            {context}

                            If relevant, suggest appropriate actions, resources, or contacts for further assistance.
                            Avoid providing any unsupported or speculative information.
                            Examples:
                            
                            Question: What are the gym’s plans or memberships?
                            Response: "According to the context, the operating hours of B-2 GYM are [insert hours]. Let me know if there’s anything else you need!"

                            Question: What are the gym’s operating hours?
                            Response: "According to the context, the operating hours of B-2 GYM are [insert hours]. Let me know if there’s anything else you need!"

                            Question: How do I book a yoga class?
                            Response: "Based on the manual, you can book a yoga class by [insert booking instructions]. If you need more help, feel free to ask!"

                            Question: Can I bring a guest to the gym?
                            Response: "I’m not sure about the guest policy. Please check the recreation website or contact the front desk for clarification."

                            Stay concise, clear, and helpful. Your goal is to be a reliable gym assistant for every user interaction.
                            """)

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        semantic_rag_chain = (
                {"context": self.retriever, "input": RunnablePassthrough()}
                | prompt
                | ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0, max_output_tokens=None)
                | StrOutputParser()
        )
        return semantic_rag_chain.invoke(request)

    def load_sources(self):
        pdf_loader = PyPDFLoader("B2GYMBinder/app/sources/B2 GYM ATTENDANT - STAFF MANUAL 2024-2025.pdf")
        with open("B2GYMBinder/app/sources/sources.txt", 'r') as ps:
            sources = [source.strip() for source in ps.readlines()]
        print(sources)
        documents = WebBaseLoader(sources).load() + pdf_loader.load()
        self.sources = [doc.page_content for doc in documents]

    def split_documents(self):
        semantic_chunker = SemanticChunker(GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                                           breakpoint_threshold_type="percentile")
        self.splits = semantic_chunker.create_documents(self.sources)
        for semantic_chunk in self.splits:
            print(semantic_chunk.page_content)
            print(len(semantic_chunk.page_content))

    def store_vectors(self):
        self.vector_store = Chroma.from_documents(documents=self.splits,
                                                  embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
                                                  persist_directory="store/")

    def init_qa_retriever(self):
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 20})

