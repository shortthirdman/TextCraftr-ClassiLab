class LLMTextClassifier:
    def __init__(
        self,
        categories: list[str],
        system_prompt_template: PromptTemplate = PromptTemplate(
            input_variables=["categories", "schema"],
            template="Classify the following text into one of the following classes: {categories}.\n "
            "Use the following schema: {schema}",
        ),
        llm_client: BaseChatModel = llm_medium,
        max_examples: int = 5,
    ):
        # Initialize model, prompt, and retrieval variables
        self.categories = categories
        self.categories_model = generate_classification_model(categories)
        self.system_prompt_template = system_prompt_template
        self.system_prompt = system_prompt_template.format(
            categories=categories, schema=self.categories_model.model_json_schema()
        )
        self.llm_classifier = llm_client.with_structured_output(self.categories_model)
        self.max_examples = max_examples
        self.examples = None
        self.vector_store = None
        self.retriever = None
	
	def fit(self, texts, labels):
        self.examples = [
            Document(page_content=text, metadata={"label": label})
            for text, label in zip(texts, labels)
        ]

        if len(self.examples) > self.max_examples:
            # Add examples to vector store
            self.vector_store = Chroma.from_documents(
                documents=self.examples,
                collection_name="llm-classifier",
                embedding=ChromaEmbeddingsAdapter(
                    embedding_functions.DefaultEmbeddingFunction()
                ),
            )
            self.retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.max_examples}
            )
	
	def predict(self, text: str) -> str:
        messages = [SystemMessage(content=self.system_prompt)]
        
        for example in self.fetch_examples(text=text):
            messages.append(HumanMessage(content=example.page_content))
            messages.append(AIMessage(content=example.metadata["label"]))

        messages.append(HumanMessage(content=text))
        prediction = self.llm_classifier.invoke(messages)

        return prediction.category


if __name__ == "__main__":
    categories = ["news", "clickbait"]
    classifier = LLMTextClassifier(categories=categories, max_examples=1)

    texts = ["Donald Trump won Michigan", "You won't believe what happened next!"]
    labels = ["news", "clickbait"]
    
    classifier.fit(texts, labels)

    text = "Donald Trump won Florida"
    result = classifier.predict(text)
    print(result)  # Should output "news" if similar to "news" examples