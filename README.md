# RAG-App-using-AWS-Bedrock-and-FAISSDB

**Before implementing the below, create a virtual environment using the command "conda create -p venv python==3.12" and activate the environment "conda activate '.....\venv'". Then install the necessary modules by "pip install -r requirements.txt".**

**To run it --> "streamlit run app.py"**

**1) Updated the vector database by loading the documents from the Data folder and creating embeddings to be stored in the document index. The vector database is created locally and named "faiss_index".**

![Screenshot1](https://github.com/user-attachments/assets/ad1e97bd-8ae4-4f4a-9320-bf65ea55a412)

**2) Firstly, try the Claude model, prompting it to generate accurate relevant content/answers from the pdf loaded earlier in the vector database.**

![Screenshot2](https://github.com/user-attachments/assets/e9a2e422-1abe-4ba7-85f5-015807f460ea)

**3) The document is fetched and several chunks are created and stored. The Embeddings are created for the chunks and get stored in FAISS vector DB locally.**

![Screenshot5](https://github.com/user-attachments/assets/6efaa54f-fa8f-4c86-915c-efb2d2e65903)

**4) Then a similarity search is performed between the query embeddings and document embeddings to retrieve the relevant content. Finally, send the relevant content and prompt to the model to generate the response.**

![Screenshot3](https://github.com/user-attachments/assets/92a5e3b0-0133-48dd-92bc-62255295abf9)

![Screenshot4](https://github.com/user-attachments/assets/b2853c96-795f-4217-aaf4-ac83b0f335d1)

**5) Secondly, try the llama3 model, and again prompt it to generate a relevant answer from the loaded pdf document.**

![Screenshot6](https://github.com/user-attachments/assets/acf9477f-2632-41fe-bb4c-159a1110b987)

**6) The relevant document is fetched and several chunks are created and stored in the document index. Then Embeddings are created from the embeddings model for the chunks and stored in FAISS vector DB locally.**

![Screenshot10](https://github.com/user-attachments/assets/7a121c0d-5ec8-470a-a7bd-af145854c12c)

**7) Then it performs the KNN / Cosine Similarity search between prompt embeddings and document embeddings to retrieve the most relevant documents since we had set the k value to be 3. So, the retrieved document and prompt are passed to the model to generate an accurate response.**

![Screenshot7](https://github.com/user-attachments/assets/bd5b30cb-b6e7-46ed-81b2-5924fc7e9135)

![Screenshot8](https://github.com/user-attachments/assets/ce7abb8c-6ed6-4ab7-a5b5-9610631bf61b)

![Screenshot9](https://github.com/user-attachments/assets/83a5fd10-db78-4c50-bb8e-0ac3243622a2)















