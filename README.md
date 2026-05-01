# Topic Modeling of Central Bank Speeches using LDA

Central bank communication plays a crucial role in shaping financial market expectations. In this project, we apply Natural Language Processing (NLP) techniques, mainly topic modeling, to extract topics from central bank speeches.

Using Latent Dirichlet Allocation (LDA), we identify the main themes discussed across institutions and analyze how these topics evolve over time. 


## Pipeline

1. **Data collection**  
   *(see `data`)*  

2. **Preprocessing**  
   *(see `notebooks/1_preprocessing_CBS_corpus.ipynb` and `src/preprocessing.py`)*  
   - The notebook explores the data  
   - Core preprocessing logic is implemented in `src/preprocessing.py`  
   - Output: structured dataset saved in `data/preprocessed/`

3. **Topic Modeling (LDA)**   

4. **Market Link (optional)**  
   *(see `notebooks/3_LDA_x_Market_Data.ipynb`)*  