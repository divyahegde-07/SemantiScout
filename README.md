
# SemantiScout

## Project Overview
The project aims to develop a robust multimodal product search engine that combines textual and visual data to enhance understanding of user preferences and deliver accurate, context-aware product recommendations. Traditional search engines often fail to capture the interplay between text and images, resulting in suboptimal results. Leveraging models like OpenAI’s CLIP for embedding generation and Pinecone for efficient vector storage, the system integrates multimodal features for improved relevance and personalization.  

Using a subset of the McAuley Amazon dataset (focusing on pet supplies, office supplies, and health and household categories), products with mismatched or incomplete data were excluded. CLIP embeddings were generated for titles and images, with data processing conducted on Google Cloud's Vertex AI for scalability and efficiency. This innovative approach demonstrates the potential of multimodal machine learning to overcome limitations of traditional search systems and enhance user satisfaction.



## Methodology

### Data Selection and Filtering
To ensure high-quality data for recommendation modeling, ~1 million products across three categories (Health and Household, Office Products, and Pet Supplies) were filtered. Key steps included:

Removing products without images to maintain consistent multimodal data.
Filtering titles with poor quality, excluding products with excessively brief or corrupted titles (two words or fewer).
Eliminating mismatched titles and images using OpenAI’s CLIP to compute cosine similarity between text and image embeddings, removing products with scores below 0.2.
These methods enhanced dataset reliability and improved the accuracy of downstream recommendations.

### Embeddings creation using CLIP
The dataset, loaded via the HuggingFace datasets library, was pre-processed and filtered before generating embeddings for three categories: Health and Household, Office Products, and Pet Supplies. CLIP was used to create text embeddings from product titles and image embeddings from high-resolution images. This process was parallelized using DASK.

### Vector Storage and Similarity Score
The obtained text and image embeddings were combined using different techniques - addition and concatenation which resulted in vectors of length 512 each and 1024 each respectively. For each category and each combining approach a separate vector database was created on Pinecone and the embeddings were upserted. For a given user query, CLIP is again used to generate a vector representation of the user query after which zero-shot classification is performed and the appropriate category database is identified. From there, the top 10 vectors are retrieved using cosine similarity.

### Zero-Shot User Query Classification and Embedding Prioritization

When a user query is entered, it is important to optimize the search process - rather than performing a search across all products in all categories, it is important to identify which category the query might pertain to and search only within that category thereby optimizing the search by limiting it to a lesser number of products.
This project leverages zero-shot classification to categorize user queries into relevant product categories (Health and Household, Office Products, and Pet Supplies) using BART-large model. 

Embedding Prioritization Based on Query Context
For user queries, words were classified as either "visualizable" (e.g., objects, colors) or "non-visualizable" (e.g., adjectives, qualities) using the Comprehend-It-Base model. Recommendations prioritized image-only embeddings for visualizable queries and re-ranked combined text and image embeddings for non-visualizable queries. A weighted strategy (75% image-only, 25% combined embeddings) was applied, optimizing recommendation relevance while balancing visual and textual features.

### **Evaluation Methodology**  

To evaluate the system, a dataset of 2,000 pet supplies, including toys and health products, was created for testing. Evaluation focused on comparing the SemantiScout system's recommendations against a baseline clustering approach using average cosine similarity.  

1. **Clustering Approach**:  
   - The top 50 products relevant to the user query were retrieved from Pinecone.  
   - K-means clustering was used to group similar products, with the optimal number of clusters determined as 8 based on the silhouette score (0.105).  
   - The system predicted the relevant cluster and retrieved the top 10 products within it for recommendations.  

2. **Cosine Similarity and Multimodal Analysis**:  
   - For quantitative evaluation, GPT-4o was used to generate concise, multimodal queries representing each recommended product by combining its title and image.  
   - The original user search query and GPT-generated queries for each recommended product were encoded using the 'all-MiniLM-L6-v2' model.  
   - Cosine similarity was calculated between the encoded vectors of the user query and each recommended product query, with the average similarity score used to assess recommendation quality.  

This methodology ensured a robust evaluation by incorporating semantic alignment and multimodal representation.


### Results and Observations
When testing out sample user queries, we experimented with different ways of combining the embeddings - adding the text and image embeddings and using only the image embeddings. These experimentations allowed us to qualitatively evaluate the effect of combining text with image vs. using image only to see which method provides the best recommendations.


####  User query: “red travel-sized electric toothbrush kit”
Results analysis: This query exemplifies CLIP's effectiveness in integrating contextual understanding when processing image features. The initial results accurately identify travel-sized electric toothbrushes but lack sensitivity to the color context, specifically the red hue present in the images. In contrast, embeddings based solely on image features capture the color context effectively, avoiding any interference from irrelevant textual information.

![Red travel-size toothbrush image+text](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/red_travel_textimage.png)

![Red travel-size toothbrush Image-only](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/red_travel_image.png)


#### User query: “Pet collar with hearts”
Results analysis: Image only embeddings outperforms in capturing shape-related visual cues. CLIP's ability to understand nuances in shape helps identify subtle variations, such as differentiating "square" from "rounded square" products. This is especially useful for applications like furniture or fashion, where small changes in shape can significantly impact style or functionality.


![Pet collar with hearts image+text](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/pet_collar_textimage.png)

![Pet collar with hearts Image-only](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/pet_collar_image.png)


#### User query: “anti-UV sunglasses with reader lens for women”
Results analysis: Here image-only embeddings has challenges with abstract concepts. Functional attributes often depend on sensory qualities (like smell or taste) or protective features (like UV-blocking properties) that are not visually observable. CLIP is unable to directly encode these qualities because they are not embedded in the image itself and are often only communicated through text descriptions.


![UV image+text](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/UV_textimage.png)

![UV Image-only](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/UV_image.png)

#### User query: “superhero dog costumes”

Below is the suggested products from clustering.
![clustering superhero costumes](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/pet_superhero_clustering.png)

And these are the products suggested by SemantiScout.
![Semantiscout superhero costumes](https://raw.githubusercontent.com/divyahegde-07/SemantiScout/main/Results/pet_superhero_SS.png)










