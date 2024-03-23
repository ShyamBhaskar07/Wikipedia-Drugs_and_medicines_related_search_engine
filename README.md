1. Collecting documents from Wikipedia:
   Perform Web-Crawling starting from a good hyper-linked page (choose drug, medicine category wiki pages)
   Extract only titles and paragraphs from webpage (save as d1.txt,d2.txt....)
   Collect as many as documents possible to get relevant results
   Save all the collected documents in a folder names "All_textfiles"

2. Constructing Indexing file for all documents
   I have chosen Inverted_index construction for faster retrieval
   Run file "inverted_index.py" to create a index file (index.txt)
   Sort "index.txt" file by running "sort.py" which will create "sorted_clean.txt" (for faster search)

3. Creating TRIES Data Structure:
   After sorting the index file, convert it into TRIEs data structure by running "new.py" file
   It will start creating folders for posting list of all words
   Basic Idea: Constant time retreival for a word in Tries data structure

4. Ranking of Documents:
   Remove stopwords and use TF-IDF Score to rank the documents
   Retreive top 10 documents based on the given query

5. User Interactive Interface:
   Run "app.py" file to redirect to UI, which will ask for a user query
   Spelling correction for the query words for good results
   Top 10 results will be displayed with 2-3 sentences and a link to original Wikipedia page

Retreival Time: Depends on the number of documents and number of query words (apprx 30s for 105000 documents, 3-4 query words)
Note : Install all the required packages
