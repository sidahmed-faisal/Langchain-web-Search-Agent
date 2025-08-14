from searchtool import *


docs  = WebBaseLoader("https://medium.com/@mehulpratapsingh/langchain-agents-for-noobs-a-complete-practical-guide-e231b6c71a4a").load()
print(docs[0].metadata)