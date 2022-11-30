#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
Introduction to Python Programming (aka Programmierkurs I, aka Python I)
Software Assignment
'''

from funcs import *


# In[2]:


class SearchEngine:

    def __init__(self, collectionName, create):
        '''
        Initialize the search engine, i.e. create or read in index. If
        create=True, the search index should be created and written to
        files. If create=False, the search index should be read from
        the files. The collectionName points to the filename of the
        document collection (without the .xml at the end). Hence, you
        can read the documents from <collectionName>.xml, and should
        write / read the idf index to / from <collectionName>.idf, and
        the tf index to / from <collectionName>.tf respectively. All
        of these files must reside in the same folder as THIS file. If
        your program does not adhere to this "interface
        specification", we will subtract some points as it will be
        impossible for us to test your program automatically!
        '''
        self.corpusname = collectionName
        
        if create == True:
            corpus = parse(os.path.join(os.getcwd(), "%s.xml" %collectionName))
            save_file = input("would you like to save the tf and idf files to your current directory? (y/n)")
            if save_file == 'y':
                print('saving files to current directory...')
                s = True
            else:
                print('the files won\'t be saved')
                s = False
            self.tf = tf_corpus(corpus, ind2word = True, save_tf = s)
            self.idf = idf(corpus, ind2word = True, save_idf = s)
        else:
            print("Reading index from file...")
            self.tf = {}
            with open("%s.tf" %collectionName, "r") as tf_file:
                for line in tf_file:
                    docid, token, value = line.split("\t")
                    if docid not in self.tf:
                        self.tf[docid] = {}
                    self.tf[docid][token] = float(value.split("\n")[0])
            
            self.idf = {}
            with open("%s.idf" %collectionName, "r") as idf_file:
                for line in idf_file:
                    token, value = line.split("\t")
                    self.idf[token] = float(value.split("\n")[0])
            print("Done.")

        
        self.docs = set(self.tf.keys())

    
    def executeQuery(self, queryTerms):
        '''
        Input to this function: List of query terms

        Returns the 10 highest ranked documents together with their
        tf.idf-sum scores, sorted score. For instance,

        [('NYT_ENG_19950101.0001', 0.07237004260325626),
         ('NYT_ENG_19950101.0022', 0.013039249597972629), ...]

        May be less than 10 documents if there aren't as many documents
        that contain the terms.
        '''
        stemmed_terms = []
        for i, q in enumerate(queryTerms):
            stemmed_terms.append(stem(q))
        r = {}
        for docid in self.docs:
            try:
                r[docid] = similarity(stemmed_terms, docid, self.tf, self.idf)
            except ZeroDivisionError:
                r[docid] = 0
        sorted_r = sorted(r.items(), key = operator.itemgetter(1), reverse = True)[0:10]
        top_docs = {doc:value for (doc,value) in sorted_r if value !=0}

        if top_docs == {}:
            return None
            
        return [(key, value) for key, value in top_docs.items()]
        
    def executeQueryConsole(self):
        '''
        When calling this, the interactive console should be started,
        ask for queries and display the search results, until the user
        simply hits enter.
        '''
        while True:
            queryinput = input("Please enter query, terms separated by whitespace: ")
            if not queryinput:
                break
            else:
                res = SearchEngine.executeQuery(self, queryinput.split(" "))
                if res == None:
                    print("Sorry, I didn't find any documents for this term.")
                else:
                    print("I found the following documents:")
                    for key, value in res:
                        print(key + " " + "(" + str(value) + ")")
        print("End of query.")


# In[5]:


if __name__ == '__main__':
    '''
    write your code here:
    * load index / start search engine
    * start the loop asking for query terms
    * program should quit if users enters no term and simply hits enter
    '''
    # Example for how we might test your program:
    # Should also work with nyt199501 !
    filename = input("Please enter file name: ") 
    searchEngine = SearchEngine(str(filename), create=False)
    searchEngine.executeQueryConsole()
    #print(searchEngine.executeQuery(['hurricane', 'philadelphia']))




