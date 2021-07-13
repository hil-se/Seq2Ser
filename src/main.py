from docEmbedding import DocEmbedding

docEmbedding = DocEmbedding()
docEmbedding.loadData()
docEmbedding.preprocess()
docEmbedding.test_seq2ser(k=3)
docEmbedding.test_average_pooling()