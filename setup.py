from setuptools import setup

# Standard set up file to include the package details
setup(name='TextMining',
      version='1.0',
      description='App for common text mining tasks',
      author='Naveen Mathew Nathan S.',
      install_requires = ['treetaggerwrapper', 'pandas', 'langdetect', 'autocorrect', 'nltk',
      'numpy', 'bs4', 'sklearn', 'fuzzywuzzy', 'pyspark', 'gensim', 'pyLDAvis', 'matplotlib',
	  'dateparser', 'tmtoolkit', 'lda', 'stanfordcorenlp'],
      author_email='pg13s_nathan@mandevian.com'
     )
