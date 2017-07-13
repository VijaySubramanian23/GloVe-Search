import mmap
import os
import numpy as np
from tqdm import *
from scipy.spatial import distance
# import matplotlib.pyplot as plt
from anytree import Node, RenderTree
# from anytree.dotexport import RenderTreeGraph

BASE_DIR = '/home/tushhar/projects/ML/word2vec'
GLOVE_DIR = BASE_DIR + '/glove.6B/'

'''let's have a global list to store unique strings'''
main_list = []
global_counter_threshhold = 0
number_of_words_per_level =5

def get_line_number(file_path):  
	fp = open(file_path, "r+")
	buf = mmap.mmap(fp.fileno(), 0)
	lines = 0
	while buf.readline():
		lines += 1
	return lines
	fp.close();

'''	print tree '''    
def print_tree(root):
	for pre, fill, node in RenderTree(root):
		print("%s%s" % (pre, node.name))


def get_top(embeddings_index,new_word,root):
	if global_counter_threshhold<number_of_words_per_level:
		main_embed = embeddings_index
		embedding_vector = main_embed.get(new_word)
		for word, value in tqdm(main_embed.iteritems(),total = len(main_embed)):
			dst = distance.euclidean(embedding_vector,value)	
			main_embed[word]=dst
			# children = []
		for key, value in sorted(main_embed.iteritems(), key=lambda (k,v): (v,k))[1:number_of_words_per_level+1]:
			print "%s: %s" % (key, value)
			# childnode = Node(key,parent=new_word_root)
			# children.append([key,childnode])
			# children.append(key)
			child_node = add_to_main_list(key,root)
			get_top(embeddings_index,key,child_node)
	else:
		return

''' add unique only '''
def add_to_main_list(value,node):
	global global_counter_threshhold
	if value not in main_list:
		main_list.append(value)
		global_counter_threshhold = 0
		print 'global_counter_threshhold = %s' % global_counter_threshhold
		return Node(value,parent=node)
	else:
		global_counter_threshhold += 1
		print 'global_counter_threshhold = %s' % global_counter_threshhold
		return node
	

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in tqdm(f,total=get_line_number(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))):
	values = line.split()
	word = values[0]
	coefs = np.asarray(values[1:], dtype='float32')
	embeddings_index[word] = coefs
f.close()
# main_embed = embeddings_index

print 'Found %s word vectors.' % len(embeddings_index)

new_word = raw_input("Enter your string: ")
root=Node(new_word) #start node
get_top(embeddings_index,new_word,root)
print main_list
print_tree(root)
# RenderTreeGraph(root).to_picture("/home/tushhar/projects/ML/word2vec/root.png")









