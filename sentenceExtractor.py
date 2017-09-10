import concepts_extractor, relation_extractor
import nltk, sys
import numpy as np
import pickle
from gensim.models.keyedvectors import KeyedVectors
from keras.models import model_from_json

import os, socket, thread
from pprint import pprint

import BNUtilities as bn
import enrichingUtilities as enrich



class sentenceExtractor(object):
	
	def __init__(self):
		## Load W2V resources
		print "[CONC-EXTRACTOR] Loading w2v resources...",
		self.random_vector = np.load("data/" + "UNK_vec.npy")
		try:
			self.word_vectors = KeyedVectors.load("data/" + "wv.w2v", mmap='r')
		except IOError:
			print('Loading W2V..')
			wv = KeyedVectors.load_word2vec_format("data/" + 'GoogleNews-vectors-negative300.bin', binary=True)
			wv.save("data/" + "wv.w2v")
			word_vectors = KeyedVectors.load("data/" + "wv.w2v", mmap='r')
		print "DONE"
		
		## Start server thread
		self.server_address = './uds_socket'
		thread.start_new_thread(self.start_server, ())
		
	def start_server(self):
		## Load keras model
		print "[CONC-EXTRACTOR] Loading/Compiling concepts extractor model...",
		sys.stdout.flush()
		json_model = open("model_conc/keras_model.json", "r")
		model = json_model.read()
		self.model = model_from_json(model)
		self.model.load_weights("model_conc/model_weights.hd")

		self.model.compile(loss='binary_crossentropy',
					optimizer='adam')
		print "DONE"
		
		print "[CONC-EXTRACTOR] Loading/Compiling relation extractor model...",
		sys.stdout.flush()
		json_model_relation = open("model_rel/keras_model.json", "r")
		model_relation = json_model_relation.read()
		self.model_relation = model_from_json(model_relation)
		self.model_relation.load_weights("model_rel/model_weights.hd")

		self.model_relation.compile(loss='binary_crossentropy',
					optimizer='adam')
		print "DONE"
		
		## Make sure the socket does not already exist
		try:
			os.unlink(self.server_address)
		except OSError:
			if os.path.exists(self.server_address):
				raise
		
		## Create a UDS socket
		sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
		sock.bind(self.server_address)
		
		## Listen for incoming connections
		sock.listen(1)

		while True:
			conn, address = sock.accept()
			conn.setblocking(0)
			#print "Got connection"
			extract_relation = False
			try:
				ser_input_tensor = ''
				while True:
					try:
						ser_input_tensor += conn.recv(4096)
					except:
						## We first send a header which specifies what to extract
						#if ser_input_tensor == "RELATION":
						#	extract_relation = True
						#	ser_input_tensor = ''
						#elif ser_input_tensor == "CONCEPTS":
						#	ser_input_tensor = ''
						if ser_input_tensor != '':
							break
				#print "Got input tensor"
				input_tensor = pickle.loads(ser_input_tensor)
				
				#if extract_relation:
				relation = self._extract_relation(input_tensor)
				#print "[server-CONC-EXTRACTOR] extracted: ", relation
				conn.sendall(pickle.dumps(relation))
				#else:
				c1, c2 = self._extract_concepts(input_tensor)
				#print "[server-CONC-EXTRACTOR] extracted: ", c1, c2
				conn.sendall(pickle.dumps(c1))
				conn.sendall(pickle.dumps(c2))
			finally:
				conn.close()
    
	def extract(self, sentence):
		## Create client socket
		sock = self._create_socket()

		## Prepare input tensor
		input_tensor = self._prepare_input_tensor(sentence)

		## Send input tensor to server
		ser_input_tensor = pickle.dumps(input_tensor)
		try:
			## Get concepts
			#sock.sendall("CONCEPTS")
			#print >>sys.stderr, 'sending "%s"' % sentence
			sock.sendall(ser_input_tensor)

			ser_rel = sock.recv(4096)
			res_rel = pickle.loads(ser_rel)
			print >>sys.stderr, 'received "%s"' % res_rel

			ser_c1 = sock.recv(4096)
			res_c1 = pickle.loads(ser_c1)
			print >>sys.stderr, 'received "%s"' % res_c1
			ser_c2 = sock.recv(4096)
			res_c2 = pickle.loads(ser_c2)
			print >>sys.stderr, 'received "%s"' % res_c2
			
		finally:
			#print >>sys.stderr, 'closing socket'
			sock.close()

		# no concept found in the sentence (wrong sentence?)
		if max(res_c1 + res_c2) == 0:
			return '', '', '', '', ''

		## Cleaning of c1 and c2
		bn_ids = bn.disambiguate(sentence)
		
		#! if there is no IDS for c1 and c2, we disambiguate only the corresponding substring
		includes_c1 = False
		includes_c2 = False
		for bn_id in bn_ids:
			tmp_incl_c1 = False
			tmp_incl_c2 = False
			for idx in range(bn_id["tokenFragment"]["start"], bn_id["tokenFragment"]["end"]+1):
				if res_c1[idx]:
					tmp_incl_c1 = True
				if res_c2[idx]:
					tmp_incl_c2 = True
			# remove ids which belongs to both c1 and c2...because prevent to split the concepts
			if tmp_incl_c1 and tmp_incl_c2:
				bn_ids.remove(bn_id)
			includes_c1 = tmp_incl_c1 or includes_c1
			includes_c2 = tmp_incl_c2 or includes_c2
			if includes_c1 and includes_c2:
				break
		
		if includes_c1:
			c1_ids, c1_str = self.assign_ids_to_sentence(self.tokenized, res_c1, bn_ids)
		else:
			c1_substring = [tok for i, tok in enumerate(self.tokenized) if res_c1[i]]
			# disambiguate again only c1
			c1_bn_ids = bn.disambiguate(' '.join(c1_substring))
			c1_ids, c1_str = self.assign_ids_to_sentence(c1_substring, [1]*len(c1_substring), c1_bn_ids)
		if includes_c2:
			c2_ids, c2_str = self.assign_ids_to_sentence(self.tokenized, res_c2, bn_ids)
		elif max(res_c2) > 0:
			c2_substring = [tok for i, tok in enumerate(self.tokenized) if res_c2[i]]
			# disambiguate again only c2
			c2_bn_ids = bn.disambiguate(' '.join(c2_substring))
			c2_ids, c2_str = self.assign_ids_to_sentence(c2_substring, [1]*len(c2_substring), c2_bn_ids)
		else:
			c2_ids = []
			c2_str = ''

		## Build relation string
		relation = ''
		rel_indexes = np.where(res_rel == 1)[0]
		if len(rel_indexes) > 0:
			for k, v in relation_extractor.RELATIONS.iteritems():
				if v == rel_indexes[0]:
					relation = k
		
		return c1_str, c1_ids, c2_str, c2_ids, relation
	
	def assign_ids_to_sentence(self, tok_sentence, mask, bn_ids):
		assigned_ids = []
		for i, token in enumerate(tok_sentence):
			if  mask[i]:
				for bn_id in bn_ids:
					if bn_id["tokenFragment"]["start"] <= i and bn_id["tokenFragment"]["end"] >= i:
						assigned_ids.append(bn_id)
		
		assigned_ids = enrich.removeOverlappingIDS(assigned_ids)

		## Build concept strings
		string = ''
		for i, assigned_id in enumerate(assigned_ids):
			if i > 0:
				btw_start = assigned_ids[i-1]["tokenFragment"]["end"] + 1
				btw_end = assigned_ids[i]["tokenFragment"]["start"]
				string += ' '.join(tok_sentence[btw_start:btw_end]) + " "
			start = assigned_id["tokenFragment"]["start"]
			end = assigned_id["tokenFragment"]["end"] + 1
			string += ' '.join(tok_sentence[start:end]) + " "
		
		return assigned_ids, string

	def _create_socket(self):
		# Create a UDS socket
		sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)

		## Connect the socket to the port where the server is listening
		#print >>sys.stderr, 'connecting to %s' % self.server_address
		try:
			sock.connect(self.server_address)
		except socket.error, msg:
			#print >>sys.stderr, msg
			sys.exit(1)

		return sock
    
	def _prepare_input_tensor(self, sentence):
		
		self.tokenized = nltk.word_tokenize(sentence.lower())

		# Cut to max sentence lenght
		self.tokenized = self.tokenized[:min(len(self.tokenized), concepts_extractor.max_len)]

		input_tensor = np.zeros((1, concepts_extractor.max_len, 301), dtype=np.float64)
		_, tag_ids = concepts_extractor.getPosTag(self.tokenized)

		# Convert with w2v
		for idx in range(concepts_extractor.max_len):
			try:
				input_tensor[0][idx] = np.append(self.word_vectors[self.tokenized[idx]], tag_ids[idx])
			except KeyError:
				input_tensor[0][idx] = np.append(self.random_vector, tag_ids[idx])
			except IndexError:
				input_tensor[0][idx] = np.append(self.random_vector, concepts_extractor.POS_TAGS["."])
		
		return input_tensor


	def _extract_concepts(self, input_tensor):
		# Predict 
		res = self.model.predict(input_tensor)[0]
		
		res_c1 = np.array(np.around(res[:concepts_extractor.max_len]).astype(int))
		res_c2 = np.array(np.around(res[concepts_extractor.max_len:]).astype(int))
		
		return res_c1, res_c2
	
	def _extract_relation(self, input_tensor):
		# Predict
		res = self.model_relation.predict(input_tensor)[0]
		# take max_value index
		max_ind = np.argmax(res)
		res[max_ind] = 1
		res_relation = np.array(np.around(res).astype(int))
		
		return res_relation
		
		
		
