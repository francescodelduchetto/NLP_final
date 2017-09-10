import urllib, urllib2, json, re, requests
from pprint import pprint


bn_key  = 'e9f3fe01-6f53-4f49-829f-8dbaf49af6b9'

def disambiguate(sentence):
	bn_disambiguate_url = 'https://babelfy.io/v1/disambiguate'
	
	request = bn_disambiguate_url + '?' + 'text=' + urllib.quote(sentence.encode('utf8')) + '&lang=EN' + '&th=.0' + '&key=' + bn_key
	request = request.encode('utf-8')
	response = urllib2.urlopen(request)
	
	bn_ids = json.load(response)
	
	return bn_ids
	
def disambiguateWithoutThresold(sentence):
	bn_disambiguate_url = 'https://babelfy.io/v1/disambiguate'
	
	request = bn_disambiguate_url + '?' + 'text=' + urllib.quote(sentence.encode('utf8')) + '&lang=EN' + '&key=' + bn_key
	request = request.encode('utf-8')
	response = urllib2.urlopen(request)
	
	bn_ids = json.load(response)
	
	return bn_ids

def getMainSenseOfWord(word):
	bn_get_id_url = 'https://babelnet.io/v4/getSenses'

	request = bn_get_id_url + '?' + 'word=' + word + '&lang=EN&key=' + bn_key
	request = request.encode('utf-8')
	response = urllib2.urlopen(request)
	
	bn_info = json.load(response)
	
	if len(bn_info) > 0:
		bn_id = bn_info[0]["synsetID"]["id"]
		return bn_id

	return None

def getMainSenseOfID(bn_id):
	bn_get_lemma_url = 'https://babelnet.io/v4/getSynset'
	
	request = bn_get_lemma_url + '?' + 'id=' + bn_id + '&key=' + bn_key
	request = request.encode('utf-8')
	response = urllib2.urlopen(request)
	bn_info = json.load(response)
	
	main_sense = bn_info["mainSense"]
	
	# remove #X
	matches = re.findall(r'(#[a-zA-Z1-9]{1})', main_sense)
	for match in matches:
		main_sense = main_sense.replace(match, '')
	
	main_sense = main_sense.replace('_', ' ')
	
	return main_sense


def saveTuplesToServer(tuples):
	url = "http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/add_items?key=" + bn_key
	
	response = requests.post(url, json=tuples)
	
	#pprint(tuples)
	print "Tuples saved to server, response: " + response.text

