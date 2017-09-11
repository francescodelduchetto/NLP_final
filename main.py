import time, thread, sys
import pickle
import telepot
from pprint import pprint
from telepot.loop import MessageLoop
from collections import deque
import re
import nltk

import sentenceExtractor
import BNUtilities as bn
import enrichingUtilities as enrich
import dataUtilities as data

bot = None

extractor = None

chat_ids = []
chat_messages = {}

relations_concepts_mapping = {}

# questions saved as <subject-relation> pairs
#questions_to_ask = []

def message_handler(msg):
	content_type, chat_type, chat_id = telepot.glance(msg)
	# Print the message arrived 
	print "[BOT] ("+ str(chat_id)+ ") >> '"+ msg["text"] +"'"
	
	if chat_id not in chat_ids:
		chat_ids.append(chat_id)
		try:
			# Initialize a new thread for the conversation
			thread.start_new_thread(start_session, (chat_id, msg["from"]["first_name"]) )
			print "[BOT] New conversation initialized with", chat_id
		except :
			print "[BOT] ERROR STARTING A NEW THREAD FOR THE CONVERSATION", chat_id
	else:
		# Save the received message in the corresponding buffer
		chat_messages[chat_id].append(msg)
		#send_message(chat_id, "Yes")

# Set-up the conversation with the new user
def start_session(chat_id, user_name):
	# Initializa the queue to hold the messages of the conversation
	chat_messages.update({chat_id : deque(maxlen=10)})
	
	message_queue = chat_messages[chat_id]
	
	### Start conversation
	welcome_message = "Hello " + user_name + ", I'm "+ bot.getMe()["first_name"] + " ! I can't wait to have a meaningful conversation with you !"
	send_message(chat_id, welcome_message)
	#time.sleep(3)
	
	### Main conversational loop
	while True:
		domain = None
		
		## Decide the type of interaction
		ask_direction_message = "Commands:\n\t/ask - You ask me a question\n\t/reply - I ask, you reply\n\t/stop - Stop the conversation\n\t/help - Show this help"
		send_message(chat_id, ask_direction_message)
		
		## select and start the session
		while True:
			msg = wait_for_answer(message_queue)
			if msg["text"].lower() == "/ask":
				send_message(chat_id, "Ok !")
				querying_session(chat_id, domain)
			elif msg["text"].lower() == "/reply":
				send_message(chat_id, "Ok !")
				enriching_session(chat_id, domain)
			elif msg["text"].lower() == "/help":
				send_message(chat_id, ask_direction_message)
			elif msg["text"].lower() == "/stop":
				stop_session(chat_id)
				send_message(chat_id, "See you again !")
				return

#>>>>> Stop session
def stop_session(chat_id):
	# save stuff
	enrich.serializeStuff()
	
	# remove the chat id
	chat_ids.remove(chat_id)
	
	print "[BOT] (" + str(chat_id) + ") X Conversation closed"

#>>>>> The robot asks questions
def enriching_session(chat_id, domain, relation=None):
	message_queue = chat_messages[chat_id]
	
	instructions_message = "Answer to question on a specific topic. \n \
			/dunno - if you don't know the answer\n \
			/nope - if the answer is non-sense or not related to the topic"
	send_message(chat_id, instructions_message)
	#time.sleep(2)
	while not domain:
		## Decide a domain
		ask_domain_message = "Select the topic: "
		for domain in enrich.getDomainsToPropose():
			ask_domain_message += "\n/"+domain.replace(",", "").replace(" ", "_")
		send_message(chat_id, ask_domain_message)
		domain_answer = wait_for_answer(message_queue)
		domain = detect_domain(domain_answer["text"])
		send_message(chat_id, "Ok")

	questions_to_ask = enrich.getQuestionToAsk(domain)

	concept_1 = questions_to_ask["concept"]
	relation = questions_to_ask["relation"]
	lemma = bn.getMainSenseOfID(concept_1)
	
	question_pattern = enrich.getQuestionPattern(relation)
	
	question = question_pattern.replace("X", lemma)
	#question = "What is the " + relation + " of " + lemma + "?"
	send_message(chat_id, question)

	answer = wait_for_answer(message_queue)
	
	answer_text = answer["text"]
	
	# Question nonsense 
	if answer_text.lower() in ["/nope", "nope"]:
		enrich.avoidQuestion(domain, concept_1, relation)
		message = "Ok, got it ! Wanna /reply again ?"
		send_message(chat_id, message)
	# Don't know answer
	elif answer_text.lower() in ["/dunno", "dunno"]:
		message = "Don't worry, type again /reply and try with another one !"
		send_message(chat_id, message)
	else:
		tok_question = nltk.word_tokenize(question)
		tok_answer = nltk.word_tokenize(answer_text)

		for text in [{"text" :question + " " + answer_text, "tok": tok_question + tok_answer, "mask" : [0] * len(tok_question) + [1] * len(tok_answer)},
					{"text" :answer_text, "tok": tok_answer, "mask" : [1] * len(tok_answer)}] :
			answer_ids = bn.disambiguateWithoutThresold(text["text"])
			if not answer_ids:
				answer_ids = bn.disambiguate(text["text"])
			if answer_ids:
				answer_ids, _ = extractor.assign_ids_to_sentence(
						text["tok"], text["mask"], answer_ids
					)

				if answer_ids:
					# save all the remaining 
					concept_2 = " ".join([id["babelSynsetID"] for id in answer_ids])
					
					tuple = { 'question': question, 'answer':answer_text, 'domains' : [domain],
							'relation': relation, 'context': '', 'c1': concept_1, 
							'c2': concept_2 }
					pprint(tuple)

					enrich.saveTuple(tuple)
					
					break
		message = "Ok great !  Wanna /reply again ?"
		send_message(chat_id, message)

#>>>>> The robot answers
def querying_session(chat_id, domain, relation=None):
	message_queue = chat_messages[chat_id]

	send_message(chat_id, "Write your question")

	user_question = wait_for_answer(message_queue)

	c1, c1_ids, c2, c2_ids, relation = extractor.extract(user_question["text"])

	print "The subject is : " + c1 + ", the object : " + c2 + ", the relation : " + relation

	answers = find_answer(c1, c1_ids, c2, c2_ids, relation)

	if answers and len(answers) > 0 :
		message = "The answer is: "
		for i, answer in enumerate(answers):
			if i < len(answers) - 1:
				message += answer + " and "
			else:
				message += answer + "."
		send_message(chat_id, message)
		pprint(answers)
	else:
		message = "I don't know :("
		send_message(chat_id, message)
		if relation and len(c1_ids) > 0:
			enrich.addQuestionToAsk(c1, c1_ids, relation)

	message = "Come on, /ask me another question !"
	send_message(chat_id, message)


## Search in the KB for an answer 
def find_answer(c1, c1_ids, c2, c2_ids, relation):
	if len(c1_ids) == 0 or not relation:
		return None
	else:
		pprint(c1_ids)
		pprint(c2_ids)

	answer_items = {}
	## First search on the temporary dataset
	tmp_tuples = enrich.findTmpTuples([t["babelSynsetID"] for t in c1_ids], relation)
	for tuple in tmp_tuples:
		matches = re.findall(r'(bn:[A-Za-z0-9]+)', tuple["c2"])
		for match in matches:
			if match in answer_items.keys():
				answer_items[match] += 1 
			else:
				answer_items.update({
						match : 1
					})

	## Reading the dataset
	with open("data/local_kb", 'rb') as dataset_file:
		concepts = relations_concepts_mapping[relation]
		
		for c1_id in c1_ids:
			print len(concepts.keys()) 
			for concept in concepts.keys():
				if c1_id["babelSynsetID"] == concept:
					print len(concepts[concept])
					for offset in concepts[concept]:
						dataset_file.seek(offset)
						item = pickle.load(dataset_file)
						# Take only those with one bn_id in c2
						matches = re.findall(r'(bn:[A-Za-z0-9]+)', item["c2"])
						for match in matches:
							if match in answer_items.keys():
								answer_items[match] += 1 
							else:
								answer_items.update({
										match : 1
									})
		dataset_file.close()

	print answer_items
	## distinguish the type of question, whether you have only c1 or c1 and c2
	lemmas = []
	if len(answer_items) > 0:
		if c2 != '' and len(c2_ids) > 0:
			lemma = "No"
			for c2_id in c2_ids:
				for answer_id in answer_items.keys():
					if c2_id["babelSynsetID"] == answer_id:
						lemma = "Yes"
						break
				if lemma == "Yes":
					break
			lemmas.append(lemma)
		else:
			## Take the most frequent only -> bcs if they are a lot it's very slow 
			max_occurrence = max(answer_items.values())
			max_indexes = [i for i, val in enumerate(answer_items.values()) if val == max_occurrence]
			lemmas = [bn.getMainSenseOfID(answer_items.keys()[i]) for i in max_indexes]

	return lemmas


# Detect the domain the user wants to talk about
def detect_domain(message):
	for domain in enrich.getDomainList():
		if message.lower().replace("/", "").replace("_", "") == domain.replace(",", "").replace(" ", ""):
			return domain

	return None


# General func for waiting for an answer in a specific conversation queue 
def wait_for_answer(message_queue):
	while len(message_queue) == 0:
		time.sleep(2)

	return message_queue.pop()


# General func which sends a message to a specific conversation
def send_message(chat_id, msg):
	if bot is not None:
		try:
			bot.sendMessage(chat_id, msg)
			print "[BOT] ("+ str(chat_id) + ") << '"+ msg + "'"
		except:
			e = sys.exc_info()[0]
			print e
			bot.sendMessage(chat_id, msg)
			print "[BOT] ("+ str(chat_id) + ") << '"+ msg + "'"
			pass

def initialize_resources():
	# relation_to_concepts is generated by extract_statistics.py
	relations_concepts_file_path = "data/relations_to_concepts"
	rc_file = open(relations_concepts_file_path, "rb")
	relations_concepts_mapping.update(pickle.load(rc_file))
	rc_file.close()


if __name__ == '__main__':
	update_database = True
	if len(sys.argv) > 1:
		if sys.argv[1] in ["--no_update", "--no_updates"]:
			update_database = False

	extractor = sentenceExtractor.sentenceExtractor()

	# Check if there are new items in the kb
	if update_database:
		print "[INIT] Database check updates..."
		sys.stdout.flush()
		data.searchDatabaseUpdates()
		print "DONE"

	# Initialize all the necessary resources
	print "[INIT] Initializing resources...",
	sys.stdout.flush()
	initialize_resources()
	print "DONE"

	# Initialize my bot
	bot_token = '428273114:AAHiEmdJ6nhfdigGCHCc7LY08IoS9RTFOXI'
	bot = telepot.Bot(bot_token)
	print "[INIT] Initialized bot: ",
	pprint(bot.getMe())


	# Start the MessageHandler thread
	MessageLoop(bot, message_handler).run_as_thread()

	while True:
		time.sleep(20)
		pass
