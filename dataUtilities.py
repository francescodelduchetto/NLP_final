import json, pickle, urllib2, sys, re

def searchDatabaseUpdates():
	number_from_address = "http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/items_number_from?"
	items_from_address = "http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/items_from?"
	kb_token = "e9f3fe01-6f53-4f49-829f-8dbaf49af6b9"
	
	## from where we have to start download?
	items_from = 0
	try:
		with open("data/items_from", "r") as if_file:
			items_from = int(if_file.read())
			if_file.close()
	except IOError:
		pass
	print "[DATASET-GEN] Items from ", str(items_from), 
	sys.stdout.flush()
	
	## See how many items are missing
	response = urllib2.urlopen(number_from_address + "id=" + str(items_from) + "&key=" + kb_token)
	missing_items = long(json.load(response))
	print ", missing " + str(missing_items) 

	## Request and save them in the local database
	with open("data/local_kb", "ab") as kb_file:
		starting_offset = kb_file.tell()
		total_items = items_from
		while missing_items > 0:
			try:
				print "[DATASET-GEN] Retrieving items from KB server...",
				sys.stdout.flush()
				response = urllib2.urlopen(items_from_address + "id=" + str(items_from) + "&key=" + kb_token)
				kb_items = json.load(response)
				n_items = len(kb_items)
				print "DONE",
			except:
				print "[DATASET-GEN] Problems connecting with KB server, consider running with --no_update"
				break
			# save in local database
			for item in kb_items:
				pickle.dump(item, kb_file, protocol=pickle.HIGHEST_PROTOCOL)
			missing_items -= n_items
			total_items += n_items
			items_from += n_items
			
			print " missing " + str(missing_items) + ", total " + str(total_items)
			
		kb_file.close()
		
		# save number of items
		with open("data/items_from", "w") as if_file:
			if_file.write(str(total_items))
			if_file.close()
		
		# update statistics with new items
		updateStatisticsFrom(starting_offset)

# update the current statistics 
# starting from items after the offset in the local kb 
def updateStatisticsFrom(starting_offset):
	print "[STAT-EXTRACTOR] Updating statistic"
	concepts_to_relations_filePath = "data/concepts_to_relations"
	relations_to_concepts_filePath = "data/relations_to_concepts"
	domains_to_concepts_filePath = "data/domain_to_concepts"
	dataset_file_path = "data/local_kb"
	
	concepts_to_relations = {}
	relations_to_concepts = {}
	domains_to_concepts = {}
	concepts_to_domain = {}
	## read concepts_to_relations
	try:
		with open(concepts_to_relations_filePath, 'rb') as f:
			concepts_to_relations = pickle.load(f)
			f.close()
	except IOError:
		with open(concepts_to_relations_filePath, "wb") as if_file:
			if_file.close()
	## read relations_to_concepts
	try:
		with open(relations_to_concepts_filePath, 'rb') as f:
			relations_to_concepts = pickle.load(f)
			f.close()
	except IOError:
		with open(relations_to_concepts_filePath, "wb") as if_file:
			if_file.close()
	## read domain_to_concepts
	try:
		with open(domains_to_concepts_filePath, 'rb') as f:
			domains_to_concepts = pickle.load(f)
			f.close()
	except IOError:
		with open(domains_to_concepts_filePath, "wb") as if_file:
			if_file.close()
	## read concepts_to_domain list (not updated by this)
	with open("chatbot_maps/BabelDomains_full/BabelDomains/babeldomains_babelnet.txt") as bn_file:
		concepts_domain = bn_file.read().lower().splitlines()
		for cd in concepts_domain:
			cd = cd.split("\t")
			concepts_to_domain.update({
					cd[0] : cd[1]
				})
	
	## updating with new items
	print "[STAT-EXTRACTOR] Reading items from "+ dataset_file_path + "...",
	sys.stdout.flush()
	with open(dataset_file_path, 'rb') as dataset_file:
		# set the starting_offset
		dataset_file.seek(starting_offset)
		while True:
			try:
				file_offset = dataset_file.tell()
				item  = pickle.load(dataset_file)
				concept = item["c1"]
				bn_ids = re.findall(r'(bn:[A-Za-z0-9]+)', concept)
				#print bn_ids
				# some bn are not directly associated to relation -> ignore pair TODO: handle this
				if len(bn_ids) == 1:
					bn_id = bn_ids[0]
					relation = item["relation"].upper()
					# updating concepts_to_relations
					if not bn_id in concepts_to_relations:
						concepts_to_relations.update({
								bn_id: {}
							})
					if not relation in concepts_to_relations[bn_id]:
						concepts_to_relations[bn_id].update({
								relation: 1
							})
					else:
						concepts_to_relations[bn_id][relation] += 1
					# updating relations_to_concepts
					if not relation in relations_to_concepts:
						relations_to_concepts.update({
								relation: {}
							})
					if not bn_id in relations_to_concepts[relation]:
						relations_to_concepts[relation].update({
								bn_id: [file_offset]
							})
					else:
						relations_to_concepts[relation][bn_id].append(file_offset)
					# updating domains_to_concepts
					if bn_id in concepts_to_domain:
						domain_of_concept = concepts_to_domain[bn_id]
						#check for domain_of_concept in domains_to_concepts
						if domain_of_concept not in domains_to_concepts:
							domains_to_concepts.update({
									domain_of_concept : {}
								})
						if bn_id not in domains_to_concepts[domain_of_concept]:
							domains_to_concepts[domain_of_concept].update({
									bn_id : 1
								})
						else:
							domains_to_concepts[domain_of_concept][bn_id] += 1
			except EOFError:
				break
	print "DONE"

	# Save dicts back in files
	with open(concepts_to_relations_filePath, 'wb') as f:
		pickle.dump(concepts_to_relations, f, pickle.HIGHEST_PROTOCOL)
		f.close()
		print "[STAT-EXTRACTOR] Saved file '" + concepts_to_relations_filePath + "'"
	with open(relations_to_concepts_filePath, 'wb') as f:
		pickle.dump(relations_to_concepts, f, pickle.HIGHEST_PROTOCOL)
		f.close()
		print "[STAT-EXTRACTOR] Saved file '" + relations_to_concepts_filePath + "'"
	with open(domains_to_concepts_filePath, 'wb') as f:
		pickle.dump(domains_to_concepts, f, pickle.HIGHEST_PROTOCOL)
		f.close()
		print "[STAT-EXTRACTOR] Saved file '" + domains_to_concepts_filePath + "'"
