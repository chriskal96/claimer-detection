from bs4 import BeautifulSoup
import json
import zipfile

import os

annotated_dataset = []

for root, dirs, files in os.walk('annotation'):
	for dir in dirs:
		if dir.endswith('.txt'):
			with zipfile.ZipFile('annotation/' + dir + '/eleni.koutli.zip', 'r') as zip_ref:
				zip_ref.extractall('annotation/' + dir + '/eleni.koutli')
			with zipfile.ZipFile('annotation/' + dir + '/vivian.stamou.zip', 'r') as zip_ref:
				zip_ref.extractall('annotation/' + dir + '/vivian.stamou')
			with zipfile.ZipFile('annotation/' + dir + '/panos.kounoudis.zip', 'r') as zip_ref:
				zip_ref.extractall('annotation/' + dir + '/panos.kounoudis')

			with open('annotation/' + dir + '/eleni.koutli/eleni.koutli.xmi', 'r') as f:
				data = f.read()
			Bs_data = BeautifulSoup(data, "xml")
			sofa = Bs_data.find('cas:Sofa')
			text = sofa.get('sofaString')
			text_start = text.find("\n\n__TEXT__\n\n") + len("\n\n__TEXT__\n\n")
			clean_text = text[text_start:]
			detection = Bs_data.find_all('custom:ClaimDetectionSchema')
			if len(detection) == 0:
				with open('annotation/' + dir + '/vivian.stamou/vivian.stamou.xmi', 'r') as f:
					data = f.read()
				Bs_data = BeautifulSoup(data, "xml")
				sofa = Bs_data.find('cas:Sofa')
				text = sofa.get('sofaString')
				detection = Bs_data.find_all('custom:ClaimDetectionSchema')
			if len(detection) == 0:
				with open('annotation/' + dir + '/panos.kounoudis/panos.kounoudis.xmi', 'r') as f:
					data = f.read()
				Bs_data = BeautifulSoup(data, "xml")
				sofa = Bs_data.find('cas:Sofa')
				text = sofa.get('sofaString')
				detection = Bs_data.find_all('custom:ClaimDetectionSchema')
			start_index = text.index('__TITLE__\n\n') + len('__TITLE__\n\n')
			end_index = text.index('\n\n', start_index)
			title = text[start_index:end_index]

			start_index = text.index('__DOMAIN__\n\nDomain: ') + len('__DOMAIN__\n\nDomain: ')
			end_index = text.index('\n\n', start_index)
			domain = text[start_index:end_index]

			claim = Bs_data.find('custom:ClaimDetectionSchema', {'Label': 'Claim'})
			if claim:
				claim_begin = int(claim.get('begin'))
				claim_end = int(claim.get('end'))
				question = 'Who said that ' + text[claim_begin:claim_end]
			else:
				question = None
			author = Bs_data.find('custom:ClaimDetectionSchema', {'Label': 'Author'})
			claimer = Bs_data.find('custom:ClaimDetectionSchema', {'Label': 'Claimer'})
			topic = Bs_data.find('custom:ClaimDetectionSchema', {'Label': 'Topic'})
			reference = Bs_data.find('custom:ClaimDetectionSchema', {'Label': 'Reference'})
			if topic:
				topic_begin = int(topic.get('begin'))
				topic_end = int(topic.get('end'))
				topic = text[topic_begin:topic_end]
			else:
				topic = None
			if reference:
				reference_begin = int(reference.get('begin'))
				reference_end = int(reference.get('end'))
				reference = text[reference_begin:reference_end]
			else:
				reference = None
			if claimer:
				claimer_begin = int(claimer.get('begin'))
				claimer_end = int(claimer.get('end'))
				answer = text[claimer_begin:claimer_end]
				answer_start = claimer_begin
			elif author:
				author_begin = int(author.get('begin'))
				author_end = int(author.get('end'))
				# answer = text[author_begin:author_end]
				answer = 'Author'
				answer_start = author_begin
			else:
				answer = None
				answer_start = None
			if claim:
				annotated_dataset.append({
					'title': title,
					'domain': domain,
					'text': text,
					'clean_text': clean_text,
					'question': question,
					'answer': answer,
					'answer_start': answer_start,
					'topic': topic,
					'reference': reference,
				})

with open('annotation/annotated_dataset_clean.json', 'w') as outfile:
	json.dump(annotated_dataset, outfile)
