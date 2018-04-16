# Parses bc3 corpus

import xml.etree.ElementTree as ET

def parseXML(xmlfile):
	tree = ET.parse(xmlfile)
	root = tree.getroot()
	for thread in root:
		print('---------- Thread with name "' + thread[0].text + '" and listno ' + thread[1].text + ' ----------')
		
		for docnum in range(2, len(thread.getchildren())):
			# { Received, From, To, (Cc), Subject, Text }
			for item in thread[docnum]:
				if item.tag == 'Subject':
					print('\n    Email subject: "' + thread[docnum][3].text + '"')
				if item.tag == 'Text':
					for sent in item:
						print('        Sentence id: ' + sent.attrib['id'])
						print('        Sentence: "' + sent.text + '"')
		print('\n')

xmlfile = open('data/corpus-tiny.xml', 'r')
parseXML(xmlfile)

xmlfile.close()
