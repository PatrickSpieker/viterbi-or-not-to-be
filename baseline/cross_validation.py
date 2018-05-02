import xml.etree.ElementTree as ET
import random
import os

with  open('data/full/corpus.FULL.xml', 'r') as corpus:
    tree = ET.parse(corpus)
    root = tree.getroot()
    threads = list(root.findall('thread'))

    for i in range(5):
        random.shuffle(threads)
        validation_set = threads[:8]
        training_set = threads[8:]

        os.makedirs('data/cross_{}/'.format(i))

        # Add annotations -- should be the full annotations
        with open('data/cross_{}/annotation.train.xml'.format(i), 'w') as output, open('data/full/annotation.FULL.xml', 'r') as annotations:
            for line in annotations:
                output.write(line)
        with open('data/cross_{}/annotation.val.xml'.format(i), 'w') as output, open('data/full/annotation.FULL.xml', 'r') as annotations:
            for line in annotations:
                output.write(line)

        # Add corpuses
        with open('data/cross_{}/corpus.train.xml'.format(i), 'wb') as output:
            output.write(b'<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n')
            output.write(b'<root>')
            for thread in training_set:
                output.write(ET.tostring(thread))
            output.write(b'</root>')

        with open('data/cross_{}/corpus.val.xml'.format(i), 'wb') as output:
            output.write(b'<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n')
            output.write(b'<root>')
            for thread in validation_set:
                output.write(ET.tostring(thread))
            output.write(b'</root>')
            

