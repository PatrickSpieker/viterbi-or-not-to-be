import os
import xml.etree.ElementTree as ET

def main():
    parse_gnue('data/gnue/tiny/corpus', 'data/gnue/tiny/annotation')

def parse_gnue(corpus_dir, annotations_dir):
    date_to_quotes = {}
    convos = []
    convo_labels = []

    for filename in os.listdir(annotations_dir):
        anno_file = os.path.join(annotations_dir, filename)
        tree = ET.parse(anno_file)
        root = tree.getroot()
        for section in root.findall('section'):
            startdate = section.attrib['startdate'].split(' ')
            enddate = section.attrib['enddate'].split(' ')
            
            start_day = int(startdate[0])
            end_day = int(enddate[0])
            start_mo = month_to_num(startdate[1])
            end_mo = month_to_num(enddate[1])
            start_year = int(startdate[2])
            end_year = int(enddate[2])
            
            # Map chatlog date to selected quotes
            for summ in section.findall('p'):
                for year in range(start_year, end_year + 1):
                    for mo in range(start_mo, end_mo + 1):
                        for day in range(start_day, end_day + 1):
                            # Build date string (chatlog file name)
                            year_str = str(year)
                            mo_str = str(mo)
                            day_str = str(day)
                            if mo < 10:
                                mo_str = '0' + mo_str
                            if day < 10:
                                day_str = '0' + day_str
                            date_str = year_str + '-' + mo_str + '-' + day_str
                            
                            # Build up date to quotes mapping
                            if date_str not in date_to_quotes:
                                date_to_quotes[date_str] = set()
                            for quote in summ.findall('quote'):
                                date_to_quotes[date_str].add(quote.text.replace('\n', ''))
    
    for filename in os.listdir(corpus_dir):
        date_convo = []
        date_convo_labels = []
        corpus_file = open(os.path.join(corpus_dir, filename), 'r')
        quotes = date_to_quotes[filename]
        for line in corpus_file.readlines():
            line = line.replace('\n', '')
            date_convo.append(line)
            for q in quotes:
                if q.replace(',', '') in line.replace(',', ''):
                    date_convo_labels.append(1)
                else:
                    date_convo_labels.append(0)
        convos.append(date_convo)
        convo_labels.append(date_convo_labels)
    
    return convos, convo_labels

def month_to_num(month):
    return {
        'Jan' : 1,
        'Feb' : 2,
        'Mar' : 3,
        'Apr' : 4,
        'May' : 5,
        'Jun' : 6,
        'Jul' : 7,
        'Aug' : 8,
        'Sep' : 9, 
        'Oct' : 10,
        'Nov' : 11,
        'Dec' : 12
    }[month]

if __name__ == '__main__':
    main()
