import os
import xml.etree.ElementTree as ET

def main():
    reformat_gnue('../data/gnue/full/raw/corpus', '../data/gnue/full/raw/annotation')

def reformat_gnue(corpus_dir, annotations_dir):
    global_thread_index = 0
    for filename in os.listdir(annotations_dir):
        anno_file = os.path.join(annotations_dir, filename)
        tree = ET.parse(anno_file)
        root = tree.getroot()
        for section in root.findall('section'):
            startdate = section.attrib['startdate'].split(' ')
            if 'enddate' in section.attrib:
                enddate = section.attrib['enddate'].split(' ')
            else:
                enddate = startdate
            
            start_day = int(startdate[0])
            end_day = int(enddate[0])
            start_mo = month_to_num(startdate[1])
            end_mo = month_to_num(enddate[1])
            start_year = int(startdate[2])
            end_year = int(enddate[2])
            
            # Create new corpus file containing chunked conversation
            corpus_thread_file = open(os.path.join('../data/gnue/full/processed/corpus', 'corpus-' + str(global_thread_index) + '.txt'), 'wb')
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
                        with open(os.path.join(corpus_dir, date_str), 'rb') as f:
                            contents = f.read()
                            corpus_thread_file.write(contents)

            # Create new annotation file with the corresponding summary xml
            anno_output_path = os.path.join('../data/gnue/full/processed/annotation', 'annotation-' + str(global_thread_index) + '.txt')
            newtree = ET.ElementTree(section)
            newtree.write(anno_output_path)

            global_thread_index += 1

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
