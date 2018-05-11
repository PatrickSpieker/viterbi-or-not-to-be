import subprocess
import re
import glob
import os
import pdb

class Evaluation:
    def __init__(self, metrics, debug, examples):
        self.metrics = metrics
        self.debug = debug
        self.examples = examples

    def rouge_evaluation(self):
        # Run ROUGE evaluation
        rouge_result = subprocess.run(['java', '-jar', 'rouge2-1.2.1.jar'], cwd='evaluation', stdout=subprocess.PIPE)
        rouge = rouge_result.stdout.decode('utf-8')

        if self.debug:
            print(rouge)

        if self.examples:
            for f in glob.glob('output/system/*.txt'):
                print('=== SYSTEM-GENERATED EXAMPLE LOCATED AT {} =='.format(f))
                with open(f, 'r') as system_file:
                    print(system_file.read())

        results = {}

        for metric in self.metrics:
            results[metric] = []

        for line in rouge.splitlines():
            metric_match = re.search('^ROUGE-(.).*', line)
            average_match = re.search('.*Average_F:(.......).*', line)
            if metric_match:
                results[metric_match.group(1)].append(average_match.group(1))

        #  Generate sums
        result = ''
        for metric, data in results.items():
            total = 0
            for point in data:
                total += float(point)
            result += '{}: {}\n'.format(metric, total / len(data))

        print(result)