# import NB_bc3
import NB_bc3_reference
import subprocess
import re

# Generate reference summaries
# NB_bc3_reference.main()

# Generate model summaries
# NB_bc3.main()

# Run ROUGE evaluation
rouge_result = subprocess.run(['java', '-jar', 'rouge2-1.2.1.jar'], stdout=subprocess.PIPE)
rouge = rouge_result.stdout.decode('utf-8')

# Decode ROUGE to produce averages
results = {
    'L': [],
    '1': [],
    '2': []
}

for line in rouge.splitlines():
    metric_match = re.search('^ROUGE-(.).*', line)
    average_match = re.search('.*Average_F:(.......).*', line)
    if metric_match:
        results[metric_match.group(1)].append(average_match.group(1))

for metric, data in results.items():
    total = 0
    for point in data:
        total += float(point)
    print('{}: {}'.format(metric, total / len(data)))
