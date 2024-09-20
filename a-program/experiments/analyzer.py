import flgo.experiment.analyzer as fea

def analyze(task, option):
    records = fea.load_records(task, ['MyAlgorithm'], option)
    for record in records:
        print("Algorithm:", record.algorithm)
        print("Task:", record.task)
        print("Options:", record.option)
        print("Logs:")
        for key, value in record.log.items():
            print(f"  {key}: {value}")
        print("-" * 40)