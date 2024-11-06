import json

def read_file(input_file):
    return [x.rstrip() for x in open(input_file, encoding="utf-8").readlines()]

def read_jsonl(input_file):
    return [json.loads(x) for x in open(input_file, mode='r', encoding='utf-8').readlines()]

def read_jsonl_key(input_file, key):
    return [json.loads(x)[key] for x in open(input_file, mode='r', encoding='utf-8').readlines()]

def write_jsonl(lines, output_file):
    with open(output_file, mode='w', encoding='utf-8') as f:
        for line in lines:
            json.dump(line, f, ensure_ascii=False)
            f.write('\n')

def write_file(lines, output_file):
	lines = [str(line)+"\n" for line in lines]
	with open(output_file, "w", encoding="utf-8") as f:
		f.writelines(lines)
