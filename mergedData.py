import json
from pathlib import Path

# 디렉토리 설정
directory_path = Path('/Users/choizeus/smu/capstone/capstone_hardware/chatBot/colabTest/최종데이터')  # JSON 파일이 저장된 디렉토리
output_file_path = directory_path / 'all.json'

def process_file(file_path):
    """ 파일을 읽고 JSON 객체를 파싱하여 리스트로 반환 """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        if not content:  # 파일이 비어 있는 경우
            print(f"파일 {file_path}이 비어 있습니다.")
            return []
        try:
            # 파일 내용을 JSON으로 로드 (단일 객체 예상)
            return [json.loads(content)]
        except json.JSONDecodeError:
            # JSON 오류 처리: 여러 객체를 감지하고 분리
            objects = []
            decoder = json.JSONDecoder()
            idx = 0
            try:
                while idx < len(content):
                    obj, end_idx = decoder.raw_decode(content[idx:])
                    objects.append(obj)
                    idx += end_idx
                return objects
            except json.JSONDecodeError as e:
                print(f"파일 {file_path}에서 JSON 파싱 오류: {e}")
                return []

# 모든 파일을 읽고 새 파일에 내용을 쓰기
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    for file_name in directory_path.glob('*.json'):
        objects = process_file(file_name)
        for obj in objects:
            json.dump(obj, outfile, ensure_ascii=False)
            outfile.write('\n')

print(f"모든 파일이 {output_file_path}로 합쳐졌습니다.")
