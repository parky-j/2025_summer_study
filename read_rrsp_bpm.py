import pandas as pd
import os

def read_rrsp_bpm_from_csv(file_path):
    """
    CSV 파일에서 rrsp_bpm_final 컬럼만 읽어서 리스트로 반환
    """
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # rrsp_bpm_final 컬럼만 추출하여 리스트로 변환
        rrsp_bpm_list = df['rrsp_bpm_final'].tolist()
        
        return rrsp_bpm_list
    
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    except KeyError:
        print(f"rrsp_bpm_final 컬럼이 파일에 없습니다: {file_path}")
        return []
    except Exception as e:
        print(f"파일 읽기 중 오류 발생: {e}")
        return []

def read_all_rrsp_bpm_from_directory(directory_path):
    """
    디렉토리 내의 모든 CSV 파일에서 rrsp_bpm_final 컬럼을 읽어서 딕셔너리로 반환
    """
    result = {}
    
    try:
        # 디렉토리 내의 모든 CSV 파일 찾기
        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                rrsp_bpm_list = read_rrsp_bpm_from_csv(file_path)
                
                if rrsp_bpm_list:
                    result[filename] = rrsp_bpm_list
                    print(f"{filename}: {len(rrsp_bpm_list)}개 데이터 로드됨")
    
    except Exception as e:
        print(f"디렉토리 읽기 중 오류 발생: {e}")
    
    return result

# 사용 예시
if __name__ == "__main__":
    # 단일 파일 읽기
    file_path = "split_samples/1-1/sample_00330.csv"
    rrsp_bpm_list = read_rrsp_bpm_from_csv(file_path)
    
    print(f"단일 파일에서 읽은 rrsp_bpm_final 데이터:")
    print(f"데이터 개수: {len(rrsp_bpm_list)}")
    print(f"처음 10개: {rrsp_bpm_list[:10]}")
    print(f"마지막 10개: {rrsp_bpm_list[-10:]}")
    
    print("\n" + "="*50 + "\n")
    
    # 디렉토리 내 모든 파일 읽기
    directory_path = "split_samples/1-1"
    all_data = read_all_rrsp_bpm_from_directory(directory_path)
    
    print(f"\n디렉토리에서 읽은 파일 수: {len(all_data)}")
    
    # 첫 번째 파일의 데이터 샘플 출력
    if all_data:
        first_file = list(all_data.keys())[0]
        print(f"\n{first_file} 파일의 데이터 샘플:")
        print(f"처음 10개: {all_data[first_file][:10]}")
        print(f"마지막 10개: {all_data[first_file][-10:]}") 