# =====================================================================================
# 최적화된 PEG 분석 시스템 사용 예시
# =====================================================================================

import json
import requests
from typing import Dict, List

def call_optimized_analysis_api(request_data: Dict) -> Dict:
    """최적화된 분석 API 호출 예시"""
    
    # API 엔드포인트 (실제 환경에 맞게 수정)
    api_url = "http://localhost:8000/analyze_cells_optimized"
    
    try:
        response = requests.post(api_url, json=request_data, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 호출 실패: {e}")
        return {"status": "error", "message": str(e)}

def example_usage():
    """사용 예시"""
    
    # 1. 간단한 멀티셀 분석 요청
    simple_request = {
        "cell_names": ["Cell77", "Cell78", "Cell79", "Cell80"],
        "peg_metrics": ["airmacdlthruavg(Kbps)", "airmaculthruavg(Kbps)"],
        "period1": "N-1",
        "period2": "N",
        "threshold": 30.0
    }
    
    print("=== 간단한 멀티셀 분석 요청 ===")
    result = call_optimized_analysis_api(simple_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 2. 대규모 셀 분석 요청
    large_scale_request = {
        "cell_names": [f"Cell{i}" for i in range(1, 101)],  # 100개 셀
        "peg_metrics": [
            "airmacdlthruavg(Kbps)",
            "airmaculthruavg(Kbps)", 
            "airmacdlpacketlossrate(%)",
            "airmaculpacketlossrate(%)"
        ],
        "period1": "N-7",
        "period2": "N",
        "threshold": 25.0
    }
    
    print("\n=== 대규모 셀 분석 요청 ===")
    result = call_optimized_analysis_api(large_scale_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 3. 장기 트렌드 분석 요청
    trend_analysis_request = {
        "cell_names": ["Cell77", "Cell78", "Cell79"],
        "peg_metrics": ["airmacdlthruavg(Kbps)"],
        "period1": "N-30",
        "period2": "N",
        "threshold": 20.0
    }
    
    print("\n=== 장기 트렌드 분석 요청 ===")
    result = call_optimized_analysis_api(trend_analysis_request)
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    example_usage()