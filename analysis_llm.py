# =====================================================================================
# PEG 데이터 LLM 기반 심층 분석 및 전문가용 리포트 생성기 (HTML 탭 오류 최종 수정)
# =====================================================================================

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from fastmcp import FastMCP
import datetime
import logging
import io
import base64
import subprocess

# --- 로깅 기본 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 1. FastMCP 서버 인스턴스 생성 ---
mcp = FastMCP(name="PEG LLM 분석기")

# --- 2. 데이터 초기 처리 및 차트 생성 함수 ---
def perform_initial_data_processing(peg_results_df: dict, threshold: float) -> (dict, dict):
    logging.info("데이터 초기 처리 및 차트 생성을 시작합니다.")
    all_reports_data, charts_base64 = [], {}
    for peg_name, df in peg_results_df.items():
        pivot = df.pivot(index='cell_name', columns='period', values='avg_value').fillna(0)
        if 'N-1' not in pivot.columns or 'N' not in pivot.columns:
            continue
        pivot['rate(%)'] = ((pivot['N'] - pivot['N-1']) / pivot['N-1'].replace(0, float('nan'))) * 100
        pivot['anomaly'] = (pivot['rate(%)'].abs() >= threshold)
        pivot.reset_index(inplace=True); pivot['peg'] = peg_name
        all_reports_data.append(pivot[['peg', 'cell_name', 'N-1', 'N', 'rate(%)', 'anomaly']])
        plt.figure(figsize=(8, 5)); pivot[['N-1', 'N']].plot(kind='bar', ax=plt.gca())
        plt.title(f"{peg_name} \n(Period N vs N-1)", fontsize=12); plt.ylabel("Average Value"); plt.xlabel("Cell Name")
        plt.xticks(ticks=range(len(pivot['cell_name'])), labels=pivot['cell_name'], rotation=45, ha='right'); plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        charts_base64[peg_name] = base64.b64encode(buf.read()).decode('utf-8'); plt.close()
        logging.info(f"'{peg_name}'에 대한 비교 차트 생성 완료.")
    if not all_reports_data:
        raise ValueError("분석할 유효한 데이터가 없습니다. 각 PEG에 N-1과 N 기간 데이터가 모두 쌍으로 존재하는지 확인하세요.")
    final_df = pd.concat(all_reports_data).round(2)
    logging.info("모든 PEG 데이터의 초기 처리 및 차트 생성을 완료했습니다.")
    return final_df, charts_base64

# --- 3. LLM 분석을 위한 프롬프트 생성 함수 (한국어 분석용) ---
def create_llm_analysis_prompt(processed_data: pd.DataFrame) -> str:
    logging.info("LLM 분석을 위한 프롬프트 생성을 시작합니다. (분석 언어: 한국어)")
    data_summary_str, cell_name = processed_data.to_string(index=False), processed_data['cell_name'].iloc[0]
    prompt = f"""
    ### **Instruction:**
    You are a world-class expert in 3GPP mobile network optimization... (프롬프트 내용은 이전과 동일)
    ### **Required JSON Output Format:**
    ```json
    {{
      "comprehensive_summary": "...",
      "potential_issues": ["..."],
      "debugging_points": ["..."],
      "detailed_peg_analysis": {{"PEG_NAME_1": "..."}}
    }}
    ```
    """
    logging.info("LLM 프롬프트 생성 완료. (분석 언어: 한국어)")
    return prompt

# --- 4. LLM API 호출 함수 (subprocess + curl) ---
def query_llm(prompt: str) -> dict:
    logging.info("내부 vLLM 모델에 curl을 사용하여 분석을 요청합니다.")
    endpoints, payload = ['http://10.251.204.93:10000', 'http://100.105.188.117:8888'], {"model": "Gemma-3-27B", "messages": [{"role": "user", "content": prompt}], "temperature": 0.2, "max_tokens": 4096}
    json_payload = json.dumps(payload)
    for endpoint in endpoints:
        try:
            logging.info(f"엔드포인트 접속 시도: {endpoint}")
            command = ['curl', f'{endpoint}/v1/chat/completions', '-H', 'Content-Type: application/json', '-d', json_payload, '--max-time', '180']
            process = subprocess.run(command, capture_output=True, check=False, encoding='utf-8', errors='ignore')
            if process.returncode != 0: logging.error(f"'{endpoint}'에 대한 curl 명령어 실행 실패. stderr: {process.stderr.strip()}"); continue
            if not process.stdout: logging.error("LLM 서버로부터 받은 응답(stdout)이 비어있습니다."); continue
            response_json = json.loads(process.stdout)
            if 'error' in response_json: logging.error(f"API에서 에러 응답 수신: {response_json['error']}"); continue
            if 'choices' not in response_json or not response_json['choices']: logging.error(f"LLM 응답에 'choices' 키가 없거나 비어있습니다. 전체 응답: {response_json}"); continue
            analysis_content = response_json['choices'][0]['message']['content']
            if not analysis_content or not analysis_content.strip(): logging.error("LLM 응답의 'content' 필드가 비어있습니다."); continue
            cleaned_json_str = analysis_content
            if '{' in cleaned_json_str:
                start_index, end_index = cleaned_json_str.find('{'), cleaned_json_str.rfind('}')
                if start_index != -1 and end_index != -1: cleaned_json_str = cleaned_json_str[start_index : end_index + 1]; logging.info("응답 문자열에서 JSON 데이터를 성공적으로 추출했습니다.")
                else: logging.error("JSON 시작/끝 문자를 찾았으나 유효한 범위를 추출하지 못했습니다."); continue
            else: logging.error("응답 내용에 JSON 시작 문자인 '{'가 없습니다."); continue
            analysis_result = json.loads(cleaned_json_str)
            logging.info(f"'{endpoint}'에서 성공적으로 분석 결과를 수신했습니다.")
            return analysis_result
        except json.JSONDecodeError as e: logging.error(f"정리된 JSON 문자열 파싱에 실패했습니다. 오류: {e}"); logging.error(f"파싱 시도한 내용: {cleaned_json_str[:1000]}..."); continue
        except Exception as e: logging.error(f"'{endpoint}' 처리 중 예기치 못한 오류 발생: {type(e).__name__}: {e}", exc_info=True); continue
    raise ConnectionError("모든 LLM API 엔드포인트에 연결하지 못했습니다.")

# --- 5. 다중 탭 HTML 리포트 생성 함수 ---
def generate_multitab_html_report(llm_analysis: dict, charts: dict, output_dir: str) -> str:
    logging.info("다중 탭 HTML 리포트 생성을 시작합니다.")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    report_filename = f"Cell_Analysis_Report_{timestamp}.html"
    report_path = os.path.join(output_dir, report_filename)
    
    summary_html = llm_analysis.get('comprehensive_summary', 'N/A').replace('\n', '<br>')
    issues_html = ''.join([f'<li>{item}</li>' for item in llm_analysis.get('potential_issues', [])])
    debugging_html = ''.join([f'<li>{item}</li>' for item in llm_analysis.get('debugging_points', [])])
    
    detailed_analysis_parts = []
    for peg, analysis in llm_analysis.get('detailed_peg_analysis', {}).items():
        analysis_html = analysis.replace('\n', '<br>')
        detailed_analysis_parts.append(f"<h2>{peg}</h2><div class='peg-analysis-box'><p>{analysis_html}</p></div>")
    detailed_analysis_html = "".join(detailed_analysis_parts)
    
    charts_html = ''.join([f'<div class="chart-item"><img src="data:image/png;base64,{b64_img}" alt="{peg} Chart"></div>' for peg, b64_img in charts.items()])
    
    # ▼▼▼ 수정된 부분: f-string 내의 모든 CSS와 JS 중괄호를 {{, }}로 이스케이프 처리 ▼▼▼
    html_template = f"""
    <!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8"><title>Cell 종합 분석 리포트</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; background-color: #f4f7f6; color: #333; }}
        .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff; box-shadow: 0 0 15px rgba(0,0,0,0.1); border-radius: 8px; }}
        h1 {{ color: #005f73; border-bottom: 3px solid #005f73; padding-bottom: 10px; }} 
        h2 {{ color: #0a9396; border-bottom: 2px solid #e9d8a6; padding-bottom: 5px; margin-top: 30px;}}
        .tab-container {{ width: 100%; }} 
        .tab-nav {{ display: flex; border-bottom: 2px solid #dee2e6; }}
        .tab-nav-link {{ padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; border-bottom: 3px solid transparent; }}
        .tab-nav-link.active {{ border-bottom-color: #005f73; color: #005f73; font-weight: bold; }}
        .tab-content {{ display: none; padding: 20px 5px; animation: fadeIn 0.5s; }} 
        .tab-content.active {{ display: block; }}
        @keyframes fadeIn {{ from {{ opacity: 0; }} to {{ opacity: 1; }} }}
        ul {{ list-style-type: '✓ '; padding-left: 20px; }} 
        li {{ margin-bottom: 10px; line-height: 1.6; }}
        .summary-box, .peg-analysis-box {{ background-color: #e9f5f9; border-left: 5px solid #0a9396; padding: 15px; margin: 15px 0; border-radius: 5px; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .chart-item {{ text-align: center; }} 
        img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
    </style></head><body>
    <div class="container"><h1>Cell 종합 분석 리포트</h1><p><strong>생성 시각:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <div class="tab-container"><div class="tab-nav">
        <button class="tab-nav-link active" onclick="openTab(event, 'summary')">종합 리포트</button>
        <button class="tab-nav-link" onclick="openTab(event, 'detailed')">PEG 상세 분석</button>
        <button class="tab-nav-link" onclick="openTab(event, 'charts')">비교 차트</button>
    </div>
    <div id="summary" class="tab-content active">
        <h2>종합 분석 요약</h2><div class="summary-box"><p>{summary_html}</p></div>
        <h2>잠재적 문제점</h2><div class="summary-box"><ul>{issues_html}</ul></div>
        <h2>디버깅 시작점</h2><div class="summary-box"><ul>{debugging_html}</ul></div>
    </div>
    <div id="detailed" class="tab-content">{detailed_analysis_html}</div>
    <div id="charts" class="tab-content"><div class="chart-grid">{charts_html}</div></div></div></div>
    <script>
        function openTab(evt, tabName) {{{{
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tab-content");
            for (i = 0; i < tabcontent.length; i++) {{{{
                tabcontent[i].style.display = "none";
            }}}}
            tablinks = document.getElementsByClassName("tab-nav-link");
            for (i = 0; i < tablinks.length; i++) {{{{
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }}}}
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }}}}
    </script></body></html>
    """
    # ▲▲▲ 수정 끝 ▲▲▲
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_template)
    
    logging.info(f"HTML 리포트 생성 완료. 경로: {report_path}")
    return report_path

# --- 6. MCP 도구 정의 ---
def _analyze_cell_performance_logic(request: dict) -> dict:
    logging.info("="*20 + " Cell 성능 분석 로직 실행 시작 " + "="*20)
    try:
        peg_data_dict = request['peg_results']
        threshold = request.get('threshold', 30.0)
        output_dir = request.get('output_dir', "/app/backend/analysis_output")
        peg_dfs = {peg_name: pd.DataFrame(data) for peg_name, data in peg_data_dict.items()}
        processed_data, charts_base64 = perform_initial_data_processing(peg_dfs, threshold)
        prompt = create_llm_analysis_prompt(processed_data)
        llm_analysis = query_llm(prompt)
        report_path = generate_multitab_html_report(llm_analysis, charts_base64, output_dir)
        logging.info(f"분석 리포트가 성공적으로 생성되었습니다. 경로: {report_path}")
        logging.info("="*20 + " Cell 성능 분석 로직 실행 종료 (성공) " + "="*20)
        return {"status": "success", "message": f"전문가 분석 리포트가 성공적으로 생성되었습니다: {report_path}", "report_path": report_path}
    except ValueError as e: return {"status": "error", "message": f"데이터 처리 오류: {str(e)}"}
    except ConnectionError as e: return {"status": "error", "message": f"LLM 연결 오류: {str(e)}"}
    except Exception as e:
        logging.exception("분석 로직 실행 중 예기치 못한 오류가 발생했습니다.")
        return {"status": "error", "message": f"예상치 못한 오류 발생: {str(e)}"}

@mcp.tool
def analyze_cell_performance_with_llm(request: dict) -> dict:
    return _analyze_cell_performance_logic(request)

# --- 7. 서버 실행 ---
if __name__ == '__main__':
    logging.info("stdio 모드로 MCP를 실행합니다.")
    mcp.run(transport="stdio")