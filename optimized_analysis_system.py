# =====================================================================================
# 최적화된 PEG 데이터 분석 시스템 (MCP 직접 DB 연동)
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
import sqlite3
import psycopg2
from typing import Dict, List, Tuple, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from dataclasses import dataclass
from enum import Enum

# --- 로깅 설정 ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 데이터 클래스 정의 ---
class PeriodType(Enum):
    N = "N"
    N_MINUS_1 = "N-1"
    N_MINUS_7 = "N-7"
    N_MINUS_30 = "N-30"

@dataclass
class AnalysisRequest:
    cell_names: List[str]
    peg_metrics: List[str]
    period1: PeriodType
    period2: PeriodType
    threshold: float = 30.0
    output_format: str = "html"  # html, pdf, json

@dataclass
class AnalysisResult:
    summary: Dict
    charts: Dict[str, str]  # base64 encoded images
    detailed_analysis: Dict
    report_path: str

# --- 1. FastMCP 서버 인스턴스 생성 ---
mcp = FastMCP(name="최적화된 PEG 분석기")

# --- 2. 데이터베이스 연결 및 쿼리 클래스 ---
class DatabaseManager:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.connection = None
    
    def connect(self):
        """데이터베이스 연결"""
        try:
            if self.db_config['type'] == 'postgresql':
                self.connection = psycopg2.connect(
                    host=self.db_config['host'],
                    database=self.db_config['database'],
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    port=self.db_config.get('port', 5432)
                )
            elif self.db_config['type'] == 'sqlite':
                self.connection = sqlite3.connect(self.db_config['path'])
            logging.info("데이터베이스 연결 성공")
        except Exception as e:
            logging.error(f"데이터베이스 연결 실패: {e}")
            raise
    
    def query_period_data(self, cell_names: List[str], peg_metrics: List[str], 
                         period: PeriodType) -> pd.DataFrame:
        """특정 기간의 데이터 조회"""
        if not self.connection:
            self.connect()
        
        # 실제 DB 스키마에 맞게 쿼리 수정 필요
        cell_list = "','".join(cell_names)
        peg_list = "','".join(peg_metrics)
        
        query = f"""
        SELECT cell_name, peg_metric, avg_value, period
        FROM peg_statistics 
        WHERE cell_name IN ('{cell_list}')
        AND peg_metric IN ('{peg_list}')
        AND period = '{period.value}'
        ORDER BY cell_name, peg_metric
        """
        
        try:
            df = pd.read_sql_query(query, self.connection)
            logging.info(f"{period.value} 기간 데이터 조회 완료: {len(df)} 행")
            return df
        except Exception as e:
            logging.error(f"데이터 조회 실패: {e}")
            raise

# --- 3. 고성능 데이터 처리 클래스 ---
class DataProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def process_comparison_data(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                              threshold: float) -> pd.DataFrame:
        """두 기간 데이터 비교 분석"""
        # 데이터 병합
        merged = pd.merge(df1, df2, on=['cell_name', 'peg_metric'], 
                         suffixes=('_1', '_2'))
        
        # 변화율 계산
        merged['change_rate'] = ((merged['avg_value_2'] - merged['avg_value_1']) / 
                               merged['avg_value_1'].replace(0, np.nan)) * 100
        
        # 이상치 판별
        merged['anomaly'] = merged['change_rate'].abs() >= threshold
        
        # 성능 등급 분류
        merged['performance_grade'] = merged['change_rate'].apply(
            lambda x: 'A' if x >= 10 else 'B' if x >= 0 else 'C' if x >= -10 else 'D'
        )
        
        return merged
    
    async def generate_charts_async(self, processed_data: pd.DataFrame, 
                                  peg_metrics: List[str]) -> Dict[str, str]:
        """비동기 차트 생성"""
        charts = {}
        
        for metric in peg_metrics:
            metric_data = processed_data[processed_data['peg_metric'] == metric]
            if metric_data.empty:
                continue
            
            # 차트 생성
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 변화율 분포
            ax1.hist(metric_data['change_rate'].dropna(), bins=20, alpha=0.7)
            ax1.set_title(f'{metric} - 변화율 분포')
            ax1.set_xlabel('변화율 (%)')
            ax1.set_ylabel('셀 수')
            
            # 성능 등급 분포
            grade_counts = metric_data['performance_grade'].value_counts()
            ax2.bar(grade_counts.index, grade_counts.values)
            ax2.set_title(f'{metric} - 성능 등급 분포')
            ax2.set_ylabel('셀 수')
            
            plt.tight_layout()
            
            # base64 인코딩
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            charts[metric] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close()
        
        return charts

# --- 4. 지능형 분석 엔진 ---
class IntelligentAnalyzer:
    def __init__(self):
        self.analysis_rules = self._load_analysis_rules()
    
    def _load_analysis_rules(self) -> Dict:
        """분석 규칙 로드"""
        return {
            'critical_threshold': 50.0,  # 급격한 성능 저하 임계값
            'improvement_threshold': 20.0,  # 성능 개선 임계값
            'correlation_threshold': 0.7,  # 상관관계 임계값
            'anomaly_detection': {
                'std_multiplier': 2.0,  # 표준편차 배수
                'iqr_multiplier': 1.5   # IQR 배수
            }
        }
    
    def analyze_performance_trends(self, processed_data: pd.DataFrame) -> Dict:
        """성능 트렌드 분석"""
        analysis = {
            'summary': {},
            'critical_issues': [],
            'improvements': [],
            'recommendations': []
        }
        
        # 전체 통계
        total_cells = len(processed_data)
        anomaly_cells = len(processed_data[processed_data['anomaly']])
        improvement_cells = len(processed_data[processed_data['change_rate'] > 0])
        
        analysis['summary'] = {
            'total_cells': total_cells,
            'anomaly_cells': anomaly_cells,
            'improvement_cells': improvement_cells,
            'anomaly_rate': (anomaly_cells / total_cells) * 100 if total_cells > 0 else 0,
            'improvement_rate': (improvement_cells / total_cells) * 100 if total_cells > 0 else 0
        }
        
        # 급격한 성능 저하 셀 식별
        critical_degradation = processed_data[
            processed_data['change_rate'] < -self.analysis_rules['critical_threshold']
        ]
        
        for _, row in critical_degradation.iterrows():
            analysis['critical_issues'].append({
                'cell_name': row['cell_name'],
                'metric': row['peg_metric'],
                'change_rate': row['change_rate'],
                'severity': 'CRITICAL'
            })
        
        # 성능 개선 셀 식별
        significant_improvements = processed_data[
            processed_data['change_rate'] > self.analysis_rules['improvement_threshold']
        ]
        
        for _, row in significant_improvements.iterrows():
            analysis['improvements'].append({
                'cell_name': row['cell_name'],
                'metric': row['peg_metric'],
                'change_rate': row['change_rate'],
                'severity': 'IMPROVEMENT'
            })
        
        # 권장사항 생성
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """분석 결과 기반 권장사항 생성"""
        recommendations = []
        
        if analysis['summary']['anomaly_rate'] > 20:
            recommendations.append("전체 셀의 20% 이상에서 성능 저하가 감지되어 네트워크 최적화가 필요합니다.")
        
        if len(analysis['critical_issues']) > 0:
            recommendations.append(f"{len(analysis['critical_issues'])}개 셀에서 급격한 성능 저하가 발생했습니다. 즉시 조치가 필요합니다.")
        
        if analysis['summary']['improvement_rate'] > 30:
            recommendations.append("성능 개선이 많이 이루어졌습니다. 최적화 방법을 다른 셀에도 적용해보세요.")
        
        return recommendations

# --- 5. 리포트 생성기 ---
class ReportGenerator:
    def __init__(self, output_dir: str = "/app/backend/analysis_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_html_report(self, analysis_result: AnalysisResult, 
                           request: AnalysisRequest) -> str:
        """HTML 리포트 생성"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
        filename = f"Optimized_Cell_Analysis_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)
        
        # HTML 템플릿 생성
        html_content = self._create_html_template(analysis_result, request)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"HTML 리포트 생성 완료: {filepath}")
        return filepath
    
    def _create_html_template(self, analysis_result: AnalysisResult, 
                            request: AnalysisRequest) -> str:
        """HTML 템플릿 생성"""
        summary = analysis_result.summary
        charts = analysis_result.charts
        
        # 차트 HTML 생성
        charts_html = ''
        for metric, chart_b64 in charts.items():
            charts_html += f'''
            <div class="chart-container">
                <h3>{metric}</h3>
                <img src="data:image/png;base64,{chart_b64}" alt="{metric} Chart">
            </div>
            '''
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>최적화된 셀 성능 분석 리포트</title>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; margin: 0; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                              gap: 20px; margin-bottom: 30px; }}
                .summary-card {{ background: white; padding: 20px; border-radius: 10px; 
                              box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); 
                            gap: 30px; }}
                .chart-container {{ background: white; padding: 20px; border-radius: 10px; 
                                 box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .chart-container img {{ max-width: 100%; height: auto; }}
                .critical {{ color: #dc3545; font-weight: bold; }}
                .improvement {{ color: #28a745; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>최적화된 셀 성능 분석 리포트</h1>
                    <p>생성 시간: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p>분석 기간: {request.period1.value} vs {request.period2.value}</p>
                </div>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>전체 셀 수</h3>
                        <p class="critical">{summary.get('total_cells', 0)}</p>
                    </div>
                    <div class="summary-card">
                        <h3>이상 셀 수</h3>
                        <p class="critical">{summary.get('anomaly_cells', 0)}</p>
                    </div>
                    <div class="summary-card">
                        <h3>개선 셀 수</h3>
                        <p class="improvement">{summary.get('improvement_cells', 0)}</p>
                    </div>
                    <div class="summary-card">
                        <h3>이상율</h3>
                        <p class="critical">{summary.get('anomaly_rate', 0):.1f}%</p>
                    </div>
                </div>
                
                <div class="chart-grid">
                    {charts_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template

# --- 6. 메인 분석 엔진 ---
class OptimizedAnalysisEngine:
    def __init__(self, db_config: Dict):
        self.db_manager = DatabaseManager(db_config)
        self.data_processor = DataProcessor()
        self.analyzer = IntelligentAnalyzer()
        self.report_generator = ReportGenerator()
    
    async def analyze_cells(self, request: AnalysisRequest) -> AnalysisResult:
        """메인 분석 프로세스"""
        logging.info("최적화된 셀 분석 시작")
        
        # 1. 데이터베이스에서 데이터 조회
        df1 = self.db_manager.query_period_data(
            request.cell_names, request.peg_metrics, request.period1
        )
        df2 = self.db_manager.query_period_data(
            request.cell_names, request.peg_metrics, request.period2
        )
        
        # 2. 데이터 처리 및 비교 분석
        processed_data = self.data_processor.process_comparison_data(
            df1, df2, request.threshold
        )
        
        # 3. 지능형 분석 수행
        analysis = self.analyzer.analyze_performance_trends(processed_data)
        
        # 4. 차트 생성 (비동기)
        charts = await self.data_processor.generate_charts_async(
            processed_data, request.peg_metrics
        )
        
        # 5. 리포트 생성
        analysis_result = AnalysisResult(
            summary=analysis['summary'],
            charts=charts,
            detailed_analysis=analysis,
            report_path=""
        )
        
        report_path = self.report_generator.generate_html_report(
            analysis_result, request
        )
        analysis_result.report_path = report_path
        
        logging.info("최적화된 셀 분석 완료")
        return analysis_result

# --- 7. MCP 도구 정의 ---
@mcp.tool
def analyze_cells_optimized(request: dict) -> dict:
    """최적화된 셀 분석 도구"""
    try:
        # 요청 파싱
        analysis_request = AnalysisRequest(
            cell_names=request['cell_names'],
            peg_metrics=request['peg_metrics'],
            period1=PeriodType(request['period1']),
            period2=PeriodType(request['period2']),
            threshold=request.get('threshold', 30.0)
        )
        
        # 데이터베이스 설정 (실제 환경에 맞게 수정 필요)
        db_config = {
            'type': 'postgresql',  # 또는 'sqlite'
            'host': 'localhost',
            'database': 'peg_statistics',
            'user': 'username',
            'password': 'password',
            'port': 5432
        }
        
        # 분석 엔진 생성 및 실행
        engine = OptimizedAnalysisEngine(db_config)
        result = asyncio.run(engine.analyze_cells(analysis_request))
        
        return {
            "status": "success",
            "message": "최적화된 분석이 완료되었습니다.",
            "report_path": result.report_path,
            "summary": result.summary
        }
        
    except Exception as e:
        logging.error(f"분석 중 오류 발생: {e}")
        return {
            "status": "error",
            "message": f"분석 실패: {str(e)}"
        }

# --- 8. 서버 실행 ---
if __name__ == '__main__':
    logging.info("최적화된 MCP 서버를 실행합니다.")
    mcp.run(transport="stdio")