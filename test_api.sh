#!/bin/bash

# SDN + AI系统API接口测试脚本

echo "=================================================="
echo "SDN + AI 网络异常检测系统 API 接口测试"
echo "=================================================="
echo

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

BASE_URL="http://localhost:8080"

# 测试函数
test_api() {
    local endpoint=$1
    local description=$2

    echo -e "${BLUE}测试:${NC} $description"
    echo -e "${YELLOW}端点:${NC} $endpoint"

    response=$(curl -s -w "%{http_code}" "$BASE_URL$endpoint")
    http_code="${response: -3}"
    content="${response%???}"

    if [ "$http_code" -eq 200 ]; then
        echo -e "${GREEN}✅ 状态码: $http_code (成功)${NC}"
        if [ -n "$content" ] && [ "$content" != "{}" ]; then
            echo -e "${GREEN}✅ 返回数据:${NC}"
            echo "$content" | head -3
        else
            echo -e "${YELLOW}⚠️  返回空数据 (可能正在初始化)${NC}"
        fi
    else
        echo -e "${RED}❌ 状态码: $http_code (失败)${NC}"
        echo -e "${RED}错误信息:${NC} $content"
    fi

    echo
    echo "----------------------------------------"
    echo
}

# 等待服务启动
echo "等待SDN + AI系统完全启动..."
sleep 3

# 测试所有API端点
echo "开始API接口测试..."
echo

# 1. 主页面
test_api "/" "主页面 (Web界面)"

# 2. 健康检查
test_api "/api/health" "系统健康检查"

# 3. 基础统计
test_api "/api/stats" "基础统计数据"

# 4. 流量图表数据
test_api "/api/traffic_plot" "流量图表数据"

# 5. 异常检测图表
test_api "/api/anomaly_plot" "异常检测图表数据"

# 6. 性能图表
test_api "/api/performance_plot" "系统性能图表数据"

# 测试不存在的端点
echo -e "${BLUE}测试不存在的端点:${NC}"
test_api "/api/nonexistent" "不存在的端点 (应该返回404)"

echo "=================================================="
echo "API接口测试完成"
echo "=================================================="
echo

# 额外的功能测试
echo "额外功能验证："
echo

# 检查Web界面是否包含关键元素
echo -e "${BLUE}检查Web界面关键元素:${NC}"
main_page=$(curl -s "$BASE_URL/")

if echo "$main_page" | grep -q "SDN + AI Network Monitoring Dashboard"; then
    echo -e "${GREEN}✅ 页面标题正确${NC}"
else
    echo -e "${RED}❌ 页面标题缺失${NC}"
fi

if echo "$main_page" | grep -q "plotly"; then
    echo -e "${GREEN}✅ 图表库已加载${NC}"
else
    echo -e "${RED}❌ 图表库缺失${NC}"
fi

if echo "$main_page" | grep -q "jquery"; then
    echo -e "${GREEN}✅ jQuery库已加载${NC}"
else
    echo -e "${RED}❌ jQuery库缺失${NC}"
fi

echo
echo "🌐 监控界面: $BASE_URL"
echo "📊 实时查看系统运行状态和异常检测结果"
echo