#!/bin/bash

# SDN + AI 网络异常检测系统演示脚本
# 自动安装依赖并运行完整演示

set -e

echo "======================================================"
echo "SDN + AI 网络异常检测系统演示"
echo "======================================================"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查Python版本
check_python() {
    log_info "检查Python环境..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        log_success "Python版本: $PYTHON_VERSION"

        # 检查版本是否 >= 3.7
        if python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 7) else 1)'; then
            log_success "Python版本满足要求 (>= 3.7)"
        else
            log_error "需要Python 3.7或更高版本"
            exit 1
        fi
    else
        log_error "未找到Python3，请先安装Python"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装Python依赖包..."

    if [ -f "requirements.txt" ]; then
        # 优先使用系统包管理器安装
        log_info "尝试使用系统包管理器安装..."

        # 检测系统类型
        if command -v apt &> /dev/null; then
            # Ubuntu/Debian
            sudo apt update
            sudo apt install -y python3-numpy python3-pandas python3-sklearn python3-flask python3-plotly python3-networkx || {
                log_warning "系统包安装失败，使用pip安装..."
                install_with_pip
            }
        elif command -v yum &> /dev/null; then
            # CentOS/RHEL
            sudo yum install -y python3-numpy python3-pandas python3-scikit-learn python3-flask || {
                log_warning "系统包安装失败，使用pip安装..."
                install_with_pip
            }
        else
            log_warning "未识别的系统，使用pip安装..."
            install_with_pip
        fi

        log_success "依赖包安装完成"
    else
        log_error "未找到requirements.txt文件"
        exit 1
    fi
}

# 使用pip安装
install_with_pip() {
    log_info "使用pip安装依赖..."

    # 尝试不同的pip安装方法
    if python3 -m pip install -r requirements.txt; then
        log_success "pip安装成功"
    elif pip3 install -r requirements.txt; then
        log_success "pip3安装成功"
    elif python3 -m pip install --user -r requirements.txt; then
        log_success "pip用户安装成功"
    else
        log_error "pip安装失败，请手动安装依赖"
        echo "手动安装命令："
        echo "pip3 install numpy pandas scikit-learn flask plotly networkx"
        exit 1
    fi
}

# 检查依赖包
check_dependencies() {
    log_info "验证依赖包..."

    PACKAGES=("numpy" "pandas" "sklearn" "flask" "plotly" "networkx")
    MISSING_PACKAGES=()

    for package in "${PACKAGES[@]}"; do
        if python3 -c "import $package" 2>/dev/null; then
            log_success "✓ $package"
        else
            log_warning "✗ $package"
            MISSING_PACKAGES+=("$package")
        fi
    done

    if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
        log_error "缺少依赖包: ${MISSING_PACKAGES[*]}"
        log_info "正在尝试重新安装..."
        install_with_pip

        # 再次检查
        for package in "${MISSING_PACKAGES[@]}"; do
            if ! python3 -c "import $package" 2>/dev/null; then
                log_error "无法安装 $package，请手动安装"
                exit 1
            fi
        done
    fi

    log_success "所有依赖包验证通过"
}

# 准备演示环境
prepare_demo() {
    log_info "准备演示环境..."

    # 创建必要的目录
    mkdir -p models
    mkdir -p logs

    # 检查端口是否被占用
    if lsof -Pi :8080 -sTCP:LISTEN -t >/dev/null 2>&1; then
        log_warning "端口8080被占用，尝试停止占用进程..."
        pkill -f "python.*8080" || true
        sleep 2
    fi

    log_success "演示环境准备完成"
}

# 运行演示
run_demo() {
    log_info "启动SDN + AI网络异常检测系统演示..."

    echo ""
    echo "======================================================"
    echo "演示系统特性："
    echo "  ✓ SDN网络拓扑模拟 (4交换机6主机)"
    echo "  ✓ 智能流量生成 (正常+攻击模式)"
    echo "  ✓ AI异常检测 (机器学习+统计分析)"
    echo "  ✓ 实时Web监控界面"
    echo "  ✓ 性能测试和报告"
    echo ""
    echo "演示场景："
    echo "  1. 基线建立 (60秒)"
    echo "  2. DDoS攻击检测 (45秒)"
    echo "  3. 端口扫描检测 (30秒)"
    echo "  4. 大文件传输检测 (25秒)"
    echo "  5. 混合攻击场景 (40秒)"
    echo ""
    echo "监控界面: http://localhost:8080"
    echo "======================================================"
    echo ""

    # 运行主演示程序
    if python3 main_demo.py; then
        log_success "演示完成"
    else
        log_error "演示运行失败"
        exit 1
    fi
}

# 清理函数
cleanup() {
    log_info "清理演示环境..."

    # 停止可能的后台进程
    pkill -f "main_demo.py" 2>/dev/null || true
    pkill -f "web_dashboard.py" 2>/dev/null || true

    log_success "清理完成"
}

# 信号处理
trap cleanup EXIT INT TERM

# 主函数
main() {
    echo "开始SDN + AI网络异常检测系统演示安装和运行..."
    echo ""

    # 执行各个步骤
    check_python
    install_dependencies
    check_dependencies
    prepare_demo
    run_demo

    echo ""
    log_success "演示程序执行完成！"
    echo ""
    echo "如需重新运行演示，请执行："
    echo "  python3 main_demo.py"
    echo ""
    echo "或者："
    echo "  ./run_demo.sh"
}

# 执行主函数
main "$@"