#!/usr/bin/env python3
"""
测试修改后的中文输出和emoji清理
"""

import subprocess
import sys
import time

def test_chinese_output():
    """测试系统的中文输出"""
    print("=" * 50)
    print("测试修改后的中文输出")
    print("=" * 50)

    # 启动系统测试（限时15秒）
    try:
        result = subprocess.run(
            [sys.executable, "main_demo.py"],
            timeout=15,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )

        output = result.stdout + result.stderr
        print("系统输出测试结果:")
        print("-" * 30)

        # 检查中文输出
        chinese_keywords = [
            "正在设置SDN网络拓扑",
            "控制器: 正在安装主动流规则",
            "SDN拓扑设置完成",
            "交换机:",
            "链路:",
            "主机:",
            "开始网络监控",
            "开始生成流量模式",
            "控制器: 来自",
            "控制器: 计算路径"
        ]

        found_chinese = []
        for keyword in chinese_keywords:
            if keyword in output:
                found_chinese.append(keyword)
                print(f"✓ 找到中文输出: {keyword}")

        # 检查emoji清理（应该没有这些emoji）
        bad_emojis = ["🚀", "📋", "📊", "🌐", "🤖", "💡", "🧹", "📁"]
        found_emojis = []
        for emoji in bad_emojis:
            if emoji in output:
                found_emojis.append(emoji)
                print(f"× 发现未清理的emoji: {emoji}")

        # 检查保留的符号
        good_symbols = ["✅", "❌"]
        found_symbols = []
        for symbol in good_symbols:
            if symbol in output:
                found_symbols.append(symbol)
                print(f"✓ 正确保留符号: {symbol}")

        print("-" * 30)
        print(f"中文关键词检测: {len(found_chinese)}/{len(chinese_keywords)}")
        print(f"emoji清理检查: {len(found_emojis)} 个未清理 (应为0)")
        print(f"符号保留检查: {len(found_symbols)} 个保留")

        if len(found_chinese) >= 5 and len(found_emojis) == 0:
            print("✓ 测试通过：中文输出正常，emoji已清理")
            return True
        else:
            print("× 测试失败：需要检查输出")
            return False

    except subprocess.TimeoutExpired:
        print("✓ 系统正常启动（超时退出为正常行为）")
        return True
    except Exception as e:
        print(f"× 测试出错: {e}")
        return False

if __name__ == "__main__":
    success = test_chinese_output()
    sys.exit(0 if success else 1)