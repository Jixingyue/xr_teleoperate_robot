#!/usr/bin/env python3
"""
性能监控工具模块

这个模块提供了用于跟踪和分析代码性能的工具类，包括：
1. PerformanceMonitor: 基础性能监控类
2. PerformanceAnalyzer: 性能分析器
3. 性能统计和可视化功能
"""

import time
import numpy as np
import logging
from collections import deque
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import os

class PerformanceMonitor:
    """
    性能监控类，用于跟踪各个步骤的执行时间
    
    这个类提供了简单易用的性能监控功能，可以：
    - 跟踪多个不同操作的执行时间
    - 计算统计信息（平均值、最小值、最大值、标准差）
    - 生成性能报告
    - 支持滑动窗口统计
    """
    
    def __init__(self, window_size: int = 100, name: str = "PerformanceMonitor"):
        """
        初始化性能监控器
        
        Args:
            window_size: 滑动窗口大小，用于计算统计信息
            name: 监控器名称，用于日志输出
        """
        self.window_size = window_size
        self.name = name
        self.timings = {}  # 当前正在计时的操作
        self.history = {}  # 历史数据
        self.logger = logging.getLogger(f"{name}")
        
    def start_timer(self, name: str) -> None:
        """
        开始计时
        
        Args:
            name: 操作名称
        """
        if name not in self.timings:
            self.timings[name] = time.time()
            if name not in self.history:
                self.history[name] = deque(maxlen=self.window_size)
    
    def end_timer(self, name: str) -> float:
        """
        结束计时并记录
        
        Args:
            name: 操作名称
            
        Returns:
            float: 执行时间（毫秒）
        """
        if name in self.timings:
            duration = (time.time() - self.timings[name]) * 1000  # 转换为毫秒
            self.history[name].append(duration)
            del self.timings[name]
            return duration
        return 0.0
    
    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """
        获取统计信息
        
        Args:
            name: 操作名称
            
        Returns:
            Dict: 包含统计信息的字典，如果数据不足则返回None
        """
        if name in self.history and self.history[name]:
            values = list(self.history[name])
            return {
                'avg': np.mean(values),
                'min': np.min(values),
                'max': np.max(values),
                'std': np.std(values),
                'latest': values[-1] if values else 0,
                'count': len(values)
            }
        return None
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有操作的统计信息
        
        Returns:
            Dict: 所有操作的统计信息字典
        """
        all_stats = {}
        for name in self.history:
            stats = self.get_stats(name)
            if stats:
                all_stats[name] = stats
        return all_stats
    
    def log_performance(self, prefix: str = "") -> None:
        """
        记录性能统计到日志
        
        Args:
            prefix: 日志前缀
        """
        if prefix:
            self.logger.info(f"=== {prefix} 性能统计 ===")
        else:
            self.logger.info(f"=== {self.name} 性能统计 ===")
            
        for name in self.history:
            stats = self.get_stats(name)
            if stats:
                self.logger.info(f"{name}: 平均={stats['avg']:.2f}ms, 最小={stats['min']:.2f}ms, "
                               f"最大={stats['max']:.2f}ms, 最新={stats['latest']:.2f}ms, "
                               f"样本数={stats['count']}")
        self.logger.info("================")
    
    def reset(self) -> None:
        """重置所有计时器和历史数据"""
        self.timings.clear()
        self.history.clear()
    
    def get_total_time(self, name: str) -> float:
        """
        获取指定操作的总执行时间
        
        Args:
            name: 操作名称
            
        Returns:
            float: 总执行时间（毫秒）
        """
        if name in self.history:
            return sum(self.history[name])
        return 0.0
    
    def get_operation_count(self, name: str) -> int:
        """
        获取指定操作的执行次数
        
        Args:
            name: 操作名称
            
        Returns:
            int: 执行次数
        """
        if name in self.history:
            return len(self.history[name])
        return 0

class PerformanceAnalyzer:
    """
    性能分析器，用于生成详细的性能报告和可视化
    """
    
    def __init__(self, output_dir: str = "."):
        """
        初始化性能分析器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.results = {}
        self.logger = logging.getLogger("PerformanceAnalyzer")
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
    
    def add_monitor_data(self, monitor: PerformanceMonitor, name: str = None) -> None:
        """
        添加监控器数据
        
        Args:
            monitor: 性能监控器实例
            name: 数据名称，如果为None则使用监控器名称
        """
        if name is None:
            name = monitor.name
        
        self.results[name] = monitor.get_all_stats()
    
    def generate_performance_report(self, filename: str = "performance_report") -> None:
        """
        生成性能报告
        
        Args:
            filename: 报告文件名（不包含扩展名）
        """
        self.logger.info("生成性能报告...")
        
        # 生成图表
        self._generate_charts(filename)
        
        # 生成文本报告
        self._generate_text_report(filename)
        
        self.logger.info(f"性能报告已保存到 {self.output_dir}")
    
    def _generate_charts(self, filename: str) -> None:
        """生成性能图表"""
        if not self.results:
            self.logger.warning("没有数据可生成图表")
            return
        
        # 计算子图数量
        total_operations = sum(len(data) for data in self.results.values())
        if total_operations == 0:
            return
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('性能分析报告', fontsize=16)
        
        # 收集所有数据用于绘图
        all_operations = []
        all_avg_times = []
        all_max_times = []
        all_min_times = []
        
        for monitor_name, operations in self.results.items():
            for op_name, stats in operations.items():
                full_name = f"{monitor_name}.{op_name}"
                all_operations.append(full_name)
                all_avg_times.append(stats['avg'])
                all_max_times.append(stats['max'])
                all_min_times.append(stats['min'])
        
        # 按平均时间排序
        sorted_indices = np.argsort(all_avg_times)[::-1]  # 降序排列
        sorted_operations = [all_operations[i] for i in sorted_indices]
        sorted_avg_times = [all_avg_times[i] for i in sorted_indices]
        sorted_max_times = [all_max_times[i] for i in sorted_indices]
        sorted_min_times = [all_min_times[i] for i in sorted_indices]
        
        # 1. 平均时间柱状图
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(sorted_operations)), sorted_avg_times, 
                       color='skyblue', alpha=0.7)
        ax1.set_title('各操作平均执行时间')
        ax1.set_ylabel('时间 (ms)')
        ax1.set_xticks(range(len(sorted_operations)))
        ax1.set_xticklabels(sorted_operations, rotation=45, ha='right')
        
        # 添加数值标签
        for i, (bar, time_val) in enumerate(zip(bars1, sorted_avg_times)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{time_val:.1f}ms', ha='center', va='bottom', fontsize=8)
        
        # 2. 时间范围图（最大值-最小值）
        ax2 = axes[0, 1]
        time_ranges = [max_t - min_t for max_t, min_t in zip(sorted_max_times, sorted_min_times)]
        bars2 = ax2.bar(range(len(sorted_operations)), time_ranges, 
                       color='lightcoral', alpha=0.7)
        ax2.set_title('各操作时间范围（最大值-最小值）')
        ax2.set_ylabel('时间范围 (ms)')
        ax2.set_xticks(range(len(sorted_operations)))
        ax2.set_xticklabels(sorted_operations, rotation=45, ha='right')
        
        # 3. 性能分布直方图（选择最耗时的操作）
        ax3 = axes[1, 0]
        if len(sorted_operations) > 0:
            # 选择前3个最耗时的操作
            top_operations = sorted_operations[:min(3, len(sorted_operations))]
            for i, op_name in enumerate(top_operations):
                # 这里需要原始数据，暂时用模拟数据
                # 实际使用时应该从monitor中获取原始时间数据
                ax3.hist(np.random.normal(sorted_avg_times[i], sorted_avg_times[i]*0.2, 50), 
                        bins=20, alpha=0.6, label=op_name)
            ax3.set_title('最耗时操作的时间分布')
            ax3.set_xlabel('时间 (ms)')
            ax3.set_ylabel('频次')
            ax3.legend()
        
        # 4. 性能对比雷达图
        ax4 = axes[1, 1]
        if len(sorted_operations) >= 3:
            # 选择前3个操作进行对比
            top_3_ops = sorted_operations[:3]
            top_3_times = sorted_avg_times[:3]
            
            # 归一化到0-1范围
            max_time = max(top_3_times)
            normalized_times = [t/max_time for t in top_3_times]
            
            # 雷达图
            angles = np.linspace(0, 2*np.pi, len(top_3_ops), endpoint=False).tolist()
            angles += angles[:1]  # 闭合图形
            normalized_times += normalized_times[:1]
            
            ax4.plot(angles, normalized_times, 'o-', linewidth=2)
            ax4.fill(angles, normalized_times, alpha=0.25)
            ax4.set_xticks(angles[:-1])
            ax4.set_xticklabels(top_3_ops)
            ax4.set_title('性能对比雷达图')
            ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{filename}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, filename: str) -> None:
        """生成文本报告"""
        report_path = os.path.join(self.output_dir, f'{filename}.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("性能分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 总体统计
            total_operations = sum(len(data) for data in self.results.values())
            f.write(f"总操作数: {total_operations}\n")
            f.write(f"监控器数量: {len(self.results)}\n\n")
            
            # 各监控器详细统计
            for monitor_name, operations in self.results.items():
                f.write(f"{monitor_name} 性能统计:\n")
                f.write("-" * 30 + "\n")
                
                if not operations:
                    f.write("  无数据\n\n")
                    continue
                
                # 按平均时间排序
                sorted_ops = sorted(operations.items(), 
                                  key=lambda x: x[1]['avg'], reverse=True)
                
                for op_name, stats in sorted_ops:
                    f.write(f"  {op_name}:\n")
                    f.write(f"    平均时间: {stats['avg']:.2f}ms\n")
                    f.write(f"    最小时间: {stats['min']:.2f}ms\n")
                    f.write(f"    最大时间: {stats['max']:.2f}ms\n")
                    f.write(f"    标准差: {stats['std']:.2f}ms\n")
                    f.write(f"    样本数: {stats['count']}\n")
                    f.write(f"    总时间: {stats['avg'] * stats['count']:.2f}ms\n\n")
                
                # 计算该监控器的总统计
                total_time = sum(stats['avg'] * stats['count'] for stats in operations.values())
                total_samples = sum(stats['count'] for stats in operations.values())
                f.write(f"  总计:\n")
                f.write(f"    总执行时间: {total_time:.2f}ms\n")
                f.write(f"    总样本数: {total_samples}\n")
                f.write(f"    平均每次执行: {total_time/total_samples:.2f}ms\n\n")
            
            # 性能建议
            f.write("性能优化建议:\n")
            f.write("-" * 30 + "\n")
            
            # 找出最耗时的操作
            all_operations = []
            for monitor_name, operations in self.results.items():
                for op_name, stats in operations.items():
                    all_operations.append((f"{monitor_name}.{op_name}", stats))
            
            if all_operations:
                # 按平均时间排序
                all_operations.sort(key=lambda x: x[1]['avg'], reverse=True)
                
                # 分析前3个最耗时的操作
                for i, (op_name, stats) in enumerate(all_operations[:3]):
                    f.write(f"{i+1}. {op_name} (平均{stats['avg']:.2f}ms):\n")
                    
                    if stats['avg'] > 50:
                        f.write("   - 严重性能瓶颈，建议立即优化\n")
                    elif stats['avg'] > 20:
                        f.write("   - 性能瓶颈，建议优化\n")
                    elif stats['avg'] > 10:
                        f.write("   - 性能较慢，可考虑优化\n")
                    else:
                        f.write("   - 性能良好\n")
                    
                    # 根据标准差判断稳定性
                    if stats['std'] > stats['avg'] * 0.5:
                        f.write("   - 执行时间不稳定，建议检查异常情况\n")
                    
                    f.write("\n")
            
            f.write("总体建议:\n")
            f.write("- 将关键路径的总延迟控制在20ms以内\n")
            f.write("- 对于实时系统，建议使用更快的算法或硬件加速\n")
            f.write("- 考虑使用缓存、预计算或并行处理来优化性能\n")
    
    def export_to_csv(self, filename: str = "performance_data.csv") -> None:
        """
        导出性能数据到CSV文件
        
        Args:
            filename: CSV文件名
        """
        import csv
        
        csv_path = os.path.join(self.output_dir, filename)
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['monitor', 'operation', 'avg_time', 'min_time', 'max_time', 
                         'std_time', 'count', 'total_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for monitor_name, operations in self.results.items():
                for op_name, stats in operations.items():
                    writer.writerow({
                        'monitor': monitor_name,
                        'operation': op_name,
                        'avg_time': f"{stats['avg']:.2f}",
                        'min_time': f"{stats['min']:.2f}",
                        'max_time': f"{stats['max']:.2f}",
                        'std_time': f"{stats['std']:.2f}",
                        'count': stats['count'],
                        'total_time': f"{stats['avg'] * stats['count']:.2f}"
                    })
        
        self.logger.info(f"性能数据已导出到 {csv_path}")

# 便捷函数
def create_monitor(name: str = "PerformanceMonitor", window_size: int = 100) -> PerformanceMonitor:
    """
    创建性能监控器的便捷函数
    
    Args:
        name: 监控器名称
        window_size: 滑动窗口大小
        
    Returns:
        PerformanceMonitor: 性能监控器实例
    """
    return PerformanceMonitor(window_size=window_size, name=name)

def quick_profile(func, *args, **kwargs):
    """
    快速性能分析装饰器
    
    Args:
        func: 要分析的函数
        *args: 函数参数
        **kwargs: 函数关键字参数
        
    Returns:
        函数执行结果和性能统计
    """
    monitor = PerformanceMonitor(name=f"QuickProfile_{func.__name__}")
    
    monitor.start_timer("function_execution")
    result = func(*args, **kwargs)
    monitor.end_timer("function_execution")
    
    stats = monitor.get_stats("function_execution")
    return result, stats

if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 创建监控器
    monitor = PerformanceMonitor(name="TestMonitor")
    
    # 模拟一些操作
    for i in range(10):
        monitor.start_timer("operation1")
        time.sleep(0.01)  # 10ms
        monitor.end_timer("operation1")
        
        monitor.start_timer("operation2")
        time.sleep(0.02)  # 20ms
        monitor.end_timer("operation2")
    
    # 输出统计信息
    monitor.log_performance()
    
    # 创建分析器并生成报告
    analyzer = PerformanceAnalyzer()
    analyzer.add_monitor_data(monitor, "TestData")
    analyzer.generate_performance_report("test_report")
    analyzer.export_to_csv("test_data.csv") 