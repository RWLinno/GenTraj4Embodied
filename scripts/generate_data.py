#!/usr/bin/env python3
"""
Generate synthetic trajectory data
生成合成轨迹数据
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.data.data_generator import TrajectoryDataGenerator


def main():
    parser = argparse.ArgumentParser(description="生成合成轨迹数据")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--output-dir", type=str, default="data", help="输出目录")
    parser.add_argument("--num-trajectories", type=int, help="轨迹数量（覆盖配置）")
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 覆盖参数
    if args.num_trajectories:
        config['data']['generation']['num_trajectories'] = args.num_trajectories
    
    # 设置日志
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("data_generation", output_dir / "logs")
    
    # 创建数据生成器
    data_generator = TrajectoryDataGenerator(config['data'])
    
    # 生成数据
    logger.info("开始生成轨迹数据...")
    train_data, val_data, test_data = data_generator.generate_all_splits()
    
    # 保存数据
    data_generator.save_data(train_data, output_dir / "train.h5")
    data_generator.save_data(val_data, output_dir / "val.h5")
    data_generator.save_data(test_data, output_dir / "test.h5")
    
    # 生成统计信息
    stats = data_generator.generate_statistics(train_data + val_data + test_data)
    
    # 保存统计信息
    import json
    with open(output_dir / "data_statistics.json", 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    logger.info("数据生成完成!")
    logger.info(f"训练集: {len(train_data)} 条轨迹")
    logger.info(f"验证集: {len(val_data)} 条轨迹")
    logger.info(f"测试集: {len(test_data)} 条轨迹")


if __name__ == "__main__":
    main()