# GenTraj4Embodied：位姿轨迹数据审计器

[在线报告](https://rwlinno.github.io/GenTraj4Embodied/) · [English](README.md) · [配套多模态生成演示](https://rwlinno.github.io/trajectory-playground/)

项目从“大量模型对比原型”收缩成了一个可直接使用的小工具：给定 HDF5 轨迹文件，检查形状、非有限值、时间戳、四元数、端点和跨文件重复，并生成 JSON 与无需服务器即可打开的交互 HTML 报告。

```bash
python -m pip install -r inspector/requirements.txt
python -m inspector.audit experiments/data/val.h5 experiments/data/test.h5 --out audit-output
```

打开 `audit-output/index.html`，或读取逐记录指标 `audit-output/report.json`。自己的训练/验证/测试集可一起传入；添加 `--strict` 后，完整性问题或跨文件完全重复轨迹会使程序退出码为 1。起终点重复仅作提示，不自动等同于泄漏。

支持顶层分组、每组 `trajectory[N,7]`、`timestamps[N]`、`start_pose[7]`、`end_pose[7]` 的文件。只需要 NumPy 和 h5py，不需要训练旧模型。其他 HDF5 格式需要适配，不能直接猜测读取。

现有验证集和测试集共 4,055 条轨迹，在列出的完整性检查中没有报错，但每一组起终点都不同，因此没有可直接用于“固定条件多模态覆盖”评估的重复参考组。这不代表数据完全无用，也不代表模型不可能学会条件分布。

速度与加速度使用真实时间戳计算；单位继承输入，不能擅自称为米/秒。报告不检查碰撞、机械臂逆运动学、关节限制或任务成功，零报错不代表可以安全执行。

旧实现仍保留供参考，当前维护入口为 `inspector.audit`，不再把模型文件数量当作实验成果。[评估依据与转型说明](ASSESSMENT.md)。协议沿用 Apache-2.0。


## Research context and diagnostics · 2026-10-14

[中文研究评估与低成本路线](RESEARCH_REPORT.zh-CN.md) covers verified evidence, diffusion / flow-matching / VLA context, practical next steps, and stopping criteria. The live report now includes distribution diagnostics; research references are context, not implemented baselines.
