# Go2-W 申校简历项目条目

以下条目对应已经完成并有记录支持的 Sim-to-Sim 工作，已更新至 2026-11-12 的针对性倒立蒸馏与本地回归。项目起止时间按实际情况另填；篇幅有限时从四项中保留三项。

## 中文版

**Go2-W 轮足机器人强化学习控制与跨仿真验证｜个人项目**  
Python、PyTorch、PPO、Isaac Sim / Isaac Lab、MuJoCo

- 基于 PPO 训练并集成轮足机器人抬腿移动、轮驱、卧倒/起立及倒立等控制策略，搭建 Isaac Sim → MuJoCo 迁移与交互验证系统，对齐机器人模型、观测/动作接口和执行器控制。
- 通过积分步长对照实验定位跨引擎关节振荡，在保持 50 Hz 策略更新的同时，将相关模式内环由 200 Hz 细分至 1 kHz，使倒立返回后站立测试中的关节速度 RMS 从约 3.4–3.9 降至 0.008 rad/s。
- 设计连续动作状态机与回归测试，在六类预设场景中完成 12 次倒立及返回，无需 reset；通过朝向观测重定义与控制上下文交接，改善转向后倒立、停车与再启动的衔接。

- 定位蒸馏学生跨引擎失稳，通过目标引擎失败前状态的教师标注、分层 DAgger 与关键动作加权补训，使学生通过六类连续倒立测试；对应阶段前轮支撑锚点偏移峰值由约 0.39 m 降至 0.17 m，接近教师。

## English version

**Reinforcement Learning Control and Sim-to-Sim Validation for a Wheeled Quadruped | Personal Project**  
Python, PyTorch, PPO, Isaac Sim / Isaac Lab, MuJoCo

- Trained and integrated PPO-based controllers for legged and wheeled locomotion, posture transitions, and handstands; built an Isaac Sim-to-MuJoCo validation system with aligned robot models, observation/action interfaces, and actuator control.
- Diagnosed cross-engine joint oscillations through timestep comparison experiments; retained 50 Hz policy inference while increasing selected inner-loop updates from 200 Hz to 1 kHz, reducing post-handstand standing joint-velocity RMS from approximately 3.4–3.9 to 0.008 rad/s in controlled tests.
- Designed a state machine and regression tests for continuous skill transitions; completed 12 handstand-and-return cycles across six predefined scenarios without resets, addressing heading-dependent failures and controller handoff issues.

- Addressed cross-engine failures in a distilled handstand controller using teacher-labeled target-simulator states, stratified DAgger, and action-weighted supervision; passed six continuous handstand scenarios and reduced peak front-wheel support-anchor displacement from about 0.39 m to 0.17 m in the compared phase, close to the teacher.

## 简版项目介绍

**中文：** 本项目围绕轮足机器人强化学习策略的跨仿真验证，构建了从 Isaac Sim 训练到本地 MuJoCo 执行的控制系统。通过统一观测、动作与执行器接口，以及对数值积分、朝向输入和状态切换的实验排查，改善站立振荡和连续动作衔接，并完成六类预设场景下的 12 次倒立与返回测试。

**English:** This project investigates the cross-simulator transfer of reinforcement-learning controllers for a wheeled quadruped. It combines PPO-based control, an Isaac Sim-to-MuJoCo runtime, and continuous skill-transition tests. Timestep comparisons and observation-frame analysis informed fixes for joint oscillations and heading-dependent failures, with validation covering 12 handstand-and-return cycles across six predefined scenarios.
