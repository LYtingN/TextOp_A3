#!/usr/bin/env python3
"""检查npz文件格式是否符合训练要求"""

import numpy as np
import sys
from pathlib import Path

def check_npz_format(npz_file):
    """检查npz文件是否包含必需的键"""
    print(f"\n检查文件: {npz_file}")
    print("=" * 60)
    
    try:
        data = np.load(npz_file)
        keys = list(data.keys())
        print(f"文件中的键: {keys}")
        print("\n各键的形状:")
        for key in keys:
            arr = data[key]
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        
        # 检查必需的键
        required_keys = {
            'fps': '帧率',
            'joint_pos': '关节位置 [T, Nq]',
            'joint_vel': '关节速度 [T, Nq]',
            'body_pos_w': '身体位置 [T, N, 3]',
            'body_quat_w': '身体四元数 [T, N, 4]',
            'body_lin_vel_w': '身体线性速度 [T, N, 3]',
            'body_ang_vel_w': '身体角速度 [T, N, 3]'
        }
        
        print("\n必需键检查:")
        missing_keys = []
        for key, desc in required_keys.items():
            if key in keys:
                print(f"  ✓ {key}: {desc} - 存在")
            else:
                print(f"  ✗ {key}: {desc} - 缺失")
                missing_keys.append(key)
        
        if missing_keys:
            print(f"\n❌ 缺少以下必需的键: {missing_keys}")
            return False
        else:
            print("\n✅ 所有必需的键都存在!")
            
            # 检查形状兼容性
            print("\n形状兼容性检查:")
            joint_pos = data['joint_pos']
            joint_vel = data['joint_vel']
            body_pos_w = data['body_pos_w']
            body_quat_w = data['body_quat_w']
            body_lin_vel_w = data['body_lin_vel_w']
            body_ang_vel_w = data['body_ang_vel_w']
            
            T = joint_pos.shape[0]
            print(f"  时间步数 T = {T}")
            print(f"  关节数 Nq = {joint_pos.shape[1]}")
            
            # 检查时间维度是否一致
            shapes_ok = True
            if joint_vel.shape[0] != T:
                print(f"  ✗ joint_vel 时间维度不匹配: {joint_vel.shape[0]} != {T}")
                shapes_ok = False
            if body_pos_w.shape[0] != T:
                print(f"  ✗ body_pos_w 时间维度不匹配: {body_pos_w.shape[0]} != {T}")
                shapes_ok = False
            if body_quat_w.shape[0] != T:
                print(f"  ✗ body_quat_w 时间维度不匹配: {body_quat_w.shape[0]} != {T}")
                shapes_ok = False
            if body_lin_vel_w.shape[0] != T:
                print(f"  ✗ body_lin_vel_w 时间维度不匹配: {body_lin_vel_w.shape[0]} != {T}")
                shapes_ok = False
            if body_ang_vel_w.shape[0] != T:
                print(f"  ✗ body_ang_vel_w 时间维度不匹配: {body_ang_vel_w.shape[0]} != {T}")
                shapes_ok = False
            
            if shapes_ok:
                print("  ✅ 所有时间维度一致")
            
            # 检查body相关数据的空间维度
            if body_pos_w.shape[2] != 3:
                print(f"  ✗ body_pos_w 最后一维应该是3，实际是 {body_pos_w.shape[2]}")
                shapes_ok = False
            if body_quat_w.shape[2] != 4:
                print(f"  ✗ body_quat_w 最后一维应该是4，实际是 {body_quat_w.shape[2]}")
                shapes_ok = False
            if body_lin_vel_w.shape[2] != 3:
                print(f"  ✗ body_lin_vel_w 最后一维应该是3，实际是 {body_lin_vel_w.shape[2]}")
                shapes_ok = False
            if body_ang_vel_w.shape[2] != 3:
                print(f"  ✗ body_ang_vel_w 最后一维应该是3，实际是 {body_ang_vel_w.shape[2]}")
                shapes_ok = False
            
            if shapes_ok:
                print("  ✅ 所有空间维度正确")
            
            return shapes_ok
            
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_npz_format.py <npz_file1> [npz_file2] ...")
        sys.exit(1)
    
    all_ok = True
    for npz_file in sys.argv[1:]:
        if not Path(npz_file).exists():
            print(f"❌ 文件不存在: {npz_file}")
            all_ok = False
            continue
        if not check_npz_format(npz_file):
            all_ok = False
    
    if all_ok:
        print("\n" + "=" * 60)
        print("✅ 所有文件格式检查通过!")
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ 部分文件格式不符合要求，请检查上述输出")
        sys.exit(1)
