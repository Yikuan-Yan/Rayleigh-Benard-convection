import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ===========================
# D3Q19 离散速度集和权重
# ===========================
# 离散速度（顺序：0为静止，1-6为轴向，7-10为xy平面对角，11-14为xz平面对角，15-18为yz平面对角）
cxs = np.array([0,  1, -1,  0,  0,  0,  0,  1, -1,  1, -1,  1, -1,  1, -1,  0,  0,  0,  0])
cys = np.array([0,  0,  0,  1, -1,  0,  0,  1,  1, -1, -1,  0,  0,  0,  0,  1, -1,  1, -1])
czs = np.array([0,  0,  0,  0,  0,  1, -1,  0,  0,  0,  0,  1,  1, -1, -1,  1,  1, -1, -1])
# 权重：静止:1/3；轴向（6个）:1/18；对角（12个）:1/36
weights = np.array([1/3] + [1/18]*6 + [1/36]*12)
cs2 = 1.0/3.0

# 反向方向索引，满足 c_i = - c_{opp[i]}
opp = np.array([0, 2, 1, 4, 3, 6, 5, 10, 9, 8, 7, 14, 13, 12, 11, 18, 17, 16, 15])

# ===========================
# 模拟参数设置
# ===========================
Nx, Ny, Nz = 100, 100, 50   # 三维网格尺寸：x, y为水平，z为垂直
Pr = 0.71
Ra = 3.0e4                # 为保证数值稳定，先用较小的瑞利数
Thot, Tcold = 1.0, 0.0    # 底部固定温度为 Thot，顶部 Tcold
H = Nz - 1                # 垂直高度
nu = 0.1                  # 动力粘性系数
alpha = nu / Pr           # 热扩散率
# 根据 Ra 计算重力（垂直方向），假设热膨胀系数 beta=1
gravity = (nu**2 * Ra) / (H**3 * Pr)

# LBM 松弛时间
tau_f = max(0.6, 0.5 + 3*nu)
tau_T = 0.5 + 3*alpha
omega_f = min(1.0/tau_f, 1.9)
omega_T = 1.0/tau_T

# ===========================
# 初始化宏观变量和分布函数
# ===========================
# 注意：数组维度按 (z, y, x)
rho = np.ones((Nz, Ny, Nx))
# 设定初始温度为垂直线性梯度（从底到顶：Thot -> Tcold）加扰动
z = np.linspace(0, 1, Nz)
T_field = np.repeat((Thot - (Thot - Tcold)*z)[:, None, None], Ny, axis=1)
T_field = np.repeat(T_field, Nx, axis=2)
np.random.seed(0)
T_field += 0.01 * (np.random.rand(Nz, Ny, Nx) - 0.5)

# 初始速度置 0
u = np.zeros((Nz, Ny, Nx, 3))   # u[:,:,:,0]=u_x, 1: u_y, 2: u_z

# 初始化分布函数 f 和 g
f = np.zeros((Nz, Ny, Nx, 19))
g = np.zeros((Nz, Ny, Nx, 19))
for i in range(19):
    cu = cxs[i]*u[:,:,:,0] + cys[i]*u[:,:,:,1] + czs[i]*u[:,:,:,2]
    f[:,:,:,i] = weights[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*(u[:,:,:,0]**2 + u[:,:,:,1]**2 + u[:,:,:,2]**2))
    g[:,:,:,i] = weights[i] * T_field * (1 + 3*cu)

# ===========================
# 主循环参数
# ===========================
max_steps = 500
plot_interval = 50   # 每隔 plot_interval 步保存一次快照
snapshots = []       # 保存快照：每个元素为 (T, u_x, u_y)
# 注意：最终输出为水平面（例如 z = Nz//2）上的温度场与水平速度（u_x, u_y）

# ===========================
# 主循环：LBM 模拟
# ===========================
for step in range(1, max_steps+1):
    # --- 计算宏观变量 ---
    # 密度
    rho = np.sum(f, axis=3)
    rho = np.clip(rho, 1e-9, 1e9)
    # 速度（3分量）
    u_x = np.sum(f * cxs[None, None, None, :], axis=3) / rho
    u_y = np.sum(f * cys[None, None, None, :], axis=3) / rho
    u_z = np.sum(f * czs[None, None, None, :], axis=3) / rho
    u[:,:,:,0] = u_x; u[:,:,:,1] = u_y; u[:,:,:,2] = u_z
    # 温度
    T_field = np.sum(g, axis=3)
    
    # --- 外力：Buoyancy 力（垂直方向 z）---
    # 参考温度 T0 取 (Thot+Tcold)/2
    F_z = gravity * (T_field - 0.5*(Thot + Tcold))
    # 半步更新垂直速度
    u_z += 0.5 * F_z / rho
    u[:,:,:,2] = u_z
    
    # --- 边界条件（宏观） ---
    # 底部 z=0：无滑移，固定温度 Thot
    u[:,:,:,0][0,:,:] = 0.0
    u[:,:,:,1][0,:,:] = 0.0
    u[:,:,:,2][0,:,:] = 0.0
    T_field[0,:,:] = Thot
    # 顶部 z=Nz-1：自由边界（仅要求 u_z=0，水平速度可自由滑动），固定温度 Tcold
    u[:,:,:,2][-1,:,:] = 0.0
    T_field[-1,:,:] = Tcold
    
    # --- 计算平衡分布函数 ---
    # u²
    u_sq = u[:,:,:,0]**2 + u[:,:,:,1]**2 + u[:,:,:,2]**2
    f_eq = np.zeros_like(f)
    g_eq = np.zeros_like(g)
    for i in range(19):
        cu = cxs[i]*u[:,:,:,0] + cys[i]*u[:,:,:,1] + czs[i]*u[:,:,:,2]
        f_eq[:,:,:,i] = weights[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u_sq)
        g_eq[:,:,:,i] = weights[i] * T_field * (1 + 3*cu)
    
    # --- BGK 碰撞 ---
    f = f - omega_f * (f - f_eq)
    g = g - omega_T * (g - g_eq)
    
    # --- 加入 Buoyancy 力项（Guo 方案） ---
    for i in range(19):
        f[:,:,:,i] += weights[i]*(1-0.5*omega_f)*3*czs[i]*(F_z/rho)
    
    # --- 流动步骤（Streaming） ---
    f_stream = np.zeros_like(f)
    g_stream = np.zeros_like(g)
    # 对于每个离散方向，将 f 和 g 沿对应方向平移
    for i in range(19):
        # x: axis=2, y: axis=1, z: axis=0
        f_stream[:,:,:,i] = np.roll(np.roll(np.roll(f[:,:,:,i], cxs[i], axis=2), cys[i], axis=1), czs[i], axis=0)
        g_stream[:,:,:,i] = np.roll(np.roll(np.roll(g[:,:,:,i], cxs[i], axis=2), cys[i], axis=1), czs[i], axis=0)
    
    # --- 边界处理 ---
    # 对于水平边界，np.roll 已经实现周期性（x和y方向无需特殊处理）
    # 对于垂直边界（z方向）：
    # 底部 z=0：对所有具有 cz < 0 的方向，做 bounce-back
    for i in range(19):
        if czs[i] < 0:
            f_stream[0,:,:,i] = f[0,:,:, opp[i]]
            # 温度场：直接重置为固定温度的分布
            g_stream[0,:,:,i] = weights[i] * Thot
    # 顶部 z=Nz-1：对于所有具有 cz > 0 的方向，做反射（仅反转垂直分量）
    for i in range(19):
        if czs[i] > 0:
            f_stream[-1,:,:,i] = f[-1,:,:, opp[i]]
            g_stream[-1,:,:,i] = weights[i] * Tcold
    
    # 更新分布函数并 clip（防止数值过大）
    f = np.clip(f_stream, -1e8, 1e8)
    g = np.clip(g_stream, -1e8, 1e8)
    
    # --- 保存快照 ---
    if step % plot_interval == 0 or step == max_steps:
        # 提取中间水平平面（z = Nz//2）的温度场和水平速度场
        z_mid = Nz // 2
        T_snap = np.sum(g[z_mid,:,:,:], axis=2)  # 其实 g 还需要求和 over directions
        # 但我们直接使用 T_field 平均一层
        T_snap = T_field[z_mid,:,:].copy()
        u_x_snap = u[z_mid,:,: ,0].copy()
        u_y_snap = u[z_mid,:,: ,1].copy()
        snapshots.append((T_snap, u_x_snap, u_y_snap))
        print(f"Step {step} / {max_steps}")

# ===========================
# 可视化：仅输出俯视图（水平面：z = Nz//2）的温度场和水平速度场
# ===========================
fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(snapshots[0][0], cmap='jet', origin='lower', vmin=Tcold, vmax=Thot)
ax.set_title("Top View (Horizontal plane at z = Nz//2)")
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
quiver = ax.quiver(X, Y, snapshots[0][1], snapshots[0][2], scale=50, color='k')

def update(frame):
    T_frame, u_x_frame, u_y_frame = snapshots[frame]
    im.set_data(T_frame)
    quiver.set_UVC(u_x_frame, u_y_frame)
    ax.set_title(f"Top View, Snapshot {frame}")
    return im, quiver

anim = animation.FuncAnimation(fig, update, frames=len(snapshots), interval=200, blit=False, repeat=False)
plt.show()
