import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# ---------------------------
# LBM D2Q9 常数
# ---------------------------
cxs = np.array([0, 1, 0, -1, 0,  1, -1, -1,  1])   # x方向离散速度
cys = np.array([0, 0, 1,  0, -1,  1,  1, -1, -1])   # y方向离散速度
weights = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# ---------------------------
# 模拟参数设置
# ---------------------------
Nx, Ny = 188, 100            # 网格尺寸 (x: 0..Nx-1, y: 0..Ny-1)
Pr = 0.71
Ra = 3.0e4                  # 为保证数值稳定，先使用较小的 Rayleigh 数
Thot, Tcold = 1.0, 0.0      # 下热，上冷
H = Ny - 1                  # 流体层高度（单位格）
nu = 0.1                    # 粘性系数
alpha = nu / Pr           # 热扩散率
gravity = (nu**2 * Ra) / (H**3 * Pr)  # 根据公式计算

# LBM 松弛时间
tau_f = max(0.6, 0.5 + 3*nu)
tau_T = 0.5 + 3*alpha
omega_f = min(1.0/tau_f, 1.9)
omega_T = 1.0/tau_T

# ---------------------------
# 初始化宏观变量和分布函数
# ---------------------------
rho = np.ones((Ny, Nx))
y = np.linspace(0, 1, Ny)
T = np.repeat((Thot - (Thot - Tcold)*y)[:, None], Nx, axis=1)
np.random.seed(0)
T += 0.01 * (np.random.rand(Ny, Nx) - 0.5)

u_x = np.zeros((Ny, Nx))
u_y = np.zeros((Ny, Nx))

f = np.zeros((Ny, Nx, 9))
g = np.zeros((Ny, Nx, 9))
for i, (cx, cy, w) in enumerate(zip(cxs, cys, weights)):
    cu = cx*u_x + cy*u_y
    f[:,:,i] = w * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*(u_x**2 + u_y**2))
    g[:,:,i] = w * T * (1 + 3*cu)

# ---------------------------
# 模拟参数
# ---------------------------
max_steps = 10000
plot_interval = 50    # 每隔 plot_interval 步保存一次快照
snapshots = []        # 保存快照，每个元素为 (T, u_x, u_y)

# ---------------------------
# 主循环：LBM 模拟
# ---------------------------
for step in range(1, max_steps+1):
    # 计算宏观变量
    rho = np.sum(f, axis=2)
    rho = np.clip(rho, 1e-9, 1e9)
    u_x = np.sum(f * cxs[None, None, :], axis=2) / rho
    u_y = np.sum(f * cys[None, None, :], axis=2) / rho
    T = np.sum(g, axis=2)
    
    # Boussinesq buoyancy：以参考温度 (Thot+Tcold)/2
    F_y = gravity * (T - 0.5*(Thot + Tcold))
    u_y += 0.5 * F_y / rho

    # ---------------------------
    # 边界条件（宏观）
    # ---------------------------
    # 底部边界 (y=0)：无滑移 (u=0) 且 T=Thot
    u_x[0, :] = 0.0
    u_y[0, :] = 0.0
    T[0, :] = Thot
    # 顶部边界 (y=Ny-1)：自由边界（只要求垂直速度 u_y=0）且 T=Tcold
    u_y[-1, :] = 0.0
    T[-1, :] = Tcold

    # ---------------------------
    # 计算平衡分布函数
    # ---------------------------
    cu_sq = u_x**2 + u_y**2
    f_eq = np.zeros_like(f)
    g_eq = np.zeros_like(g)
    for i, (cx, cy, w) in enumerate(zip(cxs, cys, weights)):
        cu = np.clip(cx*u_x + cy*u_y, -1, 1)
        f_eq[:,:,i] = w * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*cu_sq)
        g_eq[:,:,i] = w * T * (1 + 3*cu)
    
    # ---------------------------
    # BGK 碰撞步
    # ---------------------------
    f = f - omega_f * (f - f_eq)
    g = g - omega_T * (g - g_eq)
    
    # 加入 Buoyancy 力项（Guo 方案）
    for i, (cx, cy, w) in enumerate(zip(cxs, cys, weights)):
        f[:,:,i] += w * (1 - 0.5*omega_f) * (3 * cy) * (F_y / rho)
    
    # ---------------------------
    # 流动步（Streaming）
    # ---------------------------
    f_stream = np.zeros_like(f)
    g_stream = np.zeros_like(g)
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        f_stream[:,:,i] = np.roll(np.roll(f[:,:,i], cx, axis=1), cy, axis=0)
        g_stream[:,:,i] = np.roll(np.roll(g[:,:,i], cx, axis=1), cy, axis=0)
    
    # ---------------------------
    # 边界处理
    # ---------------------------
    # 底部边界 (y=0)：采用 bounce-back（无滑移）
    f_stream[0,:,2] = f[0,:,4]
    f_stream[0,:,5] = f[0,:,7]
    f_stream[0,:,6] = f[0,:,8]

    # 顶部边界 (y=Ny-1)：自由边界，处理时仅反转垂直分量
    f_stream[-1,:,4] = f[-1,:,2]
    
    # 温度边界：直接重置为固定温度的平衡分布（假设 u=0）
    for i, (cx, cy, w) in enumerate(zip(cxs, cys, weights)):
        g_stream[0,:,i] = w * Thot
        g_stream[-1,:,i] = w * Tcold
    
    # 更新分布函数并剪切（扩大 clip 范围以防止过度裁剪）
    f = np.clip(f_stream, -1e5, 1e5)
    g = np.clip(g_stream, -1e5, 1e5)

    # 保存快照
    if step % plot_interval == 0 or step == max_steps:
        T_snap = np.clip(np.sum(g, axis=2), Tcold, Thot)
        snapshots.append((T_snap.copy(), u_x.copy(), u_y.copy()))
        print(f"Step {step} / {max_steps}")

# ---------------------------
# 动画：俯视图（Top View）
# ---------------------------
fig_top, ax_top = plt.subplots(figsize=(6,5))
im_top = ax_top.imshow(snapshots[0][0], cmap='jet', origin='lower', vmin=Tcold, vmax=Thot)
ax_top.set_title("Top View: Temperature & Velocity")
X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
quiver_top = ax_top.quiver(X, Y, snapshots[0][1], snapshots[0][2], scale=50, color='k')

def update_top(frame):
    T_frame, u_x_frame, u_y_frame = snapshots[frame]
    im_top.set_data(T_frame)
    quiver_top.set_UVC(u_x_frame, u_y_frame)
    ax_top.set_title(f"Top View, Snapshot {frame}")
    return im_top, quiver_top

anim_top = animation.FuncAnimation(
    fig_top, update_top, 
    frames=len(snapshots), 
    interval=1, 
    blit=False, 
    repeat=False
)

# ---------------------------
# 动画：左视图（Left View）——取 x = Nx//2 处的垂直剖面
# ---------------------------
fig_left, ax_left = plt.subplots(figsize=(4,5))
col_index = Nx // 2
line_temp, = ax_left.plot(snapshots[0][0][:, col_index], np.arange(Ny), 'r-', label='Temperature')
line_vel, = ax_left.plot(snapshots[0][2][:, col_index], np.arange(Ny), 'b-', label='Vertical Velocity')
ax_left.set_xlim(Tcold - 0.1, Thot + 0.1)
ax_left.set_ylim(0, Ny)
ax_left.invert_yaxis()
ax_left.set_xlabel("Value")
ax_left.set_ylabel("y")
ax_left.set_title("Left View: Vertical Profile")
ax_left.legend()

def update_left(frame):
    T_frame, u_x_frame, u_y_frame = snapshots[frame]
    line_temp.set_data(T_frame[:, col_index], np.arange(Ny))
    line_vel.set_data(u_y_frame[:, col_index], np.arange(Ny))
    ax_left.set_title(f"Left View, Snapshot {frame}")
    return line_temp, line_vel

anim_left = animation.FuncAnimation(
    fig_left, update_left, 
    frames=len(snapshots), 
    interval=1, 
    blit=False, 
    repeat=False
)

anim_top.save('field_w='+str(Nx)+'_h='+str(Ny)+'_TempDiff='+str(Thot-Tcold)+'.gif')
anim_left.save('curve_w='+str(Nx)+'_h='+str(Ny)+'_TempDiff='+str(Thot-Tcold)+'.gif')
plt.show()

# ---------------------------
# 对最终快照估算对流单元个数（考虑周期性边界）
# ---------------------------
final_u_y = snapshots[-1][2]
mid_row = final_u_y[Ny//2, :]
peak_count = 0
for i in range(Nx):
    left = mid_row[(i-1) % Nx]
    right = mid_row[(i+1) % Nx]
    if mid_row[i] > left and mid_row[i] > right and mid_row[i]> 0.001:
        peak_count += 1
print("Estimated convection cell count:", peak_count)
