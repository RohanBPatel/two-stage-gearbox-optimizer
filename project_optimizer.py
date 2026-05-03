from parts import *
# import signal
# import sys
from multiprocessing import Manager

np.random.seed(42)

history_path = table_path / "history.csv"

alpha_depth = 2.25
alpha_median_fos = 0.0001
alpha_F_R = 0.001

def loss(x):
    if isinstance(x, Gearbox):
        gearbox = x
    else:
        gearbox = Gearbox.from_scaled(x)

    depth = gearbox.depth # mm
    depth_m = depth / 1e3 # m
    min_fos = gearbox.min_fos
    median_fos = gearbox.median_fos

    max_F_R = max(gearbox.shaft.bearing_A.F_R, gearbox.shaft.bearing_B.F_R) / 1e3 # kN

    # min_fos = max(0.0, min_fos)

    depth_loss = alpha_depth * depth_m
    # fos_loss = (min_fos - n_target)**2
    # fos_loss = -min_fos
    fos_loss = max(0.0, -(min_fos - n_target)) - alpha_median_fos * median_fos
    F_R_loss = alpha_F_R * max_F_R

    # cur_loss = depth_loss - min_fos
    cur_loss = depth_loss + fos_loss + F_R_loss

    # print(f"{counter:5d}: {depth_loss = :.4f}, {min_fos = :.4f}, {cur_loss = :.4f}")
    # print(f"{depth_loss = :.4f}, {min_fos = :.4f}, {cur_loss = :.4f}")
    # logging.info(f"{depth_loss = :.4f}, {min_fos = :.4f}, {cur_loss = :.4f}")
    
    with open(history_path, "a") as f:
        f.write(f"{gearbox.min_fos_key},{depth_m},{min_fos},{cur_loss}\n")

    return cur_loss

def plot_iterations():
    df = pd.read_csv(history_path, names=[
        "fos_label", "depth_m", "min_fos", "cur_loss"
    ])
    
    if df.empty:
        return
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(df["cur_loss"], lw=1, alpha=0.7, label="Total Loss")
    axs[0].plot(df["cur_loss"].cummin(), label="Best Loss")
    axs[0].set_yscale("symlog")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Loss Metrics")
    axs[0].legend()
    axs[0].grid()

    ax2_twin = axs[1].twinx()
    line1 = axs[1].plot(df["depth_m"], lw=1, alpha=0.4, color="C0", label="Depth (m)")[-1]
    line2 = ax2_twin.plot(df["min_fos"], lw=1, alpha=0.4, color="C1", label="Min F.O.S.")[-1]
    
    axs[1].set_ylabel("Depth (m)", color=line1.get_color())
    ax2_twin.set_ylabel("Min Factor of Safety", color=line2.get_color())
    axs[1].set_xlabel("Evaluation Step")
    
    # lines = line1 + line2
    # labels = [l.get_label() for l in lines]
    # axs[1].legend(lines, labels)
    # axs[1].legend()
    
    axs[1].set_title("Physical Metrics")
    axs[1].grid()

    plt.tight_layout()
    plt.savefig(fig_path / "Iterations")

def main():
    with open(history_path, "w") as f:
        # clear file
        pass 

    result = differential_evolution(
        func=loss,
        bounds=optim_vars_df[["Min Opt Bounds", "Max Opt Bounds"]].values,
        maxiter=16,
        popsize=10,
        polish=True,
        # x0=x_init,
        integrality=optim_vars_df["Discrete?"].values,
        disp=True,
        workers=os.cpu_count() - 4,
    )

    # df = pd.DataFrame(list(managed_history), columns=[
    #     "depth_m", "min_fos", "depth_loss", "fos_loss", "cur_loss"
    # ])

    print(result)
    print(result.x)
    print("[" + ", ".join(map(str, result.x)) + "]")
    
    plot_iterations()

if __name__ == "__main__":
    main()