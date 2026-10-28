"""实践一入口：在前 12 小时上核对 L=3 与 L=5 的窗口。"""

from data import load_etth1, make_windows
from reporting import save_window_examples


def main():
    frame = load_etth1(limit=12)
    values = frame["OT"].to_numpy(dtype=float)
    times = frame["date"]
    print("前 12 小时的日期与油温：")
    print(frame[["date", "OT"]].to_string(index=True))
    rows = []
    for window_size in (3, 5):
        x, y, target_index = make_windows(values, window_size)
        print(f"L={window_size}: X{x.shape}, y{y.shape}, target_index{target_index.shape}")
        for j in (0, len(y) - 1):
            t = int(target_index[j])
            print(f"  {'首' if j == 0 else '尾'}条: 输入 {times.iloc[t-window_size]}–{times.iloc[t-1]}, "
                  f"{x[j].round(3).tolist()}; 目标 {times.iloc[t]} = {y[j]:.3f}")
        for window, target, t in zip(x, y, target_index):
            t = int(t)
            rows.append({"window_size": window_size, "input_start": t - window_size,
                         "input_end": t - 1, "target_index": t,
                         "target_time": times.iloc[t].isoformat(sep=" "),
                         "input_values": str(window.round(6).tolist()),
                         "target_value": float(target)})
    save_window_examples(rows)
    print(f"已保存 {len(rows)} 条记录至 results/window_examples.csv")


if __name__ == "__main__":
    main()
