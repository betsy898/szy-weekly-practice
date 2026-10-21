"""实践一入口：连接数据、模型、优化器、一次更新与结果保存。"""

from data import prepare_step_data
from models import create_step_model
from optimizers import create_optimizer
from reporting import save_step_trace
from training import run_single_update


def main():
    x, y = prepare_step_data()
    model = create_step_model()
    optimizer = create_optimizer(model, "sgd", learning_rate=0.1)
    trace = run_single_update(model, optimizer, x, y)
    save_step_trace(trace)
    for row in trace:
        print(row)
    print("Saved results/step_trace.csv")


if __name__ == "__main__":
    main()
