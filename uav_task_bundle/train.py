"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
# 将标准输出和标准错误流设置为行缓冲模式
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# 注册一个名为 "eval" 的 OmegaConf 解析器，使得在 YAML 配置文件中可以使用 ${eval:Python_expression} 语法来执行 Python 代码
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config')),
    config_name="train_uav_final"
)
def main(cfg: OmegaConf):
    # 解析 cfg 对象中所有尚未解析的引用和表达式，确保所有配置值在程序启动时都被确定下来，保证实验的可复现性。
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
