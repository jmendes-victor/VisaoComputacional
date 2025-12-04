import warnings
import numpy as np

from src.config import SEED, CSV_DIR, EMB_DIR, PAPER_TARGETS
from src.utils import set_seed
from src.dataset import get_tasks_scenario_3
from src.trainer import train_engine
from src.joint_trainer import train_joint_learning


def print_results(results: dict) -> None:
    print("\n\n")
    print("=" * 115)
    print(f"{'RELATÓRIO DE REPRODUÇÃO - COMPARAÇÃO COM O ARTIGO':^115}")
    print("=" * 115)
    
    header = f"{'MÉTODO':<15} | {'MEU AACC':<10} | {'PAPER':<10} | {'MEU BWT':<10} | {'PAPER':<10} | {'MEU IM':<10} | {'PAPER':<10}"
    print(header)
    print("-" * 115)

    def print_row(name: str, key: str) -> None:
        my_acc, my_bwt, my_im = results[key]
        tgt_acc = PAPER_TARGETS[key]['AACC']
        tgt_bwt = PAPER_TARGETS[key]['BWT']
        tgt_im = PAPER_TARGETS[key]['IM']
        
        bwt_str = f"{my_bwt:.4f}" if key != 'JT' else "N/A"
        im_str = f"{my_im:.4f}" if key != 'JT' else "N/A"
        tgt_bwt_str = f"{tgt_bwt:.4f}" if key != 'JT' else "N/A"
        tgt_im_str = f"{tgt_im:.4f}" if key != 'JT' else "N/A"

        print(f"{name:<15} | {my_acc:<10.4f} | {tgt_acc:<10.4f} | {bwt_str:<10} | {tgt_bwt_str:<10} | {im_str:<10} | {tgt_im_str:<10}")

    print_row("Fine-Tuning", 'FT')
    print_row("ER (Replay)", 'ER')
    print_row("Ours (AKD)", 'OURS')
    print("-" * 115)
    print_row("Joint Training", 'JT')
    print("=" * 115)


def main():
    warnings.filterwarnings("ignore")
    
    print("=" * 60)
    print(" Reprodução do Experimento (Cenário E3)")
    print("=" * 60)
    
    cl_tasks = get_tasks_scenario_3(CSV_DIR, EMB_DIR)
    
    results = {}
    
    set_seed(SEED)
    acc_ft, bwt_ft, diag_ft = train_engine("FT", cl_tasks)
    
    set_seed(SEED)
    acc_er, bwt_er, diag_er = train_engine("ER", cl_tasks)

    set_seed(SEED)
    acc_ours, bwt_ours, diag_ours = train_engine("OURS", cl_tasks)
    
    set_seed(SEED)
    acc_jt, jt_per_task = train_joint_learning(CSV_DIR, EMB_DIR)

    im_ft = np.mean(diag_ft - jt_per_task)
    im_er = np.mean(diag_er - jt_per_task)
    im_ours = np.mean(diag_ours - jt_per_task)

    results['FT'] = (acc_ft, bwt_ft, im_ft)
    results['ER'] = (acc_er, bwt_er, im_er)
    results['OURS'] = (acc_ours, bwt_ours, im_ours)
    results['JT'] = (acc_jt, 0.0, 0.0)

    print_results(results)
    
    return results


if __name__ == "__main__":
    main()
