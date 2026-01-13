import json
import os
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

#from utils import get_sent_embeddings
import json
from tqdm import tqdm
from utils import answer_masked_questions, Sample_MASK_COT, get_Fillin_output, convert_yes_no_to_bool
# Argument parsing
def main():
    parser = argparse.ArgumentParser(description="LLM-only evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b"]
    )
    parser.add_argument("--dataset_name", type=str, default="MQuAKE-CF-3k-v2",
                        choices=["MQuAKE-CF-3k-v2", "strategyqa"],
                        help="Dataset to evaluate")
    args = parser.parse_args()

    # Load configuration from config.json
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

    # Extract configurations
    edit_num = config["edit_num"]
    beta = config["beta"]
    alpha = config["alpha"]
    nsample = config["nsample"]
    model_name = config["model_name"]
    seed = config["seed"]
    model_name = args.model_name 

    # Load dataset
    dataset_name = args.dataset_name
    if dataset_name.lower() == "strategyqa":
        dataset_path = f"./datasets/{dataset_name}/strategyqa_with_masks.json"
    else:
        dataset_path = f"./datasets/{dataset_name}/{dataset_name}.json"

    # Load dataset
    with open(dataset_path, "r") as f:

        all_dataset = json.load(open(f"{dataset_path}", "r"))

    if edit_num == 0:
        edit_num = len(all_dataset)
    #all_dataset=all_dataset[:4]
    all_dataset_list = [all_dataset[i:i+edit_num] for i in range(0, len(all_dataset), edit_num)]
    print(f"Dataset size: {len(all_dataset)}")



    #pdb.set_trace()
    MODEL_NAME_TO_PATH = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-7b":"Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
    }
    model_path = MODEL_NAME_TO_PATH[model_name]
    #model_path = MODEL_NAME_TO_PATH[model_name]

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float32 if "qwen2.5-7b" in model_name else torch.float16,
            device_map='auto',
            low_cpu_mem_usage=True)
    llmtokenizer = AutoTokenizer.from_pretrained(model_path)

    llmtokenizer.pad_token = llmtokenizer.eos_token
    llmtokenizer.padding_side = "left"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    mask_prompts = json.load(open("./prompt/mask.json"))



    # ===================== Boucle principale ===================== #
    output_path = f"./output/LLM_with_MSK/{model_name}_{dataset_name}_{nsample}"
    os.makedirs(output_path, exist_ok=True)

    for dataset in all_dataset_list:
        tot = 0
        correct = 0
        multiple_cor_base = 0
        multiple_cor_bon = 0
        if "strategyqa" in dataset_name:  # Ensure you're handling strategyqa dataset
            results = []
            for d in tqdm(dataset):
                questions = d.get("decomposition_cot")
                q= d.get("question", [])
                true_ans = d.get("answer")
                facts = d.get("facts")
                if not questions:
                    continue
            
                # Generate predicted masks
                predicted_masks = answer_masked_questions(questions, model, llmtokenizer, facts=facts)
                print(predicted_masks)
            
                # Convert model's MASK ANS to boolean
                pred_ans = convert_yes_no_to_bool(predicted_masks.get("MASK ANS", ""))
                print(f'pred_ans : {pred_ans}')
                # Skip answers that are not valid 'yes' or 'no'
                if pred_ans not in [True, False]:
                    continue
                print(f'Model answer: {pred_ans}')
                result_data = {
                    "question": q,
                    "correct_answers": str(true_ans),
                    "Model_answer":  str(pred_ans),
                }
                # Ground truth
                results.append(result_data)
                print(f'Ground truth : {true_ans}')
                # Compare only if ground truth exists
                if true_ans is not None:
                    tot += 1
                    if pred_ans == true_ans:
                        correct += 1
                    print(f'correct/total: {correct}/{tot}')
                    print(f"Accuracy: {correct/tot:.4f}")
            
            accuracy = correct / tot if tot > 0 else 0.0
            print(f"Accuracy: {accuracy:.4f} ({correct}/{tot})")
            with open(f"{output_path}/result.json", "w") as f:
                json.dump(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "accuracy": accuracy,
                        "results": results,
                    
                    },
                    f,
                    indent=4
                )
            print(f"Results saved to {output_path}/result.json")
            
            
                
        else:      
            results = []
            cate_scores_res = []
        
            for d in tqdm(dataset):
                correct_answers = [d["answer"].lower()] + [a.lower() for a in d["answer_alias"]]
                tot += 1
                q = d["questions"][0]
                print(f'The question is : {q}')
                result_data = {
                    "question": q,
                    "correct_answers": correct_answers,
                    "right_cot": [a["cloze"] + " " + a["answer"] for a in d["single_hops"]],
                }
                print(f' Correct Answer are : {correct_answers}')
                mask_cots_data = Sample_MASK_COT(q, model, llmtokenizer, num_samples=6)
                reses = get_Fillin_output(mask_cots_data, model, llmtokenizer)
                cate_scores = [None]*len(reses)  # placeholder si pas de scoring externe
                
                # Base Multi-hop Accuracy
                flag = False
                for res in reses:
                    if res["mask_cot"]["idx"] >= 2:
                        continue
                    if res["new"]["new_answer"] and res["new"]["new_answer"].lower() in correct_answers:
                        flag = True
                    break
                print(f'Model Answeris : {res["new"]["new_answer"]}')
                # BoN Multi-hop Accuracy
                bon_flag = False
                is_right, rpp_scores, ki_scores = [], [], []
                for res, pe_cate in zip(reses, cate_scores):
                    
                    if res["new"]["new_answer"]:
                        is_right.append(res["new"]["new_answer"].lower().strip() in correct_answers)
                        rpp_scores.append(-np.mean(res["mask_cot"]['cate_el']))
                        ki_scores.append(sum([1 if y["right"] else 0 for y in pe_cate]) / len(pe_cate) if pe_cate else 0)
                if is_right:
                    if len(rpp_scores) > nsample // 2:
                        max_values, max_indices = torch.topk(torch.tensor(rpp_scores), nsample // 2)
                    else:
                        max_indices = list(range(len(rpp_scores)))
                    idx_both = max_indices[np.argmax([ki_scores[i] for i in max_indices])]
                    if is_right[idx_both]:
                        bon_flag = True
        
                if flag:
                    multiple_cor_base += 1
                if bon_flag:
                    multiple_cor_bon += 1
        
                results.append(result_data)
                cate_scores_res.append(cate_scores)
        
                print(f"DecKER Base Multi-hop Accuracy: {multiple_cor_base}/{tot} = {multiple_cor_base/tot}")
                print(f"DecKER BoN Multi-hop Accuracy: {multiple_cor_bon}/{tot} = {multiple_cor_bon/tot}")
            accuracy_multiple_cor_base = multiple_cor_base / tot
            accuracy_multiple_cor_bon = multiple_cor_bon / tot
            with open(f"{output_path}/result.json", "w") as f:
                json.dump(
                    {
                        "model": model_name,
                        "dataset": dataset_name,
                        "accuracy_multiple_cor_base": accuracy_multiple_cor_base,
                        "accuracy_multiple_cor_bon": accuracy_multiple_cor_bon,
                        "total": tot,
                        "correct_multiple_cor_base": multiple_cor_base,
                        "correct_multiple_cor_bon": multiple_cor_bon,
                        "results": results
                    },
                    f,
                    indent=4
                )

            print(f"\nFinal Accuracy multiple_cor_base: {accuracy_multiple_cor_base:.4f}")
            print(f"\nFinal Accuracy multiple_cor_bon: {accuracy_multiple_cor_bon:.4f}")
            print(f"Results saved to {output_path}/result.json")

# =========================================================
# Run main
# =========================================================
if __name__ == "__main__":
    main()


