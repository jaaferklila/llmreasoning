# =========================================================
# Imports
# =========================================================
import json
import os
import re
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import llm_only_answer, contains_correct_answer, convert_yes_no_to_bool


def main():
    parser = argparse.ArgumentParser(description="LLM-only evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b"]
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MQuAKE-CF-3k-v2",
        choices=["MQuAKE-CF-3k-v2", "strategyqa"],
        help="Dataset to evaluate"
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--edit_num", type=int, default=0)
    parser.add_argument("--nsample", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    edit_num = args.edit_num
    nsample = args.nsample

    print("#" * 60)
    print(f"Using model: {model_name}")
    print("#" * 60)
    print(f"Dataset: {dataset_name}")
    print("#" * 60)
    print(f"CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")

    # =========================================================
    # Load dataset
    # =========================================================

    if dataset_name.lower() == "strategyqa":
        dataset_path = f"./datasets/{dataset_name}/strategyqa_with_masks.json"
    else:
       dataset_path = f"./datasets/{dataset_name}/{dataset_name}.json"

# Load dataset
    with open(dataset_path, "r") as f:
        all_dataset = json.load(open(f"{dataset_path}", "r"))

    if edit_num == 0:
        edit_num = len(all_dataset)
    #all_dataset=all_dataset[:1]
    all_dataset_list = [all_dataset[i:i+edit_num] for i in range(0, len(all_dataset), edit_num)]
    print(f"Dataset size: {len(all_dataset)}")
    print("#" * 60)
    # =========================================================
    # Load model
    # =========================================================
    MODEL_NAME_TO_PATH = {
        "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
        "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
        #"llama3.1-8b": "/home/users/jklila/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    #"qwen2.5-7b": "/home/users/jklila/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    #"qwen2.5-14b": "/home/users/jklila/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    #"qwen2.5-32b": "/home/users/jklila/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    #"llama-3.1-70B":"/home/users/jklila/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
    }

    model_path = MODEL_NAME_TO_PATH[model_name]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if model_name == "qwen2.5-7b" else torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    torch.set_grad_enabled(False)

    model_class_name = model.__class__.__name__.lower()
    if "llama" in model_class_name:
        end_token_ids = [128001, 128009]
    elif "qwen" in model_class_name:
        end_token_ids = [151645, 151643]
    else:
        end_token_ids = [tokenizer.eos_token_id]

    # =========================================================
    # Load prompts
    # =========================================================
    mask_prompts = json.load(open("./prompt/llm_only.json"))
    strategy_prompts = json.load(open("./prompt/strategyqa_problems.json"))

    # =========================================================
    # Evaluation loop
    # =========================================================
    tot = 0
    correct = 0
    results = []

    output_path = f"./output/LLM_Only/{model_name}_{dataset_name}"
    os.makedirs(output_path, exist_ok=True)

    for dataset in all_dataset_list:
        for d in dataset:

            if dataset_name.lower() == "strategyqa":
                question = d.get("question")
                true_answer = d.get("answer")
                #facts = d.get("facts", [])
                print(f'The question is: {question}')
                print(f'The answer is: {true_answer}')
                correct_answers=[d.get("answer")]
                model_answer = llm_only_answer(
                question,
                model,
                tokenizer,
                end_token_ids,
                mask_prompts,
                strategy_prompts,
                dataset=dataset_name
            )
                pred=convert_yes_no_to_bool(model_answer)
                print(f'The model answer is: {pred}')
                tot += 1
                if(pred==true_answer):
                    correct += 1
                print(f'correct/total: {correct}/{tot}')
                accuracy = correct / tot if tot > 0 else 0.0
                print(f"Accuracy: {accuracy:.4f}")
                

            else:
                question = d["questions"][0]
                correct_answers = (
                    [d["answer"].lower()]
                    + [a.lower() for a in d.get("answer_alias", [])]
                )
                

                model_answer = llm_only_answer(
                    question,
                    model,
                    tokenizer,
                    end_token_ids,
                    mask_prompts,
                    strategy_prompts,
                    dataset=dataset_name
                )
                print(f'The correct answer is: {correct_answers}')
                print(f'The model answer is: {model_answer}')
                if contains_correct_answer(model_answer, correct_answers):
                    correct += 1
                tot += 1
                print(f'correct/total: {correct}/{tot}')
                results.append({
                    "question": question,
                    "correct_answers": correct_answers,
                    "model_answer": model_answer
                })

                if nsample > 0 and tot >= nsample:
                  break

                if nsample > 0 and tot >= nsample:
                   break

    # =========================================================
    # Save results
    # =========================================================
    final_accuracy = correct / tot if tot > 0 else 0.0

    with open(f"{output_path}/result.json", "w") as f:
            json.dump(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "accuracy": final_accuracy,
                                "total": tot,
                                "correct": correct,
                                "results": results
                            },
                            f,
                            indent=4
                        )

    print(f"\nFinal Accuracy: {final_accuracy:.4f}")
    print(f"Results saved to {output_path}/result.json")


# =========================================================
# Run main
# =========================================================
if __name__ == "__main__":
    main()

# =========================================================
# Imports
# =========================================================
import json
import os
import re
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# =========================================================
# Fonctions utilitaires
# =========================================================
def llm_only_answer(question, model, tokenizer, end_token_ids, mask_prompts, strategy_prompts, dataset, max_new_tokens=64):
    if dataset.lower() == "strategyqa":
        messages = strategy_prompts + [{"role": "user", "content": question}]
    else:
        messages = mask_prompts + [{"role": "user", "content": question}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=end_token_ids,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(
        outputs[0][input_ids.shape[-1]:],
        skip_special_tokens=True
    ).strip()

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def contains_correct_answer(model_answer, correct_answers):
    model_answer_norm = normalize_text(model_answer)
    for ans in correct_answers:
        if normalize_text(ans) in model_answer_norm:
            return True
    return False

def convert_yes_no_to_bool(answer_str):
    """Convert any form of 'yes'/'no' to True/False."""
    answer = answer_str.lower()
    if "yes" in answer:
        return True
    elif "no" in answer:
        return False
    else:
        return None  # or keep answer_str
# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="LLM-only evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="llama3.1-8b",
        choices=["llama3.1-8b", "qwen2.5-7b", "qwen2.5-14b","qwen2.5-32b","llama-3.1-70B"]
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MQuAKE-CF-3k-v2",
        choices=["MQuAKE-CF-3k-v2", "strategyqa"],
        help="Dataset to evaluate"
    )
    parser.add_argument("--cuda_visible_devices", type=str, default="0")
    parser.add_argument("--edit_num", type=int, default=0)
    parser.add_argument("--nsample", type=int, default=0)
    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    edit_num = args.edit_num
    nsample = args.nsample

    print("#" * 60)
    print(f"Using model: {model_name}")
    print("#" * 60)
    print(f"Dataset: {dataset_name}")
    print("#" * 60)
    print(f"CUDA_VISIBLE_DEVICES: {args.cuda_visible_devices}")

    # =========================================================
    # Load dataset
    # =========================================================

    if dataset_name.lower() == "strategyqa":
        dataset_path = f"./datasets/{dataset_name}/strategyqa_with_masks.json"
    else:
       dataset_path = f"./datasets/{dataset_name}/{dataset_name}.json"

# Load dataset
    with open(dataset_path, "r") as f:
        all_dataset = json.load(open(f"{dataset_path}", "r"))

    if edit_num == 0:
        edit_num = len(all_dataset)
    #all_dataset=all_dataset[:1]
    all_dataset_list = [all_dataset[i:i+edit_num] for i in range(0, len(all_dataset), edit_num)]
    print(f"Dataset size: {len(all_dataset)}")
    print("#" * 60)
    # =========================================================
    # Load model
    # =========================================================
    MODEL_NAME_TO_PATH = {
    "llama3.1-8b": "/home/users/jklila/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
    "qwen2.5-7b": "/home/users/jklila/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    "qwen2.5-14b": "/home/users/jklila/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
    "qwen2.5-32b": "/home/users/jklila/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct/snapshots/5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
    "llama-3.1-70B":"/home/users/jklila/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b"
    }
    model_path = MODEL_NAME_TO_PATH[model_name]
    # =========================================================
# Select dtype (H100 / A100 friendly)
# =========================================================

# =========================================================
# Attention implementation (Qwen3 FIX)
# =========================================================
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
    model_path,
    offload_folder="/tmp/offload",
    quantization_config=quantization_config,
    attn_implementation="eager",  # fixes H100 cuDNN SDPA issue
    device_map="auto",             # automatically splits model across GPUs
    low_cpu_mem_usage=True,
    local_files_only=True,
    )

    model.eval()
    torch.set_grad_enabled(False)

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.eval()
    torch.set_grad_enabled(False)

    model_class_name = model.__class__.__name__.lower()
    if "llama" in model_class_name:
        end_token_ids = [128001, 128009]
    elif "qwen" in model_class_name:
        end_token_ids = [151645, 151643]
    else:
        end_token_ids = [tokenizer.eos_token_id]

    # =========================================================
    # Load prompts
    # =========================================================
    mask_prompts = json.load(open("./prompt/llm_only.json"))
    strategy_prompts = json.load(open("./prompt/strategyqa_problems.json"))

    # =========================================================
    # Evaluation loop
    # =========================================================
    tot = 0
    correct = 0
    results = []

    output_path = f"./output/LLM_Only/{model_name}_{dataset_name}"
    os.makedirs(output_path, exist_ok=True)

    for dataset in all_dataset_list:
        for d in dataset:

            if dataset_name.lower() == "strategyqa":
                question = d.get("question")
                true_answer = d.get("answer")
                #facts = d.get("facts", [])
                print(f'The question is: {question}')
                print(f'The answer is: {true_answer}')
                correct_answers=[d.get("answer")]
                model_answer = llm_only_answer(
                question,
                model,
                tokenizer,
                end_token_ids,
                mask_prompts,
                strategy_prompts,
                dataset=dataset_name
            )
                pred=convert_yes_no_to_bool(model_answer)
                print(f'The model answer before normalisation is: {model_answer}')
                print(f'The final model answer is: {pred}')
                tot += 1
                if(pred==true_answer):
                    correct += 1
                print(f'correct/total: {correct}/{tot}')
                accuracy = correct / tot if tot > 0 else 0.0
                print(f"Accuracy: {accuracy:.4f}")
                

            else:
                question = d["questions"][0]
                correct_answers = (
                    [d["answer"].lower()]
                    + [a.lower() for a in d.get("answer_alias", [])]
                )
                

                model_answer = llm_only_answer(
                    question,
                    model,
                    tokenizer,
                    end_token_ids,
                    mask_prompts,
                    strategy_prompts,
                    dataset=dataset_name
                )
                print(f'The correct answer is: {correct_answers}')
                print(f'The model answer is: {model_answer}')
                if contains_correct_answer(model_answer, correct_answers):
                    correct += 1
                tot += 1
                print(f'correct/total: {correct}/{tot}')
                results.append({
                    "question": question,
                    "correct_answers": correct_answers,
                    "model_answer": model_answer
                })

                if nsample > 0 and tot >= nsample:
                  break

                if nsample > 0 and tot >= nsample:
                   break

    # =========================================================
    # Save results
    # =========================================================
    final_accuracy = correct / tot if tot > 0 else 0.0

    with open(f"{output_path}/result.json", "w") as f:
            json.dump(
                            {
                                "model": model_name,
                                "dataset": dataset_name,
                                "accuracy": final_accuracy,
                                "total": tot,
                                "correct": correct,
                                "results": results
                            },
                            f,
                            indent=4
                        )

    print(f"\nFinal Accuracy: {final_accuracy:.4f}")
    print(f"Results saved to {output_path}/result.json")


# =========================================================
# Run main
# =========================================================
if __name__ == "__main__":
    main()

