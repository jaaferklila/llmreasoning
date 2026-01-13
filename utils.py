import torch
import numpy as np
import re
import json
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

mask_prompts = json.load(open("./prompt/mask.json"))
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
        pad_token_id=tokenizer.eos_token_id,
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
def get_mask(q, model, llmtokenizer, num_return_sequences=1):
    msg = mask_prompts + [{"role": "user", "content": q}]

    generation_config = dict(
                            do_sample=True,
                            max_new_tokens=100,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_logits = True,
                            output_scores = True,
                            top_p=0.95,
                            temperature=1.2,
                            num_return_sequences=num_return_sequences+1)
        
    input_ids = llmtokenizer.apply_chat_template(
        msg,
        add_generation_prompt=True,
    )
    
    input_ids += llmtokenizer.encode("[STEP] ", add_special_tokens=False)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        logits_processor=logits_processor,
        **generation_config
    )
    with torch.no_grad():
        model_name = model.__class__.__name__
        if "llama" in model_name.lower():
            end_token_ids = [128001, 128009]
        elif "qwen" in model_name.lower():
            end_token_ids = [151645,151643]
        elif "olmo" in model_name.lower():
            end_token_ids = [100257, 100265]
        start_cate = llmtokenizer.encode("[CATEGORY", add_special_tokens=False)
        # pl, el = [[]]*num_return_sequences, [[]]*num_return_sequences
        # pl_nocate, el_nocate = [[]]*num_return_sequences, [[]]*num_return_sequences
        pl, el, pl_nocate, el_nocate = [], [], [], []
        completed = []
        for _ in range(num_return_sequences):
            pl.append([])
            el.append([])
            pl_nocate.append([])
            el_nocate.append([])
            completed.append(False)
        
        
        outputs_ = []
        for i in range(num_return_sequences):
            response = outputs.sequences[i][input_ids.shape[-1]:]   
            output_ = llmtokenizer.decode(response).strip()
            outputs_.append(output_)
        
        # for i in range(len(outputs.logits)):
        #     now_id = outputs.sequences[0][input_ids.shape[-1]+i].item()
            
        for i in range(len(outputs.logits)):
            probabilities = F.softmax(outputs.logits[i], dim=-1)
            # log_probabilities = torch.log(probabilities)
            # entropy = -probabilities * log_probabilities
            # entropy_sum2 = torch.sum(entropy, dim=-1)
            entropy = torch.special.entr(probabilities)
            entropy_sum = torch.sum(entropy, dim=-1)
            
            # print(entropy_sum)
            
            for j in range(num_return_sequences):
                now_id = outputs.sequences[j][input_ids.shape[-1]+i].item()
                # print(llmtokenizer.decode([now_id]))
                if i < len(outputs.logits) - len(start_cate) and outputs.sequences[j][input_ids.shape[-1]+i:input_ids.shape[-1]+i+len(start_cate)].tolist() == start_cate:
                    completed[j] = True

                if now_id in end_token_ids:
                    continue
                if not completed[j]:
                    pl_nocate[j].append(probabilities[j][now_id].item())
                    el_nocate[j].append(entropy_sum[j].item())
                pl[j].append(probabilities[j][now_id].item())
                el[j].append(entropy_sum[j].item())

    return outputs_, pl, el, pl_nocate, el_nocate




#sert à générer plusieurs chaînes de raisonnement (Chain-of-Thought) pour une question q,
#  mais avec des parties masquées ([MASK 1], [MASK ANS], etc.) :
def Sample_MASK_COT(q, model, llmtokenizer, num_samples=6):
    assert num_samples > 1
    mask_cots, pls, els, pls_notcate, els_notcate = get_mask(q, model, llmtokenizer, num_return_sequences=num_samples-1)
    filtered_mask_cots = []
    for j, cot in enumerate(mask_cots):
        # print(j)
        # print(cot)
        # print("=====================================")
        if "\n[CATEGORY]" not in cot:
            continue
        if "MASK ANS" not in cot:
            continue
        categories = cot.split("\n[CATEGORY]")[-1].strip()

        categories = re.findall(r'\[([^\]]+)\]', categories)
        categories = [c for c in categories if "MASK" not in c]
        mask_cot = cot.split("\n[CATEGORY]")[0]
        masked_part = re.findall(r'\[MASK [^\]]+\]', cot)
        mask_num_l = 0
        for mask in masked_part:
            try:
                mask_num_l = max(mask_num_l, int(mask[6:-1]))
            except:
                pass
        masked_part = list(set(masked_part))
        masked_part.sort()
        mask_num_l += 1
        cot_data = {
            "idx": j,
            "mask_cot": mask_cot,
            "categories": categories,
            "masked_part": masked_part,
            "masked_num": mask_num_l,
            "cate_pl": pls[j],
            "cate_el": els[j],
            "notcate_pl": pls_notcate[j],
            "notcate_el": els_notcate[j],
        }
            
        mask_map = {}
        for m in range(min(len(masked_part), len(categories))):
            mask_map[masked_part[m]] = categories[m]
        cot_data["mask_map"] = mask_map
        filtered_mask_cots.append(cot_data)
    return filtered_mask_cots



class MaxScoreLogitsProcessor(LogitsProcessor):
    def __init__(self):
        super().__init__()

    def __call__(self, input_ids, scores) -> torch.Tensor:
        max_idx = torch.argmax(scores[0])
        # scores[0][0:max_idx] = -float("inf")
        # scores[0][max_idx+1:] = -float("inf")
        new_scores = scores.clone()
        new_scores[0][0:max_idx] = -float("inf")
        new_scores[0][max_idx+1:] = -float("inf")
        return new_scores
logits_processor = LogitsProcessorList([MaxScoreLogitsProcessor()])


prompts_extract = json.load(open("./prompt/extract_with_entity.json"))
prompts_extract_no_entity = json.load(open("./prompt/extract.json"))
def get_new(sentences, model, llmtokenizer, type_=None):
    msgs = []

    for s, t in zip(sentences, type_):
        if t is None or t.strip() == "":
            msgs.append(prompts_extract_no_entity + [{"role": "user", "content": f"Sentence: {s}"}])
        else:
            msgs.append(prompts_extract + [{"role": "user", "content": f"Type of the masked entity: {t}.\nSentence: {s}"}])
          
    generation_config = dict(
        do_sample=False,
        num_beams=1,
        max_new_tokens=100,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_scores=True
    )
    
    input_ids = llmtokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        padding=True,
    )
    
    input_ids = torch.tensor(input_ids).to(model.device)
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    
    outputs_ = []
    for i in range(len(sentences)):
        response = outputs.sequences[i][input_ids.shape[-1]:]
        output_ = llmtokenizer.decode(response, skip_special_tokens=True).split("\n")[0].strip()
        if output_.endswith("."):
            output_ = output_[:-1]
        outputs_.append(output_)
    
    return outputs_


def Fillin_MASK_COT(mask_cot_data, model, llmtokenizer):
   
    mask_cot = mask_cot_data["mask_cot"]
    types = mask_cot_data["categories"]
    mask_parts = mask_cot_data["masked_part"]
    
    # Créer un mapping masque -> type
    mask_map = {}
    for i in range(min(len(mask_parts), len(types))):
        mask_map[mask_parts[i]] = types[i]
    
    # Diviser le COT en étapes
    cots = [c.strip() for c in mask_cot.split("[STEP]") if c.strip()]
    
    now_cot = []  # COT final avec masques remplis
    now_keywords = [""]  # Mots-clés extraits
    ans = ""  # Réponse finale (si [MASK ANS] est présent)
    
    for i in range(len(cots)):
        # Trouver tous les masques dans l'étape actuelle
        masked_part = re.findall(r'\[MASK [^\]]+\]', cots[i])
        
        if not masked_part:
            # Pas de masque dans cette étape
            now_keywords.append("")
            now_cot.append(cots[i])
            continue
        
        # Remplir chaque masque avec le LLM
        for mask in masked_part:
            # Préparer la phrase pour le LLM
            sentence_with_mask = cots[i]
            
            # Récupérer le type du masque
            mask_type = mask_map.get(mask, " ")
            
            # Utiliser le LLM pour remplir le masque
            mask_word = get_new(
                [sentence_with_mask], 
                model, 
                llmtokenizer, 
                [mask_type]
            )[0]
            
            # Remplacer le masque dans toutes les étapes futures
            for j in range(i, len(cots)):
                cots[j] = cots[j].replace(mask, mask_word)
            
            # Enregistrer la réponse si c'est [MASK ANS]
            if mask == "[MASK ANS]":
                ans = mask_word
            
            # Ajouter aux mots-clés
            now_keywords.append(mask_word)
        
        # Ajouter l'étape avec masques remplis
        now_cot.append(cots[i])
    
    return {
        "new_thoughts": now_cot,
        "new_answer": ans,
        "docs": [],  # Liste vide car pas de récupération
        "masked_words": [nk for nk in now_keywords[1:] if nk]
    }


def get_Fillin_output(all_mask_cot_data, model, llmtokenizer):
    """
    Fonction principale simplifiée pour traiter tous les COT masqués
    """
    results = []
    
    for mask_cot_data in all_mask_cot_data:
        # Remplir les masques pour chaque COT
        result = Fillin_MASK_COT(mask_cot_data, model, llmtokenizer)
        
        results.append({
            "new": result,
            "mask_cot": mask_cot_data,
        })
    
    return results




import re
extract_strategyqa= json.load(open("./prompt/extract_strategyqa.json"))

def generate_answers(sentences, model, llmtokenizer, force_yesno=False):
    msgs = []
    llmtokenizer.pad_token = llmtokenizer.eos_token
    llmtokenizer.padding_side = "left"

    for s in sentences:
        if force_yesno:
            msgs.append([
                {"role": "system", "content": "Answer only 'yes' or 'no'."},
                {"role": "user", "content": s}
            ])
        else:
            msgs.append(extract_strategyqa + [
                {"role": "user", "content": f"Sentence: {s}"}
            ])
    generation_config = dict(
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=100,
                            return_dict_in_generate=True,
                            output_hidden_states=True,
                            output_scores = True)
    input_ids = llmtokenizer.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        padding=True,
    )

    input_ids = torch.tensor(input_ids).to(model.device)
    outputs = model.generate(
        input_ids,
        pad_token_id=llmtokenizer.eos_token_id,
        attention_mask=input_ids.ne(llmtokenizer.eos_token_id),
        **generation_config
    )
    outputs_ = []
    for i in range(len(sentences)):
        
        response = outputs.sequences[i][input_ids.shape[-1]:]   
        output_ = llmtokenizer.decode(response, skip_special_tokens=True).split("\n")[0].strip()
        if output_.endswith("."):
            output_ = output_[:-1]
        outputs_.append(output_)
        
    return outputs_





def convert_yes_no_to_bool(answer_str):
    """Convert 'yes'/'no' to True/False."""
    answer = answer_str.strip().lower()
    if answer == "yes":
        return True
    elif answer == "no":
        return False
    else:
        return answer  # Keep as string if not yes/no
MASK_PATTERN = re.compile(r"\[(MASK\s+[A-Z0-9]+)\]")

def clean_question(question, mask_dict):
    """
    Replace known masks with answers and remove remaining masks.
    """
    # Replace known masks
    for k, v in mask_dict.items():
        question = question.replace(f"[{k}]", str(v))

    # Remove unresolved masks (model should not see them)
    question = MASK_PATTERN.sub("", question)

    return question.strip()


def answer_masked_questions(questions, model, llmtokenizer, facts=None):
    """
   Generate answers to the masked questions.
   Inject facts into each question when available.
    """
    mask_dict = {}

    for idx, q in enumerate(questions):
        masks = MASK_PATTERN.findall(q)
        cleaned_q = clean_question(q, mask_dict)

        # La dernière question est toujours une question yes/no
        force_yesno = (idx == len(questions) - 1)

        # Injecter les faits dans chaque question si disponibles
        if facts:
            fact_context = (
                            "Use the following information as background knowledge. "
                            "Do not mention it explicitly in the response.\n"
                            + " ".join(facts) + "\n\n"
                        )
            cleaned_q_with_facts = fact_context + cleaned_q
        else:
            cleaned_q_with_facts = cleaned_q

        # Générer la réponse
        answer = generate_answers([cleaned_q_with_facts], model, llmtokenizer, force_yesno=force_yesno)[0]

        # Assigner la réponse au dernier masque
        if masks:
            mask_dict[masks[-1]] = answer

    return mask_dict

