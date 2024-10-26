from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, tifa_score_single,  VQAModel
from tifascore import get_llama2_pipeline, get_llama2_question_and_answers
import json
from config import RunConfig
import os
import pandas as pd
import csv

def readCSV(eval_path):
    df = pd.read_csv(os.path.join(eval_path,'QBench.csv'),dtype={'id': str})
    return df

def assignResults(id, prompt,seed,result):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'tifa_score':None,
                'object_q':None,
                'object_s':None,
                'human_q':None,
                'human_s':None,
                'animal_q':None,
                'animal_s':None,
                'animal/human_q':None,
                'animal/human_s':None,
                'food_q':None,
                'food_s':None,
                'activity_q':None,
                'activity_s':None,
                'attribute_q':None,
                'attribute_s':None,
                'counting_q':None,
                'counting_s':None,
                'color_q':None,
                'color_s':None,
                'material_q':None,
                'material_s':None,
                'spatial_q':None,
                'spatial_s':None,
                'location_q':None,
                'location_s':None,
                'shape_q':None,
                'shape_s':None,
                'other_q':None,
                'other_s':None
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    
    for question in result["question_details"].keys(): 
        type=result["question_details"][question]["element_type"]
        score_by_type=result["question_details"][question]["scores"]
        new_row[str(type)+'_q']=question
        new_row[str(type)+'_s']=score_by_type
    
    new_row['tifa_score'] = result['tifa_score']
    return new_row

def assignQuestionDetails(id, prompt,seed,scores):
    questions={
        "question":{}
    }

    for question in scores["question_details"].keys():
        #uniqueid=id+prompt.replace(" ", "")+str(seed)

        questions['question'][question]={
                'element':scores["question_details"][question]["element"], # type: ignore
                'element_type':scores["question_details"][question]["element_type"], # type: ignore
                'choices':scores["question_details"][question]["choices"], # type: ignore
                'free_form_vqa':scores["question_details"][question]["free_form_vqa"], # type: ignore
                'multiple_choice_vqa':scores["question_details"][question]["multiple_choice_vqa"], # type: ignore
                'score_by_question':scores["question_details"][question]["scores"]
            } 
    return questions


def main(config : RunConfig):
    #Load the models
    unifiedqa_model = UnifiedQAModel(config.qa_model)
    vqa_model = VQAModel(config.vqa_model)
    #llama2 for local gpt model, from Hugging Face
    pipeline = get_llama2_pipeline(config.gpt_model)

    if not (os.path.isdir(config.eval_path)):
        print("Evaluation folder not found!")
    else:
        
        models_to_evaluate = []

        for model in os.listdir(config.eval_path):
            if(os.path.isdir((os.path.join(config.eval_path,model)))):
                #key is the model name
                models_to_evaluate.append({
                    'batch_gen_images_path':(os.path.join(config.eval_path,model)),#example:evaluation/QBench/QBench-SD14
                    'folder_name':model, #example:QBench-SD14,
                    'name':model[model.find('-')+1:]
                    })
        
        for model in models_to_evaluate:
            scores_df = pd.DataFrame({
            'id':[],
            'prompt':[],
            'seed':[], 
            'tifa_score':[],
            'object_q':[],
            'object_s':[],
            'human_q':[],
            'human_s':[],
            'animal_q':[],
            'animal_s':[],
            'animal/human_q':[],
            'animal/human_s':[],
            'food_q':[],
            'food_s':[],
            'activity_q':[],
            'activity_s':[],
            'attribute_q':[],
            'attribute_s':[],
            'counting_q':[],
            'counting_s':[],
            'color_q':[],
            'color_s':[],
            'material_q':[],
            'material_s':[],
            'spatial_q':[],
            'spatial_s':[],
            'location_q':[],
            'location_s':[],
            'shape_q':[],
            'shape_s':[],
            'other_q':[],
            'other_s':[]
            })

            question_details={
                'id_prompt_seed':{}
            }

                
            images = []
            #id,prompt,obj1,bbox1,token1,obj2,token2,obj3,token3,obj4,bbox4,token4
            prompt_collection = readCSV(config.eval_path)
            for index,row in prompt_collection.iterrows(): 
                #prompt_img_path = os.path.join(model[0],prompt[0]+'_'+prompt[1])
                prompt_gen_images_path = os.path.join(model['batch_gen_images_path'],row['id']+'_'+row['prompt'])
                #prompt = prompt[1]
                for img_filename in os.listdir(prompt_gen_images_path):
                    if not img_filename.endswith((".csv",".png")):
                        img_path = os.path.join(prompt_gen_images_path,img_filename)
                        if(os.path.isfile(img_path)):
                            images.append({
                                'prompt_gen_images_path':prompt_gen_images_path,
                                'img_path': img_path,
                                'img_filename':img_filename,
                                'prompt_id':row['id'],
                                'prompt':row['prompt'],
                                'model':model['name'],
                                'seed':img_filename.split('.')[0]
                            })
                            
            images.sort(key=lambda x: (int(x['prompt_id']), int(x['seed'])))
            
            print("Starting evaluation process")
            
            #initialization
            prompt = images[0]['prompt']
            llama2_questions=get_llama2_question_and_answers(pipeline,prompt)
            filtered_questions=filter_question_and_answers(unifiedqa_model, llama2_questions)

            for image in images:
                img_path = image['img_path']
                if(prompt != image['prompt']):#when prompt changes, questions and answers change too otherwise it's unnecessary
                    prompt = image['prompt']
                    llama2_questions = get_llama2_question_and_answers(pipeline,prompt)
                    # Filter questions with UnifiedQA
                    filtered_questions = filter_question_and_answers(unifiedqa_model, llama2_questions)

                print("----")
                print("PROMPT:",prompt)
                print("PATH:",img_path)
                
                # calculate TIFA score
                scores = tifa_score_single(vqa_model, filtered_questions, img_path)

                new_scores_row=assignResults(image['prompt_id'],image['prompt'],image['seed'],scores)
                new_question_details_rows=assignQuestionDetails(image['prompt_id'],image['prompt'],image['seed'],scores)

                scores_df = pd.concat([scores_df, pd.DataFrame([new_scores_row])], ignore_index=True)
                question_details['id_prompt_seed'][image['prompt_id']+image['prompt'].replace(" ", "")+str(image['seed'])]=new_question_details_rows

                print("SCORE: ", scores['tifa_score'])

            
            #output to csv
            scores_df.to_csv(os.path.join(model['batch_gen_images_path'],model['folder_name']+'.csv'), index=False)
            #dump question details to json
            with open(os.path.join(model['batch_gen_images_path'],model['folder_name']+'.json'), 'w') as fp:
                json.dump(question_details, fp)

if __name__ == "__main__":

    main(RunConfig())
    
    
