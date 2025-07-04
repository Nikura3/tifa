import logger
import time
from tifascore import filter_question_and_answers, UnifiedQAModel, tifa_score_single, VQAModel
from tifascore import get_llama2_pipeline, get_llama2_question_and_answers
import json
from config import RunConfig,TifaVersion
from transformers import pipeline
import os
import pandas as pd
from PIL import Image
import numpy as np
import csv
import torch
import torchvision.utils
import torchvision.transforms.functional as tf
import math
from scipy.integrate import trapezoid

# for showing the images on colab with test_obj_dect()
# import matplotlib.pyplot as plt
# from torchvision.transforms.functional import to_pil_image

#read prompt collection
def readCSV(eval_path,prompt_collection):
    df = pd.read_csv(os.path.join(eval_path,prompt_collection+'.csv'),dtype={'id': str})
    return df

# select the maximum IoU between two candidates, require list of coordinates
def selectMaximumIoU(ground_truth,candidate1,candidate2):
    maximum = 1
    
    iou1=bbIoU(ground_truth,candidate1)
    iou2=bbIoU(ground_truth,candidate2)
    
    if(iou2>=iou1):
        maximum = 2
    
    return maximum

def assignIous(id, prompt, seed, ground_truth,predictions,ious):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'obj1':None,
                'iou1':None,
                'gt_bb1':None,
                'pred_bb1':None,
                'obj2':None,
                'iou2':None,
                'gt_bb2':None,
                'pred_bb2':None,
                'obj3':None,
                'iou3':None,
                'gt_bb3':None,
                'pred_bb3':None,
                'obj4':None,
                'iou4':None,
                'gt_bb4':None,
                'pred_bb4':None
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    
    index=1
    for label in ground_truth.keys():
        new_row['obj'+str(index)]=label
        new_row['iou'+str(index)]=ious[label]
        new_row['gt_bb'+str(index)]=ground_truth[label]
        if label in predictions.keys():
            new_row['pred_bb'+str(index)]=predictions[label]
        else:
            new_row['pred_bb'+str(index)]=None
        index=index+1
    return new_row

def assignNoIous(id, prompt, seed, ground_truth):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'obj1':None,
                'iou1':None,
                'gt_bb1':None,
                'pred_bb1':None,
                'obj2':None,
                'iou2':None,
                'gt_bb2':None,
                'pred_bb2':None,
                'obj3':None,
                'iou3':None,
                'gt_bb3':None,
                'pred_bb3':None,
                'obj4':None,
                'iou4':None,
                'gt_bb4':None,
                'pred_bb4':None
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    
    index=1
    for label in ground_truth.keys():
        new_row['obj'+str(index)]=label
        new_row['iou'+str(index)]=float(0)
        new_row['gt_bb'+str(index)]=ground_truth[label]
        index=index+1
    return new_row

def assignScoresByCategory(id, prompt,seed,result):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'tifa_score':None,
                'object_s':0,
                'human_s':0,
                'animal_s':0,
                'animal/human_s':0,
                'food_s':0,
                'activity_s':0,
                'attribute_s':0,
                'counting_s':0,
                'color_s':0,
                'material_s':0,
                'spatial_s':0,
                'location_s':0,
                'shape_s':0,
                'other_s':0,
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    
    count_questions_by_type={
        'object_s':0,
        'human_s':0,
        'animal_s':0,
        'animal/human_s':0,
        'food_s':0,
        'activity_s':0,
        'attribute_s':0,
        'counting_s':0,
        'color_s':0,
        'material_s':0,
        'spatial_s':0,
        'location_s':0,
        'shape_s':0,
        'other_s':0
    }

    #count number of questions by type
    for question in result["question_details"].keys(): 
        type = result["question_details"][question]["element_type"]+'_s'
        count_questions_by_type[type]=count_questions_by_type[type]+1

    #accumulate scores
    for question in result["question_details"].keys():    
        score_by_type=result["question_details"][question]["scores"]
        type = result["question_details"][question]["element_type"]+'_s'
        new_row[type]=new_row[type]+score_by_type

    #average accuracies
    for scores in new_row.keys():
        if scores not in ["id", "prompt", "seed", "tifa_score"]:
            number_of_questions = count_questions_by_type[scores]
            if(number_of_questions != 0):
                new_row[scores]=new_row[scores]/number_of_questions

    new_row['tifa_score'] = result['tifa_score']
    return new_row

def calculate_auc(accuracies):
    k_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    acc_values = [accuracies[str(k)] for k in k_values]
    auc_value = trapezoid(acc_values, k_values)  # compute the integral
    return auc_value

def assignAccuracies(id, prompt,seed,result,accuracies):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'tifa_score':None,
                'accuracy@0.0':0,
                'accuracy@0.1':0,
                'accuracy@0.2':0,
                'accuracy@0.3':0,
                'accuracy@0.4':0,
                'accuracy@0.5':0,
                'accuracy@0.6':0,
                'accuracy@0.7':0,
                'accuracy@0.8':0,
                'accuracy@0.9':0,
                'accuracy@1.0':0,
                'auc': 0,
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    new_row['tifa_score'] = result['tifa_score']
    new_row['accuracy@0.0']=accuracies['0.0']
    new_row['accuracy@0.1']=accuracies['0.1']
    new_row['accuracy@0.2']=accuracies['0.2']
    new_row['accuracy@0.3']=accuracies['0.3']
    new_row['accuracy@0.4']=accuracies['0.4']
    new_row['accuracy@0.5']=accuracies['0.5']
    new_row['accuracy@0.6']=accuracies['0.6']
    new_row['accuracy@0.7']=accuracies['0.7']
    new_row['accuracy@0.8']=accuracies['0.8']
    new_row['accuracy@0.9']=accuracies['0.9']
    new_row['accuracy@1.0']=accuracies['1.0']
    new_row['auc'] = calculate_auc(accuracies)
    
    return new_row

def assignNoAccuracies(id, prompt,seed,result):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'tifa_score':None,
                'accuracy@0.0':0,
                'accuracy@0.1':0,
                'accuracy@0.2':0,
                'accuracy@0.3':0,
                'accuracy@0.4':0,
                'accuracy@0.5':0,
                'accuracy@0.6':0,
                'accuracy@0.7':0,
                'accuracy@0.8':0,
                'accuracy@0.9':0,
                'accuracy@1.0':0,
                'auc': 0,
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    new_row['tifa_score'] = result['tifa_score']
    new_row['accuracy@0.0']=0
    new_row['accuracy@0.1']=0
    new_row['accuracy@0.2']=0
    new_row['accuracy@0.3']=0
    new_row['accuracy@0.4']=0
    new_row['accuracy@0.5']=0
    new_row['accuracy@0.6']=0
    new_row['accuracy@0.7']=0
    new_row['accuracy@0.8']=0
    new_row['accuracy@0.9']=0
    new_row['accuracy@1.0']=0
    new_row['auc'] = 0
    return new_row

# extract the overall tifa score from the results
def assignOverallScore(id, prompt,seed,result):
    row_reference={
                'id':None,
                'prompt':None,
                'seed':None, 
                'tifa_score':None
            }
    new_row = row_reference.copy()

    new_row['id']=id
    new_row['prompt']=prompt
    new_row['seed']=seed
    new_row['tifa_score'] = result['tifa_score']

    return new_row

def assignQuestionDetails(id, prompt,seed,scores):
    questions={
        "question":{}
    }

    for question in scores["question_details"].keys():
        questions['question'][question]={
                'element':scores["question_details"][question]["element"],
                'element_type':scores["question_details"][question]["element_type"],
                'choices':scores["question_details"][question]["choices"],
                'free_form_vqa':scores["question_details"][question]["free_form_vqa"],
                'multiple_choice_vqa':scores["question_details"][question]["multiple_choice_vqa"],
                'score_by_question':scores["question_details"][question]["scores"]
            } 
    return questions

#calculate intersection over union on two bouding boxes (xmin,ymin,xmax,ymax)
def bbIoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

#compute accuracy@k
def computeAccuracyK(ious,k):
    aboveK = list(map(lambda x: 1 if x >= k else 0, ious.values()))
    return np.mean(aboveK)
    
def calculate_tifa(config : RunConfig):    
    #Load the models
    unifiedqa_model = UnifiedQAModel(config.qa_model)
    vqa_model = VQAModel(config.vqa_model)
    #llama2 for local gpt model, from Hugging Face
    pipeline = get_llama2_pipeline(config.gpt_model)

    if not (os.path.isdir(config.eval_path)):
        print("Evaluation folder not found!")
    else:
        
        #read the models to evaluate defined by directory structure
        models_to_evaluate = []
        for model in os.listdir(config.eval_path):
            if(os.path.isdir((os.path.join(config.eval_path,model)))):
                #key is the model name
                models_to_evaluate.append({
                    'batch_gen_images_path':(os.path.join(config.eval_path,model)),#example:evaluation/QBench/QBench-SD14
                    'folder_name':model, #example:QBench-SD14,
                    'name':model[model.find('-')+1:]
                    })
        model_names = [model["name"] for model in models_to_evaluate]
        print("The following models will be evaluated:", model_names)

        #for every model to evaluate, run this pipeline
        for model in models_to_evaluate:
            
            #collection of scores for each prompt divided by category type
            tifa_df = pd.DataFrame({
            'id':[],
            'prompt':[],
            'seed':[], 
            'tifa_score':[],
            'object_s':[],
            'human_s':[],
            'animal_s':[],
            'animal/human_s':[],
            'food_s':[],
            'activity_s':[],
            'attribute_s':[],
            'counting_s':[],
            'color_s':[],
            'material_s':[],
            'spatial_s':[],
            'location_s':[],
            'shape_s':[],
            'other_s':[]
            })
            
            #collection of overall tifa score for each prompt
            regular_df = pd.DataFrame({
            'id':[],
            'prompt':[],
            'seed':[], 
            'tifa_score':[]
            })

            #collection of questions for each prompt made by tifa divided by category type
            detailed_questions={
                'id_prompt_seed':{}
            }

            #collection of images to evaluate, read the directory structure and collect information    
            images = []
            #id,prompt,obj1,bbox1,token1,obj2,token2,obj3,token3,obj4,bbox4,token4
            prompt_collection = readCSV(config.eval_path, config.prompt_collection)
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
                            
            #sort the images by prompt_id and seed for clarity        
            images.sort(key=lambda x: (int(x['prompt_id']), int(x['seed'])))
            
            #logger for errors
            err = open(model['folder_name']+"_errors.err", 'w')
            err.write(model['folder_name']+"\n")
            
            print("Starting evaluation process for ", model['name'])
            
            #initialize logger to map memory usage
            l=logger.Logger(os.path.join(config.eval_path,config.prompt_collection+'-'+images[0]['model']),config.tifa_version)
            
            #initialize the variables needed for the evalation
            prompt = images[0]['prompt']
            llama2_questions=get_llama2_question_and_answers(pipeline,prompt)
            filtered_questions=filter_question_and_answers(unifiedqa_model, llama2_questions)

            if(len(filtered_questions)==0):
                filtered_questions=llama2_questions
                print("Warning: all the questions were filtered out!")
                err.write(images[0]['prompt_id']+"-"+images[0]['prompt']+"\n")

            if(len(llama2_questions)==0):
                err.write("Error: no questions generated by llama2, investigate!"+images[0]['prompt_id']+"-"+images[0]['prompt']+"\n")
                err.close()
                raise Exception("Error: no questions generated by llama2, investigate!")

            for image in images:
                img_path = image['img_path']
                if(prompt != image['prompt']):#when prompt changes, questions and answers change too otherwise it's unnecessary
                    prompt = image['prompt']
                    llama2_questions = get_llama2_question_and_answers(pipeline,prompt)
                    # Filter questions with UnifiedQA
                    filtered_questions = filter_question_and_answers(unifiedqa_model, llama2_questions)
                    
                    if(len(filtered_questions)==0):
                        filtered_questions=llama2_questions
                        print("Warning: all the questions were filtered out!")
                        err.write(image['prompt_id']+"-"+image['prompt']+"\n")

                    if(len(llama2_questions)==0):
                        err.write("Error: no questions generated by llama2, investigate!"+image['prompt_id']+"-"+image['prompt']+"\n")
                        err.close()
                        raise Exception("Error: no questions generated by llama2, investigate!")

                print("----")
                print("PROMPT:",prompt)
                print("PATH:",img_path)
                
                #start stopwatch
                start=time.time()
                
                # calculate TIFA score
                scores = tifa_score_single(vqa_model, filtered_questions, img_path)
                
                #end stopwatch
                end = time.time()
                #save to logger
                l.log_time_run(start,end)

                new_scores_row=assignScoresByCategory(image['prompt_id'],image['prompt'],image['seed'],scores)
                new_overall_score_row=assignOverallScore(image['prompt_id'],image['prompt'],image['seed'],scores)
                new_question_details_rows=assignQuestionDetails(image['prompt_id'],image['prompt'],image['seed'],scores)

                tifa_df = pd.concat([tifa_df, pd.DataFrame([new_scores_row])], ignore_index=True)
                regular_df = pd.concat([regular_df, pd.DataFrame([new_overall_score_row])], ignore_index=True)
                detailed_questions['id_prompt_seed'][image['prompt_id']+image['prompt'].replace(" ", "")+str(image['seed'])]=new_question_details_rows

                print("SCORE: ", scores['tifa_score'])

            
            #output scores by category type to csv
            tifa_df.to_csv(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_tifa.csv'), index=False)
            #output tifa overall score to csv
            regular_df.to_csv(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_regular.csv'), index=False)
            #dump question details to json
            with open(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_detailed_questions.json'), 'w') as fp:
                json.dump(detailed_questions, fp)
            #close the error log file
            err.close()  
            #log gpu statistics
            l.log_gpu_memory_instance()
            #save to the performance log to csv
            l.save_log_to_csv(config.tifa_version)

def calculate_extended_tifa(config : RunConfig):
    
    #Load the models
    unifiedqa_model = UnifiedQAModel(config.qa_model)
    vqa_model = VQAModel(config.vqa_model)
    #llama2 for local gpt model, from Hugging Face
    tifa_pipeline = get_llama2_pipeline(config.gpt_model)
    #Zero shot object detection pipeline
    object_detector = pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble", device="cuda")

    if not (os.path.isdir(config.eval_path)):
        print("Evaluation folder not found!")
    else:
        #read the models to evaluate defined by directory structure
        models_to_evaluate = []
        for model in os.listdir(config.eval_path):
            if(os.path.isdir((os.path.join(config.eval_path,model)))):
                #key is the model name
                models_to_evaluate.append({
                    'batch_gen_images_path':(os.path.join(config.eval_path,model)),#example:evaluation/QBench/QBench-SD14
                    'folder_name':model, #example:QBench-SD14,
                    'name':model[model.find('-')+1:]
                    })
                
        model_names = [model["name"] for model in models_to_evaluate]
        print("The following models will be evluated:", model_names)
        
        #for every model to evaluate, run this pipeline
        for model in models_to_evaluate:
            
            #collection of scores for each prompt divided by category type
            tifa_df = pd.DataFrame({
            'id':[],
            'prompt':[],
            'seed':[], 
            'tifa_score':[],
            'object_s':[],
            'human_s':[],
            'animal_s':[],
            'animal/human_s':[],
            'food_s':[],
            'activity_s':[],
            'attribute_s':[],
            'counting_s':[],
            'color_s':[],
            'material_s':[],
            'spatial_s':[],
            'location_s':[],
            'shape_s':[],
            'other_s':[]
            })
            
            #collection of overall tifa score for each prompt
            extended_df = pd.DataFrame({
            'id':[],
            'prompt':[],
            'seed':[], 
            'tifa_score':[],
            'accuracy@0.0':[],
            'accuracy@0.1':[],
            'accuracy@0.2':[],
            'accuracy@0.3':[],
            'accuracy@0.4':[],
            'accuracy@0.5':[],
            'accuracy@0.6':[],
            'accuracy@0.7':[],
            'accuracy@0.8':[],
            'accuracy@0.9':[],
            'accuracy@1.0':[],
            })
            
            #collection of overall tifa score for each prompt
            ious_df = pd.DataFrame({
            'id':[],
            'prompt':[],
            'seed':[], 
            'obj1':[],
            'iou1':[],
            'gt_bb1':[],
            'pred_bb1':[],
            'obj2':[],
            'iou2':[],
            'gt_bb2':[],
            'pred_bb2':[],
            'obj3':[],
            'iou3':[],
            'gt_bb3':[],
            'pred_bb3':[],
            'obj4':[],
            'iou4':[],
            'gt_bb4':[],
            'pred_bb4':[],
            })
            
            #collection of questions for each prompt made by tifa divided by category type
            detailed_questions={
                'id_prompt_seed':{}
            }
            
            #collection of images to evaluate, read the directory structure and collect information
            images = [] # attributes for each generated image
            #id,prompt,obj1,bbox1,token1,obj2,token2,obj3,token3,obj4,bbox4,token4
            prompt_collection = readCSV(config.eval_path,config.prompt_collection)

            for index,row in prompt_collection.iterrows(): 
                prompt_gen_images_path = os.path.join(model['batch_gen_images_path'],row['id']+'_'+row['prompt'])
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
                                'seed':img_filename.split('.')[0],
                                'obj1': row['obj1'] if row['obj1']is not None else math.nan,
                                'bbox1':row['bbox1']if row['bbox1']is not None else math.nan,
                                'obj2': row['obj2'] if row['obj2']is not None else math.nan,
                                'bbox2':row['bbox2']if row['bbox2']is not None else math.nan,
                                'obj3': row['obj3'] if row['obj3']is not None else math.nan,
                                'bbox3':row['bbox3']if row['bbox3']is not None else math.nan,
                                'obj4': row['obj4'] if row['obj4']is not None else math.nan,
                                'bbox4':row['bbox4']if row['bbox4']is not None else math.nan,
                            })

            #sort the images by prompt_id and seed for clarity       
            images.sort(key=lambda x: (int(x['prompt_id']), int(x['seed'])))
            
            #logger for errors
            err = open(model['folder_name']+"_errors.err", 'w')
            err.write(model['folder_name']+"\n")
            
            print("Starting evaluation process")
            
            #initialize logger to map memory usage
            l=logger.Logger(os.path.join(config.eval_path,config.prompt_collection+'-'+images[0]['model']),config.tifa_version)
            
            #initialize the variables needed for the evalation
            prompt = images[0]['prompt']
            prompt_id = images[0]['prompt_id']
            
            llama2_questions=get_llama2_question_and_answers(tifa_pipeline,prompt)
            filtered_questions=filter_question_and_answers(unifiedqa_model, llama2_questions)
            
            if(len(filtered_questions)==0):
                filtered_questions=llama2_questions
                print("Warning: all the questions were filtered out!")
                err.write(images[0]['prompt_id']+"-"+images[0]['prompt']+"\n")

            if(len(llama2_questions)==0):
                err.write("Error: no questions generated by llama2, investigate!"+image['prompt_id']+"-"+image['prompt']+"\n")
                print("Warning: no questions generated by llama2, skipping this prompt:", image['prompt_id'], image['prompt'])
                continue
                # if you want to raise an exception instead of skipping the prompt, uncomment the following lines
                # err.close()
                # raise Exception("Error: no questions generated by llama2, investigate!")
            
            ground_truth = {} #ground truth bounding boxes
            if not (isinstance(images[0]['obj1'], (int,float)) and math.isnan(images[0]['obj1'])):
                ground_truth[images[0]['obj1']] = [int(x) for x in images[0]['bbox1'].split(',')]
            if not (isinstance(images[0]['obj2'], (int,float)) and math.isnan(images[0]['obj2'])):
                ground_truth[images[0]['obj2']] = [int(x) for x in images[0]['bbox2'].split(',')]
            if not (isinstance(images[0]['obj3'], (int,float)) and math.isnan(images[0]['obj3'])):
                ground_truth[images[0]['obj3']] = [int(x) for x in images[0]['bbox3'].split(',')]
            if not (isinstance(images[0]['obj4'], (int,float)) and math.isnan(images[0]['obj4'])):
                ground_truth[images[0]['obj4']] = [int(x) for x in images[0]['bbox4'].split(',')]
            
        
            for image in images:
                img_path = image['img_path']
                    
                if(prompt != image['prompt'] or prompt_id != image['prompt_id']): #when prompt changes, questions and answers change too otherwise it's unnecessary
                    prompt = image['prompt']
                    prompt_id = image['prompt_id']
                    ground_truth.clear()
                    
                    if not (isinstance(image['obj1'], (int,float)) and math.isnan(image['obj1'])):
                        ground_truth[image['obj1']]= [int(x) for x in image['bbox1'].split(',')]
                    if not (isinstance(image['obj2'], (int,float))and math.isnan(image['obj2'])):
                        ground_truth[image['obj2']]= [int(x) for x in image['bbox2'].split(',')]
                    if not (isinstance(image['obj3'], (int,float)) and math.isnan(image['obj3'])):
                        ground_truth[image['obj3']]= [int(x) for x in image['bbox3'].split(',')]
                    if not (isinstance(image['obj4'], (int,float)) and math.isnan(image['obj4'])):
                        ground_truth[image['obj4']]= [int(x) for x in image['bbox4'].split(',')]              
                    
                    # if prompt changes, generate a new set of question-answer pairs
                    llama2_questions = get_llama2_question_and_answers(tifa_pipeline,prompt)
                    # Filter questions with UnifiedQA
                    filtered_questions = filter_question_and_answers(unifiedqa_model, llama2_questions)
                    
                    if(len(filtered_questions)==0):
                        filtered_questions=llama2_questions
                        print("Warning: all the questions were filtered out!")
                        err.write(image['prompt_id']+"-"+image['prompt']+"\n")

                    if(len(llama2_questions)==0):
                        err.write("Error: no questions generated by llama2, investigate!"+image['prompt_id']+"-"+image['prompt']+"\n")
                        print("Warning: no questions generated by llama2, skipping this prompt:", image['prompt_id'], image['prompt'])
                        continue
                        # if you want to raise an exception instead of skipping the prompt, uncomment the following lines
                        # err.close()
                        # raise Exception("Error: no questions generated by llama2, investigate!")
                    
                print("----")
                print("PROMPT:",prompt)
                print("PATH:",img_path)
                
                # start stopwatch
                start=time.time()
                            
                # Standard TIFA
                scores = tifa_score_single(vqa_model, filtered_questions, img_path)
                
                # Extended TIFA enhanced with object detection
                pil_image = Image.open(img_path).convert("RGB")
                preds = object_detector(pil_image, candidate_labels=ground_truth.keys())
                
                end=time.time()
                l.log_time_run(start,end)
                
                # Regular TIFA results
                new_scores_row=assignScoresByCategory(image['prompt_id'],image['prompt'],image['seed'],scores)
                new_question_details_rows=assignQuestionDetails(image['prompt_id'],image['prompt'],image['seed'],scores)

                tifa_df = pd.concat([tifa_df, pd.DataFrame([new_scores_row])], ignore_index=True)
                detailed_questions['id_prompt_seed'][image['prompt_id']+image['prompt'].replace(" ", "")+str(image['seed'])]=new_question_details_rows
                print("SCORE: ", scores['tifa_score'])
                
                #Extended TIFA results
                predictions={} # distinct predictions, one for each element even if multiple predictions are made by the detector
                
                # basically, keep just one for each element, the one with the highest IoU
                for p in preds:
                    candidate = list(p['box'].values()) # add new entry as default
                    if (p['label'] in predictions.keys()): # check if there are two predictions of the same element, select just the highest one
                        max_iou=selectMaximumIoU(ground_truth[p['label']], # ground truth
                                        list(p['box'].values()), # candidate1
                                        predictions[p['label']] # candidate2
                                        )
                        if max_iou == 1: # if new candidate is higher than already existing one, substitute it. otherwise don't.
                            predictions[p['label']]=candidate                            
                    else:
                        predictions[p['label']]=candidate                          
                    
                
                # calculate IoU
                if (len(predictions)!=0):
                    
                    """ if (len(predictions)!= len(ground_truth)): # save disagreement if any
                        print("Some objects are not predicted by the object detector, please check!")
                        file.write(image['img_path']+" : Some objects are not predicted by the object detector, please check!\n")
                        """
                    # save the image with the predictions
                    if(os.path.exists(img_path[:-4]+"_bboxes.png")):    
                        bboxes_image=torchvision.utils.draw_bounding_boxes(tf.pil_to_tensor(Image.open(img_path[:-4]+"_bboxes.png").convert("RGB")),
                                                            torch.Tensor(list(predictions.values())),
                                                            colors=['red', 'red', 'red', 'red', 'red', 'red', 'red', 'red'],
                                                            width=4,
                                                            font='font.ttf',
                                                            font_size=20)
                        tf.to_pil_image(bboxes_image).save(os.path.join(image['prompt_gen_images_path'],image['img_filename'][:-4]+'_detection.png'))
                        
                    # a dict containing the IntesectionOverUnion between ground truth and predicted bounding boxes
                    ious={}
                    
                    for label in ground_truth.keys(): #initialize to zero all the elements
                        ious[label]=float(0)
                        
                    for label in list(predictions.keys()):
                        ious[label] = round(bbIoU(predictions[label],ground_truth[label]),6)
                        #text = text+label+" : "+ str(round(bbIoU(predictions[label],ground_truth[label]),2))+"\n" 
                    
                    accuracies = {} 
                    accuracies['0.0'] = computeAccuracyK(ious,0.0)
                    accuracies['0.1'] = computeAccuracyK(ious,0.1)
                    accuracies['0.2'] = computeAccuracyK(ious,0.2)
                    accuracies['0.3'] = computeAccuracyK(ious,0.3)
                    accuracies['0.4'] = computeAccuracyK(ious,0.4)
                    accuracies['0.5'] = computeAccuracyK(ious,0.5)
                    accuracies['0.6'] = computeAccuracyK(ious,0.6)
                    accuracies['0.7'] = computeAccuracyK(ious,0.7)
                    accuracies['0.8'] = computeAccuracyK(ious,0.8)
                    accuracies['0.9'] = computeAccuracyK(ious,0.9)
                    accuracies['1.0'] = computeAccuracyK(ious,1.0)
                    
                    new_entry_iou = assignIous(image['prompt_id'],image['prompt'],image['seed'],ground_truth,predictions,ious)
                    ious_df = pd.concat([ious_df, pd.DataFrame([new_entry_iou])], ignore_index=True)
                    
                    new_entry=assignAccuracies(image['prompt_id'],image['prompt'],image['seed'],scores,accuracies)
                    extended_df = pd.concat([extended_df, pd.DataFrame([new_entry])], ignore_index=True)
                    
                else:
                    print("Warning: No objects found by the object detector, please check!")
                    new_entry_no_iou = assignNoIous(image['prompt_id'],image['prompt'],image['seed'],ground_truth)
                    ious_df = pd.concat([ious_df, pd.DataFrame([new_entry_no_iou])], ignore_index=True)
                    
                    new_entry_no_preds=assignNoAccuracies(image['prompt_id'],image['prompt'],image['seed'],scores)
                    extended_df = pd.concat([extended_df, pd.DataFrame([new_entry_no_preds])], ignore_index=True)

            #output scores by category type to csv
            tifa_df.to_csv(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_tifa.csv'), index=False)
            #output IoUs to csv
            ious_df.to_csv(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_ious.csv'), index=False)            
            #output tifa overall score + IoU accuracies to csv
            extended_df.to_csv(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_extended.csv'), index=False)
            #dump question details to json
            with open(os.path.join(model['batch_gen_images_path'],model['folder_name']+'_detailed_questions.json'), 'w') as fp:
                json.dump(detailed_questions, fp)
            #close the error log file
            err.close()                 
            #log gpu statistics
            l.log_gpu_memory_instance()
            #save to the performance log to csv
            l.save_log_to_csv(config.tifa_version)

def main(config:RunConfig):
    
    if(config.tifa_version==TifaVersion.REGULAR):
        calculate_tifa(config)
    elif(config.tifa_version==TifaVersion.EXTENDED):
        calculate_extended_tifa(config)
    
    print("End of evaluation process")


def test_obj_dect():
    from collections import defaultdict

    detector = pipeline("zero-shot-object-detection", model="google/owlv2-base-patch16-ensemble", device="cuda")

    image_path = '/content/3.jpg'
    image = Image.open(image_path).convert("RGB")

    candidate_labels = ["A tree", "A bridge"]
    predictions = detector(image, candidate_labels=candidate_labels)

    # Define your ground truth boxes for each label
    gt_boxes = {
        "A tree": [41, 69, 214, 466],
        "A bridge": [25, 310, 482, 474]
    }

    best_predictions = {}
    for label in candidate_labels:
        max_iou = -1
        best_box = None
        for pred in predictions:
            if pred["label"] == label:
                pred_box = list(pred["box"].values())
                iou = bbIoU(pred_box, gt_boxes[label])
                if iou > max_iou:
                    max_iou = iou
                    best_box = pred_box
        if best_box:
            best_predictions[label] = best_box

    if not best_predictions:
        print("No relevant objects detected.")
        return

    predicted_labels = list(best_predictions.keys())
    predicted_bboxes = list(best_predictions.values())

    print("Filtered predictions:")
    print("Labels:", predicted_labels)
    print("Boxes:", predicted_bboxes)

    # Draw the boxes
    image_tensor = tf.pil_to_tensor(image)
    image_tensor = torchvision.utils.draw_bounding_boxes(
        image_tensor,
        torch.tensor(predicted_bboxes, dtype=torch.float),
        labels=predicted_labels,
        colors=["yellow"] * len(predicted_bboxes),
        width=4,
        font_size=25
    )

    # Draw ground truth boxes
    gt_boxes_list = list(gt_boxes.values())
    image_tensor = torchvision.utils.draw_bounding_boxes(
        image_tensor,
        torch.tensor(gt_boxes_list, dtype=torch.float),
        colors=["red", "purple"],
        width=4,
        font_size=25
    )
    
    tf.to_pil_image(image).save("pizza_bboxes.png")

    # Convert and show on colab
    # pil_img = to_pil_image(image_tensor)
    # plt.imshow(pil_img)
    # plt.axis("off")
    # plt.show()

    # Print IoUs
    for label in predicted_labels:
        print(f"gt_box for {label}:", gt_boxes[label])
        print(f"predicted_box for {label}:", best_predictions[label])
        print(f"IoU for {label}:", bbIoU(best_predictions[label], gt_boxes[label]))

if __name__ == "__main__":
    main(RunConfig())
    
    
