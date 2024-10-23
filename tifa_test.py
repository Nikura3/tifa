from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, tifa_score_single,  VQAModel
from tifascore import get_llama2_pipeline, get_llama2_question_and_answers
import json
from config import RunConfig
import os

def readCSV(prompt_collection_path):
    return [("001","A bus"), ("002","A bus and a bench")]

def main(config : RunConfig):
    #Load the models
    unifiedqa_model = UnifiedQAModel(config.qa_model)
    vqa_model = VQAModel(config.vqa_model)
    #llama2 for local gpt model, from Hugging Face
    pipeline = get_llama2_pipeline(config.gpt_model)

    if not (os.path.isdir(config.model_output_path)):
        print("Prompt collection not found!")
    else:
        # Walk through the directory and subdirectories
        

        #TODO ADD SEQUENTIAL NUMBER IN EACH PROMPT!
        models_to_evaluate = []
        for model in os.listdir(config.model_output_path):
            if(os.path.isdir((os.path.join(config.model_output_path,model)))):
                models_to_evaluate.append((os.path.join(config.model_output_path,model), model))

        prompts_to_evaluate = readCSV(config.prompt_collection_path)

        images = []
        for model in models_to_evaluate:
            for prompt in prompts_to_evaluate:
                root_img_path = os.path.join(model[0],prompt[0]+'_'+prompt[1])
                #prompt = prompt[1]
                for img_filename in os.listdir(root_img_path):
                    if not img_filename.endswith((".csv",".png")):
                        img_path = os.path.join(root_img_path,img_filename)
                        if(os.path.isfile(os.path.join(root_img_path,img_filename))):
                            images.append((root_img_path, img_path , prompt[1] ,img_filename))

        # root_img_path, img_path, prompt, image_filename
        # root_img_path: evaluation/QBench/QBench-BD_G 
        # img_path: evaluation/QBench/QBench-BG_G/001_A bus/1.jpg
        # prompt: A bus 
        # image_name: 1.jpg
        images.sort(key=lambda x: (x[0], x[2], int(x[3].split('.')[0])))
        
        print("Starting evaluation process")
        for image in images:
            root_img_path = image[0]
            img_path = image[1]
            prompt = image [2]

            print("----")
            print("PROMPT:",prompt)
            print("PATH:",img_path)

            llama2_questions = get_llama2_question_and_answers(pipeline,prompt)
        
            # Filter questions with UnifiedQA
            filtered_questions = filter_question_and_answers(unifiedqa_model, llama2_questions)

            # See the questions
            #print(filtered_questions)

            # calculate TIFA score
            result = tifa_score_single(vqa_model, filtered_questions, img_path)
            #print("SCORE:",result['tifa_score'])
            #print("-- question details: --")
            #print(result['question_details'])
            print("Questions:")
            for question in result["question_details"].keys():
                print(question," | Category: ",result["question_details"][question]["element_type"], " | Score: ",result["question_details"][question]["scores"])

if __name__ == "__main__":

    main(RunConfig())
    
    
