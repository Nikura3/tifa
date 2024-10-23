from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, tifa_score_single,  VQAModel
from tifascore import get_llama2_pipeline, get_llama2_question_and_answers
import json
from config import RunConfig
import os

def readCSV(prompt_collection_path):
    return [("001","A bus"), ("002","A bus and a bench")]

def main(config : RunConfig):
    """ #Load the models
    unifiedqa_model = UnifiedQAModel(config.qa_model)
    vqa_model = VQAModel(config.vqa_model)
    #llama2 for local gpt model, from Hugging Face
    pipeline = get_llama2_pipeline(config.gpt_model) """

    if not (os.path.isdir(config.image_path)):
        print("Prompt collection not found!")
    else:
        """ # Walk through the directory and subdirectories
        for dirpath, dirnames, filenames in os.walk(config.image_path):
            for file in filenames:
                if not file.endswith((".csv", ".png")): #to delete csv files and bounding box version of the image
                    file_paths.append((dirpath, file))
        """
        #TODO ADD SEQUENTIAL NUMBER IN EACH PROMPT!
        models_to_evaluate = []
        for model in os.listdir(config.image_path):
            if(os.path.isdir((os.path.join(config.image_path,model)))):
                models_to_evaluate.append((os.path.join(config.image_path,model), model))

        prompts_to_evaluate = readCSV(config.prompt_collection_path)

        images = []
        for model in models_to_evaluate:
            for prompt in prompts_to_evaluate:
                root_img_path = os.path.join(model[0],prompt[1])
                prompt = prompt[1]
                for file in os.listdir(root_img_path):
                    if not file.endswith((".csv",".png")):
                        if(os.path.isfile(os.path.join(root_img_path,file))):
                            images.append((root_img_path,prompt,file))

        # root_img_path, prompt, image_filename
        images.sort(key=lambda x: (x[0], x[1], int(x[2].split('.')[0])))

        """ llama2_questions = get_llama2_question_and_answers(pipeline, text)
        
        # Filter questions with UnifiedQA
        #filtered_questions = filter_question_and_answers(unifiedqa_model, gpt3_questions)
        
        # Filter questions with UnifiedQA
        filtered_questions = filter_question_and_answers(unifiedqa_model, llama2_questions)

        # See the questions
        #print(filtered_questions)

        # calculate TIFA score
        result = tifa_score_single(vqa_model, filtered_questions, img_path)

        print("-- tifa_score: --")
        print(result['tifa_score'])
        print("-- question details: --")
        print(result['question_details']) """

if __name__ == "__main__":

    main(RunConfig())
    
    
