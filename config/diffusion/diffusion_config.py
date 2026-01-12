import os

class Config:


    project_dir = "./latent_diff_exp/multi_condition_controlnet"
    data_path = "data_example/train"          
    eval_path = "data_example/test"            

    vae_resume_path = ""                 
    sd_resume_path = ""                  
    controlnet_path = ""                 
    

    num_epochs = 2000                     
    train_bc = 1                       
    eval_bc = 1                         
    sample_size = 512                   
    mode = 'train'                      
    

    in_channels = 1                     
    out_channels = 1                   
    up_and_down = (128, 256, 512, 512)  
    num_res_layers = 2                   
    scaling_factor = 0.18215            
    

    sd_num_channels = (320, 640, 1280, 1280)  
    attention_levels = (False, True, True, True)  


    od_channels = 1                     
    cfp_channels = 3                    
    vessel_channels = 1                  
    

    encoder_channels = (16, 32, 64, 128) 
    condition_fusion_type = "attention"  


    conditioning_embedding_in_channels = 3   
    conditioning_embedding_num_channels = (16, 32, 96, 256)  
    

    gva_loss_weight = 1                
    gva_start_step = 1000               
    gva_alpha = 1.0                      
    gva_beta = 0.1                       
    

    val_inter = 5                       
    save_inter = 10                     
    

    lr_controlnet = 2.5e-5              
    lr_diffusion = 2.5e-5                
    

    log_with = []                      

    data_augmentation = True             
    force_resize_to_512 = True           
    

    experiment_name = "multi_condition_controlnet"
    seed = 42                           
    
    @classmethod
    def create_directories(cls):

        os.makedirs(cls.project_dir, exist_ok=True)
        os.makedirs(os.path.join(cls.project_dir, 'image_save'), exist_ok=True)
        os.makedirs(os.path.join(cls.project_dir, 'model_save'), exist_ok=True)
        os.makedirs(os.path.join(cls.project_dir, 'logs'), exist_ok=True)
    



# 创建目录
Config.create_directories()