import numpy as np
import yaml
import inspect
from shutil import copyfile, copy
import os
import torch
from torchvision.utils import make_grid


def one_to_three(x):
    return torch.cat([x]*3, dim=1)


def save_config_to_yaml(config_obj, parrent_dir: str):
    """
    Saves the given Config object as a YAML file. The output file name is derived
    from the module name where the Config class is defined.
    
    Args:
        config_obj (Config): The Config object to be saved.
        parrent_dir (str): Parent directory where the YAML file will be saved.
    """
    os.makedirs(parrent_dir, exist_ok=True)
    
    try:
        # Handle both class and instance objects
        if inspect.isclass(config_obj):
            # If it's a class, use it directly
            module_path = inspect.getfile(config_obj)
        elif hasattr(config_obj, '__class__'):
            # If it's an instance, get its class
            module_path = inspect.getfile(config_obj.__class__)
        else:
            # Fallback: try to get file directly
            module_path = inspect.getfile(config_obj)
    except (TypeError, OSError) as e:
        print(f"Warning: Could not get module path for config object: {e}")
        # Use a default name based on the object type
        if hasattr(config_obj, '__class__'):
            base_filename = config_obj.__class__.__name__.lower()
        else:
            base_filename = "config"
        module_path = f"{base_filename}.py"
    
    # Extract the base file name without extension
    base_filename = os.path.splitext(os.path.basename(module_path))[0]
    
    # Construct the output YAML file name
    output_file = f"{base_filename}.yaml"
    output_file = os.path.join(parrent_dir, output_file)
    
    # Convert the Config object to a dictionary
    if hasattr(config_obj, '__dict__'):
        config_dict = {k: v for k, v in config_obj.__dict__.items() if not k.startswith('__')}
    else:
        # If object doesn't have __dict__, try to convert it to dict
        config_dict = dict(config_obj) if hasattr(config_obj, '__iter__') else {}
    
    # Handle special types that YAML can't serialize
    def make_serializable(obj):
        if isinstance(obj, torch.dtype):
            return str(obj)
        elif isinstance(obj, type):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return {k: make_serializable(v) for k, v in obj.__dict__.items() if not k.startswith('__')}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        else:
            return obj
    
    # Make the config dictionary serializable
    serializable_config = make_serializable(config_dict)
    
    # Save the dictionary as a YAML file
    try:
        with open(output_file, 'w') as yaml_file:
            yaml.dump(serializable_config, yaml_file, sort_keys=False, default_flow_style=False)
        print(f"Config saved to: {output_file}")
    except Exception as e:
        print(f"Error saving config to YAML: {e}")
        # Try to save as much as possible
        safe_config = {}
        for k, v in serializable_config.items():
            try:
                yaml.dump({k: v}, open(os.devnull, 'w'))
                safe_config[k] = v
            except:
                safe_config[k] = str(v)
        
        with open(output_file, 'w') as yaml_file:
            yaml.dump(safe_config, yaml_file, sort_keys=False, default_flow_style=False)
        print(f"Config saved with fallback serialization to: {output_file}")
    
    return serializable_config


def copy_yaml_to_folder(yaml_file, folder):
    """
    将一个 YAML 文件复制到一个文件夹中
    :param yaml_file: YAML 文件的路径
    :param folder: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(folder, exist_ok=True)
    
    # 获取 YAML 文件的文件名
    file_name = os.path.basename(yaml_file)
    
    # 将 YAML 文件复制到目标文件夹中
    copy(yaml_file, os.path.join(folder, file_name))


def force_remove_empty_dir(path):
    try:
        os.rmdir(path)
        print(f"Directory '{path}' removed successfully.")
    except FileNotFoundError:
        print(f"Directory '{path}' not found.")
    except OSError as e:
        print(f"Error removing directory '{path}': {e}")


def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        for key in config.keys():
            if type(config[key]) == list:
                config[key] = tuple(config[key])
        return config


def get_parameters(fn, original_dict):
    new_dict = dict()
    arg_names = inspect.getfullargspec(fn)[0]
    for k in original_dict.keys():
        if k in arg_names:
            new_dict[k] = original_dict[k]
    return new_dict


def write_config(config_path, save_path):
    copyfile(config_path, save_path)


def check_dir(dire):
    if not os.path.exists(dire):
        os.makedirs(dire)
    return dire


def combine_tensors_2_tb(tensor_list:list=None):
    image = torch.cat(tensor_list, dim=-1)
    image = (make_grid(image, nrow=1).unsqueeze(0)+1)/2
    return image.clamp(0, 1)