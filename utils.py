import logging
from tqdm import tqdm

def get_dict(temp, seg, text_entities, flag=False):
    data_list = []
    for i in tqdm(range(len(temp))):
        temp_dict = {}
        temp_dict["text"] = temp[i]
        temp_dict["label"] = {"element": {}}
        pos = 0
        if flag:
#             data_list.append(temp_dict)
            if len(temp[i]):
                data_list.append(temp_dict)
        else:
            for word in seg[i]:
                if word in text_entities:
                    start = temp[i].find(word, pos)
                    end = start + len(word) - 1
                    try:
                        temp_dict["label"]["element"][word].append([start, end])
                    except:
                        temp_dict["label"]["element"][word] = []
                        temp_dict["label"]["element"][word].append([start, end])
                pos += len(word)
            if len(temp[i]) and temp_dict["label"]["element"]!={}:
                data_list.append(temp_dict)
    return data_list

def get_Data_(data_list):
    word_list=[]
    label_list=[]
    # 先读取到内存中，然后逐行处理
    for line in data_list:
        # loads()：用于处理内存中的json对象，strip去除可能存在的空格
    #     print(line.strip())
        json_line = line
    #     print(json_line['text'])

        text = json_line['text']
        words = list(text)
        # 如果没有label，则返回None
        label_entities = json_line.get('label', None)
        labels = ['O'] * len(words)

        if label_entities is not None:
            for key, value in label_entities.items():
                for sub_name, sub_index in value.items():
                    for start_index, end_index in sub_index:
    #                     print(sub_name, ''.join(words[start_index:end_index + 1]))
                        assert ''.join(words[start_index:end_index + 1]) == sub_name
                        if start_index == end_index:
                            labels[start_index] = 'S-' + key
                        else:
                            labels[start_index] = 'B-' + key
                            labels[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
        if len(words) < 510:
    #         print(words,labels)
            word_list.append(words)
            label_list.append(labels)
    return word_list,label_list
        # 保存成二进制文件
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)