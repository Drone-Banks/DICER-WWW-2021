# DICER-WWW-2021
Source code for WWW 2021 paper "Dual Side Deep Context-aware Modulation for Social Recommendation"


# Code
Author: Bairan Fu (email: qaq.febr2.qaq@gmail.com)

If you use this code, please cite our paper:
```
@inproceedings{DICER-WWW-21,
  author    = {Bairan Fu and Wenming Zhang and Guangneng Hu and Xinyu Dai and 
                Shujian Huang and Jiajun Chen},
  title     = {Dual Side Deep Context-aware Modulation for SocialRecommendation},
  booktitle = {Proceedings of the Web Conference 2021 (WWW '21), April 19--23, 2021, Ljubljana, Slovenia},
  year      = {2021}
}
```

# Environment Settings

Python: 3.8.5

PyTorch: 1.7.1

DGL: 0.5.3


# Example to run the codes

Set the absolute path in `run.py`, `rank_task.py` and `example_final.ini`.

Run DICER:
```
python Runs/run.py --model_name final --data_name example
```

Raw Datasets (Ciao and Epinions) can be downloaded at http://www.cse.msu.edu/~tangjili/trust.html

For more details, you can refer to our paper.
