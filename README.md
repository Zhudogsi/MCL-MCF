# MCL-MCF
Hierarchical Alleviation of Heterogeneity in Multimodal Sentiment Analysis

### Introduction
Inspired by the fusion of objects in the natural world, we conceptualize multimodal fusion as a continuous process, dividing the entire procedure into
three steps. We design MCL-MCF. MCF simulates the continuous fusion process, while MCL, through multilevel alleviation of heterogeneity, assists MCF
in achieving multilevel fusion. Their collaborative operation yields optimal fusion results.
![image](https://github.com/Zhudogsi/MCL-MCF/assets/44200919/651e72c0-20f5-4936-adb2-9c0d2779937e)

### Usage
1.Download the CMU-MOSI and CMU-MOSEI dataset from [Google Drive](https://drive.google.com/drive/folders/1djN_EkrwoRLUt7Vq_QfNZgCl_24wBiIK) or [Baidu Disk](https://pan.baidu.com/share/init?surl=Wxo4Bim9JhNmg8265p3ttQ) (extraction code: g3m2)  
2. environment
```
conda env create -f environment.yml
conda activate MCL
```
3. starting 
```python
python main.py
```
### Thanks
We are grateful for the open source baseline of MMIM. We built MCL-MCF on it. For configuration and dataset related open source, please refer to MMIM (https://github.com/declare-lab/Multimodal-Infomax?tab=readme-ov-file).
Please cite this paper if you find our work useful for your research:
```bibtex
@inproceedings{han2021improving,
  title={Improving Multimodal Fusion with Hierarchical Mutual Information Maximization for Multimodal Sentiment Analysis},
  author={Han, Wei and Chen, Hui and Poria, Soujanya},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={9180--9192},
  year={2021}
}
