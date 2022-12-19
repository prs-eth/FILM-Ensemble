# FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation
Mehmet Ozgur Turkoglu, Alexander Becker, Hüseyin Anil Gündüz, Mina Rezaei, Bernd Bischl, Rodrigo Caye Daudt, Stefano D'Aronco, Jan Dirk Wegner, and Konrad Schindler. - **NeurIPS 2022**.


[ \[Paper on ArXiv\] ](https://arxiv.org/abs/2206.00050) [ \[Poster\] ](https://neurips.cc/media/PosterPDFs/NeurIPS%202022/53071.png?t=1669493329.1428604)


## 🌌 Overview

<img src="assets/method.png" alt="Method overview">


## ✏️ 📄 Citation

If you find our work useful or interesting, please cite our paper:

```latex
@inproceedings{turkoglu2022film,
  title={FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation},
  author={Turkoglu, Mehmet Ozgur and Becker, Alexander and G{\"u}nd{\"u}z, H{\"u}seyin Anil and Rezaei, Mina and Bischl, Bernd and Daudt, Rodrigo Caye and D'Aronco, Stefano and Wegner, Jan Dirk and Schindler, Konrad},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```

## Reproducing results: Cifar-10 / Cifar-100
 
### Installation
```
pip install -r requirements.txt
```

### Training
```
python main.py -e {number of ensemble}
