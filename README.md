```
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python main.py --valid --device 0 -c .\weights\Mar18-1605_5000.params
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 1 --relative S -c WEIGHTS\Mar22-1924_82500.params
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 0 --relative M -c WEIGHTS\Mar22-1921_95000.params
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 0 -c WEIGHTS\Mar23-1239_87500.params
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 0 --relative M --short_data
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 0,1 --relative M --batch 8 -n hybridnet
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 0,1 --relative M --batch 8 -n hybridnet-coarse --lr_mult 0.05
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python adam.yaml \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py --device 0,1 --relative M --batch 8 -n hybridnet-coarse 
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py base.yaml --device 0,1 --relative M --batch 8 -n hybridnet-coarse 

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py spynet_leaky.yaml --device 0,1 --relative M --batch 8 -n hybridnet-coarse 
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py resnet_leaky_adam.yaml --device 3 --relative M --batch 8 -n hybridnet-coarse 
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py resnet_leaky_adam_ul.yaml --device 2 --relative M --batch 8 -n hybridnet-coarse 
```