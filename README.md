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
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py resnet_leaky_adam_ul.yaml --device 2 --relative M --batch 8 -n flownet

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n hybridnet-coarse -d 1 --relative M --batch 8 resnet_spydecoder.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 2 flownet_dil.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n hybridnet-coarse -d 0 --relative M --batch 8 resnet_decoder_refine.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 2 flownet_dilation_deformable.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 1 flownet.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 2 flownet_dilation56.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 0 flownet_dilation_impl.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 1 flownet_dilation_control5.yaml

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 3 flownet_dilation_control6.yaml

cd \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D
python -m reader.server

\\msra-ts006\c$\Program Files\Python36\python.exe \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 0 flownet.yaml -c a01May15

\\msra-ts006\c$\Program Files\Python36\python.exe \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 0 flownet.yaml -c a01May15

\\msra-ts006\c$\Program Files\Python36\python.exe \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 4 -d 0,1 flownet_dilation_assp_nopool.yaml

\\msra-ts006\c$\Program Files\Python36\python.exe \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -n flownet --relative M --batch 8 -d 0 flownet_dilation_control5_deform.yaml --fake_data
```

## Validation
```
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -c a38Apr25 --valid -d 0
\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -c ad3Apr23-2128 --valid -d 0 \\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -c 9e4Apr25-1928 --valid -d 0

\\msra-ts1\c$\Users\v-dinliu\AppData\Local\Continuum\anaconda3\python \\msralab\ProjectData\eHealth02\v-dinliu\Flow2D\main.py -c 2beMay11 --valid -d 0
```