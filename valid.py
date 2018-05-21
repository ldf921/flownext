import os
from reader import sintel

def predict(pipe, prefix, batch_size = 8):
    sintel_dataset = sintel.list_data(sintel.sintel_path)
    if not os.path.exists(prefix):
        os.mkdir(prefix)
    
    flo = sintel.Flo(1024, 436)

    for div in ('test',):
        for k, dataset in sintel_dataset[div].items():
            output_folder = os.path.join(prefix, k)
            if not os.path.exists(output_folder):
                os.mkdir(output_folder)

            img1, img2 = [[sintel.load(p) for p in data] for data in zip(*dataset)]
            for flow, entry in zip(pipe.predict(img1, img2, batch_size=batch_size), dataset):
                img1 = entry[0]
                fname = os.path.basename(img1).replace('.png', '.flo')
                seq = os.path.basename(os.path.dirname(img1))
                seq_output_folder = os.path.join(output_folder, seq)
                if not os.path.exists(seq_output_folder):
                    os.mkdir(seq_output_folder)
                flo.save(flow, os.path.join(seq_output_folder, fname))



            