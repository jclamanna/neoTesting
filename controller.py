import sys
import yaml
import os
from shutil import rmtree

from utils import load_datasets, encode_sequences 
from mi3gpu.utils.seqload import writeSeqs
from newVaes import SVAE
import tensorflow as tf

from Vis import Vis
import Hams
import Covars
import r20
from Loss import plotLoss
from Latent import plotLatent1d, plotLatent2d


def load_seq():
    seqs_path = "data"
    sequences_name_pattern = metricsConfig["sequences_name_pattern"]+".csv"
    train_seqs, _, val_seqs, _, test_seqs, _ = load_datasets(seqs_path, sequences_name_pattern)
    train_seqs = encode_sequences(train_seqs, alphabet=ALPHA)
    val_seqs = encode_sequences(val_seqs, alphabet=ALPHA)
    test_seqs = encode_sequences(test_seqs, alphabet=ALPHA)

    return train_seqs, val_seqs, test_seqs

def make_dirs(paths):
    os.mkdir(paths["out_path"][:-1])
    os.mkdir(paths["model_path"][:-1])
    os.mkdir(paths["vis_path"][:-1])
    os.mkdir(paths["r20_path"][:-1])

def run_train(svae):
        train_seqs, val_seqs, test_seqs = load_seq()

        N, L = train_seqs.shape
        batch_size, latent_dim, inner_dim, epochs = hyperParams["batch_size"], hyperParams["latent_dim"], hyperParams["inner_dim"], hyperParams["epochs"]
        activation, optimizer = hyperParams["activation"], hyperParams["optimizer"]
        n_batches = N // batch_size

        svae.instantiate_model(L,q,latent_dim,batch_size, activation, inner_dim)
        svae.train_vae(train_seqs, epochs, optimizer, val_seqs)
        svae.save_model(name, paths["out_path"])

        loss = svae.getLoss()

        plotLoss(loss, paths, name)
        z_mean, st = plotLatent1d(svae.vae, q, latent_dim, train_seqs[:10000], paths, name)
        plotLatent2d(z_mean, st, latent_dim, paths, name)
        
if len(sys.argv) > 1:
    if ".yml" in sys.argv[1]:
        print("reading config_file: ", sys.argv[1])

        with open(sys.argv[1], 'r') as fp:
            conf = yaml.safe_load(fp)

        seqConfig = conf["seqConfig"]
        metricsConfig = conf["metricsConfig"]
        visConfig = conf["visConfig"]
        hyperParams = conf["hyperParams"]
    else:
        raise ValueError("Config file must be a .yml")
else:
        raise ValueError("Either config_file or args parameters must not be None")

if len(sys.argv) > 2:
    paths = {}
    name = "E" + str(hyperParams["epochs"]) + "_B" + str(hyperParams["batch_size"]) + "_L" + str(hyperParams["latent_dim"]) + "_F" + str(hyperParams["phylo_filter"])
    
    paths["out_path"] = "output/" + name + "/"
    paths["model_path"] = paths["out_path"] + "/model/"
    paths["vis_path"] = paths["out_path"] + "/vis/"
    paths["r20_path"] = paths["out_path"] + "/r20/"
    paths["data_path"] = "data/"
    paths["bvms_path"] = "bvms/"
    paths["covars_path"] = "covars/"

    svae=SVAE()

    if sys.argv[2] == "new":
        ALPHA = metricsConfig['ALPHA']
        q = len(ALPHA)
        METRIC = "val_loss"
        if not os.path.exists(paths["out_path"][:-1]):
            make_dirs(paths)
            run_train(svae)
        else:
            if metricsConfig["retrain"]:
                rmtree(paths["out_path"][:-1])
                make_dirs(paths)
                run_train(svae)
            else:
                raise ValueError("Model with specified hyperparams already found, load it or change \"retrain\" to true in the config!")

    elif sys.argv[2] == "load":
        svae.load_model(name, paths["out_path"])
        # est_svae = tf.keras.estimator.model_to_estimator(keras_model=svae)
else:
    raise ValueError("Must provide model option: load or new")

if len(sys.argv) > 3:

    if "g" in sys.argv[3]:
        N = 216000
        fn = paths["out_path"] + 'gen_{}'.format(name)

        with open(fn, 'wb') as f:
            for seqs in svae.generate(N):
                writeSeqs(f, seqs)

    if "m" in sys.argv[3]:
        hammin = Hams.Hams(seqConfig, visConfig, metricsConfig, paths, name)
        covar = Covars.Covars(seqConfig, visConfig, metricsConfig, paths, name)
        hom = r20.r20(seqConfig, visConfig, metricsConfig, paths, name)
        
        if metricsConfig["which_vis"]["hams"]:
            hammin.make_hams()
        if metricsConfig["which_vis"]["covars"]:
            covar.make_covars()
        if metricsConfig["which_vis"]["homs"]:
            hom.make_r20()

# else:
#     raise ValueError("Must provide use case: g for generate, m for metrics")


