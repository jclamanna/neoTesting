import yaml
import os

with open("config.yml") as file:
    config = yaml.load(file, Loader=yaml.FullLoader)

    epochs = [100, 1000, 2000, 4000]
    latent = [4,5,6,7]
    batch = [200, 54000]
    with open("jobCommands", "w") as j:
        for e in epochs:
            config["hyperParams"]["epochs"] = e
            for l in latent:
                config["hyperParams"]["latent_dim"] = l
                for b in batch:
                    config["hyperParams"]["batch_size"] = b
                    name = "E" + str(e)+"_B" + str(b) + "_L" + str(l) + "_F95_"
                    with open("configs/"+name+ "config.yml", "w") as file:
                        yaml.dump(config, file)
                    with open("jobs/"+name+"job.sh", "w") as f:
                        f.write("#!/bin/bash\n\n"+
                            "export PATH = /home/tun60633/anaconda3/bin: $PATH\n"+
                            "source activate erf\n"+
                            "cd /home/tun60633/NewSetup\n"+
                            "module load cuda\n"
                        )
                        f.write("python controller.py configs/" + name + "config.yml new gm")
                        j.write("qsub jobs/"+name+"job.sh -q gpu -l walltime=8:00:00\n")



        
